# fa4_ring — FP8-capable ring attention on FlashAttention-4

Context-parallel **causal** attention that shards a long sequence across `cp_size`
GPUs (zig-zag load balanced) and computes per-chunk attention with **Dao
FlashAttention-4** (CuTe DSL). Unlike FlashInfer's `parallel_attention` (bf16-only,
no causal) and rtp-llm's CP ring (bf16-only), this path keeps **FP8 KV** end-to-end by
routing through FA4's `_flash_attn_fwd` descale knobs.

> Status: **v1, forward-only (inference).** Supports single-sequence and **packed varlen
> (multi-sequence)** per rank. The ring algorithm (zig-zag + 3-pattern + online-softmax
> merge, single & varlen) is **CPU-validated against dense causal attention** (err ~1e-7,
> incl. GQA/MQA); the FA4 kernel call + NCCL multi-GPU + FP8 paths require a
> Hopper/Blackwell box to validate. FA4 itself is beta (`flash-attn-4 4.0.0bN`).
>
> ⚠️ **Qwen3.5 note:** Qwen3.5 is `qwen3_next_vl` = **3:1 linear-attn(GDN) : full-attn**.
> This module rings the **full-attention layers only** (≈15/60). The **GDN/linear-attn
> layers cannot ring** (they are recurrences) and need a separate CP scheme (all-gather
> or chunked-scan state hand-off). So fa4_ring is necessary-but-not-sufficient for
> end-to-end Qwen3.5 prefill-CP. See "Qwen3.5 / CP support" below.

## Why ring CP does NOT blow up KV memory
Ring CP shards KV by **sequence** (each rank stores only its `1/cp_size` slice). During
attention, KV chunks **rotate** through ranks one at a time, double-buffered — a rank
holds its own shard + at most 2 transient remote buffers. Peak KV ≈ `3/cp_size` of the
full sequence, so CP *reduces* per-GPU KV pressure. (Only the all-gather CP variant
materializes the full KV; this module uses ring rotation, not all-gather.)

## Install
```bash
pip install "flash-attn-4[cu13]"     # Hopper/Blackwell; provides flash_attn.cute
pip install flashinfer-python         # cascade.merge_state / merge_state_in_place / merge_states
pip install -e .
```

## Usage
```python
import torch, torch.distributed as dist
from fa4_ring import RingConfig, ring_flash_attn, zigzag_shard, zigzag_unshard

dist.init_process_group("nccl"); rank, world = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)

# full q/k/v: [N, H, D] / [N, H_kv, D]; shard zig-zag to this rank -> [2h, ...]
lq, lk, lv = (zigzag_shard(t, rank, world) for t in (q, k, v))

cfg = RingConfig(cp_size=world, cp_rank=rank, causal=True, merge_mode="incremental")
local_out = ring_flash_attn(lq, lk, lv, cfg)   # [2h, H, D]
```

### Packed varlen (multiple sequences per rank)
```python
from fa4_ring import ring_flash_attn
from fa4_ring.zigzag import zigzag_shard_varlen

# full_cu = [0, N0, N0+N1, ...] FULL (unsharded) sequence boundaries
lq, local_cu = zigzag_shard_varlen(q, full_cu, rank, world)   # each seq zig-zag sharded
lk, _ = zigzag_shard_varlen(k, full_cu, rank, world)
lv, _ = zigzag_shard_varlen(v, full_cu, rank, world)
max_local = int((local_cu[1:] - local_cu[:-1]).max())
out = ring_flash_attn(lq, lk, lv, cfg, cu_seqlens=local_cu, max_seqlen=max_local)
```

### FP8 KV (Blackwell)
Pass FP8 (`e4m3`) `q/k/v` shards plus per-tensor descales (all-gather the per-rank
`k`/`v` descales once and index by source rank):
```python
from fa4_ring import Descales
descales = Descales(q=q_scale, k=k_scales_per_rank, v=v_scales_per_rank)
local_out = ring_flash_attn(lq_fp8, lk_fp8, lv_fp8, cfg, descales=descales)
```

## Mergers (both included; default = incremental)
| mode | how | memory | comm overlap | when faster |
|---|---|---|---|---|
| `incremental` (default) | `merge_state_in_place` once per round | O(1) | yes (merge hidden behind next rotation) | usually, esp. comm-bound |
| `batched` | stash partials, one `merge_states` | O(cp_size) | no | fast NVLink / compute-bound |

It is hardware-dependent — run `bench/bench_merge.py` to pick on your box.

## Testing
```bash
# CPU: validates the algorithm (zig-zag + 3-pattern + merge) — no GPU needed
python tests/test_zigzag.py
python tests/test_algorithm_cpu.py

# GPU: ring == dense causal attention (needs >=2 Hopper/Blackwell GPUs + flash-attn-4)
torchrun --nproc_per_node=4 tests/test_ring_correctness.py
torchrun --nproc_per_node=4 tests/test_ring_correctness.py --merge batched
torchrun --nproc_per_node=8 bench/bench_merge.py --seqlen 32768
```

## Design / algorithm
Faithful port of rtp-llm's verified `PCPAll2AllAttnOp` zig-zag ring onto FA4:
- sequence split into `2*cp_size` chunks; rank `i` owns chunk `i` (early) + `2S-1-i` (late).
- step 0: local causal. step `s>rank`: late-Q attends full received KV. step
  `0<s<=rank`: all-Q attends early-half of received KV. Online-softmax merge each step.
- KV rotation is **chain-forward** (send current K/V to `rank+1`, recv from `rank-1`
  into fresh buffers), overlapped with compute and synchronized via the NCCL work
  handles' `req.wait()` (not CUDA-stream events — `batch_isend_irecv` runs on NCCL's own
  stream, so a side-stream event would not track data arrival).

## Qwen3.5 / CP support (important)
- **Mainline vLLM / FlashInfer / rtp-llm rings are all bf16** (no FP8 KV). The ring
  prefill-CP wired in `internal-rtp/vllm` is for **MLA / DeepSeek-v2** only
  (`mla/common.py` → `context_parallel/ring.py`); **Qwen3.5 (`qwen3_next_vl`) has no CP
  wiring** there (`supports_pcp` defaults False). Full Qwen3.5 prefill-CP existed only on
  an abandoned branch. So "vLLM supports prefill-CP" = yes for DeepSeek, **not** Qwen3.5.
- For **Qwen3.5** specifically you need TWO CP schemes: this ring for the full-attn layers,
  **plus** a linear-attn (GDN) CP for the other ~45/60 layers (rtp does this via zig-zag
  position extract + all-gather + causal-conv1d state metadata). fa4_ring covers the first.

## Known limitations (v1)
- causal only. Supports single-seq and packed varlen; **no paged KV** yet (rotated KV is
  contiguous ragged).
- forward-only (inference). No backward (we call FA4's `_flash_attn_fwd`).
- FP8 path is wired but unverified on HW; FA4 FP8-KV is Blackwell-first (on Hopper sm90
  it is a known gap — FlashInfer issue #3327).
- requires `flashinfer-python` for the mergers (imported lazily; only the `cascade` ops
  are used). The package imports fine without it — you only need it at merge time.
