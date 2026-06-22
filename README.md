# fa4_ring — FP8-capable ring attention on FlashAttention-4

Context-parallel **causal** attention that shards a long sequence across `cp_size`
GPUs (zig-zag load balanced) and computes per-chunk attention with **Dao
FlashAttention-4** (CuTe DSL). Unlike FlashInfer's `parallel_attention` (bf16-only,
no causal) and rtp-llm's CP ring (bf16-only), this path keeps **FP8 KV** end-to-end by
routing through FA4's `_flash_attn_fwd` descale knobs.

> Status: **v1, forward-only (inference), single-sequence-per-rank.** bf16/fp16 verified
> by CPU algorithm simulation; FP8 + multi-GPU paths require a Hopper/Blackwell box to
> validate (see Testing). FA4 itself is beta (`flash-attn-4 4.0.0bN`).

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

## Known limitations (v1)
- causal only; single sequence per rank (no varlen batch / no paged KV yet).
- forward-only (inference). No backward (we call FA4's `_flash_attn_fwd`).
- FP8 path is wired but unverified on HW; FA4 FP8-KV is Blackwell-first (on Hopper sm90
  it is a known gap — FlashInfer issue #3327).
- requires `flashinfer-python` for the mergers (only the `cascade` ops are used).
