"""GDN context parallelism — Plan A: all-gather projected inputs, replicate the scan on
the full sequence, extract this rank's zig-zag-local output.

This is the rtp-llm production approach (qwen3_next.py::_forward_cp_prefill) and is
**compatible with fa4_ring's zig-zag sharding** — both the full-attn ring layers and the
GDN layers use one sharding. GDN is O(N) (cheap vs O(N^2) attention), so replicating its
scan across ranks is an acceptable trade for simplicity + correctness; the big CP win is
on the quadratic full-attn layers (fa4_ring).

The ``scan_fn`` is injected so this wrapper stays agnostic to GDN internals:
  * CPU test  -> a torch wrapper around :func:`fa4_ring.gdn_cp.reference.gdn_reference`.
  * GPU       -> conv1d + fused_gdn_gating + chunk_gated_delta_rule (see build_gpu_scan_fn
                 docstring), exactly as rtp does it.

``_gdn_cp_core`` is the pure (no-dist) part so the gather/restore/extract logic is
unit-testable in a single process.
"""

from typing import Callable, Optional

import torch
import torch.distributed as dist

from .metadata import GdnCpMeta

# scan_fn(natural_packed[total, F], full_cu_seqlens[num_seq+1]) -> out[total, Hv, Dv]
ScanFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def _gdn_cp_core(packed_full: torch.Tensor, scan_fn: ScanFn, meta: GdnCpMeta) -> torch.Tensor:
    """Pure CP core (no collectives): rank-major gathered packed -> restore natural order
    -> scan -> extract local. ``packed_full``: [S*L, F]. Returns local out [L, Hv, Dv]."""
    natural = packed_full.index_select(0, meta.restore_perm)        # [total, F] natural order
    full_out = scan_fn(natural, meta.full_cu_seqlens)               # [total, Hv, Dv]
    local_out = full_out.index_select(0, meta.extract_idx)          # [L, Hv, Dv] local order
    if not bool(meta.valid_mask.all()):
        local_out = local_out * meta.valid_mask[:, None, None].to(local_out.dtype)
    return local_out


def gdn_cp_allgather(
    local_packed: torch.Tensor,   # [L, F] this rank's projected GDN inputs (zig-zag local)
    scan_fn: ScanFn,
    meta: GdnCpMeta,
    *,
    group: Optional["dist.ProcessGroup"] = None,
) -> torch.Tensor:
    """All-gather + replicate-scan + extract. Returns this rank's local output [L, Hv, Dv]."""
    world = dist.get_world_size(group) if group is not None else dist.get_world_size()
    gathered = [torch.empty_like(local_packed) for _ in range(world)]
    dist.all_gather(gathered, local_packed.contiguous(), group=group)
    packed_full = torch.cat(gathered, dim=0)        # [world*L, F] rank-major
    return _gdn_cp_core(packed_full, scan_fn, meta)


def build_gpu_scan_fn(gdn_module, attn_inputs, kv_cache=None):
    """Documentation helper (GPU): build a ``scan_fn`` that runs the real GDN math on the
    gathered full sequence, mirroring rtp qwen3_next.py::_forward_cp_prefill:

        def scan_fn(natural_packed, full_cu):
            mixed_qkv, b, a = split(natural_packed, [qkv_dim, b_dim, a_dim], dim=-1)
            mixed_qkv = causal_conv1d_fn(mixed_qkv.T, gdn.conv_weights, ...,
                                         query_start_loc=full_cu, ...).T
            g, beta = fused_gdn_gating(gdn.alog, a, b, gdn.dt_bias)
            q, k, v = split_qkv(mixed_qkv, ...)
            out, _h, _final = chunk_gated_delta_rule(q, k, v, g, beta,
                                                     cu_seqlens=full_cu,
                                                     use_qk_l2norm_in_kernel=True,
                                                     output_final_state=True)
            return out.squeeze(0)

    NOTE: causal_conv1d_fn + chunk_gated_delta_rule are GPU Triton kernels (fla / rtp);
    they are NOT vendored here. Pack ``local_packed = cat([mixed_qkv, b, a], dim=-1)`` so
    the conv runs AFTER the gather (the conv needs the natural-order neighbor tokens)."""
    raise NotImplementedError(
        "GPU scan_fn is wiring-only; see this docstring and rtp qwen3_next.py::_forward_cp_prefill."
    )
