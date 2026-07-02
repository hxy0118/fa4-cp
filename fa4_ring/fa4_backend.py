"""Per-chunk attention via Dao FlashAttention-4 (CuTe DSL), forward-only.

We call the lower-level ``_flash_attn_fwd`` rather than the public autograd wrapper
``flash_attn_varlen_func`` for two reasons:

1. Only ``_flash_attn_fwd`` exposes the FP8 ``q_descale/k_descale/v_descale`` knobs
   (the public wrappers drop them). Ring inference is forward-only, so we lose nothing.
2. It accepts ``cu_seqlens`` varlen inputs and a pre-allocated ``lse``.

Calling convention mirrors the verified vllm CP ring (internal-rtp/vllm context_parallel
ring.py): **3D varlen** ``[total, H, D]`` + ``cu_seqlens_q/k`` + ``max_seqlen_q/k`` — NOT
4D batched. For the single-sequence-per-rank case we build ``cu_seqlens = [0, T]``.

Important FA4 contract details (verified against flash_attn/cute/interface.py):
  * ``_flash_attn_fwd`` returns a **4-tuple** ``(out, lse, p, row_max)``; ``p`` and
    ``row_max`` are only non-None on the qv / sparse-MLA path (never used here).
  * varlen (no-qv) LSE is returned heads-major ``[nheads, total_q]``; we transpose to
    ``[total_q, nheads]`` for FlashInfer's ``merge_state`` convention.
  * LSE is **natural log** (the bwd path multiplies by ``log2_e`` to get base-2). NOTE:
    FlashInfer's ``cascade`` mergers expect **base-2** LSE, so :mod:`fa4_ring.merge`
    scales by ``log2(e)`` before every FlashInfer call (do not pre-convert here).
  * FP8 inputs produce a **bf16** output; FP8 descales must be float32 CUDA tensors of
    shape ``(batch_size, num_head_kv) == (1, H_kv)`` for all three of q/k/v_descale.
"""

import math
from typing import Optional, Tuple

import torch

try:
    from flash_attn.cute.interface import _flash_attn_fwd as _fa4_fwd

    _HAS_FA4 = True
    _FA4_IMPORT_ERROR = None
except Exception as e:  # pragma: no cover - depends on flash-attn-4 install
    _fa4_fwd = None
    _HAS_FA4 = False
    _FA4_IMPORT_ERROR = e


def has_fa4() -> bool:
    return _HAS_FA4


def _norm_descale(d: Optional[torch.Tensor], h_kv: int) -> Optional[torch.Tensor]:
    """Normalize an FP8 descale to FA4's required shape (1, H_kv) float32, contiguous."""
    if d is None:
        return None
    d = d.to(torch.float32)
    if d.dim() == 0:  # scalar -> broadcast to all kv heads
        d = d.reshape(1, 1).expand(1, h_kv)
    elif d.dim() == 1:  # [H_kv] -> [1, H_kv]
        d = d.unsqueeze(0)
    # else: assume already [1, H_kv]
    return d.contiguous()


def fa4_attn(
    q: torch.Tensor,  # [T_q, H, D]      (varlen-packed)
    k: torch.Tensor,  # [T_k, H_kv, D]
    v: torch.Tensor,  # [T_k, H_kv, D]
    *,
    causal: bool,
    softmax_scale: Optional[float] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Varlen FA4 forward.

    ``cu_seqlens_q/k`` (int32 ``[num_seq+1]``) + ``max_seqlen_q/k`` describe the packed
    sequences; if omitted they default to a single sequence ``[0, T]`` (single-seq case).

    Returns
    -------
    out : [T_q, H, D]   (bf16 for fp8 inputs, else the input float dtype)
    lse : [T_q, H]      (float32, natural log) — FlashInfer-merge-compatible layout
    """
    if not _HAS_FA4:
        raise RuntimeError(
            f"flash-attn-4 not importable ({_FA4_IMPORT_ERROR}); "
            f"`pip install flash-attn-4` on a Hopper/Blackwell GPU."
        )
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    t_q, t_k = q.shape[0], k.shape[0]
    if cu_seqlens_q is None:
        cu_seqlens_q = torch.tensor([0, t_q], device=q.device, dtype=torch.int32)
        max_seqlen_q = t_q
    if cu_seqlens_k is None:
        cu_seqlens_k = torch.tensor([0, t_k], device=q.device, dtype=torch.int32)
        max_seqlen_k = t_k

    h_kv = k.shape[1]
    qd = _norm_descale(q_descale, h_kv)
    kd = _norm_descale(k_descale, h_kv)
    vd = _norm_descale(v_descale, h_kv)

    # 3D varlen calling convention (matches the verified vllm CP ring). _flash_attn_fwd
    # returns a 4-tuple (out, lse, p, row_max); p/row_max are MLA-only and None here.
    out, lse, _p, _row_max = _fa4_fwd(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=True,
        q_descale=qd,
        k_descale=kd,
        v_descale=vd,
    )
    # out: [T_q, H, D]; lse: [H, T_q] -> [T_q, H]
    lse = lse.transpose(0, 1).contiguous()
    return out, lse
