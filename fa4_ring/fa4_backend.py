"""Per-chunk attention via Dao FlashAttention-4 (CuTe DSL), forward-only.

We call the lower-level ``_flash_attn_fwd`` rather than the public autograd wrapper
``flash_attn_varlen_func`` for two reasons:

1. Only ``_flash_attn_fwd`` exposes the FP8 ``q_descale/k_descale/v_descale`` knobs
   (the public wrappers drop them). Ring inference is forward-only, so we lose nothing.
2. It accepts a pre-allocated ``lse`` and a fixed batched layout.

Important FA4 contract details (verified against flash_attn/cute/interface.py):
  * ``_flash_attn_fwd`` returns a **4-tuple** ``(out, lse, p, row_max)``; ``p`` and
    ``row_max`` are only non-None on the qv / sparse-MLA path (never used here).
  * For the non-qv path the LSE is returned heads-major ``[..., nheads, seqlen]``; we
    transpose to ``[seqlen, nheads]`` for FlashInfer's ``merge_state`` convention.
  * LSE is **natural log** (the bwd path multiplies by ``log2_e`` to get base-2), which
    is exactly what FlashInfer's mergers expect — no base conversion needed.
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
    q: torch.Tensor,  # [T_q, H, D]
    k: torch.Tensor,  # [T_k, H_kv, D]
    v: torch.Tensor,  # [T_k, H_kv, D]
    *,
    causal: bool,
    softmax_scale: Optional[float] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single-sequence FA4 forward.

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

    h_kv = k.shape[1]
    qd = _norm_descale(q_descale, h_kv)
    kd = _norm_descale(k_descale, h_kv)
    vd = _norm_descale(v_descale, h_kv)

    # FA4 accepts (batch, seqlen, H, D). Use batch=1 for the single chunk; this avoids
    # building cu_seqlens for the common single-sequence-per-rank case.
    # _flash_attn_fwd returns a 4-tuple (out, lse, p, row_max); p/row_max are MLA-only.
    out, lse, _p, _row_max = _fa4_fwd(
        q.unsqueeze(0),
        k.unsqueeze(0),
        v.unsqueeze(0),
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=True,
        q_descale=qd,
        k_descale=kd,
        v_descale=vd,
    )
    # out: [1, T_q, H, D] -> [T_q, H, D]
    # lse: [1, H, T_q]    -> [T_q, H]
    out = out.squeeze(0)
    lse = lse.squeeze(0).transpose(0, 1).contiguous()
    return out, lse
