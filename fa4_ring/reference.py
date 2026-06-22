"""Single-GPU references for validating the ring output.

``full_causal_reference`` computes ordinary (non-sharded) causal attention over the
whole sequence. The test gathers all per-rank shards into the full sequence, runs this
reference, then zig-zag-shards the reference output and compares it to the gathered ring
outputs. Two backends:

  * "torch" : torch.nn.functional.scaled_dot_product_attention (math ground truth,
              runs anywhere, GQA expanded manually).
  * "fa4"   : the same FA4 forward the ring uses, for an apples-to-apples comparison.
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def _expand_kv(k: torch.Tensor, v: torch.Tensor, n_q_heads: int):
    """Expand [T, H_kv, D] -> [T, H_q, D] for GQA/MQA."""
    h_kv = k.shape[1]
    if h_kv == n_q_heads:
        return k, v
    assert n_q_heads % h_kv == 0, f"H_q={n_q_heads} not a multiple of H_kv={h_kv}"
    rep = n_q_heads // h_kv
    return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)


def full_causal_reference(
    q: torch.Tensor,  # [N, H, D]
    k: torch.Tensor,  # [N, H_kv, D]
    v: torch.Tensor,  # [N, H_kv, D]
    softmax_scale: Optional[float] = None,
    backend: str = "torch",
) -> torch.Tensor:
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    if backend == "fa4":
        from .fa4_backend import fa4_attn

        out, _ = fa4_attn(q, k, v, causal=True, softmax_scale=softmax_scale)
        return out

    # torch SDPA ground truth
    kq, vq = _expand_kv(k, v, q.shape[1])
    # [N,H,D] -> [1,H,N,D]
    qt = q.transpose(0, 1).unsqueeze(0).float()
    kt = kq.transpose(0, 1).unsqueeze(0).float()
    vt = vq.transpose(0, 1).unsqueeze(0).float()
    out = F.scaled_dot_product_attention(qt, kt, vt, is_causal=True, scale=softmax_scale)
    # [1,H,N,D] -> [N,H,D]
    return out.squeeze(0).transpose(0, 1).contiguous().to(q.dtype)
