"""Pure-torch gated-delta-rule (GDN) reference recurrence.

Mirrors the fla fused-recurrent kernel (vllm fla ops/fused_recurrent.py), per (token,
v-head), state ``S`` of shape ``[head_v_dim, head_k_dim]``:

    q,k L2-normalized; q *= scale
    S   = S * exp(g_t)                 # decay (g_t scalar per token per v-head)
    u   = beta_t * (v_t - S @ k_t)     # gated delta correction
    S   = S + outer(u, k_t)            # S += u kᵀ
    o_t = S @ q_t                      # output uses POST-update state

GQA: ``num_v_heads`` is a multiple of ``num_k_heads``; v-head ``hv`` uses k-head
``hv // (Hv//Hk)``.

This is the CPU ground truth for validating the all-gather CP path; on GPU the real
``chunk_gated_delta_rule`` Triton kernel slots into the same (q,k,v,g,beta,cu)->o role.
"""

import math
from typing import Optional, Sequence

import torch


def gdn_reference(
    q: torch.Tensor,      # [T, Hk, Dk]
    k: torch.Tensor,      # [T, Hk, Dk]
    v: torch.Tensor,      # [T, Hv, Dv]
    g: torch.Tensor,      # [T, Hv]  (log-decay; S *= exp(g))
    beta: torch.Tensor,   # [T, Hv]
    cu_seqlens: Sequence[int],
    scale: Optional[float] = None,
    l2norm: bool = True,
) -> torch.Tensor:
    """Returns o: [T, Hv, Dv]. Sequences are scanned independently per cu_seqlens."""
    T, Hk, Dk = q.shape
    Hv, Dv = v.shape[1], v.shape[2]
    assert Hv % Hk == 0, f"num_v_heads {Hv} not a multiple of num_k_heads {Hk}"
    rep = Hv // Hk
    if scale is None:
        scale = 1.0 / math.sqrt(Dk)
    kv_to_k = torch.arange(Hv) // rep  # map v-head -> k-head

    out = torch.zeros(T, Hv, Dv, dtype=torch.float32)
    cu = list(cu_seqlens)
    for si in range(len(cu) - 1):
        s, e = int(cu[si]), int(cu[si + 1])
        S = torch.zeros(Hv, Dv, Dk, dtype=torch.float32)  # per v-head state
        for t in range(s, e):
            qh = q[t, kv_to_k].float()  # [Hv, Dk]
            kh = k[t, kv_to_k].float()  # [Hv, Dk]
            vh = v[t].float()           # [Hv, Dv]
            if l2norm:
                qh = qh / (qh.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
                kh = kh / (kh.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
            qh = qh * scale
            S = S * g[t].float().exp()[:, None, None]              # decay [Hv,1,1]
            Sk = (S * kh[:, None, :]).sum(-1)                      # S@k -> [Hv,Dv]
            u = (vh - Sk) * beta[t].float()[:, None]              # [Hv,Dv]
            S = S + u[:, :, None] * kh[:, None, :]                 # S += outer(u,k)
            out[t] = (S * qh[:, None, :]).sum(-1)                  # S@q -> [Hv,Dv]
    return out
