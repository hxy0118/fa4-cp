"""FA4 ring attention (causal, zig-zag load balanced).

Algorithm (faithful port of the verified rtp-llm / vllm zig-zag ring onto FA4):

For a sequence sharded zig-zag across ``cp_size`` ranks (see :mod:`zigzag`), each rank
holds ``2h`` local tokens ``[first_half(early) | second_half(late)]`` plus the same
counts of K/V. We do ``cp_size`` steps; in step ``s`` a rank attends its local Q to the
KV that originated on rank ``src = (rank - s) % cp_size`` (which has chained around the
ring to us by then):

  * step 0 (src == rank, local): full causal attention over the local 2h tokens.
  * step ``s <= rank``:  the received KV is *earlier*, so **all local Q** see only the
    **first-half (early) of the remote KV**: non-causal ``attn(q=local_q, k=remote_k[:h])``.
  * step ``s > rank``:   the received KV is *later*, so only the **second-half (late)
    local Q** can see it: non-causal ``attn(q=local_q[h:], k=remote_k)`` merged into the
    late rows.

Communication (verified-correct pattern; no manual CUDA streams/events):
each step forwards the *current* K/V one hop around the ring (send to ``rank+1``, recv
from ``rank-1``) into **freshly allocated** receive buffers, overlapping the in-flight
transfer with this step's attention, then ``req.wait()`` before swapping. NCCL's
``batch_isend_irecv`` completion is tracked via the returned work handles — recording a
CUDA event on a side stream the transfer never ran on would NOT track data arrival.

Memory note: a rank only ever holds its own ``1/cp_size`` KV shard plus one in-flight
receive buffer — never the full KV. Ring CP *reduces* per-GPU KV pressure (it does not
replicate). (Fresh ``empty_like`` buffers per step are reclaimed by the caching
allocator; peak extra KV ~ 1-2 shards.)
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.distributed as dist

from .config import RingConfig
from .fa4_backend import fa4_attn
from .merge import make_merger
from .zigzag import half_indices


@dataclass
class Descales:
    """FP8 descales. ``k``/``v`` are indexed by the *source* rank whose KV is being
    attended (all-gather these once before the loop). Each entry must be a float32 CUDA
    tensor broadcastable to FA4's required ``(1, H_kv)`` (a scalar or ``[H_kv]`` is
    normalized automatically in :func:`fa4_attn`)."""

    q: Optional[torch.Tensor] = None              # local q descale
    k: Optional[List[torch.Tensor]] = None        # k descale per source rank
    v: Optional[List[torch.Tensor]] = None        # v descale per source rank


def _global_rank(group, r: int) -> int:
    if group is None:
        return r
    return dist.get_global_rank(group, r)


def ring_flash_attn(
    local_q: torch.Tensor,  # [2h, H, D]   (zig-zag local shard)
    local_k: torch.Tensor,  # [2h, H_kv, D]
    local_v: torch.Tensor,  # [2h, H_kv, D]
    cfg: RingConfig,
    *,
    group: Optional["dist.ProcessGroup"] = None,
    descales: Optional[Descales] = None,
) -> torch.Tensor:
    """Run causal ring attention; returns this rank's local output ``[2h, H, D]``."""
    S = cfg.cp_size
    rank = cfg.cp_rank
    scale = cfg.softmax_scale
    dev = local_q.device
    local_len = local_q.shape[0]
    h = local_len // 2
    assert local_len % 2 == 0, "local sequence must be even (zig-zag first/second half)"

    first_idx, second_idx = half_indices(local_len, dev)

    qd = descales.q if descales else None

    def kd(src):
        return descales.k[src] if (descales and descales.k) else None

    def vd(src):
        return descales.v[src] if (descales and descales.v) else None

    # The merger's first partial is ALWAYS step 0 (full causal -> finite LSE on every
    # local row). Later steps may contribute -inf LSE on uncovered rows; that is the
    # identity under the online-softmax merge precisely because step 0 seeded every row
    # with a finite value. Do not reorder step 0.
    merger = make_merger(cfg.merge_mode)

    # ---- step 0: local causal ----
    out0, lse0 = fa4_attn(
        local_q, local_k, local_v,
        causal=True, softmax_scale=scale,
        q_descale=qd, k_descale=kd(rank), v_descale=vd(rank),
    )
    merger.add(out0, lse0)

    if S == 1:
        return merger.result()

    # ---- ring rotation (chain forward, overlapped) ----
    g_send = _global_rank(group, (rank + 1) % S)
    g_recv = _global_rank(group, (rank - 1) % S)

    k, v = local_k, local_v
    reqs = None
    for step in range(1, S):
        # Receive the next hop's K/V into FRESH buffers while we compute this step.
        next_k = torch.empty_like(k)
        next_v = torch.empty_like(v)
        reqs = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, k, g_send, group=group),
            dist.P2POp(dist.irecv, next_k, g_recv, group=group),
            dist.P2POp(dist.isend, v, g_send, group=group),
            dist.P2POp(dist.irecv, next_v, g_recv, group=group),
        ])

        src = (rank - step) % S
        if step <= rank:
            # all local Q attend the first half (early) of the received KV
            k_half = k.index_select(0, first_idx).contiguous()
            v_half = v.index_select(0, first_idx).contiguous()
            o, l = fa4_attn(
                local_q, k_half, v_half,
                causal=False, softmax_scale=scale,
                q_descale=qd, k_descale=kd(src), v_descale=vd(src),
            )
            merger.add(o, l)
        else:
            # late-Q (second half) attends the full received KV
            q_sec = local_q.index_select(0, second_idx).contiguous()
            o, l = fa4_attn(
                q_sec, k, v,
                causal=False, softmax_scale=scale,
                q_descale=qd, k_descale=kd(src), v_descale=vd(src),
            )
            out_full = local_q.new_zeros((local_len, o.shape[1], o.shape[2]), dtype=o.dtype)
            lse_full = torch.full((local_len, o.shape[1]), float("-inf"),
                                  device=dev, dtype=torch.float32)
            out_full.index_copy_(0, second_idx, o)
            lse_full.index_copy_(0, second_idx, l)
            merger.add(out_full, lse_full)

        # Block on the transfer, then adopt the received K/V for the next hop.
        for req in reqs:
            req.wait()
        k, v = next_k, next_v
        reqs = None

    return merger.result()
