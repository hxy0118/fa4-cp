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

Communication (no manual CUDA streams/events):
the KV chains one hop per step (send to ``rank+1``, recv from ``rank-1``) into
**freshly allocated** receive buffers. The block used at step ``s`` must originate on
``src = (rank - s) % S``, so we ``req.wait()`` + swap to the received block *before*
computing step ``s`` (computing against the pre-rotation block was an off-by-one). To
keep the transfer overlapped, step ``s`` prefetches step ``s+1``'s block and runs its
attention while that hop is in flight. NCCL's ``batch_isend_irecv`` completion is
tracked via the returned work handles — recording a CUDA event on a side stream the
transfer never ran on would NOT track data arrival.

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
from .zigzag import get_half_index


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
    local_q: torch.Tensor,  # [T, H, D]   (zig-zag local shard; T = sum of 2*h_j)
    local_k: torch.Tensor,  # [T, H_kv, D]
    local_v: torch.Tensor,  # [T, H_kv, D]
    cfg: RingConfig,
    *,
    group: Optional["dist.ProcessGroup"] = None,
    descales: Optional[Descales] = None,
    cu_seqlens: Optional[torch.Tensor] = None,  # local int32 [num_seq+1]; None -> single seq
    max_seqlen: Optional[int] = None,
) -> torch.Tensor:
    """Run causal ring attention; returns this rank's local output ``[T, H, D]``.

    Single sequence per rank: leave ``cu_seqlens=None`` (built as ``[0, T]``).
    Multiple packed sequences: pass the LOCAL ``cu_seqlens`` (each entry already the
    zig-zag ``2*h_j`` shard length) and ``max_seqlen`` (max local shard length).
    """
    S = cfg.cp_size
    rank = cfg.cp_rank
    scale = cfg.softmax_scale
    dev = local_q.device
    total = local_q.shape[0]

    if cu_seqlens is None:
        cu_seqlens = torch.tensor([0, total], device=dev, dtype=torch.int32)
        max_seqlen = total
    assert max_seqlen is not None, "max_seqlen required when cu_seqlens is given"

    front_idx = get_half_index(cu_seqlens, front=True)   # early-half rows (-> half KV)
    back_idx = get_half_index(cu_seqlens, front=False)    # late-half rows  (-> half Q)
    half_cu = cu_seqlens // 2
    half_max = max_seqlen // 2

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
        cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
        q_descale=qd, k_descale=kd(rank), v_descale=vd(rank),
    )
    merger.add(out0, lse0)

    if S == 1:
        return merger.result()

    # ---- ring rotation (chain forward, compute overlapped with the NEXT hop) ----
    g_send = _global_rank(group, (rank + 1) % S)
    g_recv = _global_rank(group, (rank - 1) % S)

    def _rotate(cur_k, cur_v):
        # Forward the current K/V one hop (send to rank+1, recv from rank-1) into FRESH
        # buffers. Returns the receive buffers and the in-flight NCCL work handles.
        nk = torch.empty_like(cur_k)
        nv = torch.empty_like(cur_v)
        rq = dist.batch_isend_irecv([
            dist.P2POp(dist.isend, cur_k, g_send, group=group),
            dist.P2POp(dist.irecv, nk, g_recv, group=group),
            dist.P2POp(dist.isend, cur_v, g_send, group=group),
            dist.P2POp(dist.irecv, nv, g_recv, group=group),
        ])
        return nk, nv, rq

    # The KV block held at loop-iteration ``step`` must be the one that originated on
    # ``src = (rank - step) % S`` (it has chained ``step`` hops around the ring to us).
    # Step 0 (own KV, src == rank) is already merged above, so prime the pipeline by
    # fetching step-1's block first; there is nothing to overlap that first hop with.
    # Thereafter, prefetch step+1's block so its transfer overlaps step's attention.
    k, v = local_k, local_v
    next_k, next_v, reqs = _rotate(k, v)
    for step in range(1, S):
        for req in reqs:
            req.wait()
        k, v = next_k, next_v  # now originates from src = (rank - step)
        if step + 1 < S:
            # Prefetch the next hop; sends the current k/v (read-only here) onward while
            # this step's attention runs. Kept alive until the wait() at the next iter top.
            next_k, next_v, reqs = _rotate(k, v)

        src = (rank - step) % S
        if step <= rank:
            # all local Q attend the first half (early) of the received KV
            k_half = k.index_select(0, front_idx).contiguous()
            v_half = v.index_select(0, front_idx).contiguous()
            o, l = fa4_attn(
                local_q, k_half, v_half,
                causal=False, softmax_scale=scale,
                cu_seqlens_q=cu_seqlens, cu_seqlens_k=half_cu,
                max_seqlen_q=max_seqlen, max_seqlen_k=half_max,
                q_descale=qd, k_descale=kd(src), v_descale=vd(src),
            )
            merger.add(o, l)
        else:
            # late-Q (second half) attends the full received KV
            q_sec = local_q.index_select(0, back_idx).contiguous()
            o, l = fa4_attn(
                q_sec, k, v,
                causal=False, softmax_scale=scale,
                cu_seqlens_q=half_cu, cu_seqlens_k=cu_seqlens,
                max_seqlen_q=half_max, max_seqlen_k=max_seqlen,
                q_descale=qd, k_descale=kd(src), v_descale=vd(src),
            )
            out_full = local_q.new_zeros((total, o.shape[1], o.shape[2]), dtype=o.dtype)
            lse_full = torch.full((total, o.shape[1]), float("-inf"),
                                  device=dev, dtype=torch.float32)
            out_full.index_copy_(0, back_idx, o)
            lse_full.index_copy_(0, back_idx, l)
            merger.add(out_full, lse_full)

    return merger.result()
