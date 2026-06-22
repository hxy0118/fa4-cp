"""Zig-zag context-parallel sharding for causal load balancing.

A sequence of ``N`` tokens is split into ``2 * cp_size`` equal chunks of size
``h = N / (2 * cp_size)``. Rank ``i`` owns:

  * chunk ``i``                  -> "first half"  (early positions [i*h, (i+1)*h))
  * chunk ``2*cp_size - 1 - i``  -> "second half" (late  positions [(2S-1-i)*h, (2S-i)*h))

stored concatenated as the local ``[2h]`` sequence ``[first_half | second_half]``.

Why zig-zag: a plain contiguous split makes the last rank attend to the whole
sequence (full causal triangle) while rank 0 attends almost nothing. Pairing an
early chunk with a late chunk equalizes the causal work across ranks (~1x each
instead of up to 2x on the tail rank).

Within the local ``[2h]`` layout, ``first_half`` positions are strictly *earlier*
than ``second_half`` positions, so a plain ``causal=True`` attention over the
concatenation is correct for the round-0 (local) step.
"""

from typing import List

import torch


def local_seqlen(n: int, cp_size: int) -> int:
    assert n % (2 * cp_size) == 0, f"seqlen {n} must be divisible by 2*cp_size={2 * cp_size}"
    return n // cp_size


def zigzag_shard(x: torch.Tensor, cp_rank: int, cp_size: int) -> torch.Tensor:
    """Take the full [N, ...] sequence and return this rank's local [2h, ...] shard."""
    n = x.shape[0]
    assert n % (2 * cp_size) == 0, f"seqlen {n} must be divisible by 2*cp_size={2 * cp_size}"
    h = n // (2 * cp_size)
    first = x[cp_rank * h : (cp_rank + 1) * h]
    j = 2 * cp_size - 1 - cp_rank
    second = x[j * h : (j + 1) * h]
    return torch.cat([first, second], dim=0).contiguous()


def zigzag_unshard(local_shards: List[torch.Tensor], cp_size: int) -> torch.Tensor:
    """Inverse of :func:`zigzag_shard`: stitch per-rank [2h, ...] shards back to [N, ...]."""
    assert len(local_shards) == cp_size
    h = local_shards[0].shape[0] // 2
    chunks: List[torch.Tensor] = [None] * (2 * cp_size)  # type: ignore[list-item]
    for i, loc in enumerate(local_shards):
        chunks[i] = loc[:h]
        chunks[2 * cp_size - 1 - i] = loc[h:]
    return torch.cat(chunks, dim=0)


def half_indices(local_len: int, device) -> tuple:
    """Return (first_half_idx, second_half_idx) into the local [2h] sequence.

    first_half  = early KV that late-Q on other ranks still needs (-> half_kv).
    second_half = late Q that needs extra KV from other ranks    (-> half_q).
    """
    h = local_len // 2
    first = torch.arange(0, h, device=device)
    second = torch.arange(h, local_len, device=device)
    return first, second


def get_half_index(cu_seqlens: torch.Tensor, *, front: bool) -> torch.Tensor:
    """Varlen generalization of :func:`half_indices`: long indices of the front (early)
    or back (late) half of EACH packed sequence described by ``cu_seqlens`` (int32
    ``[num_seq+1]``). Each local sequence is ``[first_half | second_half]`` so its
    midpoint is ``(start+end)//2``."""
    device = cu_seqlens.device
    cu = cu_seqlens.tolist()
    parts = []
    for i in range(len(cu) - 1):
        start, end = cu[i], cu[i + 1]
        assert (end - start) % 2 == 0, "each local sequence length must be even (zig-zag)"
        mid = (start + end) // 2
        parts.append(torch.arange(start, mid, device=device) if front
                     else torch.arange(mid, end, device=device))
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.long, device=device)


def zigzag_shard_varlen(x: torch.Tensor, full_cu_seqlens, cp_rank: int, cp_size: int):
    """Shard multiple packed sequences zig-zag for this rank.

    ``x``: [sum(N_j), ...] packed; ``full_cu_seqlens``: int list/tensor [num_seq+1] of the
    FULL (un-sharded) sequence boundaries. Returns (local_packed, local_cu_seqlens) where
    local_cu_seqlens describes this rank's ``2*h_j`` per-sequence shards.
    """
    cu = full_cu_seqlens.tolist() if torch.is_tensor(full_cu_seqlens) else list(full_cu_seqlens)
    shards, local_cu = [], [0]
    for j in range(len(cu) - 1):
        seg = x[cu[j]:cu[j + 1]]
        loc = zigzag_shard(seg, cp_rank, cp_size)
        shards.append(loc)
        local_cu.append(local_cu[-1] + loc.shape[0])
    local = torch.cat(shards, dim=0).contiguous() if shards else x.new_empty((0,) + x.shape[1:])
    return local, torch.tensor(local_cu, dtype=torch.int32, device=x.device)
