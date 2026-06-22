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
