"""Index metadata for GDN (linear-attention) context parallelism, Plan A
(all-gather + replicate scan).

Layout
------
Every CP rank holds the SAME set of sequences, each zig-zag sharded (same sharding as
:mod:`fa4_ring.zigzag`, so the full-attn ring and GDN share one sharding). For a sequence
of length ``N_j``: ``h_j = N_j / (2*cp_size)``; rank ``r`` owns chunk ``r`` (early) and
chunk ``2S-1-r`` (late), stored locally as ``[first_half | second_half]`` (len ``2h_j``).

A rank's local packed tensor is the per-sequence shards concatenated:
``[seq0_shard | seq1_shard | ...]`` of total length ``L = sum_j 2h_j``.

After ``all_gather`` we get a **rank-major** packing ``[rank0_local | rank1_local | ...]``
of length ``S*L``. To run the GDN recurrence we must restore **natural token order**
(per sequence, 0..N_j-1, sequences concatenated), length ``total = sum_j N_j == S*L``.

This module builds:
  * ``restore_perm`` [total]   : ``gathered[restore_perm]`` -> natural order.
  * ``extract_idx``  [L]       : ``full_natural_out[extract_idx]`` -> this rank's local
                                  output (in local ``[first|second]`` order).
  * ``full_cu_seqlens`` [num_seq+1] : natural-order sequence boundaries (for the scan).

The all-gather path is correct **by construction** (gather full -> same scan -> extract);
the only thing that can be wrong is this index math, which is fully CPU-unit-tested.
"""

from dataclasses import dataclass
from typing import List, Sequence

import torch


@dataclass
class GdnCpMeta:
    restore_perm: torch.Tensor      # [total] long
    extract_idx: torch.Tensor       # [L] long
    valid_mask: torch.Tensor        # [L] bool (all True for the even / no-pad case)
    full_cu_seqlens: torch.Tensor   # [num_seq+1] int32 (natural order)
    local_len: int                  # L
    total_len: int                  # total == S*L


def build_gdn_cp_metadata(
    full_seq_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    device="cpu",
) -> GdnCpMeta:
    """Build GDN-CP index metadata for this rank. ``full_seq_lens`` are the FULL
    (un-sharded) sequence lengths; each must be divisible by ``2*cp_size``."""
    S = cp_size
    local_off = [0]
    natural_off = [0]
    for n in full_seq_lens:
        assert n % (2 * S) == 0, f"seq len {n} must be divisible by 2*cp_size={2 * S}"
        h = n // (2 * S)
        local_off.append(local_off[-1] + 2 * h)
        natural_off.append(natural_off[-1] + n)
    L = local_off[-1]
    total = natural_off[-1]
    assert total == S * L

    def natural_in_seq(r: int, p: int, h: int) -> int:
        # local pos p in [0,2h): first half -> chunk r (early); second half -> chunk 2S-1-r (late)
        return r * h + p if p < h else (2 * S - 1 - r) * h + (p - h)

    restore = [0] * total
    for r in range(S):
        for j, n in enumerate(full_seq_lens):
            h = n // (2 * S)
            for p in range(2 * h):
                g_slot = r * L + local_off[j] + p
                nat = natural_off[j] + natural_in_seq(r, p, h)
                restore[nat] = g_slot

    extract: List[int] = [0] * L
    for j, n in enumerate(full_seq_lens):
        h = n // (2 * S)
        for p in range(2 * h):
            extract[local_off[j] + p] = natural_off[j] + natural_in_seq(cp_rank, p, h)

    return GdnCpMeta(
        restore_perm=torch.tensor(restore, dtype=torch.long, device=device),
        extract_idx=torch.tensor(extract, dtype=torch.long, device=device),
        valid_mask=torch.ones(L, dtype=torch.bool, device=device),
        full_cu_seqlens=torch.tensor(natural_off, dtype=torch.int32, device=device),
        local_len=L,
        total_len=total,
    )
