"""CPU unit tests for zig-zag sharding (no GPU needed).

    python tests/test_zigzag.py     # or: pytest tests/test_zigzag.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa4_ring.zigzag import (  # noqa: E402
    get_half_index,
    half_indices,
    zigzag_shard,
    zigzag_shard_varlen,
    zigzag_unshard,
)


def test_shard_unshard_roundtrip():
    for cp_size in (1, 2, 4, 8):
        n = 4 * cp_size * 3  # divisible by 2*cp_size
        x = torch.arange(n).reshape(n, 1).float()
        shards = [zigzag_shard(x, r, cp_size) for r in range(cp_size)]
        recon = zigzag_unshard(shards, cp_size)
        assert torch.equal(recon, x), f"roundtrip failed cp_size={cp_size}"


def test_each_rank_holds_2h_tokens():
    cp_size, n = 4, 64
    h = n // (2 * cp_size)
    for r in range(cp_size):
        s = zigzag_shard(torch.arange(n).reshape(n, 1).float(), r, cp_size)
        assert s.shape[0] == 2 * h


def test_zigzag_pairs_early_and_late():
    # rank r should hold chunk r (early) and chunk 2S-1-r (late)
    cp_size, n = 4, 32
    h = n // (2 * cp_size)
    x = torch.arange(n).reshape(n, 1).float()
    for r in range(cp_size):
        s = zigzag_shard(x, r, cp_size).flatten().tolist()
        early = list(range(r * h, (r + 1) * h))
        j = 2 * cp_size - 1 - r
        late = list(range(j * h, (j + 1) * h))
        assert s == early + late, f"rank {r}: {s} != {early + late}"


def test_half_indices():
    first, second = half_indices(8, torch.device("cpu"))
    assert first.tolist() == [0, 1, 2, 3]
    assert second.tolist() == [4, 5, 6, 7]


def test_get_half_index_varlen():
    # two local sequences of length 4 and 8 packed -> [0,4,12]
    cu = torch.tensor([0, 4, 12], dtype=torch.int32)
    front = get_half_index(cu, front=True)
    back = get_half_index(cu, front=False)
    # seq0 [0,4): mid 2 -> front [0,1], back [2,3]; seq1 [4,12): mid 8 -> front [4..7], back [8..11]
    assert front.tolist() == [0, 1, 4, 5, 6, 7]
    assert back.tolist() == [2, 3, 8, 9, 10, 11]
    # half cu describes the packed half tensor
    assert (cu // 2).tolist() == [0, 2, 6]


def test_zigzag_shard_varlen():
    cp_size = 2
    # two full sequences of length 8 and 16 (each divisible by 2*cp_size=4)
    full_cu = torch.tensor([0, 8, 24], dtype=torch.int32)
    x = torch.arange(24).reshape(24, 1).float()
    for rank in range(cp_size):
        local, local_cu = zigzag_shard_varlen(x, full_cu, rank, cp_size)
        # seq0 local len = 8/2=4, seq1 local len = 16/2=8 -> local_cu [0,4,12]
        assert local_cu.tolist() == [0, 4, 12]
        assert local.shape[0] == 12


if __name__ == "__main__":
    test_shard_unshard_roundtrip()
    test_each_rank_holds_2h_tokens()
    test_zigzag_pairs_early_and_late()
    test_half_indices()
    test_get_half_index_varlen()
    test_zigzag_shard_varlen()
    print("zigzag unit tests: ALL PASS")
