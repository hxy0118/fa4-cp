"""CPU unit tests for zig-zag sharding (no GPU needed).

    python tests/test_zigzag.py     # or: pytest tests/test_zigzag.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa4_ring.zigzag import half_indices, zigzag_shard, zigzag_unshard  # noqa: E402


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


if __name__ == "__main__":
    test_shard_unshard_roundtrip()
    test_each_rank_holds_2h_tokens()
    test_zigzag_pairs_early_and_late()
    test_half_indices()
    print("zigzag unit tests: ALL PASS")
