"""Benchmark the two mergers (incremental vs batched) end-to-end on real GPUs.

The two only differ in how per-round partials are combined; this measures which wins on
*your* hardware (it is comm/overlap dependent, so benchmark before choosing).

Run:
    torchrun --nproc_per_node=8 bench/bench_merge.py --seqlen 32768 --heads 32
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa4_ring import RingConfig, ring_flash_attn, zigzag_shard  # noqa: E402


def bench(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=32768)
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    N, H, Hkv, D = args.seqlen, args.heads, args.kv_heads, args.head_dim
    torch.manual_seed(rank)
    q = torch.randn(N, H, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(N, Hkv, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(N, Hkv, D, device=dev, dtype=torch.bfloat16)
    lq = zigzag_shard(q, rank, world)
    lk = zigzag_shard(k, rank, world)
    lv = zigzag_shard(v, rank, world)

    res = {}
    for mode in ("incremental", "batched"):
        cfg = RingConfig(cp_size=world, cp_rank=rank, causal=True, merge_mode=mode)
        ms = bench(lambda: ring_flash_attn(lq, lk, lv, cfg))
        res[mode] = ms
    if rank == 0:
        print(f"seqlen={N} world={world} H={H} Hkv={Hkv} D={D}")
        for mode, ms in res.items():
            print(f"  {mode:12s}: {ms:.3f} ms/iter")
        best = min(res, key=res.get)
        print(f"  -> faster: {best}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
