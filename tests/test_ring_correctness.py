"""Multi-GPU correctness test: FA4 ring == single-GPU full causal attention.

Requires: >=2 CUDA GPUs (Hopper/Blackwell), flash-attn-4, flashinfer, NCCL.

Run:
    torchrun --nproc_per_node=4 tests/test_ring_correctness.py
    torchrun --nproc_per_node=4 tests/test_ring_correctness.py --merge batched
    torchrun --nproc_per_node=4 tests/test_ring_correctness.py --seqlen 8192 --heads 32

Each rank takes its zig-zag shard, runs the ring, then we all-gather the per-rank
outputs, stitch them back, and compare against dense causal attention computed on
rank 0 over the gathered full Q/K/V.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa4_ring import RingConfig, ring_flash_attn, zigzag_shard, zigzag_unshard  # noqa: E402
from fa4_ring.reference import full_causal_reference  # noqa: E402
from fa4_ring.zigzag import zigzag_shard_varlen  # noqa: E402


def _varlen_unshard(gathered, world, num_seqs, per):
    """Reconstruct packed [N, H, D] from per-rank packed shards (num_seqs equal seqs)."""
    local_per_seq = per // world
    out_seqs = []
    for i in range(num_seqs):
        shards = [gathered[r][i * local_per_seq:(i + 1) * local_per_seq] for r in range(world)]
        out_seqs.append(zigzag_unshard(shards, world))
    return torch.cat(out_seqs, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=4)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--merge", choices=["incremental", "batched"], default="incremental")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--ref", choices=["torch", "fa4"], default="fa4")
    ap.add_argument("--num-seqs", type=int, default=1,
                    help=">1 packs N equal sequences and exercises the varlen path")
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    N, H, Hkv, D = args.seqlen, args.heads, args.kv_heads, args.head_dim
    assert N % (2 * world) == 0, "seqlen must be divisible by 2*world"

    # Build the SAME full q/k/v on every rank (seeded) so rank0's reference matches.
    torch.manual_seed(1234)
    q = torch.randn(N, H, D, device=dev, dtype=dtype)
    k = torch.randn(N, Hkv, D, device=dev, dtype=dtype)
    v = torch.randn(N, Hkv, D, device=dev, dtype=dtype)

    cfg = RingConfig(cp_size=world, cp_rank=rank, causal=True, merge_mode=args.merge)

    if args.num_seqs > 1:
        # pack N equal sequences; each is zig-zag sharded independently (block-diagonal)
        assert N % args.num_seqs == 0, "seqlen must divide num-seqs"
        per = N // args.num_seqs
        full_cu = torch.arange(0, N + 1, per, device=dev, dtype=torch.int32)
        lq, local_cu = zigzag_shard_varlen(q, full_cu, rank, world)
        lk, _ = zigzag_shard_varlen(k, full_cu, rank, world)
        lv, _ = zigzag_shard_varlen(v, full_cu, rank, world)
        max_local = int((local_cu[1:] - local_cu[:-1]).max())
        local_out = ring_flash_attn(lq, lk, lv, cfg, cu_seqlens=local_cu, max_seqlen=max_local)
    else:
        lq = zigzag_shard(q, rank, world)
        lk = zigzag_shard(k, rank, world)
        lv = zigzag_shard(v, rank, world)
        local_out = ring_flash_attn(lq, lk, lv, cfg)  # [2h, H, D]

    # gather all per-rank outputs and stitch
    gathered = [torch.empty_like(local_out) for _ in range(world)]
    dist.all_gather(gathered, local_out.contiguous())
    if rank == 0:
        if args.num_seqs > 1:
            # reconstruct + compare each sequence independently (block-diagonal causal)
            per = N // args.num_seqs
            full_ring = _varlen_unshard(gathered, world, args.num_seqs, per).float()
            ref = torch.cat([
                full_causal_reference(q[i * per:(i + 1) * per], k[i * per:(i + 1) * per],
                                      v[i * per:(i + 1) * per], backend=args.ref)
                for i in range(args.num_seqs)
            ], dim=0).float()
        else:
            full_ring = zigzag_unshard(gathered, world).float()
            ref = full_causal_reference(q, k, v, backend=args.ref).float()
        max_err = (full_ring - ref).abs().max().item()
        mean_err = (full_ring - ref).abs().mean().item()
        cos = torch.nn.functional.cosine_similarity(
            full_ring.flatten(), ref.flatten(), dim=0
        ).item()
        tol = 2e-2 if dtype == torch.bfloat16 else 5e-3
        ok = max_err < tol and cos > 0.999
        print(
            f"[merge={args.merge} ref={args.ref} dtype={args.dtype} world={world}] "
            f"max_err={max_err:.4e} mean_err={mean_err:.4e} cos={cos:.6f} "
            f"-> {'PASS' if ok else 'FAIL'}"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
