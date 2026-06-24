"""Distributed test for GDN-CP Plan A: validates the real gdn_cp_allgather (dist.all_gather
+ restore + scan + extract) against a single-device full GDN scan.

Uses the pure-torch gdn_reference as the injected scan_fn, so it needs NO GPU kernel — it
runs on the gloo backend (CPU) or nccl (GPU). This isolates and validates the distributed
plumbing of Plan A; the real chunk_gated_delta_rule kernel slots into the same scan_fn.

    # CPU, no GPU needed:
    torchrun --nproc_per_node=4 tests/test_gdn_cp_dist.py --backend gloo
    # GPU:
    torchrun --nproc_per_node=4 tests/test_gdn_cp_dist.py --backend nccl
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa4_ring.gdn_cp import build_gdn_cp_metadata, gdn_cp_allgather, gdn_reference  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["gloo", "nccl"], default="gloo")
    ap.add_argument("--seqlen", type=int, default=96)
    ap.add_argument("--num-seqs", type=int, default=2)
    ap.add_argument("--hk", type=int, default=2)
    ap.add_argument("--hv", type=int, default=8)
    ap.add_argument("--dk", type=int, default=16)
    ap.add_argument("--dv", type=int, default=16)
    args = ap.parse_args()

    dist.init_process_group(args.backend)
    rank, world = dist.get_rank(), dist.get_world_size()
    if args.backend == "nccl":
        torch.cuda.set_device(rank)
        dev = torch.device("cuda", rank)
    else:
        dev = torch.device("cpu")

    assert args.seqlen % args.num_seqs == 0
    per = args.seqlen // args.num_seqs
    seq_lens = [per] * args.num_seqs
    Hk, Hv, Dk, Dv = args.hk, args.hv, args.dk, args.dv

    # same full inputs on every rank (seeded)
    torch.manual_seed(1234)
    total = sum(seq_lens)
    q = torch.randn(total, Hk, Dk, device=dev)
    k = torch.randn(total, Hk, Dk, device=dev)
    v = torch.randn(total, Hv, Dv, device=dev)
    g = -torch.rand(total, Hv, device=dev) * 0.5
    beta = torch.rand(total, Hv, device=dev)
    cu = [0]
    for n in seq_lens:
        cu.append(cu[-1] + n)
    F = torch.cat([q.reshape(total, -1), k.reshape(total, -1),
                   v.reshape(total, -1), g, beta], dim=-1)

    def scan_fn(packed, full_cu):
        i = 0
        qq = packed[:, i:i + Hk * Dk].reshape(-1, Hk, Dk); i += Hk * Dk
        kk = packed[:, i:i + Hk * Dk].reshape(-1, Hk, Dk); i += Hk * Dk
        vv = packed[:, i:i + Hv * Dv].reshape(-1, Hv, Dv); i += Hv * Dv
        gg = packed[:, i:i + Hv]; i += Hv
        bb = packed[:, i:i + Hv]; i += Hv
        return gdn_reference(qq.cpu(), kk.cpu(), vv.cpu(), gg.cpu(), bb.cpu(),
                             full_cu.tolist()).to(packed.device)

    meta = build_gdn_cp_metadata(seq_lens, world, rank, device=dev)
    local_packed = F.index_select(0, meta.extract_idx)             # this rank's local inputs
    local_out = gdn_cp_allgather(local_packed, scan_fn, meta)      # [L, Hv, Dv]

    # gather local outputs, stitch, compare on rank 0
    gathered = [torch.empty_like(local_out) for _ in range(world)]
    dist.all_gather(gathered, local_out.contiguous())
    if rank == 0:
        o_ref = gdn_reference(q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), cu).to(dev)
        full_cp = torch.zeros_like(o_ref)
        for r in range(world):
            m = build_gdn_cp_metadata(seq_lens, world, r, device=dev)
            full_cp[m.extract_idx] = gathered[r]
        max_err = (full_cp - o_ref).abs().max().item()
        ok = torch.allclose(full_cp, o_ref, atol=1e-4, rtol=1e-4)
        print(f"[gdn-cp backend={args.backend} world={world} seqs={seq_lens}] "
              f"max_err={max_err:.3e} -> {'PASS' if ok else 'FAIL'}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
