"""CPU validation of GDN context parallelism (Plan A: all-gather + replicate scan).

Validates the parts that can actually be wrong without a GPU:
  1. metadata uses the SAME zig-zag sharding as fa4_ring (so ring + GDN share one shard);
  2. all-gather + restore reconstructs natural token order;
  3. end-to-end: per-rank CP outputs stitched == single-device full GDN scan;
  4. the torch GDN reference is causal & deterministic.

The all-gather scan is correct by construction; the real (GPU) chunk_gated_delta_rule
kernel slots into the same scan_fn role. Run:  python tests/test_gdn_cp_cpu.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa4_ring.gdn_cp.allgather_cp import _gdn_cp_core  # noqa: E402
from fa4_ring.gdn_cp.metadata import build_gdn_cp_metadata  # noqa: E402
from fa4_ring.gdn_cp.reference import gdn_reference  # noqa: E402
from fa4_ring.zigzag import zigzag_shard  # noqa: E402


def _make_scan_fn(Hk, Dk, Hv, Dv):
    def scan(packed, cu):
        i = 0
        q = packed[:, i:i + Hk * Dk].reshape(-1, Hk, Dk); i += Hk * Dk
        k = packed[:, i:i + Hk * Dk].reshape(-1, Hk, Dk); i += Hk * Dk
        v = packed[:, i:i + Hv * Dv].reshape(-1, Hv, Dv); i += Hv * Dv
        g = packed[:, i:i + Hv]; i += Hv
        beta = packed[:, i:i + Hv]; i += Hv
        return gdn_reference(q, k, v, g, beta, cu.tolist())
    return scan


def _pack(q, k, v, g, beta):
    T = q.shape[0]
    return torch.cat([q.reshape(T, -1), k.reshape(T, -1), v.reshape(T, -1), g, beta], dim=-1)


def test_metadata_matches_zigzag_shard():
    """extract_idx (natural positions of a rank's local tokens) must equal fa4_ring's
    zigzag_shard — i.e. GDN CP and the full-attn ring use the identical sharding."""
    for cp_size in (2, 4, 8):
        N = 2 * cp_size * 3
        pos = torch.arange(N).reshape(N, 1).float()
        for r in range(cp_size):
            meta = build_gdn_cp_metadata([N], cp_size, r)
            ring_shard = zigzag_shard(pos, r, cp_size).flatten().long()
            assert torch.equal(meta.extract_idx, ring_shard), \
                f"GDN extract_idx != zigzag_shard at cp={cp_size} rank={r}"
    print("  metadata matches fa4_ring zigzag sharding: OK")


def test_restore_reconstructs_natural():
    """gather(per-rank local) then restore_perm == natural order (single & varlen)."""
    cases = [([48], 4), ([32, 16, 48], 4), ([64, 64], 8)]
    for seq_lens, cp in cases:
        total = sum(seq_lens)
        F = torch.randn(total, 5)
        metas = [build_gdn_cp_metadata(seq_lens, cp, r) for r in range(cp)]
        # each rank's local packed = its zig-zag tokens of F, in local order
        local = [F.index_select(0, m.extract_idx) for m in metas]
        gathered = torch.cat(local, dim=0)  # rank-major
        natural = gathered.index_select(0, metas[0].restore_perm)
        assert torch.allclose(natural, F), f"restore failed for {seq_lens} cp={cp}"
    print("  all-gather + restore reconstructs natural order: OK")


def test_gdn_cp_equals_single_device():
    """Per-rank Plan-A CP outputs, stitched, equal the single-device full GDN scan."""
    cases = [
        # (seq_lens, Hk, Hv, Dk, Dv, cp_size)
        ([48], 2, 2, 8, 8, 2),
        ([32, 16, 48], 4, 8, 16, 16, 4),   # GQA (Hv=2*Hk per the 16/64 Qwen3.5 ratio shape)
        ([64, 64], 2, 8, 16, 16, 8),       # GQA 4x
    ]
    all_ok = True
    for seq_lens, Hk, Hv, Dk, Dv, cp in cases:
        torch.manual_seed(0)
        total = sum(seq_lens)
        q = torch.randn(total, Hk, Dk)
        k = torch.randn(total, Hk, Dk)
        v = torch.randn(total, Hv, Dv)
        g = -torch.rand(total, Hv) * 0.5          # log-decay in (-0.5, 0): exp(g) in (0.6,1)
        beta = torch.rand(total, Hv)
        cu = [0]
        for n in seq_lens:
            cu.append(cu[-1] + n)

        o_ref = gdn_reference(q, k, v, g, beta, cu)
        F = _pack(q, k, v, g, beta)
        scan_fn = _make_scan_fn(Hk, Dk, Hv, Dv)

        metas = [build_gdn_cp_metadata(seq_lens, cp, r) for r in range(cp)]
        local = [F.index_select(0, m.extract_idx) for m in metas]
        gathered = torch.cat(local, dim=0)  # simulate all_gather (rank-major)

        full_cp = torch.zeros_like(o_ref)
        for r in range(cp):
            local_out = _gdn_cp_core(gathered, scan_fn, metas[r])  # [L, Hv, Dv]
            full_cp[metas[r].extract_idx] = local_out
        err = (full_cp - o_ref).abs().max().item()
        ok = torch.allclose(full_cp, o_ref, atol=1e-4, rtol=1e-4)
        all_ok &= ok
        print(f"  CP==single seqs={seq_lens} Hk={Hk} Hv={Hv} cp={cp}: max_err={err:.2e} {'OK' if ok else 'FAIL'}")
    return all_ok


def test_gdn_reference_causal_and_deterministic():
    torch.manual_seed(1)
    N, Hk, Hv, Dk, Dv = 12, 2, 4, 8, 8
    q = torch.randn(N, Hk, Dk); k = torch.randn(N, Hk, Dk); v = torch.randn(N, Hv, Dv)
    g = -torch.rand(N, Hv) * 0.5; beta = torch.rand(N, Hv)
    o1 = gdn_reference(q, k, v, g, beta, [0, N])
    o2 = gdn_reference(q, k, v, g, beta, [0, N])
    assert torch.equal(o1, o2), "reference not deterministic"
    # perturb the LAST token; outputs at earlier tokens must be unchanged (causality)
    v2 = v.clone(); v2[-1] += 10.0
    o3 = gdn_reference(q, k, v2, g, beta, [0, N])
    assert torch.allclose(o1[:-1], o3[:-1], atol=1e-6), "reference not causal"
    print("  gdn_reference causal & deterministic: OK")


def main():
    print("GDN-CP (Plan A) CPU validation:")
    test_metadata_matches_zigzag_shard()
    test_restore_reconstructs_natural()
    test_gdn_reference_causal_and_deterministic()
    ok = test_gdn_cp_equals_single_device()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
