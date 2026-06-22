"""CPU simulation of the ring algorithm — validates the hard part (zig-zag sharding,
the 3-pattern KV selection, and online-softmax merging) WITHOUT GPU / FA4 / NCCL.

It mirrors ``fa4_ring/ring_attention.py`` step for step, but:
  * per-chunk attention is plain torch (returns (out, lse));
  * the merge is a pure-torch online softmax;
  * the cp_size ranks are simulated sequentially in one process.

If the reconstructed full output matches dense causal attention, the orchestration
logic is correct. Run directly:  python tests/test_algorithm_cpu.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fa4_ring.zigzag import zigzag_shard, zigzag_unshard  # noqa: E402


def _expand_kv(k, v, n_q_heads):
    h_kv = k.shape[1]
    if h_kv == n_q_heads:
        return k, v
    rep = n_q_heads // h_kv
    return k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)


def cpu_attn(q, k, v, causal, scale):
    """q:[Tq,H,D], k/v:[Tk,Hkv,D] -> out:[Tq,H,D], lse:[Tq,H] (natural log)."""
    kq, vq = _expand_kv(k, v, q.shape[1])
    scores = torch.einsum("qhd,khd->hqk", q.float(), kq.float()) * scale  # [H,Tq,Tk]
    if causal:
        tq, tk = q.shape[0], k.shape[0]
        # local round-0: square causal (token i attends j<=i)
        mask = torch.tril(torch.ones(tq, tk, dtype=torch.bool))
        scores = scores.masked_fill(~mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # [H,Tq]
    p = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", p, vq.float())  # [Tq,H,D]
    return out, lse.transpose(0, 1).contiguous()  # [Tq,H]


def merge(out_a, lse_a, out_b, lse_b):
    """Online-softmax merge of two (out, lse). -inf rows in b contribute nothing."""
    m = torch.maximum(lse_a, lse_b)
    ea = torch.exp(lse_a - m)
    eb = torch.exp(lse_b - m)
    denom = (ea + eb).clamp_min(1e-20)
    out = (out_a * ea.unsqueeze(-1) + out_b * eb.unsqueeze(-1)) / denom.unsqueeze(-1)
    lse = torch.logaddexp(lse_a, lse_b)
    return out, lse


def simulate_ring(q, k, v, cp_size, scale):
    """Run the ring algorithm for all ranks sequentially; return reconstructed [N,H,D]."""
    n, H, D = q.shape
    h = n // (2 * cp_size)
    local_len = 2 * h
    rank_outs = []
    for rank in range(cp_size):
        lq = zigzag_shard(q, rank, cp_size)
        # round 0: local causal
        acc_out, acc_lse = cpu_attn(
            lq, zigzag_shard(k, rank, cp_size), zigzag_shard(v, rank, cp_size),
            causal=True, scale=scale,
        )
        for r in range(1, cp_size):
            src = (rank - r) % cp_size
            rk = zigzag_shard(k, src, cp_size)  # src sends its own local KV
            rv = zigzag_shard(v, src, cp_size)
            if r > rank:
                o, l = cpu_attn(lq[h:], rk, rv, causal=False, scale=scale)
                out_full = torch.zeros(local_len, H, D)
                lse_full = torch.full((local_len, H), float("-inf"))
                out_full[h:] = o
                lse_full[h:] = l
                acc_out, acc_lse = merge(acc_out, acc_lse, out_full, lse_full)
            else:
                o, l = cpu_attn(lq, rk[:h], rv[:h], causal=False, scale=scale)
                acc_out, acc_lse = merge(acc_out, acc_lse, o, l)
        rank_outs.append(acc_out)
    return zigzag_unshard(rank_outs, cp_size)


def dense_causal(q, k, v, scale):
    out, _ = cpu_attn(q, k, v, causal=True, scale=scale)
    return out


def run_case(n, H, Hkv, D, cp_size, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(n, H, D)
    k = torch.randn(n, Hkv, D)
    v = torch.randn(n, Hkv, D)
    scale = 1.0 / math.sqrt(D)
    ref = dense_causal(q, k, v, scale)
    got = simulate_ring(q, k, v, cp_size, scale)
    err = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
    print(f"  N={n:4d} H={H} Hkv={Hkv} D={D} cp={cp_size}: max_abs_err={err:.2e} {'OK' if ok else 'FAIL'}")
    return ok


def main():
    print("CPU ring-algorithm simulation vs dense causal attention:")
    cases = [
        # (N, H, Hkv, D, cp_size)
        (16, 4, 4, 8, 2),
        (32, 8, 2, 16, 4),   # GQA
        (64, 8, 8, 16, 4),
        (64, 4, 1, 32, 8),   # MQA
        (128, 8, 2, 64, 8),
        (48, 6, 3, 16, 2),
        (96, 8, 8, 16, 4),
    ]
    all_ok = True
    for c in cases:
        all_ok &= run_case(*c)
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
