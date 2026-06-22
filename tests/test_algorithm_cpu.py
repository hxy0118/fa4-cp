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
from fa4_ring.zigzag import (  # noqa: E402
    get_half_index,
    zigzag_shard,
    zigzag_shard_varlen,
    zigzag_unshard,
)


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


# ---------------- varlen (multi-sequence packed) ----------------

def cpu_attn_varlen(q, k, v, cu_q, cu_k, causal, scale):
    """Per-segment attention over packed sequences -> packed (out[Tq,H,D], lse[Tq,H])."""
    H, D = q.shape[1], q.shape[2]
    out = torch.zeros(q.shape[0], H, D)
    lse = torch.full((q.shape[0], H), float("-inf"))
    for j in range(len(cu_q) - 1):
        qs, qe = cu_q[j], cu_q[j + 1]
        ks, ke = cu_k[j], cu_k[j + 1]
        o, l = cpu_attn(q[qs:qe], k[ks:ke], v[ks:ke], causal=causal, scale=scale)
        out[qs:qe] = o
        lse[qs:qe] = l
    return out, lse


def simulate_ring_varlen(full_seqs, cp_size, scale):
    """full_seqs: list of (q[N,H,D],k,v). Returns list of per-seq reconstructed out."""
    # full cu_seqlens over the concatenation of all sequences
    lens = [s[0].shape[0] for s in full_seqs]
    full_cu = [0]
    for n in lens:
        full_cu.append(full_cu[-1] + n)
    Q = torch.cat([s[0] for s in full_seqs], 0)
    K = torch.cat([s[1] for s in full_seqs], 0)
    V = torch.cat([s[2] for s in full_seqs], 0)
    full_cu_t = torch.tensor(full_cu, dtype=torch.int32)

    rank_outs = []
    local_cu = None
    for rank in range(cp_size):
        lq, local_cu = zigzag_shard_varlen(Q, full_cu_t, rank, cp_size)
        lk, _ = zigzag_shard_varlen(K, full_cu_t, rank, cp_size)
        lv, _ = zigzag_shard_varlen(V, full_cu_t, rank, cp_size)
        max_seqlen = int((local_cu[1:] - local_cu[:-1]).max())
        front = get_half_index(local_cu, front=True)
        back = get_half_index(local_cu, front=False)
        half_cu = local_cu // 2

        acc_out, acc_lse = cpu_attn_varlen(lq, lk, lv, local_cu.tolist(), local_cu.tolist(),
                                           causal=True, scale=scale)
        for r in range(1, cp_size):
            src = (rank - r) % cp_size
            rk, _ = zigzag_shard_varlen(K, full_cu_t, src, cp_size)
            rv, _ = zigzag_shard_varlen(V, full_cu_t, src, cp_size)
            if r <= rank:
                o, l = cpu_attn_varlen(lq, rk[front], rv[front],
                                       local_cu.tolist(), half_cu.tolist(),
                                       causal=False, scale=scale)
                acc_out, acc_lse = merge(acc_out, acc_lse, o, l)
            else:
                o, l = cpu_attn_varlen(lq[back], rk, rv,
                                       half_cu.tolist(), local_cu.tolist(),
                                       causal=False, scale=scale)
                of = torch.zeros_like(acc_out)
                lf = torch.full_like(acc_lse, float("-inf"))
                of[back] = o
                lf[back] = l
                acc_out, acc_lse = merge(acc_out, acc_lse, of, lf)
        rank_outs.append(acc_out)

    # reconstruct each full sequence from per-rank zigzag shards
    h_per_seq = [(local_cu[j + 1] - local_cu[j]).item() // 2 for j in range(len(lens))]
    recon = []
    for j in range(len(lens)):
        shards = []
        for rank in range(cp_size):
            _, lc = zigzag_shard_varlen(Q, full_cu_t, rank, cp_size)
            shards.append(rank_outs[rank][lc[j]:lc[j + 1]])
        recon.append(zigzag_unshard(shards, cp_size))
    return recon


def run_varlen_case(seq_lens, H, Hkv, D, cp_size, seed=0):
    torch.manual_seed(seed)
    full_seqs = []
    for n in seq_lens:
        full_seqs.append((torch.randn(n, H, D), torch.randn(n, Hkv, D), torch.randn(n, Hkv, D)))
    scale = 1.0 / math.sqrt(D)
    recon = simulate_ring_varlen(full_seqs, cp_size, scale)
    ok = True
    for (q, k, v), got in zip(full_seqs, recon):
        ref = dense_causal(q, k, v, scale)
        ok &= torch.allclose(got, ref, atol=1e-4, rtol=1e-4)
    print(f"  varlen seqs={seq_lens} H={H} Hkv={Hkv} cp={cp_size}: {'OK' if ok else 'FAIL'}")
    return ok


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

    print("\nCPU ring-algorithm simulation (VARLEN, multi-sequence) vs per-seq dense causal:")
    varlen_cases = [
        # (seq_lens, H, Hkv, D, cp_size) — each seq_len divisible by 2*cp_size
        ([16, 32], 4, 4, 8, 2),
        ([32, 16, 48], 8, 2, 16, 4),   # GQA, 3 sequences
        ([64, 64], 4, 1, 16, 8),       # MQA
        ([48, 24], 6, 3, 16, 2),
    ]
    for c in varlen_cases:
        all_ok &= run_varlen_case(*c)

    print("\nALL PASS" if all_ok else "\nSOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
