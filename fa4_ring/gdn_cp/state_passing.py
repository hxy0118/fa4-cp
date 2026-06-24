"""GDN context parallelism — Plan B: sequential state-passing (skeleton / v2).

True CP for the GDN recurrence: each rank scans only its 1/cp_size of the tokens, passing
the recurrent SSM state hand-to-hand around the ring. Real compute parallelism + tiny
comm (one state matrix per head), but a serial dependency (rank r waits for r-1's state).

Per rank r (CONTIGUOUS sharding, NOT zig-zag — the recurrence needs token order):
    1. conv boundary: recv last (kernel-1) tokens of mixed_qkv from rank r-1, prepend;
       send our last (kernel-1) tokens to r+1.  (small)
    2. recv initial_state S_{r-1} from rank r-1 (P2P, rank order).
    3. out_r, S_r = chunk_gated_delta_rule(q,k,v,g,beta, initial_state=S_{r-1},
                                           output_final_state=True)   # kernel already supports this
    4. send S_r to rank r+1.

Why it is NOT the v1 default:
  * needs CONTIGUOUS sharding, which conflicts with fa4_ring's zig-zag full-attn sharding
    (a single model would then need two shardings or lose full-attn load balance);
  * serial latency O(cp_size) unless sub-chunk pipelined.

Use this only when the all-gather replicate-scan (Plan A) becomes the bottleneck at
extreme context / cp_size. The GDN kernel already exposes ``initial_state`` /
``output_final_state`` (verified in rtp/fla), so the kernel side is ready; what remains is
the contiguous-shard sharding policy + P2P state/conv-boundary exchange + pipelining.
"""


def gdn_cp_state_passing(*args, **kwargs):
    raise NotImplementedError(
        "Plan B (sequential state-passing) is a v2 optimization; v1 uses Plan A "
        "(all-gather + replicate scan) in allgather_cp.py. See this module docstring."
    )
