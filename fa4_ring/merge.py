"""Online-softmax mergers for combining ring partial results.

Both mergers consume **full-size** partials ``(out[T,H,D], lse[T,H])``: rounds that
only cover the second-half Q rows pass an ``out`` with zeros and an ``lse`` of
``-inf`` on the uncovered rows, so the merge is the identity there. This mirrors the
verified rtp-llm implementation and keeps both mergers symmetric.

  * IncrementalMerger  -> flashinfer.cascade.merge_state_in_place, one merge per
    round. O(1) extra memory; the merge of round r runs on the compute stream while
    round r+1's KV rotation is in flight on the comm stream -> the merge latency is
    typically fully hidden. Default.

  * BatchedMerger      -> flashinfer.cascade.merge_states, all cp_size partials
    stacked and combined in a single kernel at the end. Fewer kernel launches but
    O(cp_size) memory and no comm overlap.

LSE is float32 natural-log for both FA4 outputs and FlashInfer mergers, so no base
conversion is needed.

Invariant: the FIRST partial added must cover every row with a finite LSE (in the ring
schedule this is step 0, full causal). The merge kernels NaN-poison a row only if it is
``-inf`` in *both* operands; seeding with the all-rows-finite step-0 partial guarantees
that never happens. Do not add a ``-inf``-on-some-rows partial before step 0.
"""

from typing import List

import torch


def _merge_state_in_place(v, s, v_other, s_other):
    # Lazy import: flashinfer is a CUDA runtime dep; importing this module (e.g. for the
    # zigzag helpers or CPU tests) must not require it.
    from flashinfer.cascade import merge_state_in_place

    merge_state_in_place(v, s, v_other, s_other)


def _merge_states(v, s):
    from flashinfer.cascade import merge_states

    return merge_states(v, s)


class IncrementalMerger:
    """Accumulate (out, lse) in place, one round at a time."""

    def __init__(self):
        self.out = None  # [T, H, D]
        self.lse = None  # [T, H]

    def add(self, out: torch.Tensor, lse: torch.Tensor) -> None:
        if self.out is None:
            # First partial seeds the accumulator. clone() so later in-place merges
            # don't mutate the caller's round-0 output buffer.
            self.out = out.clone()
            self.lse = lse.clone()
        else:
            # merge (out, lse) INTO (self.out, self.lse), in place.
            _merge_state_in_place(self.out, self.lse, out, lse)

    def result(self) -> torch.Tensor:
        assert self.out is not None, "no partials were added"
        return self.out


class BatchedMerger:
    """Stash every round's (out, lse), combine all at once at the end."""

    def __init__(self):
        self.outs: List[torch.Tensor] = []
        self.lses: List[torch.Tensor] = []

    def add(self, out: torch.Tensor, lse: torch.Tensor) -> None:
        self.outs.append(out)
        self.lses.append(lse)

    def result(self) -> torch.Tensor:
        assert self.outs, "no partials were added"
        if len(self.outs) == 1:
            return self.outs[0]
        # merge_states wants v:[seq, num_states, H, D], s:[seq, num_states, H]
        v = torch.stack(self.outs, dim=1).contiguous()
        s = torch.stack(self.lses, dim=1).contiguous()
        out, _ = _merge_states(v, s)
        return out


def make_merger(merge_mode: str):
    if merge_mode == "incremental":
        return IncrementalMerger()
    if merge_mode == "batched":
        return BatchedMerger()
    raise ValueError(f"unknown merge_mode {merge_mode!r}")
