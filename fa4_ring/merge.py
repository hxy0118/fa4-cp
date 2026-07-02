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

LSE base: FA4 (and this module's callers) produce LSE in **natural log**, but
FlashInfer's ``cascade`` mergers interpret ``s`` in **base-2** (they weight partials
by ``2**(s - max)``). We therefore scale LSE by ``log2(e)`` before every FlashInfer
call and scale the returned LSE back. (Feeding natural-log LSE straight through is
only ~correct when the two partials have near-equal LSE, e.g. same key count; it is
badly wrong when they differ, e.g. a causal step vs a non-causal half step.)

FlashInfer fallback: the ``cascade`` merge kernels fail to launch for large head
counts (empirically ``num_heads >= 64``, ``MergeStateInPlace ... invalid argument``).
An analytic guard (``num_heads <= 32 and head_dim <= 256``) routes unsupported shapes
to an exact fp32 pure-torch online-softmax merge, so any head configuration is
supported. We do NOT trial-launch to detect support: a failed CUDA launch sets a
sticky error that poisons subsequent ops. The torch merge is exact and needs no base
conversion.

Invariant: the FIRST partial added must cover every row with a finite LSE (in the ring
schedule this is step 0, full causal). The merge kernels NaN-poison a row only if it is
``-inf`` in *both* operands; seeding with the all-rows-finite step-0 partial guarantees
that never happens. Do not add a ``-inf``-on-some-rows partial before step 0.
"""

from typing import List

import torch

# FA4 returns natural-log LSE; FlashInfer's cascade mergers weight by 2**(s-max),
# i.e. they expect base-2 LSE. log2(e) = 1/ln(2) converts natural log -> log2.
_LOG2E = 1.4426950408889634


def _fi_merge_state_in_place(v, s, v_other, s_other):
    # Lazy import: flashinfer is a CUDA runtime dep; importing this module (e.g. for the
    # zigzag helpers or CPU tests) must not require it.
    from flashinfer.cascade import merge_state_in_place

    merge_state_in_place(v, s, v_other, s_other)


def _fi_merge_states(v, s):
    from flashinfer.cascade import merge_states

    return merge_states(v, s)


def _flashinfer_supported(h: int, d: int, device) -> bool:
    """Whether flashinfer's cascade merge kernel can launch for a ``[*, h, d]`` problem.

    Its kernel launches roughly ``(head_dim / vec) x num_heads`` threads and fails with
    ``cudaErrorInvalidValue`` for large head counts (empirically ``num_heads >= 64``;
    ``num_heads <= 32`` is safe for ``head_dim <= 256``). We use a conservative *analytic*
    guard rather than a trial launch on purpose: a failed CUDA launch sets a **sticky**
    error that would poison every subsequent CUDA op, and the exact torch merge is cheap
    and universal, so routing borderline shapes to torch costs nothing."""
    return device.type == "cuda" and h <= 32 and d <= 256


def _torch_merge(out_a, lse_a, out_b, lse_b):
    """Exact fp32 online-softmax merge (natural-log LSE). ``-inf`` rows are the identity.

    Returns (out[T,H,D] fp32, lse[T,H] fp32). Where both LSE are ``-inf`` the row stays
    ``-inf`` and keeps ``out_a`` (avoids 0/0 NaN)."""
    la = lse_a.float()
    lb = lse_b.float()
    m = torch.maximum(la, lb)
    ea = (la - m).exp().unsqueeze(-1)
    eb = (lb - m).exp().unsqueeze(-1)
    denom = ea + eb
    oa = out_a.float()
    ob = out_b.float()
    # denom > 0 is always true in the ring (the seed partial is finite on every row);
    # the torch.where is cheap insurance against a both-``-inf`` row (keep oa, no 0/0).
    out = torch.where(denom > 0, (oa * ea + ob * eb) / denom, oa)
    lse = torch.logaddexp(la, lb)
    return out, lse


class IncrementalMerger:
    """Accumulate (out, lse) one round at a time (FlashInfer in place, or torch fallback)."""

    def __init__(self):
        self.out = None       # [T, H, D]
        self.lse = None       # [T, H], float32 (base-2 on the FI path, natural on torch)
        self._dtype = None    # output dtype of the seeding partial
        self._use_fi = False   # FlashInfer vs torch merge; decided at seed time

    def add(self, out: torch.Tensor, lse: torch.Tensor) -> None:
        lse = lse.float()
        if self.out is None:
            # First partial seeds the accumulator. clone() the output so the later
            # in-place FlashInfer merge doesn't mutate the caller's round-0 buffer.
            self._dtype = out.dtype
            self._use_fi = _flashinfer_supported(out.shape[1], out.shape[2], out.device)
            self.out = out.clone()
            # FlashInfer merges in place and weights by base-2 LSE, so keep the
            # accumulator LSE in base-2 there (``lse * _LOG2E`` is already a fresh
            # tensor); the torch path keeps natural-log and reassigns each round.
            self.lse = lse * _LOG2E if self._use_fi else lse.clone()
            return
        if self._use_fi:
            _fi_merge_state_in_place(self.out, self.lse, out, lse * _LOG2E)
        else:
            self.out, self.lse = _torch_merge(self.out, self.lse, out, lse)

    def result(self) -> torch.Tensor:
        assert self.out is not None, "no partials were added"
        # torch-fallback merges accumulate in fp32; .to() is a no-op if already the dtype.
        return self.out.to(self._dtype)


class BatchedMerger:
    """Stash every round's (out, lse), combine all at once at the end."""

    def __init__(self):
        self.outs: List[torch.Tensor] = []
        self.lses: List[torch.Tensor] = []

    def add(self, out: torch.Tensor, lse: torch.Tensor) -> None:
        self.outs.append(out)
        self.lses.append(lse.float())

    def result(self) -> torch.Tensor:
        assert self.outs, "no partials were added"
        if len(self.outs) == 1:
            return self.outs[0]
        h, d = self.outs[0].shape[1], self.outs[0].shape[2]
        if _flashinfer_supported(h, d, self.outs[0].device):
            # merge_states wants v:[seq, num_states, H, D], s:[seq, num_states, H] (base-2).
            v = torch.stack(self.outs, dim=1).contiguous()
            s = (torch.stack(self.lses, dim=1) * _LOG2E).contiguous()
            out, _ = _fi_merge_states(v, s)
            return out
        # exact torch fold (fp32), cast back to the input dtype.
        acc_o, acc_l = self.outs[0].float(), self.lses[0]
        for o, l in zip(self.outs[1:], self.lses[1:]):
            acc_o, acc_l = _torch_merge(acc_o, acc_l, o, l)
        return acc_o.to(self.outs[0].dtype)


def make_merger(merge_mode: str):
    if merge_mode == "incremental":
        return IncrementalMerger()
    if merge_mode == "batched":
        return BatchedMerger()
    raise ValueError(f"unknown merge_mode {merge_mode!r}")
