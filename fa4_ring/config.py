"""Configuration for FA4 ring attention."""

from dataclasses import dataclass
from typing import Optional

# Merger choices:
#   "incremental" -> flashinfer.cascade.merge_state_in_place, merged each round.
#                    O(1) extra memory, overlaps with the next round's KV rotation.
#                    Default; usually the faster choice for ring (merge hidden behind comm).
#   "batched"     -> flashinfer.cascade.merge_states, all cp_size partials combined in
#                    one kernel at the end. O(cp_size) memory, no comm overlap.
MERGE_MODES = ("incremental", "batched")


@dataclass
class RingConfig:
    cp_size: int
    cp_rank: int
    causal: bool = True
    softmax_scale: Optional[float] = None  # default: 1/sqrt(head_dim)
    merge_mode: str = "incremental"
    deterministic: bool = False

    def __post_init__(self):
        assert self.merge_mode in MERGE_MODES, (
            f"merge_mode must be one of {MERGE_MODES}, got {self.merge_mode}"
        )
        assert self.cp_size >= 1, "cp_size must be >= 1"
        assert 0 <= self.cp_rank < self.cp_size
        # This v1 only implements the causal ring (zigzag load balance). Non-causal
        # ring is a trivial subset (every round is full, no half-Q/half-KV split) and
        # is intentionally left out to keep the verified path small.
        assert self.causal, "v1 only implements causal ring attention"
