"""GDN (gated delta net / linear-attention) context parallelism for Qwen3.5.

Plan A (default): all-gather projected GDN inputs, replicate the recurrence on the full
sequence, extract this rank's zig-zag-local output. Compatible with fa4_ring's zig-zag
sharding (one sharding for the whole model: ring on full-attn layers, this on GDN layers).

Public API:
    build_gdn_cp_metadata  - zig-zag restore/extract index metadata
    gdn_cp_allgather       - Plan A all-gather CP (dist)
    GdnCpMeta              - metadata dataclass
    gdn_reference          - pure-torch GDN recurrence (CPU ground truth)
"""

from .allgather_cp import gdn_cp_allgather
from .metadata import GdnCpMeta, build_gdn_cp_metadata
from .reference import gdn_reference

__all__ = [
    "build_gdn_cp_metadata",
    "gdn_cp_allgather",
    "GdnCpMeta",
    "gdn_reference",
]
