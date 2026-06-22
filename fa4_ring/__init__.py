"""FA4 ring attention — context-parallel causal attention on Dao FlashAttention-4.

Public API:
    RingConfig         - configuration (cp_size, cp_rank, merge_mode, ...)
    ring_flash_attn    - run causal ring attention on a zig-zag local shard
    Descales           - optional FP8 per-tensor descales
    zigzag_shard / zigzag_unshard - shard a full sequence / stitch shards back
"""

from .config import RingConfig
from .merge import BatchedMerger, IncrementalMerger, make_merger
from .ring_attention import Descales, ring_flash_attn
from .zigzag import zigzag_shard, zigzag_unshard

__all__ = [
    "RingConfig",
    "ring_flash_attn",
    "Descales",
    "zigzag_shard",
    "zigzag_unshard",
    "IncrementalMerger",
    "BatchedMerger",
    "make_merger",
]
