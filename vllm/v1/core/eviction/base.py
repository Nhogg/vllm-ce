from abc import ABC, abstractmethod

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


class EvictionPolicy(ABC):
    """Selects which free block to evict next.

    The policy owns only the selection decision.
    The block pool retains ownership of ref-counting,
    hash maps, and list membership.
    """

    @abstractmethod
    def select_victim(self, free_block_queue: FreeKVCacheBlockQueue) -> KVCacheBlock:
        """Pop and return that block that should be evicted next.

        The returned block must be removed from free_block_queue by this call.
        """
        ...
