from abc import ABC, abstractmethod

from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


class EvictionPolicy(ABC):
    """Selects which free block to evict next.

    The policy owns only the selection decision.
    The block pool retains ownership of ref-counting,
    hash maps, and list membership.
    """

    # State tracking behaviors
    def on_block_cached(self, block: KVCacheBlock) -> None:
        pass

    def on_block_accessed(self, block: KVCacheBlock) -> None:
        pass

    def on_block_freed(self, block: KVCacheBlock) -> None:
        pass

    def on_block_evicted(self, block: KVCacheBlock) -> None:
        pass

    def on_reset(self) -> None:
        pass

    @abstractmethod
    def select_victim(self, free_block_queue: FreeKVCacheBlockQueue) -> KVCacheBlock:
        """Pop and return that block that should be evicted next.

        The returned block must be removed from free_block_queue by this call.
        """
        ...
