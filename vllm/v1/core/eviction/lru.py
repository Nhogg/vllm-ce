from vllm.v1.core.eviction.base import EvictionPolicy
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


class LRUEvictionPolicy(EvictionPolicy):
    def select_victim(self, free_block_queue: FreeKVCacheBlockQueue) -> KVCacheBlock:
        return free_block_queue.popleft()
