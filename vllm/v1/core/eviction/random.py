import random

from vllm.v1.core.eviction.base import EvictionPolicy
from vllm.v1.core.kv_cache_utils import FreeKVCacheBlockQueue, KVCacheBlock


class RandomEvictionPolicy(EvictionPolicy):
    def select_victim(self, free_block_queue: FreeKVCacheBlockQueue) -> KVCacheBlock:
        n = free_block_queue.num_free_blocks
        target = random.randint(0, n - 1)
        block = free_block_queue.fake_free_list_head.next_free_block
        for _ in range(target):
            block = block.next_free_block
        free_block_queue.remove(block)
        return block
