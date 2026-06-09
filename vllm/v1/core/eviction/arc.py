  from collections import OrderedDict

  from vllm.v1.core.eviction.base import EvictionPolicy
  from vllm.v1.core.kv_cache_utils import (
      BlockHashWithGroupId,
      FreeKVCacheBlockQueue,
      KVCacheBlock,
  )

  ResidentKey = tuple[BlockHashWithGroupId, int]


  class ARCEvictionPolicy(EvictionPolicy):
      """Adaptive Replacement Cache policy for prefix-cache block eviction.

      T1: resident blocks seen once.
      T2: resident blocks seen more than once.
      B1/B2: ghost lists remembering recently evicted block hashes.
      """

      def __init__(self) -> None:
          self.t1: OrderedDict[ResidentKey, None] = OrderedDict()
          self.t2: OrderedDict[ResidentKey, None] = OrderedDict()
          self.b1: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
          self.b2: OrderedDict[BlockHashWithGroupId, None] = OrderedDict()
          self.p = 0

      def _capacity(self) -> int:
          return max(1, len(self.t1) + len(self.t2))

      def _resident_key(self, block: KVCacheBlock) -> ResidentKey | None:
          if block.block_hash is None:
              return None
          return (block.block_hash, block.block_id)

      def _has_resident_hash(self, block_hash: BlockHashWithGroupId) -> bool:
          return any(key[0] == block_hash for key in self.t1) or any(
              key[0] == block_hash for key in self.t2
          )

      def _trim_ghosts(self) -> None:
          capacity = self._capacity()
          while len(self.b1) > capacity:
              self.b1.popitem(last=False)
          while len(self.b2) > capacity:
              self.b2.popitem(last=False)
          while len(self.b1) + len(self.b2) > 2 * capacity:
              if self.b2:
                  self.b2.popitem(last=False)
              else:
                  self.b1.popitem(last=False)

      def _find_and_remove(
          self,
          free_block_queue: FreeKVCacheBlockQueue,
          keys: OrderedDict[ResidentKey, None],
      ) -> KVCacheBlock | None:
          block = free_block_queue.fake_free_list_head.next_free_block
          while block is not None and block is not free_block_queue.fake_free_list_tail:
              key = self._resident_key(block)
              if key in keys:
                  free_block_queue.remove(block)
                  return block
              block = block.next_free_block
          return None

      def on_block_cached(self, block: KVCacheBlock) -> None:
          key = self._resident_key(block)
          if key is None:
              return

          block_hash = key[0]
          self.t1.pop(key, None)
          self.t2.pop(key, None)

          if block_hash in self.b1:
              delta = max(len(self.b2) // max(len(self.b1), 1), 1)
              self.p = min(self._capacity(), self.p + delta)
              self.b1.pop(block_hash, None)
              self.t2[key] = None
          elif block_hash in self.b2:
              delta = max(len(self.b1) // max(len(self.b2), 1), 1)
              self.p = max(0, self.p - delta)
              self.b2.pop(block_hash, None)
              self.t2[key] = None
          else:
              self.t1[key] = None

          self._trim_ghosts()

      def on_block_accessed(self, block: KVCacheBlock) -> None:
          key = self._resident_key(block)
          if key is None:
              return

          if key in self.t1:
              self.t1.pop(key, None)
              self.t2[key] = None
          elif key in self.t2:
              self.t2.move_to_end(key)

      def on_block_evicted(self, block: KVCacheBlock) -> None:
          key = self._resident_key(block)
          if key is None:
              return

          block_hash = key[0]
          if key in self.t1:
              self.t1.pop(key, None)
              if not self._has_resident_hash(block_hash):
                  self.b1[block_hash] = None
          elif key in self.t2:
              self.t2.pop(key, None)
              if not self._has_resident_hash(block_hash):
                  self.b2[block_hash] = None

          self._trim_ghosts()

      def on_reset(self) -> None:
          self.t1.clear()
          self.t2.clear()
          self.b1.clear()
          self.b2.clear()
          self.p = 0

      def select_victim(self, free_block_queue: FreeKVCacheBlockQueue) -> KVCacheBlock:
          if self.t1 and len(self.t1) > self.p:
              block = self._find_and_remove(free_block_queue, self.t1)
              if block is not None:
                  return block

          block = self._find_and_remove(free_block_queue, self.t2)
          if block is not None:
              return block

          block = self._find_and_remove(free_block_queue, self.t1)
          if block is not None:
              return block

          return free_block_queue.popleft()
