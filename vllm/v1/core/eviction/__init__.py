from vllm.v1.core.eviction.arc import ARCEvictionPolicy
from vllm.v1.core.eviction.base import EvictionPolicy
from vllm.v1.core.eviction.lru import LRUEvictionPolicy
from vllm.v1.core.eviction.random import RandomEvictionPolicy


def get_eviction_policy(name: str) -> EvictionPolicy:
    if name == "lru":
        return LRUEvictionPolicy()
    if name == "random":
        return RandomEvictionPolicy()
    if name == "arc":
        return ARCEvictionPolicy()
    raise ValueError(
        f"Unknown eviction policy: {name!r}. Choose 'lru', 'random', or 'arc'."
    )
