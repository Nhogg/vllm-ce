# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental geometric KV-cache eviction scaffolding (Option A).

All behavior in this package is gated behind an explicit
``additional_config["geo_kv"]`` block. When that block is absent (the default),
vLLM behaves exactly as upstream. See ``plan.md`` and ``agent_notes.md``.
"""
from vllm.v1.geo_kv.config import GeoKVConfig

__all__ = ["GeoKVConfig"]
