# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.benchmark_kv_eviction_ablation_vllm import _confidence_prompt


class _Tokenizer:
    def __init__(self, encoded):
        self.encoded = encoded

    def decode(self, token_ids, skip_special_tokens):
        assert token_ids == [10, 11]
        assert skip_special_tokens
        return "Original task"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert messages[0]["role"] == "user"
        assert "Proposed answer" in messages[0]["content"]
        assert tokenize
        assert add_generation_prompt
        return self.encoded


def test_confidence_prompt_accepts_token_list():
    tokenizer = _Tokenizer([1, 2, 3])

    token_ids = _confidence_prompt(
        tokenizer, {"token_ids": [10, 11], "task": "qasper"}, "Proposed answer"
    )

    assert token_ids == [1, 2, 3]


def test_confidence_prompt_extracts_batch_encoding_input_ids():
    tokenizer = _Tokenizer(
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    )

    token_ids = _confidence_prompt(
        tokenizer, {"token_ids": [10, 11], "task": "qasper"}, "Proposed answer"
    )

    assert token_ids == [1, 2, 3]
