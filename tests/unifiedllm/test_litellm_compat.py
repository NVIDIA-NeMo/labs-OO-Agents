# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the temporary LiteLLM compatibility shim."""

from litellm.types.utils import ModelResponse

from nooa.unifiedllm.litellm_compat import apply_reasoning_items_patch


def test_chat_converter_preserves_reasoning_items():
    apply_reasoning_items_patch()
    from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
        convert_to_model_response_object,
    )

    item = {
        "type": "reasoning",
        "id": "reasoning_1",
        "encrypted_content": "opaque",
        "summary": [],
    }
    response = {
        "id": "completion_1",
        "created": 1,
        "model": "gpt-5.6-terra",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done", "reasoning_items": [item]},
            }
        ],
    }

    converted = convert_to_model_response_object(
        response_object=response,
        model_response_object=ModelResponse(model="gpt-5.6-terra"),
    )

    assert converted.choices[0].message.reasoning_items == [item]


def test_openai_handlers_use_patched_converter():
    from litellm.litellm_core_utils.llm_response_utils import convert_dict_to_response
    from litellm.llms.openai import openai as openai_handler
    from litellm.llms.openai.chat import gpt_transformation

    converter = convert_dict_to_response.convert_to_model_response_object
    assert openai_handler.convert_to_model_response_object is converter
    assert gpt_transformation.convert_to_model_response_object is converter
