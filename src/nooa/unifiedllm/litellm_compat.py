# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Temporary, version-pinned LiteLLM compatibility fixes."""

from __future__ import annotations

import inspect
from functools import wraps
from importlib.metadata import version
from typing import Any

_LITELLM_VERSION = "1.97.0"
_CONVERTER_PARAMETERS = (
    "response_object",
    "model_response_object",
    "response_type",
    "stream",
    "start_time",
    "end_time",
    "hidden_params",
    "_response_headers",
    "convert_tool_call_to_json_mode",
)


def apply_reasoning_items_patch() -> None:
    """Retain Chat ``reasoning_items`` dropped by LiteLLM 1.97.0."""
    if version("litellm") != _LITELLM_VERSION:
        raise RuntimeError(
            f"The temporary NOOA reasoning patch requires litellm=={_LITELLM_VERSION}."
        )

    import litellm.utils as litellm_utils
    from litellm.litellm_core_utils.llm_response_utils import convert_dict_to_response
    from litellm.llms.openai import openai as openai_handler
    from litellm.llms.openai.chat import gpt_transformation

    original = convert_dict_to_response.convert_to_model_response_object
    if getattr(original, "_nooa_preserves_reasoning_items", False):
        return
    if tuple(inspect.signature(original).parameters) != _CONVERTER_PARAMETERS:
        raise RuntimeError("LiteLLM response converter signature changed; refusing to patch it.")

    @wraps(original)
    def patched(*args: Any, **kwargs: Any) -> Any:
        response_object = kwargs.get("response_object", args[0] if args else None)
        response_type = kwargs.get("response_type", args[2] if len(args) > 2 else "completion")
        stream = kwargs.get("stream", args[3] if len(args) > 3 else False)
        reasoning_items = []
        if response_type == "completion" and not stream and isinstance(response_object, dict):
            for choice in response_object.get("choices") or []:
                message = choice.get("message") if isinstance(choice, dict) else None
                reasoning_items.append(
                    message.get("reasoning_items") if isinstance(message, dict) else None
                )

        result = original(*args, **kwargs)
        for choice, items in zip(getattr(result, "choices", ()), reasoning_items, strict=False):
            if items is not None:
                choice.message.reasoning_items = items
        return result

    patched._nooa_preserves_reasoning_items = True  # type: ignore[attr-defined]
    convert_dict_to_response.convert_to_model_response_object = patched
    setattr(litellm_utils, "convert_to_model_response_object", patched)  # noqa: B010
    setattr(openai_handler, "convert_to_model_response_object", patched)  # noqa: B010
    gpt_transformation.convert_to_model_response_object = patched
