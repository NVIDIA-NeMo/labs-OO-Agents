# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect refuses to configure, plan or save media a native interface cannot send."""

from copy import deepcopy

import pytest

from nooa.unifiedllm.connect import configure_entry, plan, write


@pytest.mark.parametrize("style", ["anthropic", "responses"])
@pytest.mark.parametrize("boundary", ["configure", "plan", "write"])
def test_invalid_native_declaration_rejected_without_write(tmp_path, style, boundary):
    entry = {
        "model_name": "anthropic/model" if style == "anthropic" else "openai/model",
        "client_type": "responses" if style == "responses" else "completion",
        "api_style": style,
        "api_base": "https://policy.test/v1",
        "media_inputs": ["audio", "video"],
    }
    before = deepcopy(entry)
    path = tmp_path / "not-created" / "config.yaml"
    with pytest.raises(ValueError, match="[Uu]nsupported.*audio"):
        if boundary == "configure":
            configure_entry(entry)
        elif boundary == "plan":
            plan("alias", "model", style, entry["api_base"], "", media_inputs=entry["media_inputs"])
        else:
            write(entry, path, alias="alias")
    assert not path.parent.exists()
    assert entry == before


def test_dispatch_decides_native_rejection_and_chat_keeps_audio():
    entry = {
        "model_name": "opaque-model",
        "client_type": "completion",
        "api_style": "chat",
        "custom_llm_provider": "anthropic",
        "api_base": "https://policy.test",
        "media_inputs": ["audio"],
    }
    with pytest.raises(ValueError, match="Unsupported media input.*audio"):
        configure_entry(entry)
    chat = {**entry, "model_name": "anthropic/innocuous", "custom_llm_provider": "openai"}
    assert configure_entry(chat)["media_inputs"] == ["audio"]


def test_unknown_chat_declarations_save_unverified(tmp_path):
    entry = {
        "model_name": "unknown-private-route",
        "api_style": "anthropic",
        "client_type": "completion",
        "media_inputs": ["audio", "video"],
    }
    assert configure_entry(entry)["media_inputs"] == ["audio", "video"]
    write(entry, tmp_path / "models.yaml", alias="unknown")
    assert "passed" not in (tmp_path / "models.yaml").read_text()
