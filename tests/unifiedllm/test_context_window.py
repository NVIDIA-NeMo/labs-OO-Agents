# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nooa.unifiedllm import CompletionClient


def test_completion_client_context_window_config_is_honored():
    client = CompletionClient(model="unknown/context-window-model", context_window=262144)

    assert client.context_window == 262144


def test_context_window_config_overrides_registry_config():
    client = CompletionClient(model="unknown/context-window-model", context_window=262144)
    client._registry_config = {"context_window": 131072}

    assert client.context_window == 262144
