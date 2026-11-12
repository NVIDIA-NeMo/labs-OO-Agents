# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private compatibility configuration, never part of the Adapter protocol."""

import copy
from collections.abc import Mapping
from typing import Any


def snapshot_options(value: Any) -> Any:
    """Detach settings containers, preserving opaque SDK clients and transports."""
    if isinstance(value, Mapping):
        return {key: snapshot_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_options(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_options(item) for item in value)
    return value


def configure_legacy_adapter(adapter, options):
    """A lightweight call-local view for the built-in legacy adapters only.

    The view retains the original client, hooks and transport; only the small
    options containers are copied, once per call, not per retry or history turn.
    None means a third-party adapter, which needs only the public protocol.
    """
    from .litellm_chat import LiteLLMChatAdapter
    from .litellm_responses import LiteLLMResponsesAdapter

    if not isinstance(adapter, (LiteLLMChatAdapter, LiteLLMResponsesAdapter)):
        return None
    view = copy.copy(adapter)
    view._legacy_options = snapshot_options(options)
    return view
