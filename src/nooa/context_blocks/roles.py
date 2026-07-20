# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Message roles for provider formatting.

Extracted to its own module to break the circular import between
models.py (defines ResolvedBlock with EventBase field) and
events.py (defines EventBase with Role field).
"""

from enum import StrEnum


class Role(StrEnum):
    """Provider role for message formatting.

    Maps to OpenAI/Anthropic message roles.
    SYSTEM is for system prompt blocks.
    RUNTIME_EVENT is for internal runtime events that are never recorded.
    METADATA is for stored metadata events that are never shown to the LLM.
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    RUNTIME_EVENT = "runtime_event"
    METADATA = "metadata"
