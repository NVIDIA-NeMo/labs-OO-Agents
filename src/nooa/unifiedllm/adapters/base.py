# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The interface every LLM adapter implements."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nooa.llm_types import LLMResponse

if TYPE_CHECKING:
    from nooa.unifiedllm.request import LLMRequest


@dataclass(frozen=True)
class Capabilities:
    """Read-only facts about one model on one route.

    ``reasoning_levels`` is None when unknown and empty when unsupported.
    ``context_window`` and ``max_output_tokens`` are None when unknown.
    """

    media_kinds: frozenset[str] = frozenset()
    reasoning_levels: tuple[str, ...] | None = None
    tools: bool = True
    context_window: int | None = None
    max_output_tokens: int | None = None


@runtime_checkable
class Adapter(Protocol):
    """Translate one request into one wire call and the reply back.

    Adapters hold no retry, metrics or tracing logic; the calling client does.
    A failure to parse a reply that arrived is raised wrapped in
    ``errors._ReplyParseError``, which the client's retry never re-sends and
    unwraps to the original exception.
    """

    @property
    def capabilities(self) -> Capabilities:
        """What the model on this route accepts."""
        ...

    async def send(self, request: "LLMRequest") -> LLMResponse:
        """Make one async attempt for ``request``; no retries."""
        ...

    def send_sync(self, request: "LLMRequest") -> LLMResponse:
        """Make one sync attempt for ``request``; no retries."""
        ...
