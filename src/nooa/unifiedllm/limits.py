# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolved request limits, shared by display and context management."""

from dataclasses import dataclass
from typing import Any

REPLY_CAP_KEYS = frozenset({"max_tokens", "max_completion_tokens", "max_output_tokens"})


@dataclass(frozen=True)
class ContextLimits:
    """The selected model window minus the effective request's reply allowance.

    A fallback is a planning reserve, not a claim about the server's default.
    Unknown windows stay unknown; a reply ceiling is not a requested reply cap.
    """

    context_window: int | None
    reserved_output_tokens: int
    reserve_is_fallback: bool = False

    def __post_init__(self) -> None:
        if (
            not self.reserve_is_fallback
            and self.context_window is not None
            and self.reserved_output_tokens >= self.context_window
        ):
            raise ValueError(
                f"Configured reply cap ({self.reserved_output_tokens:,}) leaves no room for input "
                f"in the context window ({self.context_window:,}). Reduce the reply cap or "
                "correct the model's context_window before using context management."
            )

    @property
    def usable_input_tokens(self) -> int | None:
        """Remaining input capacity, or None when the window is unknown."""
        if self.context_window is None:
            return None
        return max(1, self.context_window - self.reserved_output_tokens)


def context_limits_for(
    client: Any, params: dict[str, Any] | None = None, *, fallback_reserve: int = 0
) -> ContextLimits:
    """Use UnifiedLLM's resolver, retaining support for legacy duck-typed clients.

    Custom clients can implement get_context_limits to expose their defaults;
    without it, only explicit per-call caps and the old window attributes are known.
    """
    if callable(resolve := getattr(client, "get_context_limits", None)):
        return resolve(params, fallback_reserve=fallback_reserve)
    window = getattr(client, "context_window", None) or getattr(client, "context_limit", None)
    if not isinstance(window, int) or window <= 0:
        window = None
    cap = next(
        ((params or {})[k] for k in REPLY_CAP_KEYS if (params or {}).get(k) is not None), None
    )
    return ContextLimits(window, cap if cap is not None else fallback_reserve, cap is None)


def reduced_reply_params(client: Any, params: dict[str, Any], limit: int) -> dict[str, Any]:
    """Keep custom clients usable during the runtime's one bounded recovery."""
    if callable(reduce := getattr(client, "_with_reduced_reply_limit", None)):
        return reduce(params, limit)
    key = next((k for k in REPLY_CAP_KEYS if k in params), "max_tokens")
    return {**{k: v for k, v in params.items() if k not in REPLY_CAP_KEYS}, key: limit}
