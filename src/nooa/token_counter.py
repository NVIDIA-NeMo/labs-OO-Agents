# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-counting utilities for context/event token limits.

When an LLM client does not provide a ``count_tokens`` method, users can
opt in to a simple character-based approximation rather than relying on
the LLM's built-in counter.
"""


def char_approximate_token_counter(text: str) -> int:
    """Approximate token count using character length divided by 4.

    A rough heuristic: English text averages ~4 characters per token.
    Accurate enough for context-window budget decisions; not suitable
    for billing or fine-grained limits.

    Usage::

        from nooa import char_approximate_token_counter

        class MyLLM:
            count_tokens = staticmethod(char_approximate_token_counter)
            ...

    Args:
        text: The text whose token count is approximated.

    Returns:
        Estimated number of tokens (``len(text) // 4``).
    """
    return len(text) // 4
