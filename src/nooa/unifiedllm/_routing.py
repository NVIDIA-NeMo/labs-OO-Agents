# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire format follows the client type and explicit model routing prefix."""


def api_style_for(model: str, client_type: str = "completion") -> str:
    """OpenAI-compatible names stay Chat, even when they contain 'anthropic'.

    Use ``anthropic/<wire-model>`` for native Messages and a ResponsesClient
    (``client_type: responses`` in a registry) for Responses.
    """
    if client_type == "responses":
        return "responses"
    return "anthropic" if model.startswith("anthropic/") else "chat"


def check_legacy_api_style(entry: dict, derived_style: str) -> None:
    """Do not silently change protocols when removing an older routing field."""
    if entry.get("api_style") not in (None, derived_style):
        raise ValueError(
            "api_style no longer selects the wire format; use client_type: responses "
            "for Responses or model_name: anthropic/<model> for native Messages "
            "(openai/<model> for Chat), then remove api_style"
        )
