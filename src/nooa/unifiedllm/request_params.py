# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, explicit translations of shared request settings."""

import os
from urllib.parse import urlsplit


def chat_token_limit(params: dict, field: str) -> dict:
    """Select one Chat cap spelling without consulting model/deployment names."""
    result = dict(params)
    names = {"max_tokens", "max_completion_tokens"}
    if len(names & result.keys()) > 1:
        raise ValueError("Use either max_tokens or max_completion_tokens, not both")
    extra = result.get("extra_body") or {}
    if names & extra.keys():
        raise ValueError("Chat reply limits belong at the top level, not in extra_body")
    if "max_tokens" not in result:
        return result
    if field == "auto":
        endpoint = (
            result.get("base_url")
            or result.get("api_base")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        field = (
            "max_completion_tokens"
            if urlsplit(endpoint).hostname == "api.openai.com"
            else "max_tokens"
        )
    if field == "max_completion_tokens":
        result[field] = result.pop("max_tokens")
    return result
