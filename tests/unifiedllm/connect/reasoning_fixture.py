# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline fixture shaped like native Anthropic ModelInfo / ModelCapabilities."""

MODEL_ID = "claude-opus-4-7"
NATIVE_MODEL = {
    "id": MODEL_ID,
    "type": "model",
    "display_name": "Claude Opus 4.7",
    "max_input_tokens": 1000000,
    "max_tokens": 128000,
    "capabilities": {
        "image_input": {"supported": True},
        "pdf_input": {"supported": True},
        "thinking": {
            "supported": True,
            "types": {
                "enabled": {"supported": False},
                "adaptive": {"supported": True},
                "disabled": {"supported": True},
            },
            "between_tools": {"supported": False},
        },
        "effort": {
            "supported": True,
            **{level: {"supported": True} for level in ("low", "medium", "high", "xhigh", "max")},
        },
    },
}
