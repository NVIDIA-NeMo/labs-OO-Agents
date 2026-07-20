# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test-only helpers for working with OTLP span data.

``_otlp_attrs_to_dict`` used to live in ``eval_pipeline.otlp_io``, which was
removed as dead production code. It is retained here because several tests still
need to flatten OTLP attribute arrays when asserting on span contents.
"""

from __future__ import annotations

from typing import Any


def _otlp_attrs_to_dict(attrs: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert OTLP attribute array [{key, value}] to a flat dict."""
    result: dict[str, Any] = {}
    for attr in attrs or []:
        key = attr.get("key", "")
        value_obj = attr.get("value", {})
        if "stringValue" in value_obj:
            result[key] = value_obj["stringValue"]
        elif "intValue" in value_obj:
            result[key] = int(value_obj["intValue"])
        elif "doubleValue" in value_obj:
            result[key] = float(value_obj["doubleValue"])
        elif "boolValue" in value_obj:
            result[key] = value_obj["boolValue"]
        elif "arrayValue" in value_obj:
            result[key] = value_obj["arrayValue"].get("values", [])
        elif "kvlistValue" in value_obj:
            result[key] = {
                kv["key"]: kv.get("value") for kv in value_obj["kvlistValue"].get("values", [])
            }
    return result
