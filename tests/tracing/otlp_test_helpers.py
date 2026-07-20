# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for reading OTLP JSONL test output."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _otlp_attr_value(val: dict) -> Any:
    """Extract the Python value from an OTLP ``AnyValue`` dict."""
    if "stringValue" in val:
        return val["stringValue"]
    if "intValue" in val:
        return int(val["intValue"])
    if "doubleValue" in val:
        return val["doubleValue"]
    if "boolValue" in val:
        return val["boolValue"]
    if "arrayValue" in val:
        return [_otlp_attr_value(v) for v in val["arrayValue"].get("values", [])]
    return val


def otlp_attrs_to_dict(attrs: list[dict]) -> dict[str, Any]:
    """Convert an OTLP ``[{key, value}]`` attribute list to a flat dict."""
    return {a["key"]: _otlp_attr_value(a["value"]) for a in attrs}


def read_otlp_jsonl_spans(path: Path) -> list[dict[str, Any]]:
    """Read an OTLP JSONL file and return flat span dicts.

    Each returned dict has:
      - ``name``, ``traceId``, ``spanId``, etc. (from the OTLP span)
      - ``attributes`` — flat ``{key: value}`` dict
      - ``resource_attributes`` — flat ``{key: value}`` dict from the envelope
    """
    spans: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            envelope = json.loads(line)
            for rs in envelope.get("resourceSpans", []):
                res_attrs = otlp_attrs_to_dict(rs.get("resource", {}).get("attributes", []))
                for ss in rs.get("scopeSpans", []):
                    for span in ss.get("spans", []):
                        flat = dict(span)
                        flat["attributes"] = otlp_attrs_to_dict(span.get("attributes", []))
                        flat["resource_attributes"] = res_attrs
                        spans.append(flat)
    return spans


def read_all_otlp_jsonl_spans(directory: str | Path) -> list[dict[str, Any]]:
    """Read all ``.jsonl`` files in *directory* and return flat span dicts."""
    spans: list[dict[str, Any]] = []
    for f in Path(directory).glob("*.jsonl"):
        spans.extend(read_otlp_jsonl_spans(f))
    return spans
