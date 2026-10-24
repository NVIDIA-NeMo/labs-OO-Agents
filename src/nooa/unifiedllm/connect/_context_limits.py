# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Literal context declarations bounded only by the exact selected endpoint."""

from copy import deepcopy

from ._verification import effective_route_scope


def bound_context_window(declared: int, *, model: str, endpoint_model: dict | None = None) -> dict:
    """Return declared/effective context evidence, without allocating a reply cap.

    Only positive integer limits on a record with the exact selected ID count.
    Both total-context fields and an input-only ceiling are conservative bounds;
    output limits, formats and guessed model/catalogue matches are not evidence.
    The declaration is clamped (not raised) to the smallest literal bound.
    """
    if type(declared) is not int or declared <= 0:
        raise ValueError("context_window must be a positive integer")
    server = endpoint_model or {}
    limits = []
    if server.get("id") == model:
        limits = [
            (server[key], "endpoint_input_limit" if key == "max_input_tokens" else "endpoint")
            for key in ("context_window", "context_length", "max_input_tokens")
            if type(server.get(key)) is int and server[key] > 0
        ]
    ceiling, source = min(limits, key=lambda item: item[0]) if limits else (None, "user")
    effective = min(declared, ceiling) if ceiling is not None else declared
    return {
        "source": "user",
        "value": declared,
        "declared": declared,
        "effective": effective,
        "effective_source": source if ceiling is not None and ceiling < declared else "user",
    }


def scoped_endpoint_limits(entry: dict) -> dict:
    """Reuse exact endpoint facts only in their recorded non-secret route scope.

    Old unstamped metadata remains historical. Behavior edits (temperature,
    profile, caps, etc.) do not change endpoint-fact identity.
    """
    provenance = entry.get("provenance") or {}
    try:
        matches = provenance.get("endpoint_limit_scope") == effective_route_scope(entry)
    except (ValueError, TypeError):
        matches = False
    if not matches or not provenance.get("endpoint_model_id"):
        return {}
    return {
        **deepcopy(provenance.get("endpoint_limits") or {}),
        "id": provenance["endpoint_model_id"],
    }


def apply_context_window(
    entry: dict, declared: int, *, model: str, endpoint_model: dict | None = None
) -> dict:
    """Copy an entry with a bounded explicit declaration and its provenance.

    This does not configure reply caps or refresh probes: callers must run
    configure_entry/refresh_plan afterward. Importers must validate stored endpoint facts with
    scoped_endpoint_limits before passing them as an ID-bearing endpoint_model.
    """
    result = deepcopy(entry)
    evidence = bound_context_window(declared, model=model, endpoint_model=endpoint_model)
    result["context_window"] = evidence["effective"]
    provenance = result.setdefault("provenance", {})
    provenance["context_window"] = evidence
    # Explicit ID-bearing records supplied by the caller are current endpoint
    # facts. Imported provenance must first pass scoped_endpoint_limits.
    if (endpoint_model or {}).get("id") == model:
        old = scoped_endpoint_limits(result)
        provenance["endpoint_limits"] = {
            **({"url": old["url"]} if "url" in old else {}),
            **{
                key: endpoint_model[key]
                for key in (
                    "context_window",
                    "context_length",
                    "max_input_tokens",
                    "max_output_tokens",
                )
                if type(endpoint_model.get(key)) is int and endpoint_model[key] > 0
            },
        }
        provenance["endpoint_model_id"] = model
        provenance["endpoint_limit_scope"] = effective_route_scope(result)
    return result
