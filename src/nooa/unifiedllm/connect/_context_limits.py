# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context-window values for a model entry, limited by what the endpoint reports.

The user declares a context window. If the endpoint's model listing reports a
smaller limit for exactly the same model ID, that limit is used instead. Model
listing facts are stored under the entry's ``provenance`` (the record of where
each configured value came from) together with a "route scope": a non-secret
fingerprint of the endpoint and model the facts were read from, so they are not
reused after the entry is pointed somewhere else.
"""

from copy import deepcopy

from ._verification import effective_route_scope


def bound_context_window(declared: int, *, model: str, endpoint_model: dict | None = None) -> dict:
    """Return the declared and effective context window for ``model``.

    ``endpoint_model`` is the endpoint's listing record. It is used only if its
    ``id`` equals ``model``, and only its positive integer ``context_window``,
    ``context_length`` and ``max_input_tokens`` fields count; the input-only
    limit is treated as a cautious upper bound as well. Output limits, formats
    and guessed matches against other models or catalogues are ignored.

    The effective value is the declared value lowered (never raised) to the
    smallest such limit. The result records ``declared``, ``effective`` and
    which source set the effective value. It does not set a reply token limit.

    Raises ValueError if ``declared`` is not a positive int.
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
    """Return the stored endpoint limits of ``entry`` if they still apply, else ``{}``.

    The limits apply only if the route scope stored with them equals the
    entry's current route scope (same endpoint and model). Older entries that
    have no stored scope are kept as history but not reused. Changes that do
    not affect the endpoint or model (temperature, profile, token limits, ...)
    do not change the route scope. The result includes the model ``id``.
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
    """Return a copy of ``entry`` with a declared context window applied.

    The effective window (see ``bound_context_window``) is stored as
    ``context_window`` and the evidence under ``provenance``. When
    ``endpoint_model`` matches ``model`` its limits are also stored, tagged with
    the current route scope.

    This does not set reply token limits or update the list of test calls to
    rerun: callers must run ``configure_entry``/``refresh_plan`` afterward.
    Callers importing an entry must check its stored endpoint limits with
    ``scoped_endpoint_limits`` before passing them in as ``endpoint_model``.
    """
    result = deepcopy(entry)
    evidence = bound_context_window(declared, model=model, endpoint_model=endpoint_model)
    result["context_window"] = evidence["effective"]
    provenance = result.setdefault("provenance", {})
    provenance["context_window"] = evidence
    # A listing record passed in by the caller with the matching ID is treated
    # as current endpoint data. (Stored data must pass scoped_endpoint_limits
    # before it is passed in here.)
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
