# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Connect checks for structured-output formats and the ``developer`` role.

Three optional test calls (``plan(..., capability_checks=True)``), only for
the Chat Completions and Responses interfaces:

- ``format:json_schema``: a tiny strict JSON schema in ``response_format``
  (Responses: ``text.format``).
- ``format:json_object``: JSON mode (``{"type": "json_object"}``). Sent only
  when the endpoint rejected ``json_schema`` as unsupported, or when the entry
  selects ``structured_output_format: json_object``.
- ``role:developer``: one ``developer`` message before the user message.

A rejection counts as "unsupported" only for HTTP 400/422 whose error names the
tested feature (the format field, or the developer role) together with a
refusal phrase. Any other failure leaves the fact unknown. The entry gets one
``structured_output_format`` value: ``json_schema`` when it was accepted, else
``json_object`` when that was accepted, else no key. Each format's check result
is recorded only in ``provenance["structured_output_format"]``. The role result
is written to ``developer_role`` with a ``provenance`` note. Provider error
text is never stored. Other interfaces (Anthropic
Messages) record ``developer_role: false`` without a call and have no
``json_object`` format to test.
"""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

CAPABILITY_PROBES = ("format:json_schema", "format:json_object", "role:developer")
# Interfaces whose requests carry response_format (or Responses text.format)
# and accept a developer role.
PROBED_STYLES = ("chat", "responses")
PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_REFUSAL = re.compile(
    r"unsupported|not supported|does not support|doesn't support|unavailable|not available"
    r"|unknown|unrecognized|not allowed|invalid|not permitted|must be one of|unexpected"
)
_FORMAT_TERMS = ("response_format", "json_schema", "text.format", "json schema", "format")
_ROLE_TERMS = ("developer", "role")


def is_capability_probe(name: str) -> bool:
    """True for the format/role checks of this module."""
    return name in CAPABILITY_PROBES


def capability_bodies(api_style: str, body: dict) -> list[tuple[str, dict]]:
    """Return ``(name, request body)`` for each check, built from the routing ``body``.

    Empty for interfaces other than Chat Completions and Responses.
    """
    if api_style not in PROBED_STYLES:
        return []
    key = {"responses": "input"}.get(api_style, "messages")
    result = []
    for name, fmt in (
        (
            "format:json_schema",
            {"type": "json_schema", "name": "probe", "strict": True, "schema": PROBE_SCHEMA},
        ),
        ("format:json_object", {"type": "json_object"}),
    ):
        probe = deepcopy(body)
        probe[key] = [
            {
                "role": "user",
                "content": 'Reply with the JSON object {"ok": true} and nothing else.',
            }
        ]
        if key == "input":
            probe["text"] = {"format": deepcopy(fmt)}
        elif fmt["type"] == "json_schema":
            probe["response_format"] = {
                "type": "json_schema",
                "json_schema": {k: deepcopy(v) for k, v in fmt.items() if k != "type"},
            }
        else:
            probe["response_format"] = deepcopy(fmt)
        result.append((name, probe))
    role = deepcopy(body)
    role[key] = [
        {"role": "developer", "content": "Answer with the single word ok."},
        {"role": "user", "content": "Reply now."},
    ]
    result.append(("role:developer", role))
    return result


def probe_entry(entry: dict) -> dict:
    """The entry a capability check builds its client from.

    Earlier detected facts and the user's choices are left out so the check
    sends exactly its planned format and roles.
    """
    from nooa.unifiedllm.capability_settings import CAPABILITY_KEYS

    return {k: v for k, v in entry.items() if k not in CAPABILITY_KEYS}


def observed_on_wire(name: str, wire: dict) -> bool:
    """Whether the sent request carries what check ``name`` tests."""
    if name.startswith("format:"):
        kind = name.removeprefix("format:")
        response_format = wire.get("response_format")
        text = wire.get("text")
        fmt = (
            response_format
            if isinstance(response_format, dict)
            else text.get("format")
            if isinstance(text, dict)
            else None
        )
        return isinstance(fmt, dict) and fmt.get("type") == kind
    if name == "role:developer":
        items = wire.get("messages", wire.get("input"))
        return isinstance(items, list) and any(
            isinstance(item, dict) and item.get("role") == "developer" for item in items
        )
    return False


def rejected_as_unsupported(name: str, exc: BaseException, status: Any) -> bool:
    """Whether a failed check shows the endpoint does not support the feature.

    Requires HTTP 400/422 and an error that names the tested feature together
    with a refusal phrase (for example DeepSeek's "This response_format type is
    unavailable now", or "Invalid value: 'developer'" for a role).
    """
    if status not in (400, 422):
        return False
    parts = [str(exc)]
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict):
            parts.extend(str(error.get(k, "")) for k in ("message", "param", "code"))
    text = " ".join(parts).lower()
    terms = _ROLE_TERMS if name == "role:developer" else _FORMAT_TERMS
    if name == "role:developer" and "developer" not in text:
        return False
    return any(term in text for term in terms) and bool(_REFUSAL.search(text))


def json_object_skip_reason(entry: dict, records: dict) -> str | None:
    """Why the json_object check is not sent, or None to send it."""
    if entry.get("structured_output_format") == "json_object":
        return None
    schema = records.get("format:json_schema") or {}
    if schema.get("outcome") == "unsupported":
        return None
    if schema.get("outcome") == "accepted":
        return "json_schema is supported; json_object is not needed"
    return "json_schema was not shown to be unsupported; json_object not tested"


def apply_capability_results(entry: dict, records: dict) -> None:
    """Write the detected settings from the check records into ``entry`` (in place).

    ``structured_output_format`` gets ONE value: ``json_schema`` when that
    check was accepted, else ``json_object`` when that was accepted. When
    both were rejected as unsupported the key is removed. Inconclusive checks
    leave the entry's existing value. A conclusive result replaces any
    existing value, whoever set it.
    Each format's result is recorded only under
    ``provenance["structured_output_format"]``; ``developer_role`` evidence
    under ``provenance["developer_role"]``.
    """
    provenance = entry.setdefault("provenance", {})
    now = datetime.now(UTC).isoformat()
    outcomes = {
        name.removeprefix("format:"): (records.get(name) or {}).get("outcome", "not_probed")
        for name in ("format:json_schema", "format:json_object")
    }
    detected: str | None = None
    settled = True
    if outcomes["json_schema"] == "accepted":
        detected = "json_schema"
    elif outcomes["json_schema"] == "unsupported" and outcomes["json_object"] == "accepted":
        detected = "json_object"
    elif not (
        outcomes["json_schema"] == "unsupported" and outcomes["json_object"] == "unsupported"
    ):
        settled = False  # Inconclusive: neither format is known to work or fail.
    if any(name in records for name in ("format:json_schema", "format:json_object")):
        if detected is not None:
            entry["structured_output_format"] = detected
        elif settled:
            entry.pop("structured_output_format", None)
        provenance["structured_output_format"] = {
            "source": "connect",
            "checks": outcomes,
            "value": detected,
            "checked_at": now,
        }
    role = (records.get("role:developer") or {}).get("outcome")
    if role is not None:
        value = True if role == "accepted" else False if role == "unsupported" else None
        provenance["developer_role"] = {
            "source": "connect",
            "check": role,
            "value": value,
            "checked_at": now,
        }
        if value is not None:
            entry["developer_role"] = value


def not_applicable(entry: dict) -> None:
    """Record the facts of an interface without these checks (Anthropic Messages)."""
    entry["developer_role"] = False
    entry.setdefault("provenance", {})["developer_role"] = {
        "source": "connect",
        "check": "not_applicable",
        "value": False,
        "reason": "This interface has no developer role; instructions are sent as system",
    }
