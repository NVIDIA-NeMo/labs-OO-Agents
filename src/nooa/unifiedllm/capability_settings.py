# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detected structured-output formats and developer-role support, and the settings that use them.

Connect detects two per-route facts and writes them into the model entry:

- ``structured_output_formats``: the ``response_format`` types the endpoint
  accepts, from ``{"json_schema", "json_object"}``. Absent means unknown.
- ``developer_role``: whether the endpoint accepts ``developer`` messages.
  Absent means unknown.

Two user settings choose how requests use them:

- ``structured_output_format``: ``auto`` (default), ``json_schema`` or
  ``json_object``. ``auto`` uses ``json_schema`` unless the only detected
  format is ``json_object``.
- ``instruction_role``: ``auto`` (default), ``system`` or ``developer``.
  ``auto`` uses ``developer`` only when ``developer_role`` was detected True.

An explicit choice that contradicts a known fact fails at load. Entries
without these keys send exactly the requests they sent before.
"""

import json
from typing import Any

from pydantic import BaseModel

from .errors import FailureKind, LLMFailure

STRUCTURED_OUTPUT_FORMATS = ("json_schema", "json_object")
STRUCTURED_OUTPUT_FORMAT_SETTINGS = ("auto", *STRUCTURED_OUTPUT_FORMATS)
INSTRUCTION_ROLE_SETTINGS = ("auto", "system", "developer")
# Entry keys this module owns. They describe one route, so a route override
# drops them like media_inputs and structured_output_mode.
CAPABILITY_KEYS = (
    "structured_output_formats",
    "developer_role",
    "structured_output_format",
    "instruction_role",
)


class CapabilitySettingError(LLMFailure, ValueError):
    """A capability fact or setting is invalid or contradicts a detected fact (CONFIG)."""

    def __init__(self, message: str):
        super().__init__(FailureKind.CONFIG, message)


def validate_structured_output_formats(value: object) -> frozenset[str] | None:
    """Return the detected formats as a frozenset, or None when unknown."""
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple, set, frozenset)):
        raise CapabilitySettingError(
            "structured_output_formats must be a list of json_schema/json_object or None"
        )
    formats = frozenset(value)
    if not formats <= set(STRUCTURED_OUTPUT_FORMATS) or any(type(f) is not str for f in value):
        raise CapabilitySettingError(
            "structured_output_formats may contain only json_schema and json_object"
        )
    return formats


def validate_developer_role(value: object) -> bool | None:
    """Return the detected developer-role support (True/False), or None when unknown."""
    if value is not None and type(value) is not bool:
        raise CapabilitySettingError("developer_role must be true, false or null")
    return value


def validate_structured_output_format(value: object) -> str | None:
    """Return a valid ``structured_output_format`` setting (None means auto)."""
    if value is not None and value not in STRUCTURED_OUTPUT_FORMAT_SETTINGS:
        raise CapabilitySettingError(
            "structured_output_format must be 'auto', 'json_schema' or 'json_object'"
        )
    return value


def validate_instruction_role(value: object) -> str | None:
    """Return a valid ``instruction_role`` setting (None means auto)."""
    if value is not None and value not in INSTRUCTION_ROLE_SETTINGS:
        raise CapabilitySettingError("instruction_role must be 'auto', 'system' or 'developer'")
    return value


def resolve_structured_output_format(setting: object, detected: object) -> str:
    """Choose the format typed requests use: ``json_schema`` or ``json_object``.

    ``auto``/None picks ``json_schema`` when it is supported or support is
    unknown, and ``json_object`` only when that is the only detected format.
    An explicit choice must be in the detected set when the set is known.

    Raises:
        CapabilitySettingError: For invalid values or a contradicted choice.
    """
    setting = validate_structured_output_format(setting)
    formats = validate_structured_output_formats(detected)
    if setting in (None, "auto"):
        if formats is not None and "json_schema" not in formats and "json_object" in formats:
            return "json_object"
        return "json_schema"
    if formats is not None and setting not in formats:
        known = ", ".join(sorted(formats)) or "none"
        raise CapabilitySettingError(
            f"structured_output_format {setting!r} is not supported by this route "
            f"(detected structured_output_formats: {known}); "
            "choose a detected format, use 'auto', or run Connect again"
        )
    return setting


def resolve_instruction_role(setting: object, detected: object) -> str:
    """Choose the role instruction messages use: ``system`` or ``developer``.

    ``auto``/None picks ``developer`` only when support was detected True.

    Raises:
        CapabilitySettingError: For invalid values, or ``developer`` on a route
            detected not to support it.
    """
    setting = validate_instruction_role(setting)
    supported = validate_developer_role(detected)
    if setting in (None, "auto"):
        return "developer" if supported is True else "system"
    if setting == "developer" and supported is False:
        raise CapabilitySettingError(
            "instruction_role 'developer' is not supported by this route "
            "(detected developer_role: false); use 'system' or 'auto', or run Connect again"
        )
    return setting


def json_object_instruction(output_model: type[BaseModel]) -> dict[str, str]:
    """The final system message carrying the schema in JSON mode.

    JSON mode only guarantees syntactically valid JSON, so the schema travels in
    the prompt. The text contains the word "JSON", which DeepSeek's JSON mode
    requires. The reply is still validated client-side against ``output_model``.
    """
    schema = json.dumps(output_model.model_json_schema())
    return {
        "role": "system",
        "content": f"Respond with one JSON object that conforms to this JSON schema:\n{schema}",
    }


def map_instruction_roles(messages: list[Any], role: str, developer_role: bool | None) -> list[Any]:
    """Return ``messages`` with instruction roles mapped for this route.

    - ``role == "developer"``: system messages are sent as developer messages.
    - Otherwise, when the route is known not to support ``developer``
      (``developer_role is False``), caller developer messages are sent as
      system messages.
    - Otherwise the list is returned unchanged (the same object).

    Only plain message dicts are copied; other items (replayed turns, cache
    boundaries) are kept as they are.
    """
    if role == "developer":
        source, target = "system", "developer"
    elif developer_role is False:
        source, target = "developer", "system"
    else:
        return messages
    if not any(type(m) is dict and m.get("role") == source for m in messages):
        return messages
    return [
        {**m, "role": target} if type(m) is dict and m.get("role") == source else m
        for m in messages
    ]
