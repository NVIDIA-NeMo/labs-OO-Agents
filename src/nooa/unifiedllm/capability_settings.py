# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Structured-output format and developer-role settings.

Two settings choose how requests are sent:

- ``structured_output_format``: ``json_schema`` or ``json_object``; absent
  (None) keeps the request format entries had before this setting existed
  (``json_schema``, or the native mode). Connect writes it after its
  capability check: ``json_schema`` when the endpoint accepts it, else
  ``json_object`` when that works, else no key. The per-format check results
  are kept only in the entry's ``provenance``. ``json_object`` sends JSON mode
  and the return type's schema as a final instruction message.
- ``instruction_role``: ``auto`` (default), ``system`` or ``developer``.
  ``auto`` uses ``developer`` only when ``developer_role`` was detected True.

Connect also records one detected fact, ``developer_role``: whether the
endpoint accepts ``developer`` messages (absent means unknown). An explicit
``instruction_role: developer`` that contradicts it fails at load. Entries
without these keys send exactly the requests they sent before.
"""

import json
from typing import Any

from pydantic import BaseModel

from .errors import FailureKind, LLMFailure

STRUCTURED_OUTPUT_FORMATS = ("json_schema", "json_object")
INSTRUCTION_ROLE_SETTINGS = ("auto", "system", "developer")
# Entry keys this module owns. They describe one route, so a route override
# drops them like media_inputs and structured_output_mode.
CAPABILITY_KEYS = (
    "developer_role",
    "structured_output_format",
    "instruction_role",
)


class CapabilitySettingError(LLMFailure, ValueError):
    """A capability fact or setting is invalid or contradicts a detected fact (CONFIG)."""

    def __init__(self, message: str):
        super().__init__(FailureKind.CONFIG, message)


def validate_developer_role(value: object) -> bool | None:
    """Return the detected developer-role support (True/False), or None when unknown."""
    if value is not None and type(value) is not bool:
        raise CapabilitySettingError("developer_role must be true, false or null")
    return value


def validate_structured_output_format(value: object) -> str | None:
    """Return a valid ``structured_output_format`` setting, or None when absent.

    Raises:
        CapabilitySettingError: For any other value, including ``auto``.
    """
    if value is not None and value not in STRUCTURED_OUTPUT_FORMATS:
        raise CapabilitySettingError(
            f"structured_output_format must be 'json_schema' or 'json_object' "
            f"(or absent), not {value!r}"
        )
    return value


def validate_instruction_role(value: object) -> str | None:
    """Return a valid ``instruction_role`` setting (None means auto)."""
    if value is not None and value not in INSTRUCTION_ROLE_SETTINGS:
        raise CapabilitySettingError("instruction_role must be 'auto', 'system' or 'developer'")
    return value


def resolve_structured_output_format(setting: object) -> str:
    """Choose the format typed requests use: ``json_schema`` or ``json_object``.

    Absent (None) means ``json_schema``, the request format used before this
    setting existed.

    Raises:
        CapabilitySettingError: For an invalid value.
    """
    return validate_structured_output_format(setting) or "json_schema"


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
