# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional per-model requirements, checked against immutable route facts at load.

No requirements means unknown facts are allowed. This layer never discovers
capabilities or infers tools from a request. Legacy tools=True is historical
optimism, not live verification. Explicit advertised True satisfies TOOLS; this
comparison does not certify a provider or infer support from its name.
"""

import re
from dataclasses import dataclass
from enum import EnumType, IntFlag

from .adapters.base import Capabilities
from .errors import FailureKind, LLMFailure


class _NeedType(EnumType):
    def __call__(cls, value):
        # Enum's cached member lookup precedes _missing_: validate before bool
        # and equal-valued floats can become ordinary members via that lookup.
        if type(value) not in (int, cls) or int(value) < 0 or int(value) & ~31:
            raise LLMFailure(FailureKind.CONFIG, "Invalid requirement flags")
        return super().__call__(value)


class Need(IntFlag, metaclass=_NeedType):
    """Known nonnegative bits only; negative IntFlag inversion remains invalid."""

    NONE = 0
    IMAGES = 1
    PDFS = 2
    AUDIO = 4
    VIDEO = 8
    TOOLS = 16

    @classmethod
    def _missing_(cls, value):
        if type(value) is not int or value < 0 or value & ~31:
            raise LLMFailure(FailureKind.CONFIG, "Invalid requirement flags")
        return super()._missing_(value)


def _validate_values(flags, minimum):
    if type(flags) is not Need or int(flags) < 0 or int(flags) & ~31:
        raise LLMFailure(FailureKind.CONFIG, "Invalid requirement flags")
    if minimum is not None and (type(minimum) is not int or minimum <= 0):
        raise LLMFailure(FailureKind.CONFIG, "Required context must be a positive integer or None")


@dataclass(frozen=True)
class Requirements:
    flags: Need = Need.NONE
    min_context: int | None = None

    def __post_init__(self):
        _validate_values(self.flags, self.min_context)


def _normalize_require(require):
    if require is None:
        return Requirements()
    if type(require) is Need:
        return Requirements(require)
    if type(require) is Requirements:
        _validate_values(require.flags, require.min_context)
        return require
    raise LLMFailure(FailureKind.CONFIG, "require must be Need, Requirements or None")


def _validate_requirements(capabilities, require=None, *, label="model"):
    """SDK-free one-time check, also suitable for a future injected model slot.

    Only explicit advertised facts satisfy a requirement. Historical optimistic
    tools=True is accepted, not certified. Nothing is retained per request.
    Labels are bounded alias/slot identifiers, not provider/config diagnostics.
    """
    required = _normalize_require(require)
    safe_label = label if type(label) is str and re.fullmatch(r"[\w.-]{1,64}", label) else "model"
    if required.flags == Need.NONE and required.min_context is None:
        return
    if type(capabilities) is not Capabilities:
        raise LLMFailure(FailureKind.CONFIG, f"{safe_label}: unknown capabilities")
    missing = []
    for flag, kind in (
        (Need.IMAGES, "image"),
        (Need.PDFS, "pdf"),
        (Need.AUDIO, "audio"),
        (Need.VIDEO, "video"),
    ):
        if required.flags & flag and kind not in (capabilities.media_kinds or ()):
            missing.append(f"missing/unknown {kind} capability")
    if required.flags & Need.TOOLS:
        if capabilities.tools is None:
            missing.append("unknown tools capability")
        elif capabilities.tools is not True:
            missing.append("missing tools capability")
    if required.min_context is not None:
        window = capabilities.context_window
        if type(window) is not int or window < required.min_context:
            missing.append("missing/unknown minimum context capability")
    if missing:
        raise LLMFailure(FailureKind.CONFIG, f"{safe_label}: " + "; ".join(missing))
