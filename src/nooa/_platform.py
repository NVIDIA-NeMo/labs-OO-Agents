# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform policy: native Windows runs the pure-Python core only.

Features that need POSIX (bash sessions, the sandbox, process groups, file
locks, POSIX signals) call :func:`require_posix` at the point of use, never at
import time, so ``import nooa`` and in-memory agents keep working on Windows.
We do not port those features to Windows; use WSL 2 for them.
"""

from __future__ import annotations

import sys

from nooa.errors import NemoOOAgentsRuntimeError

CONNECT_SAVE_FEATURE = "Saving a model with nooa connect (the registry write uses an fcntl lock)"
WSL_HINT = "Native Windows runs only the pure-Python core of nooa; use WSL 2 (https://learn.microsoft.com/windows/wsl/install) for this feature."


class PosixOnlyFeatureError(NemoOOAgentsRuntimeError, NotImplementedError):
    """A POSIX-only nooa feature was used on native Windows."""

    def __init__(self, feature: str) -> None:
        self.feature = feature
        super().__init__(f"{feature} requires Linux or macOS. {WSL_HINT}")


def is_native_windows() -> bool:
    """True on native Windows (not WSL, which reports ``linux``)."""
    return sys.platform == "win32"


def require_posix(feature: str) -> None:
    """Raise :class:`PosixOnlyFeatureError` naming *feature* on native Windows."""
    if is_native_windows():
        raise PosixOnlyFeatureError(feature)
