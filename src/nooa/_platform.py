# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Platform policy: native Windows is unsupported; use WSL 2.

``import nooa`` on native Windows raises :class:`UnsupportedPlatformError`
unless ``NOOA_SKIP_PLATFORM_CHECK=1`` is set (like Node.js's
``NODE_SKIP_PLATFORM_CHECK``). That check lives in the stdlib-only
:mod:`nooa._platform_policy` so it runs before other nooa modules load. With the
override, the pure-Python core and ``nooa connect`` are tested in CI; other
features may work but are unsupported.

Features that need POSIX (bash sessions, the sandbox, process groups, file
locks, POSIX signals) call :func:`require_posix` at the point of use, never at
import time. We do not port those features to Windows; use WSL 2 for them.
"""

from __future__ import annotations

from nooa._platform_policy import (
    SKIP_PLATFORM_CHECK_ENV,
    UNSUPPORTED_PLATFORM_MESSAGE,
    UnsupportedPlatformError,
    check_platform,
    is_native_windows,
    platform_check_skipped,
)
from nooa.errors import NemoOOAgentsRuntimeError

__all__ = [
    "SKIP_PLATFORM_CHECK_ENV",
    "UNSUPPORTED_PLATFORM_MESSAGE",
    "WSL_HINT",
    "PosixOnlyFeatureError",
    "UnsupportedPlatformError",
    "check_platform",
    "is_native_windows",
    "platform_check_skipped",
    "require_posix",
]

WSL_HINT = "Native Windows runs only the pure-Python core of nooa; use WSL 2 (https://learn.microsoft.com/windows/wsl/install) for this feature."


class PosixOnlyFeatureError(NemoOOAgentsRuntimeError, NotImplementedError):
    """A POSIX-only nooa feature was used on native Windows."""

    def __init__(self, feature: str) -> None:
        self.feature = feature
        super().__init__(f"{feature} requires Linux or macOS. {WSL_HINT}")


def require_posix(feature: str) -> None:
    """Raise :class:`PosixOnlyFeatureError` naming *feature* on native Windows."""
    if is_native_windows():
        raise PosixOnlyFeatureError(feature)
