# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package version helpers."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _metadata_version

_PACKAGE_NAME = "nooa"
_UNKNOWN_VERSION = "0.0.0+unknown"


def get_version(package_name: str = _PACKAGE_NAME) -> str:
    """Return the installed package version, or a source-tree fallback."""
    try:
        return _metadata_version(package_name)
    except PackageNotFoundError:
        return _UNKNOWN_VERSION


__version__ = get_version()
