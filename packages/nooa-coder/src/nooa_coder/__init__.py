# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session layer, session tree and in-process host for NOOA interactive agents."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("nooa-coder")
except PackageNotFoundError:  # pragma: no cover - running from a source tree without install
    __version__ = "0.0.0"

__all__ = ["__version__"]
