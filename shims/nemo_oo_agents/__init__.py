# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: ``import nemo_oo_agents`` → ``import nooa``."""

from nooa import *  # noqa: F401, F403
from nooa import __version__  # noqa: F401
