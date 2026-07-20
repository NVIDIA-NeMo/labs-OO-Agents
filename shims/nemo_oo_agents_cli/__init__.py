# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backward-compatibility shim: ``import nemo_oo_agents_cli`` → ``import nooa_cli``."""

from nooa_cli import *  # noqa: F401, F403
from nooa_cli import main  # noqa: F401
