# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility module for the shared interactive implementation."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("nooa_cli.interactive.mcp_registry")
