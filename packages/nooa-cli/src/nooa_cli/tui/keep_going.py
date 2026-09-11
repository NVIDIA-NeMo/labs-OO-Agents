# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility module for shared interactive policy."""

import sys
from importlib import import_module

sys.modules[__name__] = import_module("nooa_cli.interactive.keep_going")
