# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated compatibility imports; use :mod:`nooa_cli.tui.statusbar`."""

from .statusbar import StatusbarContext as ToolbarContext
from .statusbar import StatusbarProvider as ToolbarProvider
from .statusbar import StatusbarRegistry as ToolbarRegistry

__all__ = ["ToolbarContext", "ToolbarProvider", "ToolbarRegistry"]
