# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deprecated compatibility imports; use :mod:`nooa_cli.tui.statusbar`."""

from .statusbar import (
    LEGACY_TOOLBAR_ENTRY_POINT as TOOLBAR_ENTRY_POINT,
)
from .statusbar import (
    StatusbarContext as ToolbarContext,
)
from .statusbar import (
    StatusbarProvider as ToolbarProvider,
)
from .statusbar import (
    StatusbarRegistry as ToolbarRegistry,
)

__all__ = ["TOOLBAR_ENTRY_POINT", "ToolbarContext", "ToolbarProvider", "ToolbarRegistry"]
