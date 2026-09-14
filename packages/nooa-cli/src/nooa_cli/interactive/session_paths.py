# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared session location for native and protocol hosts."""

import os
from pathlib import Path

from nooa.paths import get_project_dir


def session_directory(workspace: Path | None = None) -> Path:
    """Resolve an optional isolated store without changing skills or workspace.

    ``NOOA_SESSIONS_DIR`` is useful for independent acceptance-test copies.
    Normal launches retain the existing project-local session directory.
    """
    configured = os.environ.get("NOOA_SESSIONS_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return workspace / ".nooa" / "sessions" if workspace else get_project_dir("sessions")
