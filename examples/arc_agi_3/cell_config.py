# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reject obsolete per-cell isolation requests before starting ARC agents."""

import os

REMOVED_CELL_SANDBOX = "The per-cell CodeAct OS sandbox was removed. Migrate isolation to the whole agent (--sandbox drop/full or an external container/VM). In-process cell_timeout is cooperative; hard per-cell deadlines and kernel resource limits are unavailable."


def reject_cell_sandbox_environment() -> None:
    removed = sorted(key for key in os.environ if key.startswith("ARC_SANDBOX_"))
    if removed:
        raise ValueError(f"{', '.join(removed)}: {REMOVED_CELL_SANDBOX}")
