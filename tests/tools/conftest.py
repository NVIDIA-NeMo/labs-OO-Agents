# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-start check for shell-tool test prerequisites.

`test_shell_tools_modern.py` skipif-drops 145 tests when `rg` or `grep` is
missing from `PATH`. A skip is indistinguishable from a pass in the pytest
summary, so a base image that stops shipping ripgrep would silently erase
that coverage. Fail loud in CI; warn locally.
"""

from __future__ import annotations

import os
import shutil

import pytest


def _missing_prereqs() -> list[str]:
    return [tool for tool in ("rg", "grep") if shutil.which(tool) is None]


def _prereq_message(missing: list[str]) -> str:
    return (
        f"tests/tools/ prerequisite missing on PATH: {', '.join(missing)}. "
        "Install ripgrep (`apt install ripgrep` / `brew install ripgrep`); "
        "see tests/README.md."
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    # CI must fail loud rather than silently skip 145 tests. Locally, the
    # header hook below surfaces a banner but the run continues — a
    # contributor running `pytest tests/agents/` should not be forced to
    # install a dep for a suite they are not touching.
    missing = _missing_prereqs()
    if missing and os.environ.get("CI"):
        pytest.exit(_prereq_message(missing), returncode=1)


def pytest_report_header(config: pytest.Config) -> str | None:
    missing = _missing_prereqs()
    if not missing:
        return None
    return f"WARNING: {_prereq_message(missing)}"
