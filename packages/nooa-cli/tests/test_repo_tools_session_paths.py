# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-aware path validation in RepoTools.

When a shared session is wired (e.g. a Gym-hosted seeded sandbox), the repo
root lives in the session's filesystem and a host-side ``Path.exists()`` probe
wrongly reports valid paths as missing. The existence and file probes must go
through the session when one is wired, and fall back to the host otherwise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest
from nooa_cli.tools.repo_tools import RepoTools


class _FakeSession:
    """Records ``test -e``/``test -f`` probes; answers from a fake filesystem."""

    def __init__(self, existing: set[str], files: set[str]) -> None:
        self.existing = existing
        self.files = files
        self.calls: list[str] = []

    async def run(self, command: str, timeout: float = 30.0) -> Tuple[str, str, int]:
        self.calls.append(command)
        import shlex

        parts = shlex.split(command)
        if len(parts) >= 3 and parts[0] == "test":
            target = parts[-1]
            if parts[1] == "-e":
                return ("", "", 0 if target in self.existing else 1)
            if parts[1] == "-f":
                return ("", "", 0 if target in self.files else 1)
        return ("", "", 1)


@pytest.mark.asyncio
async def test_symbols_probes_paths_via_session_when_wired() -> None:
    # Root is a container path that does NOT exist on the host.
    session = _FakeSession(existing={"/app"}, files=set())
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    # symbols(".") must probe "." via the session, not the host filesystem.
    result = await repo.symbols(".")

    assert any("test -e" in c for c in session.calls), session.calls
    # The result is NOT a path-resolution failure: the session saw /app.
    assert result.diagnostic is None


@pytest.mark.asyncio
async def test_symbols_reports_diagnostic_when_session_says_missing() -> None:
    session = _FakeSession(existing=set(), files=set())
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo.symbols("missing/pkg")

    assert result.diagnostic is not None
    assert result.diagnostic.code == "PATH_NOT_FOUND"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refs_probes_paths_via_session_when_wired() -> None:
    session = _FakeSession(existing={"/app/src"}, files=set())
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo.refs("handler", path="/app/src")

    assert any("test -e" in c for c in session.calls), session.calls
    assert result.diagnostic is None


@pytest.mark.asyncio
async def test_host_fallback_still_works_without_session(tmp_path: Path) -> None:
    repo = RepoTools(root=tmp_path)

    result = await repo.symbols("missing/package")

    assert result.diagnostic is not None
    assert result.diagnostic.resolved_path == str((tmp_path / "missing/package").resolve())  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_file_probe_uses_session_when_wired() -> None:
    session = _FakeSession(existing={"/app/a.py"}, files={"/app/a.py"})
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    # Direct probe helpers.
    assert await repo._path_exists(Path("/app/a.py")) is True
    assert await repo._path_is_file(Path("/app/a.py")) is True
    assert await repo._path_exists(Path("/app/missing.py")) is False
    assert any("test -f" in c for c in session.calls), session.calls
