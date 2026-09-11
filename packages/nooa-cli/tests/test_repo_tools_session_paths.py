# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session-aware repository processing in RepoTools.

When a shared session is wired (e.g. a Gym-hosted seeded sandbox), the repo
root lives in the session filesystem: host-side ``Path`` probes and reads
would wrongly report paths as missing or drop results. Validation, file
reads, searches, and anchor creation must all go through the session when
one is wired, and fall back to the host otherwise.
"""

from __future__ import annotations

import base64
import re as _re
import shlex
from pathlib import Path

import pytest
from nooa_cli.tools.repo_tools import RepoTools


class FakeSession:
    """A scripted session over an in-memory filesystem.

    Answers the exact command vocabulary RepoTools issues: ``test -e/-f/-d``
    probes, ``base64 -w 0`` reads, ``command -v rg``, ``rg``/``grep`` searches,
    and ``rg --files``/``find`` listings. Records every command for assertions.
    """

    def __init__(self, fs: dict[str, str], *, has_rg: bool = True) -> None:
        """Build the fake session from a mapping of absolute path to content."""
        self.fs = fs
        self.has_rg = has_rg
        self.calls: list[str] = []

    def _is_dir(self, target: str) -> bool:
        """Return True when any fake file lives under *target*."""
        return any(f.startswith(target.rstrip("/") + "/") for f in self.fs)

    async def run(self, command: str, timeout: float = 30.0) -> tuple[str, str, int]:
        """Answer one session command against the in-memory filesystem."""
        self.calls.append(command)
        try:
            tokens = shlex.split(command)
        except ValueError:
            return ("", "", 1)
        # Truncate at a bare pipe (e.g. the "| head -N" suffix); note the
        # regex pattern itself may contain quoted pipes, hence shlex first.
        if "|" in tokens:
            tokens = tokens[: tokens.index("|")]
        parts = [t for t in tokens if not t.startswith("2>")]
        if not parts:
            return ("", "", 1)

        if parts[0] == "test" and len(parts) >= 3:
            flag, target = parts[1], parts[-1]
            if flag == "-e":
                ok = target in self.fs or self._is_dir(target)
            elif flag == "-f":
                ok = target in self.fs
            elif flag == "-d":
                ok = self._is_dir(target)
            else:
                ok = False
            return ("", "", 0 if ok else 1)

        if parts[0] == "command":
            return ("", "", 0 if self.has_rg else 1)

        if parts[0] == "base64":
            target = parts[-1]
            if target in self.fs:
                return (base64.b64encode(self.fs[target].encode()).decode(), "", 0)
            return ("", "", 1)

        if parts[0] == "rg" and "--files" in parts:
            root = parts[parts.index("--files") + 1]
            files = sorted(f for f in self.fs if f.startswith(root))
            return ("\n".join(files), "", 0 if files else 1)

        if parts[0] == "find":
            root = parts[1]
            files = sorted(f for f in self.fs if f.startswith(root))
            return ("\n".join(files), "", 0 if files else 1)

        if parts[0] in ("rg", "grep"):
            pattern: str | None = None
            path: str | None = None
            for tok in parts[1:]:
                if tok.startswith("-") or tok.startswith("2>") or tok == "/dev/null":
                    continue
                if pattern is None:
                    pattern = tok
                else:
                    path = tok
            out: list[str] = []
            if pattern:
                for f in sorted(self.fs):
                    if path and not f.startswith(path):
                        continue
                    for i, line in enumerate(self.fs[f].splitlines(), 1):
                        try:
                            if _re.search(pattern, line, _re.IGNORECASE):
                                out.append(f"{f}:{i}:{line}")
                        except _re.error:
                            continue
            return ("\n".join(out), "", 0 if out else 1)

        return ("", "", 1)


FS = {
    "/app/mod.py": "def handler():\n    pass\n\n\nclass Widget:\n    def run(self):\n        pass\n",
    "/app/caller.py": "from mod import handler\n\nhandler()\n",
}


@pytest.mark.asyncio
async def test_symbols_probes_paths_via_session_when_wired() -> None:
    """symbols() must probe existence through the session, not the host."""
    session = FakeSession(FS)
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo.symbols(".")

    assert any(c.startswith("test -e") for c in session.calls), session.calls
    assert result.diagnostic is None


@pytest.mark.asyncio
async def test_symbols_reports_diagnostic_when_session_says_missing() -> None:
    """A session-reported missing path still yields a structured diagnostic."""
    session = FakeSession({})
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo.symbols("missing/pkg")

    assert result.diagnostic is not None
    assert result.diagnostic.code == "PATH_NOT_FOUND"  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_refs_probes_paths_via_session_when_wired() -> None:
    """refs() must probe existence through the session, not the host."""
    session = FakeSession(FS)
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo.refs("handler", path="/app")

    assert any(c.startswith("test -e") for c in session.calls), session.calls
    assert result.diagnostic is None


@pytest.mark.asyncio
async def test_host_fallback_still_works_without_session(tmp_path: Path) -> None:
    """Without a session, validation stays host-side and unchanged."""
    repo = RepoTools(root=tmp_path)

    result = await repo.symbols("missing/package")

    assert result.diagnostic is not None
    assert result.diagnostic.resolved_path == str((tmp_path / "missing/package").resolve())  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_file_probe_uses_session_when_wired() -> None:
    """The existence and file probes route through the wired session."""
    session = FakeSession(FS)
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    assert await repo._path_exists(Path("/app/mod.py")) is True
    assert await repo._path_is_file(Path("/app/mod.py")) is True
    assert await repo._path_exists(Path("/app/missing.py")) is False
    assert any(c.startswith("test -f") for c in session.calls), session.calls


@pytest.mark.asyncio
async def test_symbols_file_reads_session_only_content() -> None:
    """A session-only file yields symbols and anchors with real content."""
    session = FakeSession(FS)
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo.symbols("/app/mod.py")

    assert result.diagnostic is None
    assert result.matches, "expected symbol anchors from session content"
    assert "def handler" in result.text
    # Anchors carry the session-fetched line content, not empty host reads.
    assert any("handler" in anchor.text for anchor in result.matches)


@pytest.mark.asyncio
async def test_symbols_search_returns_session_only_results() -> None:
    """A definition search over session-only files returns matches and anchors."""
    session = FakeSession(FS)
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo.symbols(".", query="handler")

    assert result.diagnostic is None
    assert result.total_matches >= 1
    assert result.matches, "expected anchors built from session search results"


@pytest.mark.asyncio
async def test_refs_returns_session_only_results() -> None:
    """Reference search over session-only files returns matches and anchors."""
    session = FakeSession(FS)
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo.refs("handler", path="/app")

    assert result.diagnostic is None
    assert result.total_matches >= 1
    # The call site in caller.py must appear with content from the session.
    assert any("handler" in line for line in result.lines)


@pytest.mark.asyncio
async def test_repo_map_lists_session_files_without_rg() -> None:
    """repo_map falls back to find() when the session image lacks rg."""
    session = FakeSession(FS, has_rg=False)
    repo = RepoTools(root="/app", session=session)  # type: ignore[arg-type]

    result = await repo._repo_map()

    assert any("mod.py" in section for section in result.summary.splitlines())
    assert any(c.startswith("find") for c in session.calls), session.calls


@pytest.mark.asyncio
async def test_missing_path_diagnostic_is_not_found_on_host_collision(tmp_path: Path) -> None:
    """A path existing on the host but not in the session reports PATH_NOT_FOUND.

    Without the explicit ``reason`` the error would infer from the host
    filesystem and misclassify this collision as PATH_NOT_FILE.
    """
    (tmp_path / "pkg").mkdir()
    session = FakeSession({})  # session sees nothing under the root
    repo = RepoTools(root=tmp_path, session=session)  # type: ignore[arg-type]

    result = await repo.symbols("pkg")

    assert result.diagnostic is not None
    assert result.diagnostic.code == "PATH_NOT_FOUND"  # type: ignore[attr-defined]
    assert result.diagnostic.reason == "not_found"  # type: ignore[attr-defined]
