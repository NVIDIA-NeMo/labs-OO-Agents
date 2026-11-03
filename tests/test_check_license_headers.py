# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the repository SPDX header scanner."""

from __future__ import annotations

import subprocess
from pathlib import Path

from scripts.check_license_headers import source_python_files


def _python_file(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("print('test')\n")
    return path


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def test_source_files_follow_standard_git_excludes(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    (tmp_path / ".gitignore").write_text("/tmp/\n")
    with (tmp_path / ".git" / "info" / "exclude").open("a") as exclude:
        exclude.write("\n.codex/\n")

    tracked_source = _python_file(tmp_path / "src" / "tracked.py")
    untracked_source = _python_file(tmp_path / "src" / "new.py")
    ignored_by_gitignore = _python_file(tmp_path / "tmp" / "generated.py")
    ignored_by_info_exclude = _python_file(tmp_path / ".codex" / "external.py")
    _git(tmp_path, "add", ".gitignore", "src/tracked.py")

    found = set(source_python_files(tmp_path))

    assert tracked_source in found
    assert untracked_source in found
    assert ignored_by_gitignore not in found
    assert ignored_by_info_exclude not in found


def test_source_files_include_tracked_files_that_match_ignore_rules(tmp_path: Path) -> None:
    _git(tmp_path, "init")
    tracked_source = _python_file(tmp_path / "generated" / "tracked.py")
    _git(tmp_path, "add", "generated/tracked.py")
    (tmp_path / ".gitignore").write_text("/generated/\n")

    assert tracked_source in source_python_files(tmp_path)
