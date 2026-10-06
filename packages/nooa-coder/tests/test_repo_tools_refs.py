# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RepoTools.refs() on the host: hidden ancestors, paths outside the root, one file."""

from __future__ import annotations

from pathlib import Path

import nooa_coder.tools._tree_sitter_backend as ts_backend
import pytest
from nooa_coder.tools.repo_tools import RepoTools

from nooa.tools.shell_tools import ShellTools

MOD = "def helper():\n    return 1\n"
CALLER = "from mod import helper\n\nvalue = helper()\n"


@pytest.fixture(params=["tree-sitter", "regex"])
def backend(request, monkeypatch):
    """Run each test with the AST search and with the text fallback."""
    if request.param == "regex":
        monkeypatch.setattr(ts_backend, "TREE_SITTER_AVAILABLE", False)
    elif not ts_backend.TREE_SITTER_AVAILABLE:
        pytest.skip("tree-sitter is not installed")
    return request.param


def _repo(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "mod.py").write_text(MOD)
    (root / "caller.py").write_text(CALLER)
    return root


async def test_refs_finds_matches_under_a_dot_prefixed_ancestor(tmp_path, backend):
    root = _repo(tmp_path / ".cache" / "repo")
    result = await RepoTools(root=root).refs("helper")
    assert result.total_matches > 0
    assert any("caller.py" in line for line in result.lines)


async def test_refs_still_skips_hidden_directories_inside_the_root(tmp_path, backend):
    root = _repo(tmp_path / "repo")
    (root / ".venv").mkdir()
    (root / ".venv" / "dep.py").write_text(CALLER)
    result = await RepoTools(root=root).refs("helper")
    assert not any(".venv" in line for line in result.lines)


async def test_refs_outside_the_root_does_not_raise(tmp_path, backend):
    root = tmp_path / "a"
    root.mkdir()
    other = _repo(tmp_path / "b")
    result = await RepoTools(root=root).refs("helper", path=str(other))
    assert result.total_matches > 0


async def test_refs_scoped_to_one_file_without_a_session(tmp_path, backend):
    root = _repo(tmp_path / "repo")
    result = await RepoTools(root=root).refs("helper", path="caller.py")
    assert result.total_matches > 0
    assert all(line.startswith("caller.py:") for line in result.lines)


@pytest.mark.parametrize("has_rg", [True, False], ids=["rg", "grep"])
async def test_refs_scoped_to_one_file_through_a_session(tmp_path, monkeypatch, has_rg):
    monkeypatch.setattr(ts_backend, "TREE_SITTER_AVAILABLE", False)
    root = _repo(tmp_path / "repo")
    shell = ShellTools(cwd=str(root))
    try:
        repo = RepoTools(root=root, session=await shell._get_session())
        repo._has_rg = has_rg
        result = await repo.refs("helper", path="caller.py")
        assert result.total_matches > 0
        assert all("caller.py:" in line for line in result.lines)
    finally:
        await shell.close()


def test_the_tool_docs_promise_editable_only_when_match_has_it():
    """The model is told to filter on Match.editable only if core Match has it (#382)."""
    from nooa_coder.tools import repo_tools

    from nooa.agentdoc import doc

    rendered = str(doc(RepoTools))
    assert ("editable" in rendered) == repo_tools._MATCH_HAS_EDITABLE
    assert "read-only" not in rendered or repo_tools._MATCH_HAS_EDITABLE
