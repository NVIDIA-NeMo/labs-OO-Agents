# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository navigation works with native shells and editable UTF-8 anchors."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from nooa_cli.tools import _tree_sitter_backend as backend
from nooa_cli.tools.repo_tools import RepoTools

from nooa.tools import ShellTools


@pytest.fixture
def regex_backend(monkeypatch):
    monkeypatch.setattr(backend, "TREE_SITTER_AVAILABLE", False)


def _source(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    source = root / "中文 sample's [file].py"
    source.write_text(
        "def target():  # 定义\n"
        '    return "原始内容"\n'
        "\n"
        "result = target()  # 调用: 结果\n"
        "# target() is a comment\n",
        encoding="utf-8",
    )
    return source


@pytest.mark.parametrize("lookup", ["file", "overview", "query", "refs"])
async def test_powershell_navigation_returns_editable_utf8_anchors(
    tmp_path, monkeypatch, regex_backend, lookup
):
    root = tmp_path / ".codex" / "中文 space's [repo]"
    source = _source(root)
    shell = ShellTools(cwd=str(root), shell="powershell")
    run = AsyncMock(side_effect=AssertionError("repository navigation must not execute Bash"))
    monkeypatch.setattr(shell.session, "run", run)
    repo = RepoTools(root=root, session=shell.session)
    try:
        if lookup == "file":
            result = await repo.symbols(source.name)
        elif lookup == "overview":
            result = await repo.symbols()
        elif lookup == "query":
            result = await repo.symbols(query="target")
        else:
            result = await repo.refs("target")

        assert repo.session is shell.session
        assert len(result) == 1
        anchor = result[0]
        assert Path(anchor.resolved_path) == source
        assert anchor.start == (4 if lookup == "refs" else 1)
        original = source.read_text(encoding="utf-8")
        assert anchor.text == original.splitlines(keepends=True)[anchor.start - 1]
        replacement = "result = '修改后'\n" if lookup == "refs" else "def target():  # 修改后\n"
        await shell.replace(anchor, replacement)
        assert source.read_text(encoding="utf-8") == original.replace(anchor.text, replacement, 1)
        run.assert_not_awaited()
    finally:
        await shell.close()


async def test_bash_without_rg_falls_back_to_python_for_references(tmp_path, regex_backend):
    source = _source(tmp_path)
    session = SimpleNamespace(
        shell_name="bash",
        run=AsyncMock(return_value=("", "rg is not available", 127)),
    )
    result = await RepoTools(root=tmp_path, session=session).refs("target")

    assert len(result) == 1
    assert Path(result[0].resolved_path) == source
    assert result[0].start == 4


async def test_native_shell_navigation_and_edits(tmp_path, regex_backend):
    root = tmp_path / "中文 repo's [root]"
    source = _source(root)
    shell = ShellTools(cwd=str(root))
    repo = RepoTools(root=root, session=shell.session)
    try:
        overview = await repo.symbols()
        definitions = await repo.symbols(query="target")
        references = await repo.refs("target")
        assert len(overview) == len(definitions) == len(references) == 1
        assert definitions[0].start == 1
        assert references[0].start == 4
        assert repo.session is shell.session
        await shell.replace(references[0], "result = '修改后'\n")
        assert "result = '修改后'\n" in source.read_text(encoding="utf-8")
        assert not await repo.refs("target")
    finally:
        await shell.close()


@pytest.mark.parametrize("lookup", ["query", "refs"])
async def test_bash_rg_parses_absolute_paths_and_colons_in_content(tmp_path, regex_backend, lookup):
    source = _source(tmp_path)
    contents = source.read_text(encoding="utf-8").splitlines()
    output = (
        f"{source}:1:{contents[0]}\n"
        if lookup == "query"
        else "\n".join(f"{source}:{line}:{contents[line - 1]}" for line in (1, 4, 5)) + "\n"
    )
    session = SimpleNamespace(
        shell_name="bash",
        run=AsyncMock(side_effect=[("/usr/bin/rg\n", "", 0), (output, "", 0)]),
    )
    repo = RepoTools(root=tmp_path, session=session)
    result = await repo.symbols(query="target") if lookup == "query" else await repo.refs("target")

    assert len(result) == 1
    assert result[0].start == (1 if lookup == "query" else 4)
    assert Path(result[0].resolved_path) == source
    assert result[0].text == contents[result[0].start - 1] + "\n"
    assert contents[result[0].start - 1] in result.text


@pytest.mark.parametrize("path_kind", ["directory", "file", "absolute_file"])
async def test_python_references_accept_file_or_directory_roots(tmp_path, regex_backend, path_kind):
    root = tmp_path / ".codex" / "repo"
    source = _source(root / "src")
    paths = {
        "directory": "src",
        "file": str(source.relative_to(root)),
        "absolute_file": str(source),
    }
    result = await RepoTools(root=root).refs("target", path=paths[path_kind])

    assert len(result) == 1
    assert Path(result[0].resolved_path) == source
    assert result[0].start == 4


@pytest.mark.parametrize("lookup", ["overview", "query", "refs"])
async def test_python_navigation_skips_hidden_and_build_directories(
    tmp_path, regex_backend, lookup
):
    source = _source(tmp_path / "src")
    ignored = [".git", ".venv", ".cache", "node_modules", "__pycache__", "venv", "build", "dist"]
    for directory in ignored:
        _source(tmp_path / directory / "nested")
    (tmp_path / ".hidden.py").write_text("def target(): pass\ntarget()\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("def target(): pass\ntarget()\n", encoding="utf-8")
    repo = RepoTools(root=tmp_path)

    if lookup == "overview":
        result = await repo.symbols()
    elif lookup == "query":
        result = await repo.symbols(query="target")
    else:
        result = await repo.refs("target")

    assert len(result) == 1
    assert Path(result[0].resolved_path) == source


async def test_explicit_search_root_is_not_excluded_by_its_own_name(tmp_path, regex_backend):
    source = _source(tmp_path / "build")
    result = await RepoTools(root=tmp_path).refs("target", path="build")

    assert len(result) == 1
    assert Path(result[0].resolved_path) == source


@pytest.mark.parametrize("lookup", ["overview", "query", "refs"])
async def test_python_navigation_obeys_result_limit(tmp_path, regex_backend, lookup):
    for index in range(3):
        _source(tmp_path / f"src{index}")
    repo = RepoTools(root=tmp_path)
    if lookup == "overview":
        result = await repo.symbols(max_results=2)
    elif lookup == "query":
        result = await repo.symbols(query="target", max_results=2)
    else:
        result = await repo.refs("target", max_results=2)

    assert len(result) == 2
    assert result.truncated
    assert len(result.lines) == len(result.matches)


@pytest.mark.parametrize("has_grammar", [True, False])
async def test_native_navigation_accepts_ast_results_or_regex_fallback(
    tmp_path, monkeypatch, has_grammar
):
    source = _source(tmp_path / ".codex" / "repo")
    contents = source.read_text(encoding="utf-8").splitlines()
    extract = Mock(return_value=["     1 function target"] if has_grammar else None)
    find_refs = Mock(return_value=[(4, contents[3])] if has_grammar else None)
    monkeypatch.setattr(backend, "TREE_SITTER_AVAILABLE", True)
    monkeypatch.setattr(backend, "ts_extract_symbols", extract)
    monkeypatch.setattr(backend, "ts_find_references", find_refs)
    session = SimpleNamespace(
        shell_name="powershell",
        run=AsyncMock(side_effect=AssertionError("native navigation must not execute Bash")),
    )
    repo = RepoTools(root=source.parent, session=session)

    symbols = await repo.symbols(source.name)
    references = await repo.refs("target", path=source.name)

    assert [anchor.start for anchor in symbols] == [1]
    assert [anchor.start for anchor in references] == [4]
    assert symbols[0].text == contents[0] + "\n"
    assert references[0].text == contents[3] + "\n"
    assert Path(references[0].resolved_path) == source
    extract.assert_called_once()
    find_refs.assert_called_once()
    session.run.assert_not_awaited()


async def test_tree_sitter_navigation_uses_native_files_and_ignores_strings(tmp_path):
    if not backend.TREE_SITTER_AVAILABLE or backend._get_parser("javascript") is None:
        pytest.skip("JavaScript Tree-sitter grammar is not installed")
    root = tmp_path / ".codex" / "repo"
    root.mkdir(parents=True)
    source = root / "中文 methods.js"
    source.write_text(
        "class Example {\n"
        "  target() { return '原始'; }\n"
        "}\n"
        "new Example().target();\n"
        "const text = 'target() is not a reference';\n"
        "// target() is not a reference\n",
        encoding="utf-8",
    )
    session = SimpleNamespace(
        shell_name="powershell",
        run=AsyncMock(side_effect=AssertionError("AST navigation must not execute Bash")),
    )
    repo = RepoTools(root=root, session=session)

    symbols = await repo.symbols(source.name, query="target")
    references = await repo.refs("target")

    assert [anchor.start for anchor in symbols] == [2]
    assert [anchor.start for anchor in references] == [4]
    assert Path(references[0].resolved_path) == source
    session.run.assert_not_awaited()
