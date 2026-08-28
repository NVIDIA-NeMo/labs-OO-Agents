# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path-resolution diagnostics for RepoTools."""

from __future__ import annotations

from pathlib import Path

import pytest
from nooa_cli.tools.repo_tools import RepoTools
from pydantic import BaseModel

from nooa.tools.shell_tools import PathResolutionError


@pytest.mark.asyncio
async def test_symbols_missing_path_reports_repo_root(tmp_path: Path) -> None:
    repo = RepoTools(root=tmp_path)

    result = await repo.symbols("missing/package")

    assert not result
    assert isinstance(result.diagnostic, PathResolutionError)
    assert result.diagnostic.code == "PATH_NOT_FOUND"
    assert result.diagnostic.operation == "symbols"
    assert result.diagnostic.requested_path == "missing/package"
    assert result.diagnostic.resolved_path == str((tmp_path / "missing/package").resolve())
    assert result.diagnostic.base_name == "self.repo.root"
    assert result.diagnostic.base_path == str(tmp_path)
    assert "self.repo.root" in str(result)


@pytest.mark.asyncio
async def test_refs_missing_path_reports_repo_root(tmp_path: Path) -> None:
    repo = RepoTools(root=tmp_path)

    result = await repo.refs("target", path="missing")

    assert not result
    assert result.diagnostic is not None
    assert result.diagnostic.operation == "refs"
    assert result.diagnostic.resolved_path == str((tmp_path / "missing").resolve())


def test_repo_result_is_pydantic_model() -> None:
    from nooa_cli.tools.repo_tools import RepoResult

    result = RepoResult(query="Widget", lines=[])

    assert isinstance(result, BaseModel)
    assert result.model_dump() == {
        "query": "Widget",
        "lines": [],
        "matches": [],
        "total_matches": 0,
        "truncated": False,
        "diagnostic": None,
    }
