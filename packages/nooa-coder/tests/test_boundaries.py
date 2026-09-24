# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa_coder stays independent of the older host packages."""

import ast
from pathlib import Path

import nooa_coder

_FORBIDDEN = ("nooa_cli", "nooa_acp")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_no_module_imports_the_cli_or_acp_packages():
    root = Path(nooa_coder.__file__).parent
    sources = sorted(root.rglob("*.py"))
    assert sources
    offenders = {
        str(path.relative_to(root)): sorted(
            name for name in _imported_modules(path) if name.split(".")[0] in _FORBIDDEN
        )
        for path in sources
    }
    assert {path: names for path, names in offenders.items() if names} == {}


def test_the_coding_modules_use_no_retired_turn_or_worker_types():
    """The coding agent is on the Session layer: no RespondResult, no CodingWorker.

    session/session.py still accepts a RespondResult from older agents; that
    compatibility belongs to the Session layer and is not checked here.
    """
    root = Path(nooa_coder.__file__).parent
    retired = ("RespondResult", "RespondReason", "CodingWorker")
    hits = [
        f"{path.relative_to(root)}: {name}"
        for package in ("coding", "tools", "workspace")
        for path in sorted((root / package).rglob("*.py"))
        for name in retired
        if name in path.read_text()
    ]
    assert hits == []
