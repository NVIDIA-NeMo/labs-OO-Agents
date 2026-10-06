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
