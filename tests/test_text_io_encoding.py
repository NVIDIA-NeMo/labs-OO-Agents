# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every ``Path.read_text()`` / ``Path.write_text()`` outside tests names its encoding (#289).

Ruff's ``PLW1514`` covers ``open()`` but cannot see that a variable is a
``Path``, so these two calls need their own check.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "packages", "scripts", "util", "examples")
SKIPPED_PARTS = {"tests", "test", ".venv", "node_modules", "dist", "build", "__pycache__"}


def _missing_encoding(tree: ast.AST) -> list[int]:
    lines = []
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("read_text", "write_text")
        ):
            continue
        if any(k.arg in ("encoding", None) for k in node.keywords):
            continue
        positional = len(node.args)
        # read_text(x): a positional encoding, or importlib.metadata's read_text(filename).
        if node.func.attr == "read_text" and positional >= 1:
            continue
        if node.func.attr == "write_text" and positional != 1:
            continue
        lines.append(node.lineno)
    return lines


def test_text_reads_and_writes_name_their_encoding():
    offenders = []
    for top in SCANNED:
        for path in sorted((ROOT / top).rglob("*.py")):
            relative = path.relative_to(ROOT)
            if SKIPPED_PARTS.intersection(relative.parts) or path.name.startswith("test_"):
                continue
            if path.name == "conftest.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            offenders.extend(f"{relative}:{line}" for line in _missing_encoding(tree))
    assert not offenders, "pass encoding= (usually 'utf-8'):\n" + "\n".join(offenders)
