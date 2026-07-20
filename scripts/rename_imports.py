#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AST-based import renaming tool for bulk package consolidation.

Usage:
    python scripts/rename_imports.py --preview ROOT MAPPING_JSON
    python scripts/rename_imports.py --apply ROOT MAPPING_JSON

MAPPING_JSON is a JSON object like:
    '{"agentdoc": "nooa.agentdoc", "context_blocks": "nooa.context_blocks"}'

Examples:
    # Preview changes
    python scripts/rename_imports.py --preview . '{"agentdoc": "nooa.agentdoc"}'

    # Apply changes
    python scripts/rename_imports.py --apply . '{"agentdoc": "nooa.agentdoc"}'
"""

import ast
import json
import sys
from pathlib import Path

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    ".venv-macos",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "node_modules",
    ".nooa",
    "3p",
}


def matches(old: str, new: str, module: str):
    """If module starts with old, return the rewritten module path."""
    if module == old:
        return new
    if module.startswith(old + "."):
        return new + module[len(old) :]
    return None


def rewrite_source(source: str, mapping: dict[str, str]):
    """Rewrite import statements in source using AST-guided line replacement.

    Returns (rewritten_source, list_of_change_descriptions).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, ["SKIP: SyntaxError"]

    lines = source.splitlines(keepends=True)
    changes = []
    edits = []  # (line_index, old_line, new_line)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for old, new in mapping.items():
                    new_module = matches(old, new, alias.name)
                    if new_module is not None:
                        lineno = node.lineno - 1
                        old_line = lines[lineno]
                        new_line = old_line.replace(alias.name, new_module, 1)
                        if old_line != new_line:
                            edits.append((lineno, old_line, new_line))
                            changes.append(
                                f"  L{node.lineno}: import {alias.name} -> import {new_module}"
                            )
                        break

        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            for old, new in mapping.items():
                new_module = matches(old, new, node.module)
                if new_module is not None:
                    lineno = node.lineno - 1
                    old_line = lines[lineno]
                    new_line = old_line.replace(node.module, new_module, 1)
                    if old_line != new_line:
                        edits.append((lineno, old_line, new_line))
                        changes.append(f"  L{node.lineno}: from {node.module} -> from {new_module}")
                    break

    # Apply edits bottom-up to preserve line numbers
    seen = set()
    for lineno, _old_line, new_line in sorted(edits, key=lambda e: e[0], reverse=True):
        if lineno not in seen:
            lines[lineno] = new_line
            seen.add(lineno)

    return "".join(lines), changes


def collect_python_files(root: Path):
    """Recursively collect .py files, skipping excluded dirs."""
    results = []
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if not set(rel.parts) & EXCLUDE_DIRS:
            results.append(p)
    return results


def main():
    if len(sys.argv) != 4 or sys.argv[1] not in ("--preview", "--apply"):
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1]
    root = Path(sys.argv[2]).resolve()
    mapping = json.loads(sys.argv[3])

    files = collect_python_files(root)
    total_changes = 0
    changed_files = 0

    for f in files:
        source = f.read_text(encoding="utf-8", errors="replace")
        rewritten, changes = rewrite_source(source, mapping)

        if source != rewritten:
            changed_files += 1
            total_changes += len(changes)
            rel = f.relative_to(root)
            print(f"\n{rel}")
            for c in changes:
                print(c)

            if mode == "--apply":
                f.write_text(rewritten, encoding="utf-8")

    action = "Would change" if mode == "--preview" else "Changed"
    print(f"\n--- {action} {changed_files} file(s), {total_changes} import(s) ---")


if __name__ == "__main__":
    main()
