# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The packaged README must not point readers at files that are not in the repo."""

from __future__ import annotations

import re
from pathlib import Path

_README = Path(__file__).resolve().parents[2] / "src" / "nooa_memory" / "README.md"
_MARKDOWN_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def test_readme_does_not_cite_missing_memory_bench():
    """Issue #104: examples/memory_bench/ is not in the public repository."""
    text = _README.read_text()
    assert "memory_bench" not in text, (
        "README cites examples/memory_bench/, which is not in this repository. "
        "Point at examples/quickstart/12_memory.py instead."
    )


def test_readme_relative_markdown_links_resolve():
    """Every relative markdown link from the README must exist on disk."""
    text = _README.read_text()
    missing: list[str] = []
    for href in _MARKDOWN_LINK.findall(text):
        target = href.split("#", 1)[0].strip()
        if not target or target.startswith(("http://", "https://", "mailto:")):
            continue
        resolved = (_README.parent / target).resolve()
        if not resolved.exists():
            missing.append(href)
    assert missing == [], f"README links that do not exist: {missing}"
