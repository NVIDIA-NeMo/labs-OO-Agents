# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility tests for the optional Tree-sitter RepoTools backend."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nooa_cli.tools import _tree_sitter_backend as backend


def test_compile_query_uses_language_method_when_available(monkeypatch):
    expected = object()
    legacy_query = Mock(return_value=expected)
    language = Mock(query=legacy_query)
    constructor = Mock(side_effect=AssertionError("modern API should not be used"))
    monkeypatch.setattr(backend, "ts", SimpleNamespace(Query=constructor), raising=False)

    assert backend._compile_query(language, "(identifier) @name") is expected
    legacy_query.assert_called_once_with("(identifier) @name")
    constructor.assert_not_called()


def test_compile_query_uses_query_constructor_without_language_method(monkeypatch):
    expected = object()
    language = object()
    constructor = Mock(return_value=expected)
    monkeypatch.setattr(backend, "ts", SimpleNamespace(Query=constructor), raising=False)

    assert backend._compile_query(language, "(identifier) @name") is expected
    constructor.assert_called_once_with(language, "(identifier) @name")


def test_current_tree_sitter_extracts_symbols_and_references(tmp_path: Path):
    if not backend.TREE_SITTER_AVAILABLE:
        pytest.skip("Tree-sitter AST extra is not installed")
    if backend._get_parser("javascript") is None:
        pytest.skip("JavaScript Tree-sitter grammar is not installed")

    source = tmp_path / "sample.js"
    source.write_text(
        "class Example {\n"
        "  target() { return 1; }\n"
        "}\n"
        "new Example().target();\n"
        'const text = "target() is not a reference";\n'
        "// target() is not a reference\n"
    )

    symbols = backend.ts_extract_symbols(source, "javascript")
    references = backend.ts_find_references(source, "javascript", "target")

    assert symbols is not None
    assert any("method target" in symbol for symbol in symbols)
    assert references == [(4, "new Example().target();")]


def test_query_captures_normalizes_legacy_mapping_results(monkeypatch):
    later = SimpleNamespace(start_byte=20)
    earlier = SimpleNamespace(start_byte=10)
    query = SimpleNamespace(captures=Mock(return_value={"name": [later, earlier]}))
    constructor = Mock(side_effect=AssertionError("QueryCursor should not be used"))
    monkeypatch.setattr(backend, "ts", SimpleNamespace(QueryCursor=constructor), raising=False)

    assert backend._query_captures(query, object()) == [
        (earlier, "name"),
        (later, "name"),
    ]
    constructor.assert_not_called()


def test_typescript_definition_query_compiles_with_installed_grammar(tmp_path: Path):
    if not backend.TREE_SITTER_AVAILABLE:
        pytest.skip("Tree-sitter AST extra is not installed")
    if backend._get_parser("typescript") is None:
        pytest.skip("TypeScript Tree-sitter grammar is not installed")

    source = tmp_path / "sample.ts"
    source.write_text("class Example {}\n")

    symbols = backend.ts_extract_symbols(source, "typescript")

    assert symbols is not None
    assert any("class Example" in symbol for symbol in symbols)
