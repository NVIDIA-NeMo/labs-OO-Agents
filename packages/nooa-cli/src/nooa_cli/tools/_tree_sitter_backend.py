# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tree-sitter backend for RepoTools.

Provides AST-aware code analysis: symbol extraction and reference finding
using tree-sitter grammars. Falls back gracefully if grammars are unavailable.

Usage (internal — called by repo_tools.py)::

    from nooa_cli.tools._tree_sitter_backend import (
        ts_extract_symbols,
        ts_find_references,
        TREE_SITTER_AVAILABLE,
    )
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import tree_sitter as ts

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False

# Language grammar loaders (lazy-loaded)
_PARSERS: dict[str, ts.Parser] = {}

# Map language name → grammar module and language loader function.
_GRAMMAR_MODULES = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "rust": ("tree_sitter_rust", "language"),
}

# Tree-sitter queries for extracting definitions per language
_DEFINITION_QUERIES = {
    "python": """
        (function_definition name: (identifier) @name) @def
        (class_definition name: (identifier) @name) @def
    """,
    "javascript": """
        (function_declaration name: (identifier) @name) @def
        (class_declaration name: (identifier) @name) @def
        (method_definition name: (property_identifier) @name) @def
        (lexical_declaration
            (variable_declarator
                name: (identifier) @name
                value: (arrow_function))) @def
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @def
        (class_declaration name: (identifier) @name) @def
        (interface_declaration name: (type_identifier) @name) @def
        (type_alias_declaration name: (type_identifier) @name) @def
        (method_definition name: (property_identifier) @name) @def
    """,
    "go": """
        (function_declaration name: (identifier) @name) @def
        (method_declaration name: (field_identifier) @name) @def
        (type_declaration (type_spec name: (type_identifier) @name)) @def
    """,
    "rust": """
        (function_item name: (identifier) @name) @def
        (struct_item name: (type_identifier) @name) @def
        (enum_item name: (type_identifier) @name) @def
        (trait_item name: (type_identifier) @name) @def
        (impl_item type: (type_identifier) @name) @def
    """,
}

# Tree-sitter queries for finding references (call sites, imports, usages)
_DEFINITION_QUERIES["tsx"] = _DEFINITION_QUERIES["typescript"]

_REFERENCE_QUERIES = {
    "python": """
        (call function: (identifier) @ref)
        (call function: (attribute attribute: (identifier) @ref))
        (import_from_statement name: (dotted_name (identifier) @ref))
        (identifier) @ref
    """,
    "javascript": """
        (call_expression function: (identifier) @ref)
        (call_expression function: (member_expression property: (property_identifier) @ref))
        (identifier) @ref
    """,
    "typescript": """
        (call_expression function: (identifier) @ref)
        (call_expression function: (member_expression property: (property_identifier) @ref))
        (type_identifier) @ref
        (identifier) @ref
    """,
    "go": """
        (call_expression function: (identifier) @ref)
        (call_expression function: (selector_expression field: (field_identifier) @ref))
        (identifier) @ref
    """,
    "rust": """
        (call_expression function: (identifier) @ref)
        (call_expression function: (field_expression field: (field_identifier) @ref))
        (identifier) @ref
    """,
}
_REFERENCE_QUERIES["tsx"] = _REFERENCE_QUERIES["typescript"]


def _get_parser(lang: str) -> ts.Parser | None:
    """Get or create a tree-sitter parser for the given language."""
    if not TREE_SITTER_AVAILABLE:
        return None

    if lang in _PARSERS:
        return _PARSERS[lang]

    grammar = _GRAMMAR_MODULES.get(lang)
    if not grammar:
        return None
    mod_name, loader_name = grammar

    try:
        import importlib

        grammar_mod = importlib.import_module(mod_name)
        language_fn = getattr(grammar_mod, loader_name)
        language = ts.Language(language_fn())
        parser = ts.Parser(language)
        _PARSERS[lang] = parser
        return parser
    except (ImportError, AttributeError, Exception) as e:
        logger.debug(f"tree-sitter grammar not available for {lang}: {e}")
        return None


def _query_captures(query: ts.Query, root_node: ts.Node) -> list[tuple[ts.Node, str]]:
    legacy_captures = getattr(query, "captures", None)
    if legacy_captures is not None:
        return legacy_captures(root_node)

    captures = ts.QueryCursor(query).captures(root_node)
    if isinstance(captures, dict):
        flattened = [
            (node, capture_name) for capture_name, nodes in captures.items() for node in nodes
        ]
        return sorted(flattened, key=lambda item: item[0].start_byte)
    return captures


def ts_extract_symbols(path: Path, lang: str, max_symbols: int = 200) -> list[str] | None:
    """Extract symbol definitions using tree-sitter AST parsing.

    Returns a list of formatted symbol lines, or None if tree-sitter
    is not available for this language (caller should fall back to regex).
    """
    parser = _get_parser(lang)
    if parser is None:
        return None

    try:
        source = path.read_bytes()
        tree = parser.parse(source)
    except (OSError, Exception) as e:
        logger.debug(f"tree-sitter parse failed for {path}: {e}")
        return None

    query_src = _DEFINITION_QUERIES.get(lang)
    if not query_src:
        return None

    try:
        language = parser.language
        query = language.query(query_src)
    except Exception as e:
        logger.debug(f"tree-sitter query compilation failed for {lang}: {e}")
        return None

    symbols: list[str] = []
    captures = _query_captures(query, tree.root_node)

    # Process captures — look for @name captures paired with @def
    seen_lines: set[int] = set()
    for node, capture_name in captures:
        if capture_name != "name":
            continue
        if len(symbols) >= max_symbols:
            break

        line = node.start_point[0] + 1  # 1-indexed
        if line in seen_lines:
            continue
        seen_lines.add(line)

        name = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")

        # Determine kind from the enclosing definition node.
        parent = node.parent
        kind = _node_to_kind(parent, lang, source)

        # Calculate indent
        indent = node.start_point[1]
        prefix = "  " * (indent // 4) if indent > 0 else ""
        symbols.append(f"  {line:4d} {prefix}{kind} {name}")

    return symbols if symbols else None


def ts_find_references(
    path: Path, lang: str, name: str, max_results: int = 100
) -> list[tuple[int, str]] | None:
    """Find references to a symbol using tree-sitter AST parsing.

    Returns a list of (line_number, line_text) tuples, or None if
    tree-sitter is not available (caller should fall back to regex).

    Filters out definition sites — only returns usages/call sites.
    """
    parser = _get_parser(lang)
    if parser is None:
        return None

    try:
        source = path.read_bytes()
        tree = parser.parse(source)
    except (OSError, Exception):
        return None

    query_src = _REFERENCE_QUERIES.get(lang)
    if not query_src:
        return None

    try:
        language = parser.language
        query = language.query(query_src)
    except Exception:
        return None

    lines = source.decode("utf-8", errors="replace").splitlines()
    captures = _query_captures(query, tree.root_node)

    # Handle qualified names (e.g. "TraceExplorer.from_file" → match "from_file")
    search_name = name.split(".")[-1] if "." in name else name
    qualifier = name.split(".")[0] if "." in name else None

    results: list[tuple[int, str]] = []
    seen_lines: set[int] = set()

    for node, capture_name in captures:
        if capture_name != "ref":
            continue
        if len(results) >= max_results:
            break

        node_text = source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
        if node_text != search_name:
            continue

        line = node.start_point[0] + 1  # 1-indexed
        if line in seen_lines:
            continue

        # Skip definition sites
        parent = node.parent
        if parent and _is_definition_node(parent.type, lang):
            continue

        # If qualified name, check that qualifier appears on the same line
        if qualifier and line <= len(lines):
            line_text = lines[line - 1]
            if qualifier not in line_text:
                continue

        seen_lines.add(line)
        line_text = lines[line - 1] if line <= len(lines) else ""
        results.append((line, line_text.strip()))

    return results if results else None


def _node_to_kind(node: ts.Node | None, lang: str, source: bytes) -> str:
    """Map a tree-sitter definition node to a human-readable kind string."""
    if node is None:
        return "symbol"

    if lang == "python" and node.type == "function_definition":
        line = source.splitlines()[node.start_point[0]].lstrip()
        if line.startswith(b"async def "):
            return "async function"

    if lang == "go" and node.type == "type_spec":
        for child in node.children:
            if child.type == "struct_type":
                return "struct"
            if child.type == "interface_type":
                return "interface"

    return _node_type_to_kind(node.type, lang)


def _node_type_to_kind(node_type: str, lang: str) -> str:
    """Map tree-sitter node type to a human-readable kind string."""
    mapping = {
        "function_definition": "function",
        "function_declaration": "function",
        "function_item": "function",
        "method_definition": "method",
        "method_declaration": "method",
        "class_definition": "class",
        "class_declaration": "class",
        "struct_item": "struct",
        "enum_item": "enum",
        "trait_item": "trait",
        "impl_item": "impl",
        "interface_declaration": "interface",
        "type_alias_declaration": "type",
        "type_declaration": "type",
        "type_spec": "type",
        "lexical_declaration": "function",
        "variable_declarator": "function",
    }
    return mapping.get(node_type, "symbol")


def _is_definition_node(node_type: str, lang: str) -> bool:
    """Return True if the node type represents a definition (not a reference)."""
    def_types = {
        "function_definition",
        "function_declaration",
        "function_item",
        "method_definition",
        "method_declaration",
        "class_definition",
        "class_declaration",
        "struct_item",
        "enum_item",
        "trait_item",
        "impl_item",
        "interface_declaration",
        "type_alias_declaration",
        "type_declaration",
        "type_spec",
        "lexical_declaration",
        "variable_declarator",
    }
    return node_type in def_types
