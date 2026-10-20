# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RepoTools — code navigation that returns ShellTools Match anchors.

Use ``symbols()`` to find definitions and ``refs()`` to find usages. Both return
``RepoResult``: printable lines plus editable ``Match`` objects for
``self.shell.replace(result[i], new_text)``.
"""

import logging
import re
import shlex
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

from nooa.agentdoc import hidden, spec
from nooa.skill import Skill
from nooa.tools._shell_session import ShellSession
from nooa.tools.shell_tools import Match

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class FileMapResult:
    path: Annotated[str, spec(description="File path")]
    language: Annotated[str, spec(description="Detected language")]
    symbols: Annotated[
        list[str], spec(description="Formatted symbol lines (name, kind, line number)")
    ]
    truncated: Annotated[bool, spec(description="True if symbols were capped")] = False
    anchors: Annotated[
        list[Match],
        spec(description="ShellTools-compatible anchors for each symbol line"),
    ] = field(default_factory=list)

    @property
    def text(self) -> str:
        if not self.symbols:
            return f"[{self.path}] ({self.language}) — no symbols found"
        header = f"[{self.path}] ({self.language}, {len(self.symbols)} symbols)"
        return f"{header}\n" + "\n".join(self.symbols)

    def __str__(self) -> str:
        return self.text


@dataclass
class RepoMapResult:
    root: Annotated[str, spec(description="Repository root path")]
    summary: Annotated[str, spec(description="Formatted repo map with key files and their exports")]
    num_files: Annotated[int, spec(description="Total files analyzed")]
    truncated: Annotated[bool, spec(description="True if output was capped")] = False
    anchors: Annotated[
        list[Match],
        spec(description="ShellTools-compatible anchors for symbol lines in the map"),
    ] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self.summary

    def __str__(self) -> str:
        return self.text


@dataclass
class SymbolSearchResult:
    query: Annotated[str, spec(description="Search query")]
    matches: Annotated[list[str], spec(description="Matching symbol lines (file:line: kind name)")]
    total_matches: Annotated[int, spec(description="Total matches found")]
    truncated: Annotated[bool, spec(description="True if results were capped")] = False
    anchors: Annotated[
        list[Match],
        spec(description="ShellTools-compatible anchors for each returned match"),
    ] = field(default_factory=list)

    @property
    def text(self) -> str:
        if not self.matches:
            return f'No symbols matching "{self.query}" found.'
        parts = list(self.matches)
        if self.truncated:
            parts.append(f"\n... ({self.total_matches} total, showing first {len(self.matches)})")
        else:
            parts.append(f"\n({self.total_matches} matches)")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.text


@dataclass
class ReferenceSearchResult:
    query: Annotated[str, spec(description="Symbol name searched")]
    matches: Annotated[list[str], spec(description="Reference lines (file:line: context)")]
    total_matches: Annotated[int, spec(description="Total references found")]
    truncated: Annotated[bool, spec(description="True if results were capped")] = False
    anchors: Annotated[
        list[Match],
        spec(description="ShellTools-compatible anchors for each returned reference"),
    ] = field(default_factory=list)

    @property
    def text(self) -> str:
        if not self.matches:
            return f'No references to "{self.query}" found.'
        parts = list(self.matches)
        if self.truncated:
            parts.append(f"\n... ({self.total_matches} total, showing first {len(self.matches)})")
        else:
            parts.append(f"\n({self.total_matches} references)")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.text


@dataclass
class RepoResult:
    query: Annotated[str, spec(description="Search query or symbol name")]
    lines: Annotated[list[str], spec(description="Display lines: file:line: context")]
    matches: Annotated[
        list[Match],
        spec(description="ShellTools-compatible anchors; pass an item to self.shell.replace()"),
    ] = field(default_factory=list)
    total_matches: Annotated[int, spec(description="Total matches found")] = 0
    truncated: Annotated[bool, spec(description="True if results were capped")] = False

    @property
    def text(self) -> str:
        if not self.lines:
            return f'No matches for "{self.query}" found.'
        parts = list(self.lines)
        if self.truncated:
            parts.append(f"\n... ({self.total_matches} total, showing first {len(self.lines)})")
        else:
            noun = "match" if self.total_matches == 1 else "matches"
            parts.append(f"\n({self.total_matches} {noun})")
        return "\n".join(parts)

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.matches)

    def __bool__(self) -> bool:
        return len(self.matches) > 0

    def __iter__(self) -> Iterator[Match]:
        return iter(self.matches)

    def __getitem__(self, index: int) -> Match:
        return self.matches[index]


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------
_LANG_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".jsx": "javascript",
    ".rb": "ruby",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".sh": "bash",
    ".lua": "lua",
    ".r": "r",
}

# Regex patterns for extracting symbols per language
_SYMBOL_PATTERNS: dict[str, list[tuple[str, str]]] = {
    "python": [
        (r"^(\s*)def\s+(\w+)\s*\(", "function"),
        (r"^(\s*)async\s+def\s+(\w+)\s*\(", "async function"),
        (r"^(\s*)class\s+(\w+)", "class"),
    ],
    "javascript": [
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
        (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
        (r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(", "function"),
    ],
    "typescript": [
        (r"^\s*(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
        (r"^\s*(?:export\s+)?class\s+(\w+)", "class"),
        (r"^\s*(?:export\s+)?(?:interface|type)\s+(\w+)", "type"),
        (r"^\s*(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\(", "function"),
    ],
    "go": [
        (r"^func\s+(?:\(\w+\s+\*?\w+\)\s+)?(\w+)\s*\(", "function"),
        (r"^type\s+(\w+)\s+struct", "struct"),
        (r"^type\s+(\w+)\s+interface", "interface"),
    ],
    "rust": [
        (r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)", "function"),
        (r"^\s*(?:pub\s+)?struct\s+(\w+)", "struct"),
        (r"^\s*(?:pub\s+)?trait\s+(\w+)", "trait"),
        (r"^\s*(?:pub\s+)?enum\s+(\w+)", "enum"),
        (r"^\s*impl(?:<[^>]*>)?\s+(\w+)", "impl"),
    ],
    "java": [
        (
            r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:abstract\s+)?class\s+(\w+)",
            "class",
        ),
        (
            r"^\s*(?:public|private|protected)?\s*(?:static\s+)?(?:abstract\s+)?interface\s+(\w+)",
            "interface",
        ),
        (r"^\s*(?:public|private|protected)?\s*(?:static\s+)?[\w<>\[\]]+\s+(\w+)\s*\(", "method"),
    ],
    "ruby": [
        (r"^\s*def\s+(\w+)", "method"),
        (r"^\s*class\s+(\w+)", "class"),
        (r"^\s*module\s+(\w+)", "module"),
    ],
}

# Add common aliases
_SYMBOL_PATTERNS["tsx"] = _SYMBOL_PATTERNS["typescript"]
_SYMBOL_PATTERNS["jsx"] = _SYMBOL_PATTERNS["javascript"]
_SYMBOL_PATTERNS["cpp"] = [
    (r"^\s*(?:virtual\s+)?[\w:]+\s+(\w+)\s*\(", "function"),
    (r"^\s*class\s+(\w+)", "class"),
    (r"^\s*struct\s+(\w+)", "struct"),
    (r"^\s*namespace\s+(\w+)", "namespace"),
]
_SYMBOL_PATTERNS["c"] = _SYMBOL_PATTERNS["cpp"]


def _detect_lang(path: Path) -> str:
    return _LANG_MAP.get(path.suffix.lower(), "unknown")


def _tree_sitter_available() -> bool:
    try:
        from nooa_cli.tools._tree_sitter_backend import TREE_SITTER_AVAILABLE
    except ImportError:
        return False
    return TREE_SITTER_AVAILABLE


def _line_match(path: Path, line_no: int) -> Match | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        return None
    if not (1 <= line_no <= len(lines)):
        return None
    return Match(str(path), line_no, line_no, lines[line_no - 1], resolved_path=path)


def _symbol_anchor_pairs(path: Path, symbols: list[str]) -> list[tuple[str, Match]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    except OSError:
        return []
    pairs: list[tuple[str, Match]] = []
    for symbol in symbols:
        line_text = symbol.strip()
        if not line_text:
            continue
        line_no_text = line_text.split(maxsplit=1)[0]
        try:
            line_no = int(line_no_text)
        except ValueError:
            continue
        if 1 <= line_no <= len(lines):
            pairs.append(
                (
                    symbol,
                    Match(
                        str(path),
                        line_no,
                        line_no,
                        lines[line_no - 1],
                        resolved_path=path,
                    ),
                )
            )
    return pairs


def _symbol_anchors(path: Path, symbols: list[str]) -> list[Match]:
    return [anchor for _, anchor in _symbol_anchor_pairs(path, symbols)]


def _parse_match_line(line: str) -> tuple[str, int, str] | None:
    """Parse rg or symbol output without treating a Windows drive as a line separator."""
    match = re.match(r"^(.+?):\s*(\d+)(?::|\s+|$)(.*)$", line)
    if match is None:
        return None
    return match.group(1), int(match.group(2)), match.group(3)


def _anchor_from_match_line(line: str, root: Path) -> Match | None:
    parsed = _parse_match_line(line)
    if parsed is None:
        return None
    path_text, line_no, _ = parsed
    fpath = Path(path_text)
    if not fpath.is_absolute():
        fpath = root / fpath
    return _line_match(fpath, line_no)


def _format_match(anchor: Match, root: Path) -> str:
    path = Path(anchor.path)
    try:
        display_path = str(path.relative_to(root)) if path.is_absolute() else anchor.path
    except ValueError:
        display_path = anchor.path
    return f"{display_path}:{anchor.start}: {anchor.text.strip()}"


def _is_definition_line(content: str, name: str = "") -> bool:
    if any(
        re.match(pattern, content)
        for pattern in (
            r"^\s*(?:export\s+)?(?:async\s+)?function\s+",
            r"^\s*(?:export\s+)?(?:abstract\s+)?class\s+",
            r"^\s*(?:export\s+)?(?:interface|type)\s+",
            r"^\s*(?:async\s+)?def\s+",
            r"^\s*(?:func|fn|type|struct|trait|impl|enum|module)\s+",
        )
    ):
        return True
    return bool(
        name
        and re.match(
            rf"^\s*(?:export\s+)?(?:const|let|var)\s+{re.escape(name)}\b\s*=",
            content,
        )
    )


def _extract_symbols(path: Path, lang: str, max_symbols: int = 200) -> list[str]:
    """Extract symbol definitions from a file using tree-sitter AST (with regex fallback)."""
    # Try tree-sitter first (AST-aware, more accurate)
    try:
        from nooa_cli.tools._tree_sitter_backend import (
            TREE_SITTER_AVAILABLE,
            ts_extract_symbols,
        )

        if TREE_SITTER_AVAILABLE:
            ts_result = ts_extract_symbols(path, lang, max_symbols)
            if ts_result is not None:
                return ts_result
    except ImportError:
        pass

    # Fallback: regex-based extraction
    patterns = _SYMBOL_PATTERNS.get(lang, [])
    if not patterns:
        return []

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except (OSError, PermissionError):
        return []

    symbols: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        if len(symbols) >= max_symbols:
            break
        for pattern, kind in patterns:
            m = re.match(pattern, line)
            if m:
                # Get the name from the last capturing group
                groups = [g for g in m.groups() if g is not None]
                name = groups[-1] if groups else "?"
                indent = len(line) - len(line.lstrip())
                prefix = "  " * (indent // 4) if indent > 0 else ""
                symbols.append(f"  {i:4d} {prefix}{kind} {name}")
                break
    return symbols


# ---------------------------------------------------------------------------
# RepoTools Skill
# ---------------------------------------------------------------------------
class RepoTools(Skill):
    """Find code locations as ShellTools ``Match`` anchors.

    Two methods:
        symbols(path=".", query="")  — definitions / file or repo overview
        refs(name, path=".")         — references/usages, excluding definitions

    Results print like search output and index/iterate as editable matches::
        r = await self.repo.symbols("src/", query="Handler")
        await self.shell.replace(r[0], new_code)
    """

    __nosnapshot__ = True

    def __init__(
        self,
        root: str | Path = ".",
        session: ShellSession | None = None,
        require_tree_sitter: bool = False,
    ) -> None:
        self._root = Path(root).resolve()
        self._session = session  # shared session with ShellTools (optional)
        self._has_rg: bool | None = None  # lazy-checked
        if not _tree_sitter_available():
            message = "tree-sitter is not available; RepoTools will use regex/rg fallbacks"
            if require_tree_sitter:
                raise RuntimeError(message)
            logger.warning(message)

    def __repr__(self) -> str:
        session = "shared" if self._session is not None else "none"
        return f"RepoTools(root={str(self._root)!r}, session={session}, has_rg={self._has_rg!r})"

    @property
    @hidden
    def root(self) -> Path:
        """Repository root this tool is scoped to."""
        return self._root

    @property
    @hidden
    def session(self) -> ShellSession | None:
        """The shared shell session, if one was wired in (else None)."""
        return self._session

    async def symbols(
        self,
        path: Annotated[str, spec(description="File or directory to inspect")] = ".",
        query: Annotated[str, spec(description="Optional symbol-name substring filter")] = "",
        max_results: Annotated[int, spec(description="Maximum result lines")] = 50,
    ) -> RepoResult:
        """Find definitions under a file or directory.

        Returns printable lines plus editable ``Match`` anchors; use
        ``await self.shell.replace(result[0], new_text)`` to edit a hit.
        """
        resolved = self._resolve(path)
        query_lower = query.lower()

        if resolved.is_file():
            file_result = await self._filemap(path, max_symbols=max_results if not query else 500)
            pairs = [
                (symbol, anchor)
                for symbol, anchor in zip(file_result.symbols, file_result.anchors, strict=True)
                if not query_lower or query_lower in symbol.lower()
            ]
            anchors = [anchor for _, anchor in pairs[:max_results]]
            return RepoResult(
                query=query or path,
                lines=[_format_match(anchor, self._root) for anchor in anchors],
                matches=anchors,
                total_matches=len(pairs),
                truncated=len(pairs) > max_results,
            )

        if query:
            symbol_result = await self._search_symbol(query, path=path, max_results=max_results)
            return RepoResult(
                query=query,
                lines=symbol_result.matches,
                matches=symbol_result.anchors,
                total_matches=symbol_result.total_matches,
                truncated=symbol_result.truncated,
            )

        map_result = await self._repo_map(paths=[path], max_files=max_results)
        anchors = map_result.anchors[:max_results]
        return RepoResult(
            query=path,
            lines=[_format_match(anchor, self._root) for anchor in anchors],
            matches=anchors,
            total_matches=len(map_result.anchors),
            truncated=map_result.truncated or len(map_result.anchors) > max_results,
        )

    async def refs(
        self,
        name: Annotated[str, spec(description="Symbol or qualified name to find references for")],
        path: Annotated[str, spec(description="Directory to search")] = ".",
        max_results: Annotated[int, spec(description="Maximum result lines")] = 50,
    ) -> RepoResult:
        """Find references/usages of a symbol, excluding definitions.

        Returns printable lines plus editable ``Match`` anchors; use
        ``await self.shell.replace(result[0], new_text)`` to edit a hit.
        """
        result = await self._search_references(name, path=path, max_results=max_results)
        return RepoResult(
            query=name,
            lines=result.matches,
            matches=result.anchors,
            total_matches=result.total_matches,
            truncated=result.truncated,
        )

    async def _check_rg(self) -> bool:
        """Check availability of the Bash rg fast path, caching the result."""
        if self._has_rg is None:
            if self._session and self._session.shell_name != "bash":
                self._has_rg = False
            elif self._session:
                _, _, code = await self._session.run("command -v rg", timeout=5)
                self._has_rg = code == 0
            else:
                import shutil

                self._has_rg = shutil.which("rg") is not None
        return self._has_rg

    # ------------------------------------------------------------------
    # filemap — show symbols in a single file
    # ------------------------------------------------------------------
    async def _filemap(self, path: str, max_symbols: int = 200) -> FileMapResult:
        """Show the structure of a file: function and class definitions with line numbers.

        Useful for getting an overview of a file without reading every line.
        Shows definitions but not their bodies.

        Args:
            path: File path (relative to repo root).
            max_symbols: Maximum symbols to show (default: 200).

        Returns:
            FileMapResult with symbol listing.

        Examples:
            r = await self.repo._filemap("src/main.py")
            r = await self.repo._filemap("internal/llm/tools/edit.go")
        """
        resolved = self._resolve(path)
        if not resolved.is_file():
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().repo_failure(
                "filemap:file_not_found", f"File not found: {path}", path
            )
            return FileMapResult(
                path=path, language="unknown", symbols=[f"Error: {path} not found"]
            )

        lang = _detect_lang(resolved)
        symbols = _extract_symbols(resolved, lang, max_symbols=max_symbols)
        truncated = len(symbols) >= max_symbols

        symbol_anchor_pairs = _symbol_anchor_pairs(resolved, symbols)
        return FileMapResult(
            path=path,
            language=lang,
            symbols=[symbol for symbol, _ in symbol_anchor_pairs],
            truncated=truncated,
            anchors=[anchor for _, anchor in symbol_anchor_pairs],
        )

    # ------------------------------------------------------------------
    # repo_map — overview of repository structure and key symbols
    # ------------------------------------------------------------------
    async def _repo_map(
        self,
        paths: list[str] | None = None,
        depth: int = 3,
        max_files: int = 50,
        max_symbols_per_file: int = 20,
    ) -> RepoMapResult:
        """Generate a repository map showing key files and their top-level symbols.

        Scans the repository (or specified paths) and produces a concise
        overview of the most important files and their exports/definitions.
        Files are sorted by relevance (recently modified first).

        Args:
            paths: Specific directories to map (default: repo root).
            depth: Directory depth to scan (default: 3).
            max_files: Maximum files to include (default: 50).
            max_symbols_per_file: Max symbols per file in the map (default: 20).

        Returns:
            RepoMapResult with formatted repository overview.

        Examples:
            r = await self.repo._repo_map()
            r = await self.repo._repo_map(paths=["src/", "lib/"])
            r = await self.repo._repo_map(max_files=100, depth=4)
        """
        # Find source files, sorted by modification time (newest first)
        search_paths = paths or ["."]
        all_files: list[Path] = []

        for sp in search_paths:
            resolved = self._resolve(sp)
            if not resolved.is_dir():
                continue
            # Use rg to find files respecting gitignore, or fall back to walking
            if self._session and await self._check_rg():
                stdout, _, _ = await self._session.run(
                    f"rg --files {shlex.quote(str(resolved))} 2>/dev/null | head -{max_files * 3}",
                    timeout=15,
                )
                for line in stdout.splitlines():
                    p = Path(line.strip())
                    if p.is_file() and _detect_lang(p) != "unknown":
                        all_files.append(p)
            else:
                all_files.extend(self._iter_source_files(resolved, max_files=max_files * 3))

        # Sort by mtime (newest first) — rg --sort requires v14+.
        all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # Deduplicate and limit
        seen: set[str] = set()
        unique_files: list[Path] = []
        for f in all_files:
            key = str(f.resolve())
            if key not in seen:
                seen.add(key)
                unique_files.append(f)
        unique_files = unique_files[:max_files]

        # Build the map
        sections: list[str] = []
        anchors: list[Match] = []
        for fpath in unique_files:
            try:
                rel = fpath.relative_to(self._root)
            except ValueError:
                rel = fpath
            lang = _detect_lang(fpath)
            symbols = _extract_symbols(fpath, lang, max_symbols=max_symbols_per_file)
            symbol_anchor_pairs = _symbol_anchor_pairs(fpath, symbols)
            if symbol_anchor_pairs:
                sections.append(f"\n{rel}:")
                sections.extend(symbol for symbol, _ in symbol_anchor_pairs)
                anchors.extend(anchor for _, anchor in symbol_anchor_pairs)
            else:
                sections.append(f"\n{rel}: ({lang})")

        summary = "\n".join(sections).strip()
        if not summary:
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().repo_failure(
                "repo_map:no_files", "No source files found", str(search_paths)[:200]
            )
            summary = "(no source files found)"

        return RepoMapResult(
            root=str(self._root),
            summary=summary,
            num_files=len(unique_files),
            truncated=len(all_files) > max_files,
            anchors=anchors,
        )

    # ------------------------------------------------------------------
    # search_symbol — find definitions across the codebase
    # ------------------------------------------------------------------
    async def _search_symbol(
        self,
        name: str,
        path: str = ".",
        max_results: int = 50,
    ) -> SymbolSearchResult:
        """Search for function, class, or type definitions by name.

        Searches across all source files for symbol definitions matching
        the query (case-insensitive substring match).

        Args:
            name: Symbol name or partial name to search for.
            path: Directory to search (default: repo root).
            max_results: Maximum results (default: 50).

        Returns:
            SymbolSearchResult with matching definitions.

        Examples:
            r = await self.repo._search_symbol("calculate_score")
            r = await self.repo._search_symbol("Handler", path="src/")
        """
        resolved = self._resolve(path)
        matches: list[str] = []

        # Use grep to find definition patterns matching the name
        if self._session and await self._check_rg():
            # Build a regex that matches common definition patterns
            pattern = f"(def|class|function|func|struct|trait|interface|type|impl|module|const)\\s+\\w*{re.escape(name)}\\w*"
            stdout, _, _ = await self._session.run(
                f"rg -n -i --color=never {shlex.quote(pattern)} {shlex.quote(str(resolved))} 2>/dev/null | head -{max_results * 2}",
                timeout=30,
            )
            for line in stdout.splitlines():
                if line.strip():
                    matches.append(line.strip())
        else:
            # Fallback: walk files and check symbols (rg not available or no session)
            name_lower = name.lower()
            for fpath in self._iter_source_files(resolved, max_files=None):
                lang = _detect_lang(fpath)
                symbols = _extract_symbols(fpath, lang, max_symbols=200)
                for sym in symbols:
                    if name_lower in sym.lower():
                        try:
                            rel = fpath.relative_to(self._root)
                        except ValueError:
                            rel = fpath
                        matches.append(f"{rel}:{sym.strip()}")
                    if len(matches) >= max_results:
                        break
                if len(matches) >= max_results:
                    break

        total = len(matches)
        paired = [
            (match, anchor)
            for match, anchor in (
                (m, _anchor_from_match_line(m, self._root)) for m in matches[:max_results]
            )
            if anchor is not None
        ]
        if total == 0:
            from nooa.runtime.harness_metrics import get_harness_metrics

            get_harness_metrics().repo_failure(
                "search_symbol:no_results", f"No matches for '{name}'", f"path={path}"
            )
        truncated = total >= max_results
        return SymbolSearchResult(
            query=name,
            matches=[m for m, _ in paired],
            total_matches=total,
            truncated=truncated,
            anchors=[a for _, a in paired],
        )

    # ------------------------------------------------------------------
    # search_references — find call sites / usages
    # ------------------------------------------------------------------
    async def _search_references(
        self,
        name: Annotated[
            str,
            spec(
                description="Symbol name to find references for (e.g. 'from_file' or 'TraceExplorer.from_file')"
            ),
        ],
        path: Annotated[str, spec(description="Directory to search")] = ".",
        max_results: Annotated[int, spec(description="Maximum results")] = 50,
    ) -> ReferenceSearchResult:
        """Find all references (call sites, usages) of a symbol — excludes definitions.

        Uses tree-sitter AST analysis when available for accurate results,
        falling back to ripgrep with heuristic filtering.

        Args:
            name: Symbol name or qualified name (e.g. 'TraceExplorer.from_file').
            path: Directory to search (default: repo root).
            max_results: Maximum results (default: 50).

        Returns:
            ReferenceSearchResult with matching reference lines.

        Examples:
            r = await self.repo._search_references("from_file")
            r = await self.repo._search_references("TraceExplorer.from_file")
        """
        resolved = self._resolve(path)

        # Try tree-sitter first for accurate AST-aware reference finding
        try:
            from nooa_cli.tools._tree_sitter_backend import (
                TREE_SITTER_AVAILABLE,
                ts_find_references,
            )

            if TREE_SITTER_AVAILABLE:
                all_matches: list[str] = []
                for fpath in self._iter_source_files(resolved, max_files=None):
                    lang = _detect_lang(fpath)
                    if lang == "unknown":
                        continue
                    refs = ts_find_references(
                        fpath, lang, name, max_results=max_results - len(all_matches)
                    )
                    if refs:
                        try:
                            rel = fpath.relative_to(self._root)
                        except ValueError:
                            rel = fpath
                        for line_no, context in refs:
                            all_matches.append(f"{rel}:{line_no}: {context}")
                    if len(all_matches) >= max_results:
                        break
                if all_matches:
                    truncated = len(all_matches) >= max_results
                    paired = [
                        (match, anchor)
                        for match, anchor in (
                            (m, _anchor_from_match_line(m, self._root))
                            for m in all_matches[:max_results]
                        )
                        if anchor is not None
                    ]
                    return ReferenceSearchResult(
                        query=name,
                        matches=[m for m, _ in paired],
                        total_matches=len(all_matches),
                        truncated=truncated,
                        anchors=[a for _, a in paired],
                    )
        except ImportError:
            pass

        # Fallback: ripgrep-based reference search with heuristic filtering
        search_name = re.escape(name)
        pattern = f"\\b{search_name}\\b"

        raw_lines: Iterable[str]
        if self._session and await self._check_rg():
            cmd = f"rg -n -H --color=never {shlex.quote(pattern)} {shlex.quote(str(resolved))} 2>/dev/null | head -{max_results * 3}"
            stdout, _, code = await self._session.run(cmd, timeout=30)
            if code != 0 or not stdout:
                return ReferenceSearchResult(query=name, matches=[], total_matches=0)
            raw_lines = stdout.strip().splitlines()
        else:
            raw_lines = self._iter_reference_lines(resolved, pattern)

        # Filter out definitions, comments, and string-only lines.
        matches: list[str] = []
        anchors: list[Match] = []
        for raw in raw_lines:
            if len(matches) >= max_results:
                break
            # Parse file:line:content
            parsed = _parse_match_line(raw)
            if parsed is None:
                continue
            path_text, line_no, content = parsed
            content = content.strip()
            # Skip definitions
            if _is_definition_line(content, name.split(".")[-1]):
                continue
            # Skip comment-only lines
            stripped = content.lstrip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue
            anchor = _anchor_from_match_line(raw, self._root)
            if anchor is None:
                continue
            matches.append(f"{path_text}:{line_no}: {content}")
            anchors.append(anchor)
            if len(matches) >= max_results:
                break

        truncated = len(matches) >= max_results
        return ReferenceSearchResult(
            query=name,
            matches=matches[:max_results],
            total_matches=len(matches),
            truncated=truncated,
            anchors=anchors[:max_results],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _iter_reference_lines(self, root: Path, pattern: str) -> Iterator[str]:
        """Search until the caller has enough references, regardless of file count."""
        regex = re.compile(pattern)
        for fpath in self._iter_source_files(root, max_files=None):
            try:
                rel = fpath.relative_to(self._root)
            except ValueError:
                rel = fpath
            try:
                with fpath.open(encoding="utf-8", errors="replace") as source:
                    for line_no, line in enumerate(source, 1):
                        if regex.search(line):
                            content = line.rstrip("\r\n")
                            yield f"{rel}:{line_no}:{content}"
            except OSError:
                continue

    def _iter_source_files(self, root: Path, max_files: int | None = 500) -> Iterator[Path]:
        """Yield source files, pruning ignored directories below the requested root."""
        if max_files is not None and max_files <= 0:
            return
        if root.is_file():
            if _detect_lang(root) != "unknown":
                yield root
            return
        ignored_dirs = {"node_modules", "__pycache__", "venv", "build", "dist"}
        count = 0
        for directory, subdirs, filenames in root.walk():
            subdirs[:] = sorted(
                name for name in subdirs if not name.startswith(".") and name not in ignored_dirs
            )
            for name in sorted(filenames):
                fpath = directory / name
                if name.startswith(".") or _detect_lang(fpath) == "unknown" or not fpath.is_file():
                    continue
                # A symlink or NTFS junction inside the tree can point anywhere;
                # reading through it would hand the model a file the repository
                # does not contain. Only the resolved location is checked — a
                # link that stays inside is fine.
                try:
                    if self._root.resolve() not in fpath.resolve().parents:
                        continue
                except OSError:
                    continue
                yield fpath
                count += 1
                if max_files is not None and count >= max_files:
                    return

    def _resolve(self, path: str) -> Path:
        """Resolve a path against the repo root, and keep the result inside it.

        Absolute paths are accepted when they point into the repository: hosts
        pass anchors and workspace paths that way, and ``refs(path=...)``
        documents an absolute file as a valid root. What is refused is a path
        whose *resolved* location is outside the root, which covers the forms
        that used to slip through:

        * ``..`` segments that walk above the root;
        * a drive-relative path such as ``C:foo``, which names the process's
          current directory on that drive rather than a path under the root —
          it is neither absolute nor relative in the sense this function needs,
          so it is resolved and then rejected by the containment check;
        * a path that reaches outside through a symlink or NTFS junction, since
          ``Path.resolve`` follows both.
        """
        candidate = Path(path)
        # ``C:foo`` has a drive but no root: Windows resolves it against the
        # process's current directory *on that drive*, which may happen to sit
        # inside the repository, so the containment check below cannot see it.
        if candidate.drive and not candidate.root:
            raise ValueError(f"path {path!r} is drive-relative, not a repository path")
        resolved = (candidate if candidate.is_absolute() else self._root / candidate).resolve()
        root = self._root.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"path {path!r} resolves outside the repository root {root}")
        return resolved
