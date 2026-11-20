# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Searchable documentation indexes: official docs for objects without docstrings.

A package registers an index of :class:`DocEntry` records (name, kind, signature,
summary, text, source). ``doc()`` shows the entry of an object that has no
docstring and no curated page; :func:`search_docs` (``doc.search``) finds names by
keywords across all registered indexes::

    from nooa.agentdoc.docindex import DocEntry, register_doc_entries, search_docs

    register_doc_entries("usd", [
        DocEntry("usd.open_stage", "function", "(path: str)", "Open a stage."),
    ])
    print(search_docs("open a stage", packages=["usd", "pxr"]))

Indexes may be registered lazily (:func:`register_doc_index`); built-in ones load
their data file on first lookup or search. Ranking is BM25 over name tokens
(dotted and camelCase parts), signature, summary and text.
"""

from __future__ import annotations

import inspect
import math
import re
import sys
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DocEntry",
    "DocIndex",
    "SearchHit",
    "get_doc_index",
    "lookup_doc",
    "register_doc_entries",
    "register_doc_index",
    "register_doc_resolver",
    "search_doc_entries",
    "search_docs",
    "unregister_doc_index",
]


@dataclass(frozen=True)
class DocEntry:
    """One documented name: ``Usd.Stage.Open`` with its signature and description."""

    name: str  # Dotted name as code spells it, e.g. "UsdGeom.Subset.CreateGeomSubset"
    kind: str = ""  # module, class, method, staticmethod, function, property, token, constant...
    signature: str = ""  # "(a: int, b: str = '')"; overloads separated by " | "
    summary: str = ""  # One-line description
    text: str = ""  # Full description (may repeat the summary)
    source: str = ""  # Where the text comes from, e.g. "usd/usd/stage.h:260"

    @property
    def description(self) -> str:
        """The full text, or the summary when there is no text."""
        return self.text or self.summary


@dataclass(frozen=True)
class SearchHit:
    """A search result: the entry, its index, and its score."""

    entry: DocEntry
    package: str
    score: float


_STOPWORDS = frozenset(
    "a an the and or of for to in on at by with from into is are be been it its this that "
    "these those as i we you how do does can should what which who all any some my me".split()
)
_SUFFIXES = (
    ("ational", "ate"),
    ("ization", "ize"),
    ("ations", "ate"),
    ("ation", "ate"),
    ("encies", ""),
    ("ency", ""),
    ("ences", ""),
    ("ence", ""),
    ("ents", ""),
    ("ent", ""),
    ("ies", "y"),
    ("ings", ""),
    ("ing", ""),
    ("ed", ""),
    ("es", "e"),
    ("s", ""),
)
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _stem(word: str) -> str:
    word = word.lower()
    if len(word) <= 3:
        return word
    for suffix, repl in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) + len(repl) >= 3:
            if suffix == "s" and word.endswith("ss"):
                break
            word = word[: len(word) - len(suffix)] + repl
            break
    if len(word) > 4 and word.endswith("e"):
        word = word[:-1]
    return word


def name_tokens(name: str) -> list[str]:
    """Stemmed tokens of a code name: dots, underscores and camelCase split."""
    out = []
    for part in re.split(r"[._\W]+", name):
        for piece in _CAMEL.findall(part):
            token = _stem(piece)
            if token:
                out.append(token)
    return out


def text_tokens(text: str) -> list[str]:
    """Stemmed tokens of prose (camelCase words also split), stopwords removed."""
    out = []
    for word in _WORD.findall(text):
        lower = word.lower()
        if lower in _STOPWORDS:
            continue
        pieces = _CAMEL.findall(word)
        if len(pieces) > 1:
            out.append(_stem(word))
            out.extend(_stem(p) for p in pieces if p.lower() not in _STOPWORDS)
        else:
            out.append(_stem(word))
    return out


class DocIndex:
    """Entries of one package, with lazy BM25 search."""

    _K1 = 1.2
    _B = 0.75

    def __init__(
        self,
        package: str,
        entries: Iterable[DocEntry] = (),
        *,
        version: str = "",
        source: str = "",
        note: str = "",
    ) -> None:
        self.package = package
        self.version = version
        self.source = source
        self.note = note  # One line shown with entries, e.g. a version mismatch
        self.entries: dict[str, DocEntry] = {}
        self._lock = threading.RLock()
        self._postings: dict[str, list[tuple[int, float]]] | None = None
        self._order: list[DocEntry] = []
        self._lengths: list[float] = []
        self._leaves: dict[str, list[str]] | None = None
        self.add(entries)

    def add(self, entries: Iterable[DocEntry]) -> None:
        with self._lock:
            for entry in entries:
                self.entries[entry.name] = entry
            self._postings = None
            self._leaves = None

    def get(self, name: str) -> DocEntry | None:
        return self.entries.get(name)

    def source_label(self) -> str:
        """Where the index text comes from, e.g. "OpenUSD v26.08 C++ headers"."""
        return self.source

    def __len__(self) -> int:
        return len(self.entries)

    def leaves(self) -> dict[str, list[str]]:
        """Last name component -> full names (for did-you-mean)."""
        with self._lock:
            if self._leaves is None:
                leaves: dict[str, list[str]] = {}
                for name in self.entries:
                    leaves.setdefault(name.rsplit(".", 1)[-1], []).append(name)
                self._leaves = leaves
            return self._leaves

    def _build(self) -> None:
        postings: dict[str, dict[int, float]] = {}
        self._order = list(self.entries.values())
        self._lengths = []
        for doc_id, entry in enumerate(self._order):
            weights: dict[str, float] = {}
            owner, _, leaf = entry.name.rpartition(".")
            for token in name_tokens(leaf):
                weights[token] = weights.get(token, 0.0) + 3.0
            for token in name_tokens(owner):
                weights[token] = weights.get(token, 0.0) + 1.0
            for token in text_tokens(re.sub(r":[^,()|]*", " ", entry.signature)):
                weights[token] = weights.get(token, 0.0) + 1.0
            for token in text_tokens(entry.summary):
                weights[token] = weights.get(token, 0.0) + 2.0
            text = entry.text
            if text.startswith(entry.summary):
                text = text[len(entry.summary) :]
            for token in text_tokens(text):
                weights[token] = weights.get(token, 0.0) + 1.0
            self._lengths.append(sum(weights.values()))
            for token, weight in weights.items():
                postings.setdefault(token, {})[doc_id] = weight
        self._postings = {token: list(docs.items()) for token, docs in postings.items()}

    def search(self, query: str, limit: int = 8) -> list[tuple[float, DocEntry]]:
        """Entries matching ``query`` keywords, best first."""
        tokens = list(dict.fromkeys(text_tokens(query)))
        if not tokens:
            return []
        with self._lock:
            if self._postings is None:
                self._build()
            postings, order, lengths = self._postings, self._order, self._lengths
        assert postings is not None
        if not order:
            return []
        n = len(order)
        avg = sum(lengths) / n or 1.0
        scores: dict[int, float] = {}
        for token in tokens:
            docs = postings.get(token)
            if not docs:
                continue
            idf = math.log(1 + (n - len(docs) + 0.5) / (len(docs) + 0.5))
            for doc_id, tf in docs:
                norm = (
                    tf
                    * (self._K1 + 1)
                    / (tf + self._K1 * (1 - self._B + self._B * lengths[doc_id] / avg))
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * norm
        for doc_id in scores:
            if not order[doc_id].description:
                scores[doc_id] *= 0.8  # an undocumented name helps less
        best = sorted(scores.items(), key=lambda item: (-item[1], order[item[0]].name))[:limit]
        return [(score, order[doc_id]) for doc_id, score in best]


# ----------------------------------------------------------------------------- registry

_lock = threading.RLock()
_indexes: dict[str, DocIndex] = {}
_loaders: dict[str, Callable[[], DocIndex | None]] = {}
_resolvers: list[tuple[str, Callable[[Any], tuple[str, str] | None]]] = []

# Built-in indexes: package -> "module:function" returning a DocIndex. Loading reads
# a data file only; the documented library is not imported.
_BUILTIN_LOADERS = {"pxr": "nooa.agentdoc.adapters.pxr_index:load_pxr_index"}


def _coerce(entry: DocEntry | Mapping[str, Any]) -> DocEntry:
    if isinstance(entry, DocEntry):
        return entry
    if isinstance(entry, Mapping):
        known = {k: str(entry.get(k) or "") for k in DocEntry.__dataclass_fields__ if k in entry}
        if not known.get("name"):
            raise ValueError(f"doc entry needs a name: {entry!r}")
        return DocEntry(**known)
    raise TypeError(f"expected DocEntry or mapping, got {type(entry).__name__}")


def register_doc_entries(
    package: str,
    entries: Iterable[DocEntry | Mapping[str, Any]],
    *,
    version: str = "",
    source: str = "",
    replace: bool = False,
) -> DocIndex:
    """Add doc entries for ``package`` (a skill, wrapper or library name).

    Entries are :class:`DocEntry` objects or mappings with ``name`` and optional
    ``kind``, ``signature``, ``summary``, ``text``, ``source``. Entries with an
    existing name replace it; ``replace=True`` drops the package's old entries.
    Returns the package's index; search it with ``search_docs(q, packages=[package])``.
    """
    coerced = [_coerce(entry) for entry in entries]
    with _lock:
        index = None if replace else _get_index_locked(package)
        if index is None:
            index = DocIndex(package, version=version, source=source)
            _indexes[package] = index
            _loaders.pop(package, None)
        if version:
            index.version = version
        if source:
            index.source = source
        index.add(coerced)
        return index


def register_doc_index(package: str, index: DocIndex | Callable[[], DocIndex | None]) -> None:
    """Register a whole index, or a loader called once on first use."""
    with _lock:
        if isinstance(index, DocIndex):
            _indexes[package] = index
            _loaders.pop(package, None)
        else:
            _indexes.pop(package, None)
            _loaders[package] = index


def unregister_doc_index(package: str) -> None:
    """Remove a package's index (and loader); mainly for tests."""
    with _lock:
        _indexes.pop(package, None)
        _loaders.pop(package, None)
        _BUILTIN_DISABLED.add(package)


_BUILTIN_DISABLED: set[str] = set()


def register_doc_resolver(
    module_prefix: str, resolver: Callable[[Any], tuple[str, str] | None]
) -> None:
    """Map live objects to ``(package, entry name)``.

    ``resolver(obj)`` is consulted for classes and callables whose ``__module__``
    starts with ``module_prefix`` or is not a loaded module (also for any object when
    the prefix is "").
    It must only read attributes and return None for objects it does not know.
    """
    with _lock:
        _resolvers[:] = [(p, r) for p, r in _resolvers if p != module_prefix]
        _resolvers.append((module_prefix, resolver))


def _get_index_locked(package: str) -> DocIndex | None:
    index = _indexes.get(package)
    if index is not None:
        return index
    loader = _loaders.pop(package, None)
    if loader is None and package in _BUILTIN_LOADERS and package not in _BUILTIN_DISABLED:
        module_name, _, func = _BUILTIN_LOADERS[package].partition(":")
        _BUILTIN_DISABLED.add(package)  # load once

        def loader() -> DocIndex | None:
            import importlib

            return getattr(importlib.import_module(module_name), func)()

    if loader is None:
        return None
    try:
        index = loader()
    except Exception:  # noqa: BLE001 - a missing data file must never break doc()
        index = None
    if index is not None:
        _indexes[package] = index
    return index


def get_doc_index(package: str) -> DocIndex | None:
    """The package's index, loading it on first use; None when there is none."""
    with _lock:
        return _get_index_locked(package)


def _packages() -> list[str]:
    with _lock:
        names = list(_indexes) + [p for p in _loaders if p not in _indexes]
        names += [p for p in _BUILTIN_LOADERS if p not in names and p not in _BUILTIN_DISABLED]
        return names


def _module_of(obj: Any) -> str:
    target = obj
    if inspect.ismethod(obj):
        target = obj.__func__
    for candidate in (target, getattr(target, "__objclass__", None)):
        module = getattr(candidate, "__module__", None)
        if isinstance(module, str):
            return module
    owner = getattr(obj, "__self__", None)
    if owner is not None:
        return getattr(owner if isinstance(owner, type) else type(owner), "__module__", "") or ""
    return ""


def lookup_doc(obj: Any) -> tuple[DocIndex, DocEntry] | None:
    """The index entry documenting ``obj`` (a class or callable), if a resolver knows it."""
    with _lock:
        resolvers = list(_resolvers)
    if not resolvers:
        return None
    module = _module_of(obj)
    for prefix, resolver in resolvers:
        # Some compiled functions have no real __module__ (Boost.Python in usd-core
        # 25.11 reports the class name, "Path"): then the resolvers decide.
        if prefix and module in sys.modules and not module.startswith(prefix):
            continue
        try:
            found = resolver(obj)
        except Exception:  # noqa: BLE001 - lookups must never break doc()
            found = None
        if not found:
            continue
        index = get_doc_index(found[0])
        entry = index.get(found[1]) if index is not None else None
        if entry is not None:
            return index, entry
    return None


def search_doc_entries(
    query: str, packages: Iterable[str] | None = None, limit: int = 8
) -> list[SearchHit]:
    """Ranked hits for ``query``.

    With ``packages``, results come in that package order (each package's good
    matches, best first), so a wrapper package listed first ranks above the raw
    library. Without it, all registered indexes are merged by relative score.
    """
    limit = max(1, int(limit))
    if packages is None:
        # BM25 scores of indexes of different sizes do not compare: merge on each
        # package's relative score (its best hit = 1.0).
        merged = []
        for package in _packages():
            index = get_doc_index(package)
            if index is None:
                continue
            found = index.search(query, limit)
            if found:
                best = found[0][0] or 1.0
                merged += [(s / best, s, SearchHit(e, package, s)) for s, e in found]
        merged.sort(key=lambda item: (-item[0], -item[1], item[2].entry.name))
        return [hit for _rel, _raw, hit in merged[:limit]]
    ordered: list[SearchHit] = []
    for package in dict.fromkeys(packages):
        index = get_doc_index(package)
        if index is None:
            continue
        found = index.search(query, limit)
        if not found:
            continue
        best = found[0][0]
        ordered += [SearchHit(e, package, s) for s, e in found if s >= 0.35 * best]
    return ordered[:limit]


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def format_hit(hit: SearchHit) -> str:
    entry = hit.entry
    signature = _clip(entry.signature.split(" | ")[0], 110)
    if " | " in entry.signature:
        signature += " …"
    if signature and not signature.startswith(("(", ":")):
        signature = " " + signature  # "= 'face'" for a constant
    summary = entry.summary or _clip(entry.text, 140) or f"({entry.kind or 'no description'})"
    return f"{entry.name}{signature} — {_clip(summary, 160)}"


def search_docs(query: str, packages: Iterable[str] | None = None, limit: int = 8) -> str:
    """Search registered doc indexes by keywords; compact lines ``Name(sig) — summary``.

    ``packages`` limits and orders the indexes (e.g. ``["usd", "pxr"]``); by default
    every registered index is searched. Ends with a hint to call ``doc(name)``.
    """
    packages = list(packages) if packages is not None else None
    hits = search_doc_entries(query, packages, limit)
    if not hits:
        searched = ", ".join(packages if packages is not None else _packages()) or "none"
        return f"No documentation matches {query!r} (indexes searched: {searched})."
    lines = [format_hit(hit) for hit in hits]
    notes = []
    for package in dict.fromkeys(hit.package for hit in hits):
        index = get_doc_index(package)
        if index is not None and index.note:
            notes.append(index.note)
    lines += notes
    lines.append(f"Full text: doc(<name>), e.g. doc({hits[0].entry.name}).")
    return "\n".join(lines)


def entry_docstring(index: DocIndex, entry: DocEntry, *, with_signatures: bool = True) -> str:
    """The text doc() shows for an index entry: description, overloads, source, note."""
    parts = [entry.description or f"No description in the {index.package} docs index."]
    overloads = entry.signature.split(" | ")
    if with_signatures and len(overloads) > 1:
        parts.append("Overloads:\n" + "\n".join("    " + sig for sig in overloads))
    origin = ", ".join(p for p in (index.source_label(), entry.source) if p)
    if origin:
        parts.append(f"Source: {origin}")
    if index.note:
        parts.append(index.note)
    return "\n\n".join(parts)


# ----------------------------------------------------------------------------- doc() hooks


def _split_signature(signature: str) -> tuple[str, str]:
    """First overload as (params, return type)."""
    first = signature.split(" | ")[0].strip()
    if not first.startswith("("):
        return "(...)", ""
    depth = 0
    for k, ch in enumerate(first):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                rest = first[k + 1 :].strip()
                return first[: k + 1], rest[2:].strip() if rest.startswith("->") else ""
    return first, ""


def indexed_callable_info(obj: Any, *, concise_doc: bool = False):
    """CallableInfo for a callable from its docs index entry, or None."""
    found = lookup_doc(obj)
    if found is None:
        return None
    from nooa.agentdoc._info import CallableInfo

    index, entry = found
    params, returns = _split_signature(entry.signature)
    if concise_doc:
        docstring = entry.summary or None
    else:
        docstring = entry_docstring(index, entry)
    if entry.kind == "staticmethod" and docstring:
        docstring = "Static. " + docstring
    return CallableInfo(name=entry.name, signature=params, return_type=returns, docstring=docstring)


def enrich_type_info(cls: type, info: Any) -> Any:
    """Fill a compiled class's missing docstring and method docs from its docs index."""
    with _lock:
        if not _resolvers:
            return info
    import dataclasses

    found = lookup_doc(cls)
    docstring = info.docstring
    if not docstring and found is not None:
        index, entry = found
        docstring = entry_docstring(index, entry, with_signatures=False)
        overloads = [s for s in entry.signature.split(" | ") if s]
        if overloads:
            docstring += "\n\nConstructors:\n" + "\n".join(
                f"    {cls.__name__}{s}" for s in overloads
            )
    methods = []
    by_qualname = {getattr(k, "__qualname__", ""): k for k in getattr(cls, "__mro__", ())}
    for method in info.methods:
        if method.docstring or method.signature != "(...)":
            methods.append(method)
            continue
        qual, _, attr = method.name.rpartition(".")
        owner = by_qualname.get(qual)
        raw = vars(owner).get(attr) if owner is not None else None
        if isinstance(raw, (staticmethod, classmethod)):
            raw = raw.__func__
        better = indexed_callable_info(raw, concise_doc=True) if raw is not None else None
        if better is None:
            methods.append(method)
            continue
        methods.append(
            dataclasses.replace(
                method,
                signature=better.signature,
                return_type=better.return_type,
                docstring=better.docstring,
            )
        )
    return dataclasses.replace(info, docstring=docstring, methods=methods)


def enrich_module_info(module: Any, info: Any) -> Any:
    """Fill a module's undocumented compiled functions from its docs index."""
    with _lock:
        if not _resolvers:
            return info
    import dataclasses

    functions = []
    for function in info.functions or []:
        if function.docstring or function.signature != "(...)":
            functions.append(function)
            continue
        raw = getattr(module, function.name.rsplit(".", 1)[-1], None)
        better = indexed_callable_info(raw, concise_doc=True) if raw is not None else None
        functions.append(
            function
            if better is None
            else dataclasses.replace(
                function,
                signature=better.signature,
                return_type=better.return_type,
                docstring=better.docstring,
            )
        )
    return dataclasses.replace(info, functions=functions)


def official_addendum(obj: Any, curated_doc: str | None, *, max_chars: int = 700) -> str | None:
    """The index description to append to a curated entry, when it adds something.

    Returns None when there is no entry or the curated text already contains the
    official summary.
    """
    found = lookup_doc(obj)
    if found is None:
        return None
    index, entry = found
    if not entry.description:
        return None
    probe = " ".join(entry.summary.split())[:60].lower()
    if probe and probe in " ".join((curated_doc or "").split()).lower():
        return None
    first = entry.description.split("\n\n")[0]
    text = first if len(first) <= max_chars else first[: max_chars - 1].rstrip() + "…"
    more = "" if text == entry.description else " (more: doc.search or the C++ docs)"
    label = index.source_label() or f"{index.package} docs"
    lines = [
        f"Official description ({label}{', ' + entry.source if entry.source else ''}):",
        text + more,
    ]
    if index.note:
        lines.append(index.note)
    return "\n".join(lines)
