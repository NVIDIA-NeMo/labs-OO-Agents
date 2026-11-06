# SPDX-FileCopyrightText: Copyright (c) 2026 Sampoorn Nagpal
# SPDX-License-Identifier: Apache-2.0
"""An example-local immutable document snapshot; no model or vector store required."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import asdict, dataclass, replace
from math import log


@dataclass(frozen=True)
class Section:
    key: str
    title: str
    text: str
    parent: str | None = None


@dataclass(frozen=True)
class OutlineEntry:
    section_id: str
    path: tuple[str, ...]


@dataclass(frozen=True)
class Evidence:
    section_id: str
    path: tuple[str, ...]
    text: str
    start: int
    end: int
    truncated: bool


def encode(items: list[OutlineEntry] | list[Evidence]) -> str:
    """The exact UTF-8 JSON representation used for the example's input budgets."""
    return json.dumps([asdict(item) for item in items], ensure_ascii=False, separators=(",", ":"))


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", text.casefold())


class DocumentIndex:
    """A single document revision with explicit parent links and section-local offsets.

    Keys are unique within a document, while opaque section IDs include the
    entire snapshot's hash. Editing any section invalidates old IDs. Internal
    nodes can own text: that text remains independently retrievable.
    """

    def __init__(self, document_id: str, sections: list[Section]):
        if not document_id or any(not s.key or not s.title for s in sections):
            raise ValueError("Document ID, section keys and titles must be nonempty")
        by_key = {s.key: s for s in sections}
        if len(by_key) != len(sections):
            raise ValueError("Duplicate section key")
        paths: dict[str, tuple[str, ...]] = {}
        for section in sections:
            path: list[str] = []
            visited: set[str] = set()
            current: Section | None = section
            while current is not None:
                if current.key in visited:
                    raise ValueError("Cyclic section hierarchy")
                visited.add(current.key)
                path.append(current.title)
                if current.parent is not None and current.parent not in by_key:
                    raise ValueError(f"Unknown parent: {current.parent}")
                current = by_key.get(current.parent) if current.parent is not None else None
            paths[section.key] = tuple(reversed(path))
        raw = json.dumps(
            {"document_id": document_id, "sections": [asdict(s) for s in sections]},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.revision = hashlib.sha256(raw.encode()).hexdigest()
        self.document_id = document_id
        self._sections = {
            f"{self.revision}:{section.key}": section
            for section in sections
            if section.text.strip()
        }
        self._paths = {sid: paths[s.key] for sid, s in self._sections.items()}
        self._terms = {sid: Counter(_words(s.text)) for sid, s in self._sections.items()}

    def outline(self, max_chars: int) -> tuple[str, bool]:
        """Return bounded JSON plus completeness; never silently route on a partial tree."""
        if max_chars < 2:
            raise ValueError("Outline budget must fit an empty JSON array")
        entries: list[OutlineEntry] = []
        for sid in self._sections:
            entry = OutlineEntry(sid, self._paths[sid])
            if len(encode(entries + [entry])) > max_chars:
                return encode(entries), False
            entries.append(entry)
        return encode(entries), True

    def validate_ids(self, ids: list[str], k: int) -> list[str]:
        """Check membership before deduplication and enforce the distinct-section limit."""
        if k < 1:
            raise ValueError("k must be positive")
        if any(sid not in self._sections for sid in ids):
            raise ValueError("Unknown or stale section ID")
        unique = list(dict.fromkeys(ids))
        if len(unique) > k:
            raise ValueError("Too many sections selected")
        return unique

    def search(self, query: str, k: int) -> list[str]:
        """Small content-only BM25 fallback (k1=1.2, b=0.75); ties use source order."""
        if k < 1:
            raise ValueError("k must be positive")
        n = len(self._terms)
        if not n:
            return []
        average = sum(sum(terms.values()) for terms in self._terms.values()) / n
        if not average:
            return []
        query_terms = set(_words(query))
        df = {word: sum(word in terms for terms in self._terms.values()) for word in query_terms}
        scores: dict[str, float] = {}
        for sid, terms in self._terms.items():
            length = sum(terms.values())
            scores[sid] = sum(
                log(1 + (n - df[word] + 0.5) / (df[word] + 0.5))
                * terms[word]
                * 2.2
                / (terms[word] + 1.2 * (0.25 + 0.75 * length / average))
                for word in query_terms
                if terms[word]
            )
        return [sid for sid in sorted(scores, key=lambda sid: -scores[sid]) if scores[sid] > 0][:k]

    def read(self, ids: list[str], *, k: int, max_chars: int) -> list[Evidence]:
        """Pack evidence into a JSON character budget, including IDs/path/offset metadata.

        Offsets are Unicode character offsets within the section's original text.
        At most the final retained body is shortened. No text from later sections
        can be cited unless it was actually included in this returned evidence.
        """
        if max_chars < 2:
            raise ValueError("Evidence budget must fit an empty JSON array")
        selected = self.validate_ids(ids, k)
        evidence: list[Evidence] = []
        for sid in selected:
            text = self._sections[sid].text
            item = Evidence(sid, self._paths[sid], text, 0, len(text), False)
            if len(encode(evidence + [item])) <= max_chars:
                evidence.append(item)
                continue
            low, high = 0, len(text) - 1
            while low < high:
                middle = (low + high + 1) // 2
                shortened = replace(item, text=text[:middle], end=middle, truncated=True)
                if len(encode(evidence + [shortened])) <= max_chars:
                    low = middle
                else:
                    high = middle - 1
            if low:
                evidence.append(replace(item, text=text[:low], end=low, truncated=True))
            break
        return evidence
