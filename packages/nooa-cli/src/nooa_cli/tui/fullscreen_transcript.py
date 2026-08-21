# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Renderer-owned transcript state for the alternate-screen TUI."""

from __future__ import annotations

from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from unicodedata import normalize

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
from rich.cells import split_graphemes
from wcwidth import wcswidth

from .terminal_safety import (
    hyperlink_at_plain_offset,
    project_prompt_toolkit_ansi,
    safe_hyperlink_spans,
    sanitize_transcript_ansi,
    strip_safe_ansi,
)

_MAX_PROJECTED_WIDTHS = 2


@dataclass(frozen=True, slots=True)
class ViewportAnchor:
    """A unique source location plus semantic position in one retained record."""

    record_id: int
    source_offset: int
    semantic_offset: int = 0


@dataclass(frozen=True, slots=True)
class ViewportState:
    """Whether new output is followed or a logical source location is pinned."""

    follows_tail: bool = True
    anchor: ViewportAnchor | None = None


@dataclass(frozen=True, slots=True)
class _SelectionHit:
    """Record-relative text interval occupied by one clicked terminal cell."""

    record_id: int
    before: int
    after: int


@dataclass(frozen=True, slots=True)
class _Record:
    record_id: int
    ansi: str
    plain: str
    has_separator: bool = False
    hyperlinks: tuple[tuple[int, int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class _ProjectedRow:
    anchor: ViewportAnchor
    fragments: tuple[tuple[str, str], ...]
    source_spans: tuple[tuple[int, int], ...] = ()
    hyperlinks: tuple[tuple[int, int, str], ...] = ()


class FullscreenTranscriptModel:
    """Ordered transcript plus bounded, incremental visual-row projections."""

    def __init__(self) -> None:
        self._records: list[_Record] = []
        self._projectable_record_count = 0
        self._next_record_id = 0
        self._viewport = ViewportState()
        self._projection_cache: OrderedDict[int, list[_ProjectedRow]] = OrderedDict()
        self._projection_index_cache: OrderedDict[int, dict[int, tuple[list[int], list[int]]]] = (
            OrderedDict()
        )
        self._formatted_cache: OrderedDict[tuple[int, int, int, int], FormattedText] = OrderedDict()
        self._ends_newline = False
        self._selection_anchor: _SelectionHit | None = None
        self._selection_active: _SelectionHit | None = None
        self._record_index_cache: tuple[dict[int, _Record], dict[int, int]] | None = None

    @property
    def text(self) -> str:
        """ANSI-free logical text, primarily for assertions and export."""
        return "".join(record.plain for record in self._records)

    @property
    def viewport(self) -> ViewportState:
        return self._viewport

    def formatted_text(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> FormattedText:
        """Return safe rows, optionally virtualized to the visible viewport."""
        if width is None:
            width = max(1, *(max(0, wcswidth(line)) for line in self.text.split("\n")))
        width = max(1, width)
        rows = self._projection(width)
        top = 0
        top_padding = 0
        if height is not None:
            height = max(1, height)
            history_fits = len(rows) <= height
            top = 0 if history_fits else self.top_row(width=width, height=height)
            rows = rows[top : top + height]
            if history_fits:
                # Short histories belong next to the bottom chrome regardless
                # of whether navigation has explicitly anchored their sole page.
                top_padding = height - len(rows)
        key = (width, top, len(rows), top_padding)
        cached = self._formatted_cache.get(key)
        if cached is not None:
            self._formatted_cache.move_to_end(key)
            return cached
        result = self._format_rows(rows, top_padding=top_padding)
        self._formatted_cache[key] = result
        while len(self._formatted_cache) > _MAX_PROJECTED_WIDTHS:
            self._formatted_cache.popitem(last=False)
        return result

    def cursor_position(self, *, width: int, height: int = 1) -> Point:
        """Expose a cursor within the virtualized visible transcript."""
        rows = self._projection(width)
        if not rows:
            return Point(x=0, y=0)
        top = self.top_row(width=width, height=height)
        visible = rows[top : top + max(1, height)]
        if self._viewport.follows_tail:
            top_padding = max(0, max(1, height) - len(visible))
            return Point(
                x=self._row_text_length(visible[-1]),
                y=top_padding + len(visible) - 1,
            )
        return Point(x=0, y=0)

    def top_row(self, *, width: int, height: int) -> int:
        """Return the exact first visual row for the current viewport."""
        rows = self._projection(width)
        if not rows:
            return 0
        if self._viewport.follows_tail or self._viewport.anchor is None:
            return max(0, len(rows) - max(1, height))
        index = self._row_index_for_anchor(width, rows, self._viewport.anchor)
        if index is None:
            self.jump_to_tail()
            return max(0, len(rows) - max(1, height))
        return index

    def scroll_visual_lines(self, delta: int, *, width: int, height: int) -> None:
        """Move the top visual row; positive deltas move toward the tail."""
        rows = self._projection(width)
        if not rows:
            self.jump_to_tail()
            return
        current = self.top_row(width=width, height=height)
        tail_top = max(0, len(rows) - max(1, height))
        target = max(0, min(tail_top, current + delta))
        if target >= tail_top:
            self.jump_to_tail()
        else:
            self._viewport = ViewportState(False, rows[target].anchor)
        self._formatted_cache.clear()

    def jump_to_start(self, *, width: int) -> None:
        rows = self._projection(width)
        if rows:
            self._viewport = ViewportState(False, rows[0].anchor)
            self._formatted_cache.clear()

    def jump_to_tail(self) -> None:
        self._viewport = ViewportState()
        self._formatted_cache.clear()

    def begin_selection(self, *, x: int, y: int, width: int, height: int) -> None:
        """Start a logical selection at one visible transcript cell."""
        hit = self._selection_hit(x=x, y=y, width=width, height=height)
        self._selection_anchor = hit
        self._selection_active = hit
        self._formatted_cache.clear()

    def update_selection(self, *, x: int, y: int, width: int, height: int) -> None:
        """Extend the current selection to one visible transcript cell."""
        if self._selection_anchor is None:
            return
        self._selection_active = self._selection_hit(x=x, y=y, width=width, height=height)
        self._formatted_cache.clear()

    def hyperlink_at(self, *, x: int, y: int, width: int, height: int) -> str | None:
        """Return the safe OSC-8 target under one visible transcript cell."""
        hit = self._selection_hit(x=x, y=y, width=width, height=height, clamp=False)
        if hit is None:
            return None
        records, _bases = self._record_indexes()
        record = records.get(hit.record_id)
        if record is None:
            return None
        return hyperlink_at_plain_offset(record.ansi, hit.before)

    def clear_selection(self) -> None:
        """Discard renderer-owned selection without changing the viewport."""
        self._selection_anchor = None
        self._selection_active = None
        self._formatted_cache.clear()

    def selected_text(self) -> str:
        """Return exact ANSI-free logical text covered by the selection."""
        if self._selection_anchor is None or self._selection_active is None:
            return ""
        selected = self._selection_bounds()
        if selected is None:
            return ""
        start, stop = selected
        pieces: list[str] = []
        offset = 0
        for record in self._records:
            record_stop = offset + len(record.plain)
            if record_stop > start and offset < stop:
                pieces.append(
                    record.plain[max(0, start - offset) : min(len(record.plain), stop - offset)]
                )
            if record_stop >= stop:
                break
            offset = record_stop
        return "".join(pieces)

    def _selection_hit(
        self,
        *,
        x: int,
        y: int,
        width: int,
        height: int,
        clamp: bool = True,
    ) -> _SelectionHit | None:
        width = max(1, width)
        height = max(1, height)
        rows = self._projection(width)
        top = self.top_row(width=width, height=height)
        top_padding = max(0, height - len(rows)) if len(rows) <= height else 0
        visual_y = max(0, min(height - 1, y))
        if visual_y < top_padding:
            return None
        row_index = max(0, min(len(rows) - 1, top + visual_y - top_padding))
        row = rows[row_index]

        display = self._row_text(row)
        display_spans = self._grapheme_spans([("", char) for char in display])
        records, bases = self._record_indexes()
        record = records.get(row.anchor.record_id)
        if record is None:
            return None
        if not display_spans or not row.source_spans:
            if not clamp:
                return None
            insertion = min(
                len(record.plain), self._character_offset(record, row.anchor.source_offset)
            )
            return _SelectionHit(record.record_id, insertion, insertion)

        cell = max(0, x)
        occupied = 0
        total_cells = sum(max(1, cells) for _start, _stop, cells in display_spans)
        if not clamp and cell >= total_cells:
            return None
        selected = len(display_spans) - 1
        for index, (_start, _stop, cells) in enumerate(display_spans):
            next_occupied = occupied + max(1, cells)
            if cell < next_occupied:
                selected = index
                break
            occupied = next_occupied
        selected = min(selected, len(row.source_spans) - 1)
        start, stop = row.source_spans[selected]
        return _SelectionHit(record.record_id, start, stop)

    def append(self, source: str, *, record_id: int | None = None) -> None:
        safe = sanitize_transcript_ansi(source)
        plain = strip_safe_ansi(safe)
        first_record = not self._records
        had_projectable_content = self._projectable_record_count > 0
        separator = "" if first_record or self._ends_newline else "\n"
        if record_id is None:
            record_id = self._next_record_id
            self._next_record_id += 1
        else:
            self._next_record_id = max(self._next_record_id, record_id + 1)
        record_ansi = separator + safe
        record = _Record(
            record_id,
            record_ansi,
            separator + plain,
            bool(separator),
            safe_hyperlink_spans(record_ansi),
        )
        prior_ends_newline = self._ends_newline
        self._records.append(record)
        if record.plain:
            self._projectable_record_count += 1
        self._record_index_cache = None
        # Joining is defined by the retained record, including its separator.
        # An empty/ANSI-only source after unterminated text therefore completes
        # that line and must affect the next append exactly as replace() does.
        self._ends_newline = record.plain.endswith("\n") if record.plain else prior_ends_newline

        # An empty projection contains one synthetic display row. It is not a
        # retained record and must never be extended into the first real one.
        if first_record or (not had_projectable_content and bool(record.plain)):
            # The empty model exposes one synthetic row that is not retained
            # history. If navigation pinned that row, the first projectable
            # append must resume tail-following rather than retain an anchor
            # that cannot exist in the rebuilt projection.
            if not had_projectable_content and not self._viewport.follows_tail:
                self.jump_to_tail()
            self._clear_caches()

        # Existing width projections are updated from only the appended record.
        # Drop the synthetic trailing row first; the new record supplies it.
        for width, rows in list(self._projection_cache.items()):
            index = self._projection_index_cache.get(width)
            if prior_ends_newline and rows and not self._row_text(rows[-1]):
                removed_index = len(rows) - 1
                removed = rows.pop()
                if index is not None:
                    self._remove_index_row(index, removed, removed_index)
            appended = self._project_record(record, width)
            if separator and appended and not self._row_text(appended[0]):
                # The separator terminates the preceding row; projecting the
                # record independently must not create another blank row.
                appended.pop(0)
            start = len(rows)
            rows.extend(appended)
            if self._ends_newline:
                rows.append(self._empty_tail_row(record))
            if index is not None:
                self._extend_index(index, rows, start)
        self._formatted_cache.clear()

    def replace(
        self,
        sources: list[str],
        *,
        record_ids: list[int] | None = None,
    ) -> None:
        """Reproject semantic records while preserving explicit record identity."""
        if record_ids is not None and len(record_ids) != len(sources):
            raise ValueError("record_ids must have one item per source")
        old_anchor = self._viewport.anchor
        old_records = {record.record_id: record for record in self._records}
        available: dict[str, list[int]] = {}
        for record in self._records:
            available.setdefault(record.plain.lstrip("\n"), []).append(record.record_id)
        rebuilt: list[_Record] = []
        accumulated_plain = ""
        for index, source in enumerate(sources):
            safe = sanitize_transcript_ansi(source)
            plain = strip_safe_ansi(safe)
            separator = "" if index == 0 or accumulated_plain.endswith("\n") else "\n"
            if record_ids is not None:
                record_id = record_ids[index]
            else:
                matches = available.get(plain)
                record_id = matches.pop(0) if matches else self._next_record_id
                if not matches and record_id == self._next_record_id:
                    self._next_record_id += 1
            self._next_record_id = max(self._next_record_id, record_id + 1)
            record_ansi = separator + safe
            rebuilt.append(
                _Record(
                    record_id,
                    record_ansi,
                    separator + plain,
                    bool(separator),
                    safe_hyperlink_spans(record_ansi),
                )
            )
            accumulated_plain += separator + plain
        self._records = rebuilt
        self._projectable_record_count = sum(bool(record.plain) for record in rebuilt)
        self._record_index_cache = None
        self._ends_newline = accumulated_plain.endswith("\n")
        self.clear_selection()
        self._clear_caches()
        if old_anchor is not None:
            replacement = next(
                (record for record in rebuilt if record.record_id == old_anchor.record_id),
                None,
            )
            old_record = old_records.get(old_anchor.record_id)
            if replacement is None or old_record is None:
                # The empty model exposes one synthetic display row whose
                # record id is not retained identity. Never transfer that
                # anchor to a first real record with the same id.
                self.jump_to_tail()
            elif old_record.plain != replacement.plain:
                source_offset = self._source_offset_for_semantic(
                    replacement.plain,
                    old_anchor.semantic_offset,
                    fallback=old_anchor.source_offset,
                )
                self._viewport = ViewportState(
                    False,
                    ViewportAnchor(
                        old_anchor.record_id,
                        source_offset,
                        old_anchor.semantic_offset,
                    ),
                )

    def clear(self) -> None:
        self._records.clear()
        self._projectable_record_count = 0
        self._record_index_cache = None
        self._ends_newline = False
        self.clear_selection()
        self.jump_to_tail()
        self._clear_caches()

    def evict_prefix(self, count: int) -> None:
        """Evict oldest records and repair retained joining semantics."""
        count = max(0, min(count, len(self._records)))
        if not count:
            return
        old_anchor = self._viewport.anchor
        evicted = self._records[:count]
        del self._records[:count]
        self._projectable_record_count -= sum(bool(record.plain) for record in evicted)
        stripped_record_id: int | None = None
        if self._records and self._records[0].has_separator:
            first = self._records[0]
            stripped_record_id = first.record_id
            record_ansi = first.ansi[1:]
            self._records[0] = _Record(
                first.record_id,
                record_ansi,
                first.plain[1:],
                False,
                safe_hyperlink_spans(record_ansi),
            )
            # Selection endpoints are record-local character offsets. Removing
            # the synthetic joining newline must not move a selection that is
            # wholly contained in the surviving record.
            self._selection_anchor = self._shift_hit_after_prefix_strip(
                self._selection_anchor, stripped_record_id
            )
            self._selection_active = self._shift_hit_after_prefix_strip(
                self._selection_active, stripped_record_id
            )
        retained_ids = {record.record_id for record in self._records}
        self._record_index_cache = None
        self._ends_newline = bool(self._records and self._records[-1].plain.endswith("\n"))
        if (
            self._selection_anchor is not None
            and self._selection_active is not None
            and (
                self._selection_anchor.record_id not in retained_ids
                or self._selection_active.record_id not in retained_ids
            )
        ):
            self._selection_anchor = self._selection_active = None
        self._clear_caches()
        if old_anchor is not None and old_anchor.record_id in retained_ids:
            self._viewport = ViewportState(False, old_anchor)
        else:
            self.jump_to_tail()

    @staticmethod
    def _shift_hit_after_prefix_strip(
        hit: _SelectionHit | None, record_id: int
    ) -> _SelectionHit | None:
        if hit is None or hit.record_id != record_id:
            return hit
        return _SelectionHit(
            hit.record_id,
            max(0, hit.before - 1),
            max(0, hit.after - 1),
        )

    def _clear_caches(self) -> None:
        self._projection_cache.clear()
        self._projection_index_cache.clear()
        self._formatted_cache.clear()

    def _projection(self, width: int) -> list[_ProjectedRow]:
        width = max(1, width)
        cached = self._projection_cache.get(width)
        if cached is not None:
            self._projection_cache.move_to_end(width)
            return cached
        rows: list[_ProjectedRow] = []
        previous_ended_newline = True
        for record_index, record in enumerate(self._records):
            projected = self._project_record(record, width)
            if (
                projected
                and not self._row_text(projected[0])
                and (not previous_ended_newline or (record_index > 0 and not rows))
            ):
                projected.pop(0)
            rows.extend(projected)
            if record.plain:
                previous_ended_newline = record.plain.endswith("\n")
        if self._records and self._ends_newline:
            rows.append(self._empty_tail_row(self._records[-1]))
        if not rows:
            rows.append(_ProjectedRow(ViewportAnchor(0, 0), (("", ""),)))
        result = rows
        self._projection_cache[width] = result
        while len(self._projection_cache) > _MAX_PROJECTED_WIDTHS:
            evicted_width, _ = self._projection_cache.popitem(last=False)
            self._projection_index_cache.pop(evicted_width, None)
        return result

    @classmethod
    def _project_record(cls, record: _Record, width: int) -> list[_ProjectedRow]:
        # A semantically empty/ANSI-only record contributes no cells. Any
        # separator required by joining is retained in ``record.plain`` and is
        # projected normally (for example ``"\n"`` after unterminated text).
        if not record.plain:
            return []
        safe = project_prompt_toolkit_ansi(record.ansi)
        fragments = list(to_formatted_text(ANSI(safe)))
        styled_chars: list[tuple[str, str]] = []
        for style, text, *_ in fragments:
            styled_chars.extend((style, char) for char in text)

        rows: list[_ProjectedRow] = []
        row: list[tuple[str, str]] = []
        row_source_spans: list[tuple[int, int]] = []
        row_cells = 0
        source_offset = 0
        semantic_offset = 0
        row_source_offset = 0
        row_semantic_offset = 0
        for start, stop, cells in cls._grapheme_spans(styled_chars):
            cluster = styled_chars[start:stop]
            cluster_text = "".join(char for _, char in cluster)
            if cluster_text == "\n":
                rows.append(
                    _ProjectedRow(
                        ViewportAnchor(record.record_id, row_source_offset, row_semantic_offset),
                        tuple(row),
                        tuple(row_source_spans),
                        record.hyperlinks,
                    )
                )
                source_offset += 1
                row = []
                row_source_spans = []
                row_cells = 0
                row_source_offset = source_offset
                row_semantic_offset = semantic_offset
                continue
            # Keep valid extended graphemes intact. The transcript window
            # installs each cluster as one screen atom with this terminal-cell
            # width, avoiding prompt_toolkit's code-point width accounting for
            # flags, ZWJ emoji, modifiers, and keycaps. NFC is still useful for
            # canonically composable text. Only a cluster physically wider
            # than the entire viewport gets a viewport-local fallback; source
            # text and exports remain unchanged.
            normalized = normalize("NFC", cluster_text)
            if len(normalized) == 1:
                cluster = [(cluster[0][0] if cluster else "", normalized)]
                cluster_text = normalized
            cells = max(cells, 0)
            if cells > width:
                # No terminal can faithfully fit this cluster in the viewport.
                # Preserve the valid Unicode and clip only its layout footprint.
                cells = width
            if row and cells and row_cells + cells > width:
                rows.append(
                    _ProjectedRow(
                        ViewportAnchor(record.record_id, row_source_offset, row_semantic_offset),
                        tuple(row),
                        tuple(row_source_spans),
                        record.hyperlinks,
                    )
                )
                row = []
                row_source_spans = []
                row_cells = 0
                row_source_offset = source_offset
                row_semantic_offset = semantic_offset
            row.extend(cluster)
            row_source_spans.append((start, stop))
            row_cells += cells
            source_offset += 1
            if not cluster_text.isspace():
                semantic_offset += 1
            if row_cells >= width and stop < len(styled_chars) and styled_chars[stop][1] != "\n":
                rows.append(
                    _ProjectedRow(
                        ViewportAnchor(record.record_id, row_source_offset, row_semantic_offset),
                        tuple(row),
                        tuple(row_source_spans),
                        record.hyperlinks,
                    )
                )
                row = []
                row_source_spans = []
                row_cells = 0
                row_source_offset = source_offset
                row_semantic_offset = semantic_offset
        if row or not rows or (styled_chars and styled_chars[-1][1] != "\n"):
            rows.append(
                _ProjectedRow(
                    ViewportAnchor(record.record_id, row_source_offset, row_semantic_offset),
                    tuple(row),
                    tuple(row_source_spans),
                    record.hyperlinks,
                )
            )
        return rows

    @staticmethod
    def _grapheme_spans(chars: list[tuple[str, str]]) -> list[tuple[int, int, int]]:
        """Return terminal grapheme spans without allowing spans across newlines."""
        text = "".join(char for _, char in chars)
        spans: list[tuple[int, int, int]] = []
        offset = 0
        lines = text.split("\n")
        for line_index, line in enumerate(lines):
            rich_spans, _ = split_graphemes(line)
            index = 0
            while index < len(rich_spans):
                start, stop, cells = rich_spans[index]
                # Rich intentionally treats regional indicators separately.
                # Terminal wrapping needs a flag pair to remain one cluster.
                if index + 1 < len(rich_spans):
                    next_start, next_stop, next_cells = rich_spans[index + 1]
                    first = line[start:stop]
                    second = line[next_start:next_stop]
                    if FullscreenTranscriptModel._is_regional(
                        first
                    ) and FullscreenTranscriptModel._is_regional(second):
                        stop = next_stop
                        cells += next_cells
                        index += 1
                spans.append((offset + start, offset + stop, cells))
                index += 1
            offset += len(line)
            if line_index + 1 < len(lines):
                spans.append((offset, offset + 1, 0))
                offset += 1
        return spans

    @staticmethod
    def _is_regional(value: str) -> bool:
        return len(value) == 1 and 0x1F1E6 <= ord(value) <= 0x1F1FF

    @classmethod
    def _empty_tail_row(cls, record: _Record) -> _ProjectedRow:
        # Use exactly the newline-aware span accounting from ``_project_record``.
        # Rich's public splitter can merge consecutive newlines into one span,
        # which would make this final synthetic anchor non-monotonic and break
        # the bisected per-record projection index.
        chars = [("", char) for char in record.plain]
        spans = cls._grapheme_spans(chars)
        semantic_length = sum(
            1 for start, stop, _cells in spans if not record.plain[start:stop].isspace()
        )
        return _ProjectedRow(
            ViewportAnchor(record.record_id, len(spans), semantic_length),
            (("", ""),),
        )

    def _format_rows(self, rows: Sequence[_ProjectedRow], *, top_padding: int = 0) -> FormattedText:
        fragments: list[tuple[str, str]] = []
        for _ in range(top_padding):
            fragments.append(("", "\n"))
        selected = self._selection_bounds()
        records, record_bases = self._record_indexes()
        for index, row in enumerate(rows):
            if index:
                fragments.append(("", "\n"))
            record = records.get(row.anchor.record_id)
            if record is None or not row.source_spans:
                fragments.extend(row.fragments)
                continue
            display_chars = [(style, char) for style, text in row.fragments for char in text]
            display_spans = self._grapheme_spans(display_chars)
            base = record_bases[record.record_id]
            link_index = 0
            link_markers: dict[str, str] = {}
            for offset, (start, stop, _cells) in enumerate(display_spans):
                highlighted = False
                link: str | None = None
                if offset < len(row.source_spans):
                    source_start, source_stop = row.source_spans[offset]
                    if selected is not None:
                        highlighted = (
                            base + source_start < selected[1] and base + source_stop > selected[0]
                        )
                    while (
                        link_index < len(row.hyperlinks)
                        and row.hyperlinks[link_index][1] <= source_start
                    ):
                        link_index += 1
                    if link_index < len(row.hyperlinks):
                        link_start, link_stop, target = row.hyperlinks[link_index]
                        if link_start < source_stop and link_stop > source_start:
                            link = target
                if row.hyperlinks:
                    sequence = f"\x1b]8;;{link}\x1b\\" if link is not None else "\x1b]8;;\x1b\\"
                    fragments.append(("[ZeroWidthEscape]", sequence))
                for style, char in display_chars[start:stop]:
                    if link is not None:
                        marker = link_markers.setdefault(
                            link, sha256(link.encode()).hexdigest()[:12]
                        )
                        style = f"{style} class:native-hyperlink-{marker}".strip()
                    if highlighted:
                        style = f"{style} class:selected".strip()
                    fragments.append((style, char))
        return FormattedText(fragments or [("", "")])

    @classmethod
    def _character_offset(cls, record: _Record, source_offset: int) -> int:
        """Map a grapheme offset to a record-local character offset."""
        if source_offset <= 0:
            return 0
        spans = cls._grapheme_spans([("", char) for char in record.plain])
        if source_offset >= len(spans):
            return len(record.plain)
        return spans[source_offset][0]

    def _selection_bounds(self) -> tuple[int, int] | None:
        if self._selection_anchor is None or self._selection_active is None:
            return None
        _records, bases = self._record_indexes()
        anchor_base = bases.get(self._selection_anchor.record_id)
        active_base = bases.get(self._selection_active.record_id)
        if anchor_base is None or active_base is None:
            return None
        anchor_before = anchor_base + self._selection_anchor.before
        active_before = active_base + self._selection_active.before
        anchor_after = anchor_base + self._selection_anchor.after
        active_after = active_base + self._selection_active.after
        return min(anchor_before, active_before), max(anchor_after, active_after)

    def _record_indexes(self) -> tuple[dict[int, _Record], dict[int, int]]:
        """Return cached identity and logical-offset indexes for retained records."""
        if self._record_index_cache is not None:
            return self._record_index_cache
        records: dict[int, _Record] = {}
        bases: dict[int, int] = {}
        offset = 0
        for record in self._records:
            records[record.record_id] = record
            bases[record.record_id] = offset
            offset += len(record.plain)
        self._record_index_cache = (records, bases)
        return self._record_index_cache

    def _row_index_for_anchor(
        self,
        width: int,
        rows: Sequence[_ProjectedRow],
        anchor: ViewportAnchor,
    ) -> int | None:
        index = self._projection_index_cache.get(width)
        if index is None:
            index = {}
            self._extend_index(index, rows, 0)
            self._projection_index_cache[width] = index
            while len(self._projection_index_cache) > _MAX_PROJECTED_WIDTHS:
                self._projection_index_cache.popitem(last=False)
        else:
            self._projection_index_cache.move_to_end(width)
        record_rows = index.get(anchor.record_id)
        if record_rows is None:
            return None
        offsets, indices = record_rows
        position = bisect_right(offsets, anchor.source_offset) - 1
        return indices[max(0, position)]

    @staticmethod
    def _extend_index(
        index: dict[int, tuple[list[int], list[int]]],
        rows: Sequence[_ProjectedRow],
        start: int,
    ) -> None:
        for row_index in range(start, len(rows)):
            anchor = rows[row_index].anchor
            offsets, indices = index.setdefault(anchor.record_id, ([], []))
            offsets.append(anchor.source_offset)
            indices.append(row_index)

    @staticmethod
    def _remove_index_row(
        index: dict[int, tuple[list[int], list[int]]],
        row: _ProjectedRow,
        row_index: int,
    ) -> None:
        record_rows = index.get(row.anchor.record_id)
        if record_rows is None:
            return
        offsets, indices = record_rows
        if indices and indices[-1] == row_index:
            offsets.pop()
            indices.pop()
        if not indices:
            index.pop(row.anchor.record_id, None)

    @classmethod
    def _source_offset_for_semantic(
        cls,
        text: str,
        semantic_offset: int,
        *,
        fallback: int,
    ) -> int:
        # Remapping must use the exact source coordinate system used by
        # ``_project_record``. In particular, regional indicators are merged
        # into one flag grapheme and newlines remain distinct spans.
        spans = cls._grapheme_spans([("", char) for char in text])
        if semantic_offset <= 0:
            # Preserve a distinct leading-whitespace/blank-row location when
            # there is no semantic token available as a stronger landmark.
            return min(max(0, fallback), len(spans))
        semantic = 0
        for source, (start, stop, _cells) in enumerate(spans):
            if text[start:stop].isspace():
                continue
            if semantic >= semantic_offset:
                return source
            semantic += 1
        return len(spans)

    @staticmethod
    def _row_text(row: _ProjectedRow) -> str:
        return "".join(text for _, text in row.fragments)

    @staticmethod
    def _row_text_length(row: _ProjectedRow) -> int:
        return sum(len(text) for _, text in row.fragments)
