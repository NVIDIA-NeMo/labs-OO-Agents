# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference resolution — memories that point at live state instead of copying it.

A ``MemoryRef`` (``kind:key``) resolves by **strict name lookup, never eval**:
memory content can originate from other agents on a shared store, so evaluating
stored strings would be an injection primitive. A key is a dict key / block name
/ relative path / id — nothing more.

Kinds:

* ``var``     — ``agent.vars[key]`` (host-provided persistent vars, e.g. the TUI)
* ``context`` — ``agent.context_manager[key]`` (a context block's current value)
* ``file``    — a path **relative to the working dir, contained within it**
* ``todo``    — ``agent.todo.get(key)`` (the session todo skill, when attached)
* ``memory``  — another record in the same store (prose cross-reference;
  typed relations stay edges)

Resolution returns LIVE (current value) or DANGLING (target gone / not visible
from this agent) — DANGLING renders the write-time ``preview`` snapshot, clearly
stamped. That is the honest semantic for cross-agent recall: reader B sees
writer A's snapshot, labeled as such.
"""

from __future__ import annotations

import time
from collections.abc import MutableMapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from nooa_memory.schema import REF_KINDS, MemoryRef

if TYPE_CHECKING:
    from nooa_memory.store import MemoryStore

_MAX_VALUE_CHARS = 200  # clip for previews and rendered live values
_MAX_FILE_CHARS = 4000  # hard cap on file reads before clipping


class ResolvedRef(BaseModel):
    """A reference plus its resolution outcome."""

    ref: MemoryRef
    status: Literal["LIVE", "DANGLING"]
    value_repr: str  # clipped live value, or the stale preview when DANGLING


def parse_ref(spec: str) -> tuple[str, str]:
    """Split ``kind:key``; raise on malformed input (no fallback)."""
    kind, sep, key = spec.partition(":")
    key = key.strip()
    if not sep or kind not in REF_KINDS or not key:
        raise ValueError(
            f"malformed reference {spec!r}; expected '<kind>:<key>' with kind in {REF_KINDS}"
        )
    return kind, key


def _clip(text: str, limit: int = _MAX_VALUE_CHARS) -> str:
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def _contained_file(key: str) -> Path:
    """Resolve a file key relative to the working dir; refuse escapes.

    Raises ``ValueError`` — used at capture time (own writes). At resolve time
    the caller maps the raise to DANGLING, because a stored ref may come from
    another agent and must not be able to read outside the working dir.
    """
    root = Path.cwd().resolve()
    candidate = Path(key)
    if candidate.is_absolute():
        raise ValueError(f"file reference must be relative to the working dir: {key!r}")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"file reference escapes the working dir: {key!r}")
    return resolved


def _lookup(agent: object, store: MemoryStore, kind: str, key: str) -> str | None:
    """Current value behind a reference, clipped — or None when it doesn't resolve.

    The ``getattr`` capability probes are deliberate: ``vars``/``todo`` are
    host-provided surfaces (TUI agents have them, bare library agents may not).
    """
    if kind == "var":
        vars_map = getattr(agent, "vars", None)
        if isinstance(vars_map, MutableMapping) and key in vars_map:
            return _clip(repr(vars_map[key]))
        return None
    if kind == "context":
        cm = getattr(agent, "context_manager", None)
        if cm is None:
            return None
        try:
            return _clip(str(cm[key]))
        except Exception:
            return None
    if kind == "file":
        try:
            path = _contained_file(key)
        except ValueError:
            return None
        if not path.is_file():
            return None
        try:
            return _clip(path.read_text(errors="replace")[:_MAX_FILE_CHARS])
        except OSError:
            return None
    if kind == "todo":
        todo_skill = getattr(agent, "todo", None)
        get = getattr(todo_skill, "get", None)
        item = get(key) if callable(get) else None
        if item is None:
            return None
        title = getattr(item, "title", None) or str(item)
        status = getattr(item, "status", None)
        return _clip(f"{title} [{status}]" if status else title)
    if kind == "memory":
        try:
            resolved_id = store.resolve_id(key)
        except ValueError:
            return None  # ambiguous prefix from stored/foreign content: degrade
        m = store.get(resolved_id) if resolved_id else None
        if m is None:
            return None
        head = m.title or m.content
        return _clip(f"[{m.type.value}] {head}")
    return None  # unreachable: parse_ref/MemoryRef validate the kind


def capture(agent: object, store: MemoryStore, spec: str) -> MemoryRef:
    """Build a reference from its ``kind:key`` form, snapshotting a preview now.

    Malformed specs raise; a target that doesn't resolve *yet* is allowed
    (preview stays None and resolution reports DANGLING until it appears).
    """
    kind, key = parse_ref(spec)
    if kind == "file":
        _contained_file(key)  # validate containment at write time (raises)
    return MemoryRef(kind=kind, key=key, preview=_lookup(agent, store, kind, key))


def resolve(agent: object, store: MemoryStore, ref: MemoryRef) -> ResolvedRef:
    """Resolve a stored reference against the *reading* agent's live state."""
    value = _lookup(agent, store, ref.kind, ref.key)
    if value is not None:
        return ResolvedRef(ref=ref, status="LIVE", value_repr=value)
    stale = ref.preview if ref.preview is not None else "(no snapshot captured)"
    return ResolvedRef(ref=ref, status="DANGLING", value_repr=stale)


def render(resolved: ResolvedRef) -> str:
    """One agent-facing line for a resolved reference."""
    tag = resolved.ref.spec()
    if resolved.status == "LIVE":
        return f"ref {tag} (LIVE) → {resolved.value_repr}"
    stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(resolved.ref.captured_at))
    return f"ref {tag} (DANGLING; snapshot @ {stamp}) → {resolved.value_repr}"
