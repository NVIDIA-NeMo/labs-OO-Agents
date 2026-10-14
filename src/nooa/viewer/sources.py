# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional read sources, discovered from installed Python entry points.

Catalog reads do not import traces. Detail reads materialize portable OTLP and
message-journal records locally so rendering, search and annotations share the
same implementation as local traces. Remote stores are never written to.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from collections.abc import Iterable
from functools import lru_cache
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any, Protocol

from . import otlp_store


class TraceSource(Protocol):
    """An installed source factory receives its name and JSON options.

    Catalog rows use the local store's session shape: id, name, experiment,
    modified (epoch seconds), span_count (or None), and eval metadata. IDs must
    be globally unique and stable. version changes invalidate cached details.
    load_session returns OTLP bodies and nooaJournal blocks/call records.
    """

    version: str

    def list_sessions(self) -> list[dict[str, Any]]: ...

    def load_session(self, session_id: str) -> Iterable[dict[str, Any]]: ...


@lru_cache(maxsize=1)
def configured_sources() -> dict[str, TraceSource]:
    config = os.environ.get("NOOA_VIEWER_SOURCES_CONFIG")
    if not config:
        return {}
    entries = entry_points(group="nooa.viewer.sources")
    result = {}
    for item in json.loads(Path(config).read_text())["sources"]:
        name = item["name"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in result:
            raise ValueError("Source names must be unique letters, digits, underscores or hyphens")
        matches = [entry for entry in entries if entry.name == item["plugin"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one installed trace-source plugin: {item['plugin']}")
        result[name] = matches[0].load()(name=name, options=item.get("options", {}))
    return result


class SelectableRunSource(Protocol):
    """Optional capability: validate a run and return updated JSON options.

    Preparation must not mutate the active source or write to a remote store.
    The viewer validates a replacement catalog and persists its configuration
    before activating it. Endpoint-specific run validation stays in the plugin.
    """

    def prepare_run(self, run_id: str) -> dict[str, Any]: ...


_selection_lock = threading.Lock()


def add_run(name: str, run_id: str) -> dict[str, Any]:
    """Persist and activate a source selection without restarting the viewer."""
    with _selection_lock:
        if name not in configured_sources():
            raise KeyError(name)
        path = Path(os.environ["NOOA_VIEWER_SOURCES_CONFIG"])
        config = json.loads(path.read_text())
        item = next((s for s in config["sources"] if s["name"] == name), None)
        if item is None:
            raise KeyError(name)
        matches = [e for e in entry_points(group="nooa.viewer.sources") if e.name == item["plugin"]]
        if len(matches) != 1:
            raise ValueError("The configured source plugin is no longer uniquely installed")
        factory = matches[0].load()
        current = factory(name=name, options=item.get("options", {}))
        prepare = getattr(current, "prepare_run", None)
        if not callable(prepare):
            raise NotImplementedError("This source does not support selecting runs")
        options = prepare(run_id)
        replacement = factory(name=name, options=options)
        # Validate across sources too: names may have overlapping prefixes.
        catalog = [
            session
            for owner, _, session in _catalog({**configured_sources(), name: replacement}).values()
            if owner == name
        ]
        changed = options != item.get("options", {})
        if changed:
            item["options"] = options
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
                    temporary = Path(f.name)
                    os.chmod(temporary, path.stat().st_mode & 0o777)
                    json.dump(config, f, indent=2)
                    f.write("\n")
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(temporary, path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        # Replace only this source. Readers already using the old instance may
        # finish normally; subsequent requests see the new, preloaded catalog.
        configured_sources()[name] = replacement
        return {
            "source": name,
            "run_id": run_id,
            "added": changed,
            "experiments": sorted({s["experiment"] for s in catalog if s.get("experiment")}),
        }


def _catalog(plugins=None) -> dict[str, tuple[str, TraceSource, dict[str, Any]]]:
    rows = {}
    for name, source in (configured_sources() if plugins is None else plugins).items():
        for session in source.list_sessions():
            sid = session["id"]
            if sid in rows:
                raise ValueError(f"Trace-source session ID collision: {sid}")
            if not sid.startswith(name + "-"):
                raise ValueError("Source session IDs must use their configured name as a namespace")
            rows[sid] = (name, source, session)
    return rows


def list_sessions(experiment=None, eval_only=False, batch_id=None):
    if not configured_sources():
        return otlp_store.list_sessions(experiment, eval_only, batch_id)
    rows = {row["id"]: row for row in otlp_store.list_sessions(experiment, eval_only, batch_id)}
    cached_rows = {row["id"]: row for row in otlp_store.list_sessions()}
    for sid, (name, _source, session) in _catalog().items():
        # Catalog membership is authoritative even when the portable payload
        # omitted evaluation fields or used a different experiment name.
        rows.pop(sid, None)
        if experiment is not None and session.get("experiment") != experiment:
            continue
        if batch_id is not None and session.get("batch_id") != batch_id:
            continue
        if eval_only and not session.get("eval"):
            continue
        cached = cached_rows.get(sid)
        # The catalog owns grades; the local cache owns measured trace metrics.
        rows[sid] = {**session, "source": name}
        if cached:
            rows[sid]["span_count"] = cached["span_count"]
            rows[sid]["eval"] = {**cached.get("eval", {}), **session.get("eval", {})}
            availability = cached.get("eval", {}).get("trace_available")
            if availability is not None:
                rows[sid]["eval"]["trace_available"] = availability
    return list(rows.values())


def list_experiments():
    if not configured_sources():
        return otlp_store.list_experiments()
    return sorted({s["experiment"] for s in list_sessions() if s.get("experiment")})


_lock_guard = threading.Lock()
_locks: dict[str, threading.Lock] = {}


def ensure_session(session_id: str):
    plugins = configured_sources()
    if not plugins or not any(session_id.startswith(name + "-") for name in plugins):
        return
    if otlp_store.session_exists(session_id):
        resource = otlp_store.get_session_resource(session_id)
        owner = resource.get("viewer.source.name")
        if owner in plugins:
            if resource.get("viewer.source.version") == plugins[owner].version:
                return  # Opening a cached trace does not need a remote catalog.
            raise ValueError("Cached source version changed; clear its cache before reopening")
    entry = _catalog().get(session_id)
    if entry is None:
        return
    name, source, _ = entry
    with _lock_guard:
        lock = _locks.setdefault(session_id, threading.Lock())
    with lock:
        if otlp_store.session_exists(session_id):
            resource = otlp_store.get_session_resource(session_id)
            if resource.get("viewer.source.name") != name:
                raise ValueError(f"Source cannot overwrite an existing local session: {session_id}")
            if resource.get("viewer.source.version") == source.version:
                return
            # Do not silently replace traces with annotations during upgrades.
            raise ValueError("Cached source version changed; clear its cache before reopening")
        records = list(source.load_session(session_id))
        body = {"resourceSpans": []}
        journals = []
        for record in records:
            for resource in record.get("resourceSpans", []):
                attrs = resource.setdefault("resource", {}).setdefault("attributes", [])
                identity = next(
                    (a["value"].get("stringValue") for a in attrs if a["key"] == "session.id"), None
                )
                if identity != session_id:
                    raise ValueError("Source returned a mismatched session identity")
                attrs.extend(
                    [
                        {"key": "viewer.source.name", "value": {"stringValue": name}},
                        {"key": "viewer.source.version", "value": {"stringValue": source.version}},
                    ]
                )
                body["resourceSpans"].append(resource)
            journal = record.get("nooaJournal")
            if journal and journal.get("type") != "manifest":
                if journal.get("session_id") != session_id:
                    raise ValueError("Source returned a mismatched journal identity")
                if journal["type"] not in ("blocks", "call"):
                    raise ValueError("Unsupported source journal record")
                if journal["type"] == "call" and journal["call"].get("session_id") != session_id:
                    raise ValueError("Source returned a mismatched call identity")
                if journal["type"] == "call":
                    # Call IDs in the native journal table are globally keyed.
                    # Portable traces may reuse IDs, so scope them by session.
                    journal = {
                        **journal,
                        "call": {
                            **journal["call"],
                            "call_id": json.dumps([session_id, journal["call"]["call_id"]]),
                        },
                    }
                journals.append(journal)
        if not any(
            scope.get("spans")
            for resource in body["resourceSpans"]
            for scope in resource.get("scopeSpans", [])
        ):
            raise ValueError("Source returned no OTLP trace")

        def write():
            # A native exporter may have published this identity while the
            # remote fetch was running. Recheck ownership on the writer thread.
            if otlp_store.session_exists(session_id):
                resource = otlp_store.get_session_resource(session_id)
                if resource.get("viewer.source.name") != name:
                    raise ValueError(
                        f"Source cannot overwrite an existing local session: {session_id}"
                    )
                if resource.get("viewer.source.version") != source.version:
                    raise ValueError(
                        "Cached source version changed; clear its cache before reopening"
                    )
                return
            otlp_store.ingest_session_with_journal(body, journals)

        # Serialize with native ingestion and annotation writes.
        from .main import _write_executor

        _write_executor.submit(write).result()
