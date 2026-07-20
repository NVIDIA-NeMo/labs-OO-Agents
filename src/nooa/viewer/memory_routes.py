# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory viewer API routes — read-only views over MemoryStore sqlite files.

Serves the web viewer's Memory tab (Records / Dashboard / Explain) on top of
``nooa_memory``: record listing + detail (with the per-memory Usage
panel), store-level KPIs, and the retrieval-explain debugger.

Caveats:

* Opening a store runs its idempotent schema migration — a small write to the
  file. Endpoints never call ``add``/``save``/``archive``; the migration on
  open is the only write this module can cause.
* ``/explain`` embeds the query with the offline ``HashingEmbedder`` at
  ``dim`` dimensions (default 256 = ``EmbeddingConfig`` default). Scores are
  only meaningful for stores written with the hashing backend at the same dim;
  a dim mismatch fails loudly with 422.
"""

from __future__ import annotations

import atexit
import sqlite3
from collections import OrderedDict
from contextlib import suppress
from pathlib import Path

from fastapi import APIRouter, HTTPException

try:
    from nooa_memory.config import EmbeddingConfig, ForgetPolicy, RetrievalConfig
    from nooa_memory.embeddings import get_embedder
    from nooa_memory.forgetting import ForgettingEngine
    from nooa_memory.observability import per_memory_usage, store_kpis
    from nooa_memory.retrieval import RetrievalEngine
    from nooa_memory.schema import Memory
    from nooa_memory.store import MemorySchemaError, MemoryStore

    _HAS_MEMORY = True
except ImportError:
    _HAS_MEMORY = False

router = APIRouter(prefix="/api/memory")

_ALLOWED_SUFFIXES = (".sqlite", ".db")
_PREVIEW_CHARS = 200
_EDGE_PREVIEW_CHARS = 120
# Candidate pool for ?q= keyword search before type/status filtering + pagination.
_SEARCH_POOL = 500

# One store per resolved path — reopening on every request would re-run the
# migration and rebuild the in-memory vector index. refresh_if_changed() keeps
# a cached index current when another process writes the file. The cache is a
# bounded LRU: each entry holds a sqlite connection + vector index, so growth
# must evict (and close) the least-recently-used store.
_MAX_STORES = 8
_stores: OrderedDict[str, MemoryStore] = OrderedDict()


def close_stores() -> None:
    """Drain the store cache, closing every connection. Idempotent."""
    while _stores:
        _, store = _stores.popitem(last=False)
        with suppress(Exception):
            store.close()


atexit.register(close_stores)


def _resolve_db(db: str) -> Path:
    """Validate ``db``: sqlite-ish suffix, inside the working dir, existing file.

    The viewer serves the project directory it was started in, and the ``dbs``
    discovery endpoint only offers paths under it. Containment is enforced here
    because opening a store applies its schema migration — without it, any
    readable sqlite file on the host could be opened (and touched) via ``?db=``.
    """
    root = Path.cwd().resolve()
    path = Path(db).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if path.suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(status_code=422, detail=f"db must end with .sqlite or .db: {db}")
    if not path.is_relative_to(root):
        raise HTTPException(
            status_code=403, detail=f"db must live under the working directory: {db}"
        )
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Memory db not found: {db}")
    return path


def _get_store(db: str) -> MemoryStore:
    key = str(_resolve_db(db))
    store = _stores.get(key)
    if store is None:
        try:
            store = MemoryStore(key)
        except MemorySchemaError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except sqlite3.DatabaseError as e:
            raise HTTPException(status_code=422, detail=f"Not a sqlite database: {db} ({e})") from e
        _stores[key] = store
        while len(_stores) > _MAX_STORES:
            _, evicted = _stores.popitem(last=False)
            with suppress(Exception):
                evicted.close()
    else:
        _stores.move_to_end(key)
        store.refresh_if_changed()
    return store


def _validate_owner(owner: str | None) -> str | None:
    """Reject SQL LIKE wildcards in the ``owner`` filter.

    Role scopes expand to ``owner LIKE role||'@%'`` in the store, so a ``%``
    or ``_`` here would match across agents in a shared store. Stored owners
    can never contain them (``MemoryConfig._validate_owner``) — mirror that
    for the query parameter.
    """
    if owner and any(ch in owner for ch in "%_"):
        raise HTTPException(status_code=422, detail=f"owner may not contain '%' or '_': {owner}")
    return owner


def _forgetting(store: MemoryStore) -> ForgettingEngine:
    return ForgettingEngine(store, ForgetPolicy())


def _head(text: str, chars: int) -> str:
    return " ".join(text.split())[:chars]


def _record_row(m: Memory) -> dict:
    return {
        "id": m.id,
        "type": m.type.value,
        "status": m.status,
        "owner": m.owner,
        "importance": m.importance,
        "importance_label": m.importance_label(),
        "title": m.title,
        "preview": _head(m.content, _PREVIEW_CHARS),
        "tags": m.tags,
        "created_at": m.created_at,
        "last_accessed_at": m.last_accessed_at,
        "archived": m.archived,
        # Deliberate + spontaneous fetches, mirroring observability._fetches
        # (reinforced_count is a write-side dedup signal, not a fetch).
        "fetches": m.recalled_count + m.searched_count + m.injected_count + m.deref_count,
        "edge_count": len(m.edges),
    }


@router.get("/dbs")
def list_dbs() -> list[dict]:
    """Discover memory DBs under the viewer's cwd.

    Scans ``<cwd>/.nooa/memory/*.sqlite`` (project stores) and
    ``<cwd>/.nooa/sessions/*-memory.db`` (per-session TUI stores). This is
    only the cheap local discovery feeding the DB selector — every other
    endpoint honors an explicit ``?db=`` path regardless of this list.
    """
    root = Path.cwd() / ".nooa"
    candidates = sorted((root / "memory").glob("*.sqlite"))
    candidates += sorted((root / "sessions").glob("*-memory.db"))
    out: list[dict] = []
    for p in candidates:
        if not p.is_file():
            continue
        stat = p.stat()
        out.append({"path": str(p), "size_bytes": stat.st_size, "mtime": stat.st_mtime})
    return out


@router.get("/records")
def list_records(
    db: str,
    owner: str | None = None,
    type: str | None = None,
    status: str | None = None,
    q: str | None = None,
    include_archived: bool = False,
    page: int = 1,
    limit: int = 50,
) -> dict:
    """Paginated record listing with owner/type/status filters + keyword search.

    With ``q``, rows come from ``keyword_search`` in rank order (archived rows
    are never keyword-matched); otherwise all memories sorted by last access.
    ``owner`` matches that owner's rows plus unowned shared rows (store
    semantics).
    """
    store = _get_store(db)
    owner = _validate_owner(owner)
    limit = max(1, min(limit, 500))
    page = max(1, page)

    if q:
        ids = store.keyword_search(q, _SEARCH_POOL, owner=owner)
        memories = [m for mid in ids if (m := store.get(mid)) is not None]
    else:
        memories = store.all_memories(include_archived=include_archived, owner=owner)
        memories.sort(key=lambda m: m.last_accessed_at, reverse=True)

    if type is not None:
        memories = [m for m in memories if m.type.value == type]
    if status is not None:
        memories = [m for m in memories if m.status == status]

    total = len(memories)
    start = (page - 1) * limit
    return {
        "records": [_record_row(m) for m in memories[start : start + limit]],
        "total": total,
        "page": page,
        "limit": limit,
        "has_more": start + limit < total,
    }


@router.get("/record")
def get_record(db: str, id: str) -> dict:
    """Full detail for one memory: all fields + usage panel + hydrated edges."""
    store = _get_store(db)
    m = store.get(id)
    if m is None:
        raise HTTPException(status_code=404, detail=f"Memory not found: {id}")

    edges: list[dict] = []
    for e in store.neighbors(id):
        target = store.get(e.target_id)
        edges.append(
            {
                "target_id": e.target_id,
                "type": e.type.value,
                "weight": e.weight,
                "target_type": target.type.value if target else None,
                "target_preview": (
                    _head(target.title or target.content, _EDGE_PREVIEW_CHARS) if target else None
                ),
            }
        )

    payload = m.model_dump(mode="json")
    payload["edges"] = edges  # replace raw Edge dumps with hydrated previews
    payload["importance_label"] = m.importance_label()
    payload["salience_label"] = m.salience_label()
    payload["confidence_label"] = m.confidence_label()
    payload["usage"] = per_memory_usage(m, forgetting=_forgetting(store))
    return payload


@router.get("/stats")
def get_stats(db: str) -> dict:
    """Store-level KPI payload for the dashboard (see observability.store_kpis)."""
    store = _get_store(db)
    return store_kpis(store, forgetting=_forgetting(store))


@router.get("/explain")
def explain(db: str, q: str, k: int = 10, dim: int = 256, owner: str | None = None) -> list[dict]:
    """Dry-run retrieval debugger: the scored candidate table, no touch/logging.

    The query is embedded with the offline ``HashingEmbedder`` at ``dim``
    dimensions (default 256 matches the default ``EmbeddingConfig``). Only
    valid against stores whose embeddings were written by the same hashing
    backend/dim — a mismatched ``dim`` fails with 422 rather than returning
    wrong scores silently.
    """
    if k < 1:
        raise HTTPException(status_code=422, detail="k must be >= 1")
    if dim < 1:
        raise HTTPException(status_code=422, detail="dim must be >= 1")
    owner = _validate_owner(owner)
    store = _get_store(db)
    engine = RetrievalEngine(store, get_embedder(EmbeddingConfig(dim=dim)), RetrievalConfig())
    try:
        return engine.explain(q, k=k, owner=owner)
    except ValueError as e:
        raise HTTPException(
            status_code=422,
            detail=f"explain failed — embedding dim mismatch with the store? ({e})",
        ) from e
