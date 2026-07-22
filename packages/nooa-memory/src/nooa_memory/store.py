# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SQLite-centric memory store + pluggable vector index.

Everything lives in one SQLite file selected by ``MemoryConfig.path`` or the
embedding host. TUI defaults to one memory DB per session; other hosts may choose
workspace/project paths explicitly:

* ``memories``      — full record as JSON + promoted columns for filtering/sort
                      + the embedding as a float32 blob.
* ``memory_edges``  — the directed, typed association graph.

The vector index is abstracted behind ``VectorIndex`` so the brute-force numpy
default can later be swapped for ``sqlite-vec`` / Chroma without touching callers.
The default ``NumpyVectorIndex`` does exact cosine KNN in memory over
L2-normalised vectors.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nooa_memory.schema import Edge, EdgeType, Memory, MemoryType
from nooa_memory.vector_backends import (
    NumpyVectorIndex,
    VectorIndex,
    make_vector_index,
)

if TYPE_CHECKING:
    from nooa_memory.config import VectorConfig


def _vec_to_blob(vec: np.ndarray) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


# Stamp of the DDL shape this code understands (PRAGMA user_version).
SCHEMA_VERSION = 2


class MemorySchemaError(RuntimeError):
    """The store file's schema is newer than this code understands."""


# Note: the two v2 columns (owner/status) appear in the CREATE TABLE for fresh
# files; their indexes are created in ``_migrate`` because on a v1 file the
# columns don't exist until the ALTERs run.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id            TEXT PRIMARY KEY,
    type          TEXT NOT NULL,
    content       TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 5.0,
    salience      REAL NOT NULL DEFAULT 0.0,
    strength      INTEGER NOT NULL DEFAULT 1,
    created_at    REAL NOT NULL,
    last_accessed REAL NOT NULL,
    access_count  INTEGER NOT NULL DEFAULT 0,
    archived      INTEGER NOT NULL DEFAULT 0,
    owner         TEXT NOT NULL DEFAULT '',
    status        TEXT,
    embedding     BLOB,
    data          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(type);
CREATE INDEX IF NOT EXISTS idx_mem_archived ON memories(archived);

CREATE TABLE IF NOT EXISTS memory_edges (
    src        TEXT NOT NULL,
    dst        TEXT NOT NULL,
    type       TEXT NOT NULL,
    weight     REAL NOT NULL DEFAULT 1.0,
    created_at REAL NOT NULL,
    PRIMARY KEY (src, dst, type)
);
CREATE INDEX IF NOT EXISTS idx_edge_src ON memory_edges(src);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON memory_edges(dst);

CREATE TABLE IF NOT EXISTS maintenance_log (
    ts     REAL NOT NULL,
    kind   TEXT NOT NULL,
    report TEXT NOT NULL
);
"""


class MemoryStore:
    """Persistent store for memories + their association graph + vectors."""

    def __init__(
        self,
        path: str | Path = ":memory:",
        *,
        vector_index: VectorIndex | None = None,
        vector_config: VectorConfig | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()
        # An explicit index wins; otherwise dispatch on the config's backend
        # (numpy / sqlite_vec / chroma_*). Default is the zero-dependency numpy index.
        if vector_index is not None:
            self._index: VectorIndex = vector_index
        elif vector_config is not None:
            self._index = make_vector_index(
                vector_config, dim=embedding_dim, conn=self._conn, path=self.path
            )
        else:
            self._index = NumpyVectorIndex()
        self._load_index()
        self._data_version = self._current_data_version()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _migrate(self) -> None:
        """Bring the file to ``SCHEMA_VERSION``; refuse files from the future.

        Hybrid on purpose: the upgrades themselves are introspection-guarded
        idempotent ALTERs (a half-migrated file converges), while the version
        stamp does the one thing introspection cannot — fail loudly when the
        file was written by *newer* code (shared project stores can be opened
        by processes running different checkouts).
        """
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise MemorySchemaError(
                f"memory store {self.path!r} has schema v{version}, but this code "
                f"understands up to v{SCHEMA_VERSION} — update nooa"
            )
        self._conn.executescript(_SCHEMA)
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(memories)").fetchall()}
        if "owner" not in cols:
            self._conn.execute("ALTER TABLE memories ADD COLUMN owner TEXT NOT NULL DEFAULT ''")
        if "status" not in cols:
            self._conn.execute("ALTER TABLE memories ADD COLUMN status TEXT")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_owner ON memories(owner)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_mem_status ON memories(status)")
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def _current_data_version(self) -> int:
        return int(self._conn.execute("PRAGMA data_version").fetchone()[0])

    def refresh_if_changed(self) -> bool:
        """Reload the derived vector index if another connection committed.

        SQLite's ``data_version`` moves only on *other* connections' commits,
        so this is a cheap no-op in the single-writer case. Only the in-memory
        numpy index needs rebuilding; sqlite-vec and Chroma read shared storage
        directly. SQL reads are always current and never need this.
        """
        version = self._current_data_version()
        if version == self._data_version:
            return False
        self._data_version = version
        if isinstance(self._index, NumpyVectorIndex):
            self._index = NumpyVectorIndex()
            self._load_index()
        return True

    def _load_index(self) -> None:
        cur = self._conn.execute(
            "SELECT id, embedding FROM memories WHERE archived = 0 AND embedding IS NOT NULL"
        )
        for row in cur.fetchall():
            self._index.add(row["id"], _blob_to_vec(row["embedding"]))

    def _row_to_memory(self, row: sqlite3.Row) -> Memory:
        data = json.loads(row["data"])
        edges = [
            Edge(target_id=e["dst"], type=EdgeType(e["type"]), weight=e["weight"])
            for e in self._edge_rows(row["id"])
        ]
        data["edges"] = [e.model_dump(mode="json") for e in edges]
        # Promoted columns are authoritative (archive() flips the column;
        # v1 payloads predate owner/status), so they override whatever the
        # serialised payload captured at write time.
        data["archived"] = bool(row["archived"])
        data["owner"] = row["owner"]
        data["status"] = row["status"]
        return Memory.model_validate(data)

    def _edge_rows(self, src: str) -> list[dict]:
        cur = self._conn.execute("SELECT dst, type, weight FROM memory_edges WHERE src = ?", (src,))
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # writes
    # ------------------------------------------------------------------
    def add(self, memory: Memory, embedding: np.ndarray | None = None) -> Memory:
        """Insert (or replace) a memory and its edges. Persists the embedding."""
        blob = _vec_to_blob(embedding) if embedding is not None else None
        payload = memory.model_dump(mode="json", exclude={"edges"})
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, type, content, importance, salience, strength, created_at,
                last_accessed, access_count, archived, owner, status, embedding, data)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                memory.id,
                memory.type.value,
                memory.content,
                memory.importance,
                memory.salience,
                memory.strength,
                memory.created_at,
                memory.last_accessed_at,
                memory.access_count,
                int(memory.archived),
                memory.owner,
                memory.status,
                blob,
                json.dumps(payload),
            ),
        )
        # Rewrite edges for this source.
        self._conn.execute("DELETE FROM memory_edges WHERE src = ?", (memory.id,))
        for e in memory.edges:
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_edges (src, dst, type, weight, created_at)"
                " VALUES (?,?,?,?,?)",
                (memory.id, e.target_id, e.type.value, e.weight, e.created_at),
            )
        self._conn.commit()
        if embedding is not None and not memory.archived:
            self._index.add(memory.id, embedding)
        elif memory.archived:
            self._index.remove(memory.id)
        return memory

    def save(self, memory: Memory) -> None:
        """Persist mutations to an existing memory (keeps the stored embedding)."""
        payload = memory.model_dump(mode="json", exclude={"edges"})
        self._conn.execute(
            """UPDATE memories SET type=?, content=?, importance=?, salience=?,
               strength=?, last_accessed=?, access_count=?, archived=?, owner=?,
               status=?, data=? WHERE id=?""",
            (
                memory.type.value,
                memory.content,
                memory.importance,
                memory.salience,
                memory.strength,
                memory.last_accessed_at,
                memory.access_count,
                int(memory.archived),
                memory.owner,
                memory.status,
                json.dumps(payload),
                memory.id,
            ),
        )
        self._conn.execute("DELETE FROM memory_edges WHERE src = ?", (memory.id,))
        for e in memory.edges:
            self._conn.execute(
                "INSERT OR REPLACE INTO memory_edges (src, dst, type, weight, created_at)"
                " VALUES (?,?,?,?,?)",
                (memory.id, e.target_id, e.type.value, e.weight, e.created_at),
            )
        self._conn.commit()
        if memory.archived:
            self._index.remove(memory.id)

    def add_edge(
        self, src: str, dst: str, type: EdgeType = EdgeType.RELATED, weight: float = 1.0
    ) -> None:
        from nooa_memory.schema import _now

        self._conn.execute(
            "INSERT OR REPLACE INTO memory_edges (src, dst, type, weight, created_at)"
            " VALUES (?,?,?,?,?)",
            (src, dst, type.value, weight, _now()),
        )
        self._conn.commit()

    def archive(self, id: str) -> None:
        self._conn.execute("UPDATE memories SET archived = 1 WHERE id = ?", (id,))
        self._conn.commit()
        self._index.remove(id)

    def delete(self, id: str) -> None:
        self._conn.execute("DELETE FROM memories WHERE id = ?", (id,))
        self._conn.execute("DELETE FROM memory_edges WHERE src = ? OR dst = ?", (id, id))
        self._conn.commit()
        self._index.remove(id)

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------
    def get(self, id: str) -> Memory | None:
        row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (id,)).fetchone()
        return self._row_to_memory(row) if row else None

    def resolve_id(self, id_or_prefix: str) -> str | None:
        """Exact id, or a unique id prefix (>=6 chars, as rendered in recalled
        lines). Raises on an ambiguous prefix; returns None when nothing matches."""
        row = self._conn.execute("SELECT id FROM memories WHERE id = ?", (id_or_prefix,)).fetchone()
        if row is not None:
            return str(row["id"])
        if len(id_or_prefix) < 6:
            return None
        rows = self._conn.execute(
            "SELECT id FROM memories WHERE id LIKE ? LIMIT 2", (id_or_prefix + "%",)
        ).fetchall()
        if len(rows) > 1:
            raise ValueError(f"memory id prefix {id_or_prefix!r} is ambiguous")
        return str(rows[0]["id"]) if rows else None

    def rename_owner(self, old: str, new: str) -> int:
        """Re-stamp every row owned by ``old`` to ``new``; returns rows changed.

        The promoted column is authoritative on read, so the stale ``owner``
        inside the data JSON is harmless. Used to heal legacy owner spellings
        (e.g. the TUI's module:qualname keys) into the canonical short name.
        """
        if old == new or not old:
            return 0  # never mass-claim the unowned/shared namespace
        cur = self._conn.execute("UPDATE memories SET owner = ? WHERE owner = ?", (new, old))
        self._conn.commit()
        return cur.rowcount

    def owner_of(self, id: str) -> str | None:
        """The owner column alone (cheap visibility checks during graph spread)."""
        row = self._conn.execute("SELECT owner FROM memories WHERE id = ?", (id,)).fetchone()
        return None if row is None else str(row["owner"])

    def get_embedding(self, id: str) -> np.ndarray | None:
        row = self._conn.execute("SELECT embedding FROM memories WHERE id = ?", (id,)).fetchone()
        if row and row["embedding"] is not None:
            return _blob_to_vec(row["embedding"])
        return None

    def neighbors(self, id: str) -> list[Edge]:
        return [
            Edge(target_id=e["dst"], type=EdgeType(e["type"]), weight=e["weight"])
            for e in self._edge_rows(id)
        ]

    @staticmethod
    def _owner_clause(owner: str) -> tuple[str, list[object]]:
        """SQL for one owner scope: exact instance for ``role@inst``, role
        scope (all instances + bare-role rows) for a bare role; unowned rows
        always match."""
        if "@" in owner:
            return "owner IN (?, '')", [owner]
        return "(owner = ? OR owner LIKE ? OR owner = '')", [owner, owner + "@%"]

    @classmethod
    def _filters(cls, *, include_archived: bool, owner: str | None) -> tuple[str, list[object]]:
        """WHERE clause for the shared read filters.

        ``owner=None`` means no filtering (the manager resolves policy —
        "mine" / "*" / a name — before calling the store); a concrete scope
        follows the hierarchical-owner semantics of ``_owner_clause``.
        """
        clauses: list[str] = []
        params: list[object] = []
        if not include_archived:
            clauses.append("archived = 0")
        if owner is not None:
            clause, owner_params = cls._owner_clause(owner)
            clauses.append(clause)
            params.extend(owner_params)
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def all_memories(
        self, *, include_archived: bool = False, owner: str | None = None
    ) -> list[Memory]:
        where, params = self._filters(include_archived=include_archived, owner=owner)
        q = "SELECT * FROM memories" + where
        return [self._row_to_memory(r) for r in self._conn.execute(q, params).fetchall()]

    def iter_memories(
        self, *, include_archived: bool = False, owner: str | None = None
    ) -> Iterator[Memory]:
        yield from self.all_memories(include_archived=include_archived, owner=owner)

    def count(self, *, include_archived: bool = False, owner: str | None = None) -> int:
        where, params = self._filters(include_archived=include_archived, owner=owner)
        q = "SELECT COUNT(*) AS n FROM memories" + where
        return int(self._conn.execute(q, params).fetchone()["n"])

    def _owned_ids(self, ids: list[str], owner: str) -> set[str]:
        if not ids:
            return set()
        placeholders = ",".join("?" * len(ids))
        clause, owner_params = self._owner_clause(owner)
        rows = self._conn.execute(
            f"SELECT id FROM memories WHERE id IN ({placeholders}) AND {clause}",
            [*ids, *owner_params],
        ).fetchall()
        return {r["id"] for r in rows}

    def knn(
        self, query_vec: np.ndarray, k: int, *, owner: str | None = None
    ) -> list[tuple[str, float]]:
        """Top-k by cosine similarity (excludes archived).

        With ``owner``, results are post-filtered to that owner + unowned rows.
        The ``VectorIndex`` protocol has no metadata filter, so the index is
        oversampled (4x, growing) until ``k`` survive or it is exhausted.
        """
        self.refresh_if_changed()
        if owner is None:
            return self._index.query(query_vec, k)
        total = len(self._index)
        n = min(total, max(k * 4, k))
        while True:
            ranked = self._index.query(query_vec, n)
            allowed = self._owned_ids([mid for mid, _ in ranked], owner)
            out = [(mid, score) for mid, score in ranked if mid in allowed]
            if len(out) >= k or n >= total:
                return out[:k]
            n = min(total, n * 4)

    def keyword_search(
        self,
        text: str,
        k: int,
        *,
        mem_type: MemoryType | None = None,
        owner: str | None = None,
    ) -> list[str]:
        """Cheap sparse fallback: rank by overlapping-token LIKE matches."""
        from nooa_memory.embeddings import _TOKEN_RE

        tokens = list(dict.fromkeys(_TOKEN_RE.findall(text.lower())))[:12]
        if not tokens:
            return []
        scored: dict[str, int] = {}
        for tok in tokens:
            clause = "SELECT id FROM memories WHERE archived = 0 AND lower(content) LIKE ?"
            params: list[object] = [f"%{tok}%"]
            if mem_type is not None:
                clause += " AND type = ?"
                params.append(mem_type.value)
            if owner is not None:
                owner_sql, owner_params = self._owner_clause(owner)
                clause += f" AND {owner_sql}"
                params.extend(owner_params)
            for row in self._conn.execute(clause, params).fetchall():
                scored[row["id"]] = scored.get(row["id"], 0) + 1
        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return [mid for mid, _ in ranked[:k]]

    def log_maintenance(self, kind: str, report: dict) -> None:
        """Append one reflect()/prune() run to the store-level maintenance history.

        The one deliberate exception to record-level self-containment: this
        history describes the *store*, not any single memory — it still lives
        in the same SQLite file (nothing external to track).
        """
        from nooa_memory.schema import _now

        self._conn.execute(
            "INSERT INTO maintenance_log (ts, kind, report) VALUES (?,?,?)",
            (_now(), kind, json.dumps(report)),
        )
        self._conn.commit()

    def maintenance_history(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT ts, kind, report FROM maintenance_log ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"ts": r["ts"], "kind": r["kind"], "report": json.loads(r["report"])} for r in rows]

    def close(self) -> None:
        self._conn.close()
