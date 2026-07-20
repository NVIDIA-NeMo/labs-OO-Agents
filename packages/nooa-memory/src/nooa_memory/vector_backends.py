# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pluggable vector-index backends.

All backends implement the ``VectorIndex`` protocol and return cosine scores
(``1.0`` == identical). The ``memories.embedding`` blob column in the store is
the source of truth; a backend is a *derived* nearest-neighbour index that the
store repopulates on open (so swapping backends never loses data).

* ``NumpyVectorIndex``     — exact brute-force cosine, in-process, no extra deps.
* ``SqliteVecVectorIndex`` — ANN inside the SAME SQLite file via the ``sqlite-vec``
                             extension (``uv pip install sqlite-vec``).
* ``ChromaVectorIndex``    — Chroma embedded (persistent/ephemeral) or HTTP server
                             (``uv pip install chromadb``).

Select one with ``MemoryConfig.vector.backend``; ``make_vector_index`` dispatches.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from nooa_memory.config import VectorConfig


@runtime_checkable
class VectorIndex(Protocol):
    """A nearest-neighbour index over L2-normalised vectors (cosine scores)."""

    def add(self, id: str, vector: np.ndarray) -> None: ...
    def remove(self, id: str) -> None: ...
    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...
    def __len__(self) -> int: ...


# ---------------------------------------------------------------------------
# numpy (default, zero-dependency)
# ---------------------------------------------------------------------------
class NumpyVectorIndex:
    """Exact brute-force cosine KNN. Fine to ~10^5 vectors."""

    def __init__(self) -> None:
        self._vecs: dict[str, np.ndarray] = {}
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []

    def add(self, id: str, vector: np.ndarray) -> None:
        self._vecs[id] = vector.astype(np.float32)
        self._matrix = None  # invalidate cache

    def remove(self, id: str) -> None:
        if self._vecs.pop(id, None) is not None:
            self._matrix = None

    def _rebuild(self) -> None:
        self._ids = list(self._vecs.keys())
        if self._ids:
            self._matrix = np.vstack([self._vecs[i] for i in self._ids])
        else:
            self._matrix = np.zeros((0, 0), dtype=np.float32)

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._matrix is None:
            self._rebuild()
        assert self._matrix is not None
        if not self._ids:
            return []
        q = vector.astype(np.float32)
        scores = self._matrix @ q  # cosine, since both sides are normalised
        k = min(k, len(self._ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]

    def __len__(self) -> int:
        return len(self._vecs)


# ---------------------------------------------------------------------------
# sqlite-vec (vectors live in the SAME SQLite file as the store)
# ---------------------------------------------------------------------------
class SqliteVecVectorIndex:
    """ANN via the ``sqlite-vec`` extension, sharing the store's connection.

    Keeps the "everything in one SQLite file" property: the vec0 virtual table
    sits alongside the ``memories``/``memory_edges`` tables. Vectors are
    L2-normalised, so the L2 distance vec0 returns converts to cosine via
    ``cos = 1 - d^2/2``.
    """

    def __init__(self, conn: sqlite3.Connection, dim: int, table: str = "vec_memories") -> None:
        if dim <= 0:
            raise ValueError("sqlite-vec backend requires a positive embedding dim")
        try:
            import sqlite_vec
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "The 'sqlite_vec' backend requires the sqlite-vec package. "
                "Install it with: uv pip install sqlite-vec"
            ) from e
        self._conn = conn
        self._table = table
        self._dim = dim
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        self._serialize = sqlite_vec.serialize_float32
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {table} "
            f"USING vec0(memory_id TEXT PRIMARY KEY, embedding float[{dim}])"
        )
        conn.commit()

    def add(self, id: str, vector: np.ndarray) -> None:
        self._conn.execute(
            f"INSERT OR REPLACE INTO {self._table}(memory_id, embedding) VALUES (?, ?)",
            (id, self._serialize(vector.astype(np.float32).tolist())),
        )
        self._conn.commit()

    def remove(self, id: str) -> None:
        self._conn.execute(f"DELETE FROM {self._table} WHERE memory_id = ?", (id,))
        self._conn.commit()

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            f"SELECT memory_id, distance FROM {self._table} "
            f"WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (self._serialize(vector.astype(np.float32).tolist()), k),
        ).fetchall()
        out: list[tuple[str, float]] = []
        for mid, dist in rows:
            cos = 1.0 - (float(dist) ** 2) / 2.0  # L2-on-unit-vectors -> cosine
            out.append((mid, max(-1.0, min(1.0, cos))))
        return out

    def __len__(self) -> int:
        return int(self._conn.execute(f"SELECT count(*) FROM {self._table}").fetchone()[0])


# ---------------------------------------------------------------------------
# Chroma (embedded persistent / ephemeral / HTTP server)
# ---------------------------------------------------------------------------
class ChromaVectorIndex:
    """Vector index backed by ChromaDB (cosine space)."""

    def __init__(
        self, config: VectorConfig, *, dim: int | None = None, path: str = ":memory:"
    ) -> None:
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:  # pragma: no cover - exercised only without the dep
            raise ImportError(
                "The Chroma backends require the chromadb package. "
                "Install it with: uv pip install chromadb"
            ) from e
        settings = Settings(anonymized_telemetry=False)
        name = config.collection
        if config.backend == "chroma_http":
            self._client = chromadb.HttpClient(
                host=config.host or "localhost", port=config.port or 8000, settings=settings
            )
        elif path and path != ":memory:":
            chroma_dir = str(Path(path).with_suffix(".chroma"))
            self._client = chromadb.PersistentClient(path=chroma_dir, settings=settings)
        else:
            # Ephemeral clients share one in-process system, so a fixed collection
            # name leaks dim/contents across independent in-memory stores. Give each
            # store its own collection.
            import uuid

            self._client = chromadb.EphemeralClient(settings=settings)
            name = f"{config.collection}-{uuid.uuid4().hex[:8]}"
        self._collection = self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, id: str, vector: np.ndarray) -> None:
        self._collection.upsert(ids=[id], embeddings=[vector.astype(np.float32).tolist()])

    def remove(self, id: str) -> None:
        self._collection.delete(ids=[id])

    def query(self, vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        res = self._collection.query(
            query_embeddings=[vector.astype(np.float32).tolist()], n_results=k
        )
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        # Chroma cosine distance = 1 - cosine_similarity.
        return [(i, 1.0 - float(d)) for i, d in zip(ids, dists, strict=False)]

    def __len__(self) -> int:
        return int(self._collection.count())


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
def make_vector_index(
    config: VectorConfig,
    *,
    dim: int | None = None,
    conn: sqlite3.Connection | None = None,
    path: str = ":memory:",
) -> VectorIndex:
    """Build the vector index named by ``config.backend``."""
    backend = config.backend
    if backend == "numpy":
        return NumpyVectorIndex()
    if backend == "sqlite_vec":
        if conn is None:
            raise ValueError("sqlite_vec backend needs the store's sqlite connection")
        if dim is None:
            raise ValueError("sqlite_vec backend needs the embedding dim")
        return SqliteVecVectorIndex(conn, dim)
    if backend in ("chroma_embedded", "chroma_http"):
        return ChromaVectorIndex(config, dim=dim, path=path)
    raise ValueError(f"Unknown vector backend: {backend!r}")
