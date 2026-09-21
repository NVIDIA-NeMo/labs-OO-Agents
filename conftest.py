# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root-level pytest fixtures shared across all test directories."""

import sqlite3

import pytest

from nooa.storage.sqlite import _ensure_schema


@pytest.fixture(autouse=True)
def isolate_offline_transport_override(request, monkeypatch):
    """A developer's soak switch must not collapse offline transport matrices.

    Tests of the switch set it explicitly. Live release checks keep their own
    fail-fast guard: a configured switch must never silently relabel evidence.
    """
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.delenv("NOOA_LLM_TRANSPORT", raising=False)


@pytest.fixture
def sqlite_conn():
    """In-memory SQLite connection with schema initialized."""
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    yield conn
    conn.close()
