# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root-level pytest fixtures shared across all test directories."""

import sqlite3

import pytest

from nooa.storage.sqlite import _ensure_schema


@pytest.fixture
def symlink_file():
    """Create a file symlink, skipping only an unavailable Windows privilege."""

    def create(link, target):
        try:
            link.symlink_to(target)
        except OSError as error:
            if getattr(error, "winerror", None) == 1314:
                pytest.skip("Windows file symlinks require Developer Mode or symlink privilege")
            raise

    return create


@pytest.fixture
def sqlite_conn():
    """In-memory SQLite connection with schema initialized."""
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    yield conn
    conn.close()


@pytest.fixture(autouse=True)
def _no_connect_session_call_pacing(monkeypatch):
    """connect's session check paces its three real calls to dodge spurious
    5xx/rate limits on live endpoints; no test hits a live endpoint, so
    disable it everywhere rather than paying the wall-clock cost per test.
    """
    try:
        from nooa.unifiedllm.connect import _session
    except ImportError:
        return
    monkeypatch.setattr(_session, "SESSION_CALL_PACING_SECONDS", 0)
