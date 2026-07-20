# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root-level pytest fixtures shared across all test directories."""

import sqlite3

import pytest

from nooa.storage.sqlite import _ensure_schema


@pytest.fixture
def sqlite_conn():
    """In-memory SQLite connection with schema initialized."""
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    yield conn
    conn.close()
