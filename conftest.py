# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root-level pytest fixtures shared across all test directories."""

import sqlite3

import pytest

from nooa.storage.sqlite import _ensure_schema


@pytest.fixture(autouse=True)
def _isolate_config_dirs(tmp_path, monkeypatch):
    """Keep tests out of the developer's real NOOA config directories.

    Project config discovery normally walks up from the checkout. Any test
    exercising a settings writer would therefore mutate this repository's
    ``.nooa/settings.yaml`` unless both writable config roots are isolated.
    """
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "project"))


@pytest.fixture
def sqlite_conn():
    """In-memory SQLite connection with schema initialized."""
    conn = sqlite3.connect(":memory:")
    _ensure_schema(conn)
    yield conn
    conn.close()
