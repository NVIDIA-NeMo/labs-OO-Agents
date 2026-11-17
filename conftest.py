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


@pytest.fixture(autouse=True)
def _trust_test_folders(monkeypatch, tmp_path_factory):
    """Trust pytest's temp folders (nooa.workspace_trust), as a user who ran /trust would.

    Tests build workspaces there and expect their code skills to load. A
    test of the untrusted case requests ``untrusted_folders``.
    """
    from nooa import workspace_trust

    real = workspace_trust.trusted_folders
    base = tmp_path_factory.getbasetemp().resolve()
    monkeypatch.setattr(workspace_trust, "trusted_folders", lambda: [*real(), base])
    return real


@pytest.fixture
def untrusted_folders(monkeypatch, _trust_test_folders):
    """Undo ``_trust_test_folders``: only the user and NEMO_OO_SETTINGS files trust folders."""
    from nooa import workspace_trust

    monkeypatch.setattr(workspace_trust, "trusted_folders", _trust_test_folders)
