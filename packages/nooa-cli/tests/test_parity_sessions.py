# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sqlite3

import pytest
from nooa_cli.interactive.parity import prepare_sessions

from nooa.sessions import SessionStore
from nooa.storage import SessionAlreadyActiveError


def test_prepares_identical_independent_session_stores(tmp_path):
    store = SessionStore(tmp_path / "source")
    with store.create(session_id="parity", working_directory=str(tmp_path)) as handle:
        handle.record_user_message("seed")
        source = handle.path
    memory = sqlite3.connect(source.with_name("parity-memory.db"))
    try:
        memory.execute("CREATE TABLE fixture (value TEXT)")
        memory.execute("INSERT INTO fixture VALUES ('remember')")
        memory.commit()
    finally:
        memory.close()
    output = tmp_path / "copies"
    manifest = prepare_sessions(source, output)
    assert set(manifest["sha256"]) == {"parity.db"}
    # An old adjacent memory database is not part of the shared session contract.
    assert not (output / "native" / "parity-memory.db").exists()
    assert not (output / "pool" / "parity-memory.db").exists()
    for name in manifest["sha256"]:
        assert (output / "native" / name).read_bytes() == (output / "pool" / name).read_bytes()
    with SessionStore(output / "native").open("parity") as native:
        with SessionStore(output / "pool").open("parity") as pool:
            native.record_user_message("native only")
            assert [turn.content for turn in pool.turns()] == ["seed"]
    assert [turn.content for turn in store.load_turns("parity")] == ["seed"]
    with pytest.raises(FileExistsError):
        prepare_sessions(source, output)
    assert (output / "manifest.json").is_file()


def test_rejects_live_source_without_creating_output(tmp_path):
    store = SessionStore(tmp_path / "source")
    with store.create(session_id="active") as handle:
        with pytest.raises(SessionAlreadyActiveError):
            prepare_sessions(handle.path, tmp_path / "copies")
    assert not (tmp_path / "copies").exists()
