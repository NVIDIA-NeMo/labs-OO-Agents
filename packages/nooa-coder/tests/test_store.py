# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SessionStore: durable session files, their metadata and listing."""

from nooa_coder.session.items import SessionInfo, Usage
from nooa_coder.session.store import SessionStore


def test_store_returns_the_single_session_info_model(sessions_dir):
    store = SessionStore(sessions_dir)
    with store.create(model="m", agent="pkg:Agent") as handle:
        assert isinstance(handle.info, SessionInfo)
        assert handle.info.status == "on_disk"
        assert handle.info.usage == Usage()
        session_id = handle.id

    info = store.get(session_id)
    assert isinstance(info, SessionInfo)
    assert (info.id, info.model, info.agent) == (session_id, "m", "pkg:Agent")
    assert [i.id for i in store.list()] == [session_id]
