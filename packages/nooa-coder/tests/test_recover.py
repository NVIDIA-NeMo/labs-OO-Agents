# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Recovering a session marked in use: SessionStore.in_use() and SessionStore.fork()."""

import asyncio
import contextlib
import hashlib
import subprocess
import sys
import textwrap

import pytest
from coder_test_agents import BLOCKING_CELL, ScriptedModels, cell, done, fresh_events
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore

TIMEOUT = 20


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mark_foreign(store, session_id):
    """A record another machine left: the kernel lock is not visible across the mount."""
    lock = store.path_for(session_id).with_suffix(".lock")
    lock.write_text("4242 other-box")
    return lock


async def _session_with_history(root_options, sessions_dir, *, title="Fix the parser"):
    """A closed root session with two answered turns and a checkpoint."""
    models = ScriptedModels(
        {
            None: [
                cell("self.v.note = 'kept'\nreturn_result(Done(explanation='noted'))"),
                done("second answer"),
            ]
        }
    )
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(root_options)
        await asyncio.wait_for(root.prompt("first question"), TIMEOUT)
        await asyncio.wait_for(root.prompt("second question"), TIMEOUT)
        await root.wait_for_checkpoint()
        await root.set_title(title, user_set=True)
    finally:
        await registry.close_all()
    return root.id


async def test_a_stale_session_is_listed_as_in_use_and_forked_without_touching_it(
    root_options, sessions_dir
):
    store = SessionStore(sessions_dir)
    original_id = await _session_with_history(root_options, sessions_dir)
    with store.open(original_id) as handle:
        handle.set_mode("ask")
        handle.set_reasoning("high")
    lock = _mark_foreign(store, original_id)
    path = store.path_for(original_id)
    before = _digest(path)
    assert [info.id for info in store.list()] == []

    [in_use] = store.in_use(workspace=root_options.workspace)
    assert (in_use.id, in_use.title, in_use.owner) == (
        original_id,
        "Fix the parser",
        "pid 4242 on other-box",
    )
    assert in_use.last_write == pytest.approx(path.stat().st_mtime)
    assert _digest(path) == before  # listing wrote nothing

    fork = store.fork(original_id)

    assert fork.id != original_id
    assert fork.forked_from == original_id
    assert fork.title == "Fix the parser (recovered)"
    assert (fork.mode, fork.reasoning, fork.workspace) == (
        "ask",
        "high",
        str(root_options.workspace),
    )
    assert [info.id for info in store.list()] == [fork.id]
    assert store.get(fork.id).forked_from == original_id
    assert store.load_transcript(fork.id) == store.load_transcript(original_id)
    assert store.snapshot_ids(fork.id)[0] == store.snapshot_ids(original_id)[0]
    # The fork is closed; the original is untouched and still hidden.
    assert not store.is_active(fork.id)
    assert store.path_for(fork.id).with_suffix(".lock").read_text() == ""
    assert _digest(path) == before
    assert lock.read_text() == "4242 other-box"
    assert [info.id for info in store.in_use()] == [original_id]


async def test_the_fork_loads_with_the_saved_agent_state(root_options, sessions_dir):
    store = SessionStore(sessions_dir)
    original_id = await _session_with_history(root_options, sessions_dir)
    _mark_foreign(store, original_id)
    fork = store.fork(original_id, title="Parser, again")
    assert fork.title == "Parser, again"

    registry = SessionRegistry(store, agent_factory=ScriptedModels())
    try:
        loaded = await registry.load(fork.id)
        assert loaded._agent.v.note == "kept"
    finally:
        await registry.close_all()


_HOLDER = textwrap.dedent(
    """
    import sys
    from nooa_coder.session.store import SessionStore

    handle = SessionStore(sys.argv[1]).open(sys.argv[2])
    print("held", flush=True)
    sys.stdin.read()
    handle.close()
    """
)


async def test_a_session_another_process_holds_is_forked_and_the_holder_is_undisturbed(
    root_options, sessions_dir
):
    store = SessionStore(sessions_dir)
    original_id = await _session_with_history(root_options, sessions_dir)
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, str(sessions_dir), original_id],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        lock = store.path_for(original_id).with_suffix(".lock")
        record = lock.read_text()
        assert record.startswith(f"{holder.pid} ")

        [in_use] = store.in_use()
        assert (in_use.id, in_use.owner) == (original_id, f"local process (pid {holder.pid})")

        fork = store.fork(original_id)
        assert fork.forked_from == original_id
        assert store.load_transcript(fork.id) == store.load_transcript(original_id)
        assert store.is_active(original_id)
        assert lock.read_text() == record
    finally:
        holder.stdin.close()
        assert holder.wait(TIMEOUT) == 0
    assert not store.is_active(original_id)


async def test_items_the_original_never_read_are_not_queued_again_in_the_fork(
    registry, root_options, models, sessions_dir
):
    started, _block = fresh_events()
    models.scripts[None] = [cell(BLOCKING_CELL)]
    root = await registry.create(root_options)
    first = asyncio.ensure_future(root.prompt("start"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    unread = await root.submit("NEVER-READ")
    # A crash: the loop dies mid-turn and the file is let go.
    root._agent.turns._task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await root._agent.turns._task
    root.handle.close()
    first.cancel()
    store = SessionStore(sessions_dir)
    _mark_foreign(store, root.id)

    fork = store.fork(root.id)

    discarded = store.load_rows(fork.id, frozenset({"ItemDiscarded"}))
    assert [raw["item_id"] for _, raw in discarded] == [unread.item_id]
    assert discarded[0][1]["reason"]
    later = ScriptedModels({None: [done("fine")]})
    fresh = SessionRegistry(store, agent_factory=later)
    try:
        loaded = await fresh.load(fork.id)
        assert store.load_rows(fork.id, frozenset({"ItemRequeued"})) == []
        assert await asyncio.wait_for(loaded.prompt("what now?"), TIMEOUT)
        messages = str(later.llms[None].calls[0].messages)
        # The agent is told what was dropped, and it is not handed the item again.
        assert "not carried over" in messages
        assert messages.count("NEVER-READ") == 1
    finally:
        await fresh.close_all()


async def test_the_agent_is_told_which_subagent_sessions_were_not_carried_over(
    registry, root_options, sessions_dir
):
    root = await registry.create(root_options)
    await registry.create(root.options.inherit(name="Review auth", retain=True), parent_id=root.id)
    await registry.close_all()
    store = SessionStore(sessions_dir)
    _mark_foreign(store, root.id)
    before = {info.id for info in store.list(roots_only=False)}

    fork = store.fork(root.id)

    # Children are not copied: the only new file is the fork.
    assert {info.id for info in store.list(roots_only=False)} == before | {fork.id}
    later = ScriptedModels({None: [done("fine")]})
    fresh = SessionRegistry(store, agent_factory=later)
    try:
        loaded = await fresh.load(fork.id)
        await asyncio.wait_for(loaded.prompt("carry on"), TIMEOUT)
        messages = str(later.llms[None].calls[0].messages)
        assert "Review auth" in messages and "not carried over" in messages
    finally:
        await fresh.close_all()


def test_a_damaged_file_is_forked_event_by_event(tmp_path):
    """A file that lost its last pages (as a crash on a shared mount can leave it)."""
    store = SessionStore(tmp_path / "sessions")
    with store.create(workspace=str(tmp_path)) as handle:
        handle.set_title("Damaged work", user_set=True)
        for index in range(300):
            handle.set_mode("ask" if index % 2 else "auto")
        handle.storage.save_snapshot_json('{"old": true}')
        for index in range(300):
            handle.set_mode("ask" if index % 2 else "auto")
    path = store.path_for(handle.id)
    with open(path, "r+b") as file:
        file.truncate(path.stat().st_size - 3 * 4096 - 1000)
    _mark_foreign(store, handle.id)
    before = _digest(path)

    fork = store.fork(handle.id)

    assert fork.forked_from == handle.id
    assert fork.title == "Damaged work (recovered)"
    [(_, recovered)] = store.load_rows(fork.id, frozenset({"SessionRecovered"}))
    assert recovered["copied_by_event"] is True
    assert recovered["skipped_events"] > 0 or recovered["end_unreadable"] is True
    assert store.snapshot_ids(fork.id) != []
    assert _digest(path) == before
    assert fork.id in [info.id for info in store.list()]


def test_a_crash_mid_write_leaves_a_journal_the_fork_can_still_read(tmp_path):
    """A hot rollback journal: a read-only reader cannot roll it back, a copy can."""
    store = SessionStore(tmp_path / "sessions")
    with store.create(workspace=str(tmp_path)) as handle:
        handle.set_title("Interrupted", user_set=True)
    path = store.path_for(handle.id)
    crash = textwrap.dedent(
        """
        import os, sqlite3, sys
        db = sqlite3.connect(sys.argv[1], isolation_level=None)
        db.execute("PRAGMA cache_size=1")
        db.execute("BEGIN")
        db.execute("UPDATE events SET data = data")
        db.execute(
            "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 2000) "
            "INSERT INTO snapshots SELECT 'x' || i, 'later', '{}' || hex(randomblob(200)) FROM n"
        )
        os._exit(0)
        """
    )
    subprocess.run([sys.executable, "-c", crash, str(path)], check=True)
    journal = path.with_name(path.name + "-journal")
    assert journal.exists()
    _mark_foreign(store, handle.id)
    before = (_digest(path), _digest(journal))

    fork = store.fork(handle.id)

    assert fork.title == "Interrupted (recovered)"
    [(_, recovered)] = store.load_rows(fork.id, frozenset({"SessionRecovered"}))
    assert recovered["copied_by_event"] is True
    assert (recovered["skipped_events"], recovered["end_unreadable"]) == (0, False)
    # The copy was rolled back: the half-written snapshots are not in it.
    assert store.snapshot_ids(fork.id) == []
    assert (_digest(path), _digest(journal)) == before


def test_an_untitled_session_gets_a_plain_title(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    with store.create(workspace=str(tmp_path)) as handle:
        pass
    _mark_foreign(store, handle.id)
    assert store.fork(handle.id).title == "Recovered session"


def test_old_tui_claims_and_workspaces_in_the_in_use_listing(tmp_path):
    import json
    import os

    store = SessionStore(tmp_path / "sessions")
    with store.create(workspace=str(tmp_path / "one")) as claimed:
        pass
    with store.create(workspace=str(tmp_path / "two")) as elsewhere:
        pass
    with store.create(workspace=str(tmp_path / "one")) as free:
        pass
    claim_dir = store.path_for(claimed.id).with_suffix(".active")
    claim_dir.mkdir()
    (claim_dir / "owner-a.json").write_text(json.dumps({"pid": os.getpid()}))
    _mark_foreign(store, elsewhere.id)

    listed = {info.id: info.owner for info in store.in_use(workspace=tmp_path / "one")}
    assert listed == {claimed.id: f"old TUI (pid {os.getpid()})"}
    assert free.id not in {info.id for info in store.in_use()}
