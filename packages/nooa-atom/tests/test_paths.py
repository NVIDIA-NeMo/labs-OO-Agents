# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Names and paths in the session tree: ``/root``, ``/root/mid``, ``/root/mid/leaf``."""

import pytest
from atom_test_agents import ScriptedModels
from nooa_atom.session.items import ChildFailedError
from nooa_atom.session.names import check_name, is_valid_name
from nooa_atom.session.port import current_port
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore


def test_a_name_is_letters_digits_underscores_and_hyphens_not_a_keyword():
    for good in ("tests", "review_pr_462", "_x", "CI", "ci-check", "review-pr", "sh-1", "_a-"):
        assert is_valid_name(good) and check_name(good, "job") == good  # never rewritten
    for bad in ("Fix parser", "1st", "-x", "class", "", " ", None, "a/b", "a#1", "a.b", "a b"):
        assert not is_valid_name(bad)
        with pytest.raises(ValueError, match="is not valid: use a name such as"):
            check_name(bad, "child")


def test_channel_and_person_names_are_reserved():
    for reserved in ("jobs", "delegates", "user", "user_messages", "system_messages"):
        assert is_valid_name(reserved)  # an identifier, but refused by check_name
        with pytest.raises(ValueError, match=f"name '{reserved}' is reserved: Waiting"):
            check_name(reserved, "job")
    # A queue-manager channel in use (a skill's 'mesh') is refused too.
    with pytest.raises(ValueError, match="child name 'mesh' is reserved"):
        check_name("mesh", "child", ["jobs", "mesh"])
    assert check_name("mesh", "child") == "mesh"  # free when no such channel exists


async def _tree(registry, root_options):
    root = await registry.create(root_options)
    mid = await registry.create(root.options.inherit(name="mid", kept=True), parent_id=root.id)
    leaf = await registry.create(mid.options.inherit(name="leaf", kept=True), parent_id=mid.id)
    return root, mid, leaf


async def test_paths_follow_the_parent_chain(registry, root_options):
    root, mid, leaf = await _tree(registry, root_options)
    assert [registry.path(s.id) for s in (root, mid, leaf)] == [
        "/root",
        "/root/mid",
        "/root/mid/leaf",
    ]
    port = root._agent.session
    assert port.path == "/root" and mid._agent.session.path == "/root/mid"
    [ref] = port.children()
    assert (ref.name, ref.path) == ("mid", "/root/mid")
    assert registry.child_ref(leaf).path == "/root/mid/leaf"
    # The uuid stays on the ref but is not in its repr.
    assert (
        ref.id == mid.id
        and "(id=" not in repr(ref)
        and ", id=" not in repr(ref)
        and "path='/root/mid'" in repr(ref)
    )


async def test_a_deeper_session_is_read_only(registry, root_options):
    root, mid, leaf = await _tree(registry, root_options)
    port = root._agent.session
    token = current_port.set(port)
    try:
        deep = port.descendant("/root/mid/leaf")
        assert (deep.id, deep.parent_id, deep.path) == (leaf.id, mid.id, "/root/mid/leaf")
        assert deep.status == "retained"  # live status, read through the parent
        with pytest.raises(ChildFailedError, match="control it through its parent"):
            await deep.send("hello")
        with pytest.raises(ChildFailedError, match="control it through its parent"):
            await deep.close()
        with pytest.raises(ChildFailedError, match="idle with nothing queued"):
            await deep.wait()
        with pytest.raises(KeyError, match="no session at"):
            port.descendant("/root/mid/nope")
    finally:
        current_port.reset(token)


async def test_an_open_childs_name_is_taken_until_it_is_closed(registry, root_options):
    root, mid, _leaf = await _tree(registry, root_options)
    port = root._agent.session
    with pytest.raises(ValueError, match="name 'mid' is already used by child /root/mid"):
        port.check_child_name("mid")
    with pytest.raises(ValueError, match="is not valid"):
        port.check_child_name("ci check")
    port.check_child_name("review-pr")  # a hyphenated name is free
    for reserved in ("jobs", "delegates", "user_messages"):
        with pytest.raises(ValueError, match=f"child name '{reserved}' is reserved"):
            port.check_child_name(reserved)
    root._agent.queue_manager.queue("mesh")  # a channel in use is refused as a name
    with pytest.raises(ValueError, match="child name 'mesh' is reserved"):
        port.check_child_name("mesh")
    mid._agent.session.check_child_name("mid")  # names are per session
    await registry.close_child(root, mid.id)
    port.check_child_name("mid")  # free again


async def test_names_survive_a_reload_and_older_records_get_one(sessions_dir, root_options):
    models = ScriptedModels()
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    root = await registry.create(root_options)
    kept = await registry.create(root.options.inherit(name="keeper", kept=True), parent_id=root.id)
    # Before names were identifiers, a child was named by its description.
    old = await registry.create(
        root.options.inherit(name="Fix the parser", kept=True), parent_id=root.id
    )
    assert [c.name for c in registry.children(root.id)] == ["keeper", "child_2"]
    assert registry.path(old.id) == "/root/child_2"
    await registry.close_all()

    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    try:
        assert [(c.name, c.status) for c in fresh.children(root.id)] == [
            ("keeper", "on_disk"),
            ("child_2", "on_disk"),
        ]
        loaded = await fresh.load(old.id)
        assert loaded.name == "child_2"
        assert fresh.path(kept.id) == "/root/keeper"
        assert fresh.path(old.id) == "/root/child_2"
    finally:
        await fresh.close_all()


async def test_older_siblings_get_distinct_names_that_stay_stable(sessions_dir, root_options):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=ScriptedModels())
    root = await registry.create(root_options)
    # Two records from before names were identifiers, then a child really named child_1.
    first = await registry.create(root.options.inherit(name="Fix A", kept=True), parent_id=root.id)
    second = await registry.create(root.options.inherit(name="Fix B", kept=True), parent_id=root.id)
    named = await registry.create(
        root.options.inherit(name="child_1", kept=True), parent_id=root.id
    )
    # Places 1 and 2, bumped past child_1 and past each other: never the same name.
    assert registry.path(second.id) == "/root/child_3"  # derived first, for the later sibling
    assert [c.name for c in registry.children(root.id)] == ["child_2", "child_3", "child_1"]
    assert registry.path(first.id) == "/root/child_2"
    assert registry.path(named.id) == "/root/child_1"
    await registry.close_all()

    # Recorded when first derived: a later process keeps them, even once a sibling is gone.
    store = SessionStore(sessions_dir)
    assert store.delete(first.id)
    fresh = SessionRegistry(store, agent_factory=ScriptedModels())
    try:
        assert [c.name for c in fresh.children(root.id)] == ["child_3", "child_1"]
        assert fresh.path(second.id) == "/root/child_3"
        assert (await fresh.load(second.id)).name == "child_3"
    finally:
        await fresh.close_all()
