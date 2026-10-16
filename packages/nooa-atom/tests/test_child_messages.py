# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A child's ``self.message()`` reaches its parent as a ``ChildMessage`` on delegates."""

import asyncio
import json

from atom_test_agents import cell, done, until
from nooa_atom.session.items import ChildMessage, ChildRef

from nooa.interactive import AgentMessage, Done

TIMEOUT = 20


# A parent turn that records what arrived on delegates and keeps waiting
# until a result (not a message) has arrived.
def collect():
    return cell(
        "items = list(notification.get('delegates', []))\n"
        "is_message = lambda i: type(i).__name__ == 'ChildMessage'\n"
        "self.vars['seen'] = list(self.vars.get('seen') or []) + [\n"
        "    f'{type(i).__name__}:' + (i.text if is_message(i) else i.done.explanation)\n"
        "    for i in items\n"
        "]\n"
        "if not all(is_message(i) for i in items):\n"
        "    return_result(Done(explanation='collected'))\n"
        "else:\n"
        "    return_result(Waiting(explanation='more to come', on=['delegates']))"
    )


def _delegates(registry, session_id):
    """(item class name, payload) of each item admitted on delegates, in order."""
    rows = registry.store.load_rows(session_id, frozenset({"ItemAdmitted"}))
    found = []
    for _type, raw in rows:
        if raw.get("channel") != "delegates":
            continue
        item = json.loads(raw["item_json"])
        kind = str(raw["item_type"]).rsplit(":", 1)[-1].rsplit(".", 1)[-1]
        payload = item.get("text") if kind == "ChildMessage" else item.get("done", {})
        found.append((kind, payload if kind == "ChildMessage" else payload.get("explanation")))
    return found


async def _spawn_and_collect(registry, root_options, models, *, keep):
    how = ", kept=True" if keep else ""
    models.scripts[None] = [
        cell(
            f"await self.session.delegate('Kid', 'work'{how})\n"
            "return_result(Waiting(explanation='child working', on=['delegates']))"
        ),
        collect(),
        collect(),
        collect(),
    ]
    # The messages and the turn end come from one cell, as close together as they get.
    models.scripts["Kid"] = [
        cell(
            "self.message('first')\n"
            "self.message('second')\n"
            "return_result(Done(explanation='kid done', message='final words'))"
        )
    ]
    root = await registry.create(root_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    return root, outcome


async def test_a_batch_childs_messages_arrive_in_order_before_its_result(
    registry, root_options, models
):
    root, outcome = await _spawn_and_collect(registry, root_options, models, keep=False)
    assert outcome == Done(explanation="collected")
    assert _delegates(registry, root.id) == [
        ("ChildMessage", "first"),
        ("ChildMessage", "second"),
        ("ChildResult", "kid done"),
    ]
    # The Done's own message comes with the ChildResult, not as a ChildMessage.
    assert root._agent.v.seen == [
        "ChildMessage:first",
        "ChildMessage:second",
        "ChildResult:kid done",
    ]


async def test_a_kept_childs_messages_arrive_in_order_before_its_result(
    registry, root_options, models
):
    root, outcome = await _spawn_and_collect(registry, root_options, models, keep=True)
    assert outcome == Done(explanation="collected")
    assert _delegates(registry, root.id) == [
        ("ChildMessage", "first"),
        ("ChildMessage", "second"),
        ("ChildResult", "kid done"),
    ]
    [child_options] = [o for o in models.built if o.name == "Kid"]
    assert child_options.turn_method == "handle"


async def test_the_childs_messages_stay_in_its_own_events(registry, root_options, models):
    await _spawn_and_collect(registry, root_options, models, keep=True)
    [kid] = registry.children(next(iter(registry.sessions)))
    rows = registry.store.load_rows(kid.id, frozenset({AgentMessage.__name__}))
    assert [raw["content"] for _t, raw in rows][:2] == ["first", "second"]


async def test_a_message_wakes_a_parent_waiting_on_delegates(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "await self.session.delegate('Kid', 'work', kept=True)\n"
            "return_result(Waiting(explanation='child working', on=['delegates']))"
        ),
        cell(
            "[m] = notification['delegates']\n"
            "return_result(Done(explanation=f'{type(m).__name__}: {m.text} from {m.child.name}'))"
        ),
    ]
    # The child only messages and then waits itself: no result is delivered.
    models.scripts["Kid"] = [
        cell(
            "self.message('progress')\n"
            "return_result(Waiting(explanation='idle', on=['user_messages']))"
        )
    ]
    root = await registry.create(root_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="ChildMessage: progress from Kid")
    [(_t, raw)] = [
        r
        for r in registry.store.load_rows(root.id, frozenset({"ItemAdmitted"}))
        if r[1].get("channel") == "delegates"
    ]
    assert raw["source"] == "child:Kid"


async def test_a_message_does_not_resolve_wait(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "child = await self.session.delegate('Kid', 'work')\n"
            "done = (await child.wait()).done\n"
            "return_result(Done(explanation=f'waited: {done.explanation}'))"
        ),
        collect(),
    ]
    models.scripts["Kid"] = [
        cell("self.message('note')\nreturn_result(Done(explanation='kid done'))")
    ]
    root = await registry.create(root_options)
    ended = []
    root.subscribe(lambda e: ended.append(e) if e.kind == "turn_ended" else None)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="waited: kid done")
    # The result went to the waiter; only the message is on delegates, and
    # it wakes the parent for one more turn (which keeps waiting).
    assert _delegates(registry, root.id) == [("ChildMessage", "note")]
    await until(lambda: len(ended) == 2)
    assert root._agent.v.seen == ["ChildMessage:note"]


async def test_a_grandchilds_messages_go_only_to_its_parent(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "c = await self.session.delegate('Mid', 'm')\n"
            "d = (await c.wait()).done\n"
            "return_result(Done(explanation=d.explanation))"
        )
    ]
    models.scripts["Mid"] = [
        cell(
            "g = await self.session.delegate('Leaf', 'l')\n"
            "d = (await g.wait()).done\n"
            "self.v.mid_seen = d.explanation\n"
            "return_result(Done(explanation='mid ok'))"
        ),
        # Mid's later turn for the leaf's message.
        done("mid saw the leaf's message"),
    ]
    models.scripts["Leaf"] = [
        cell("self.message('leaf note')\nreturn_result(Done(explanation='leaf ok'))")
    ]
    root = await registry.create(root_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="mid ok")
    [mid] = registry.children(root.id)
    assert _delegates(registry, mid.id) == [("ChildMessage", "leaf note")]
    assert _delegates(registry, root.id) == []


async def test_a_roots_messages_go_to_the_host_not_delegates(registry, root_options, models):
    models.scripts[None] = [
        cell("self.message('hello host')\nreturn_result(Done(explanation='ok'))")
    ]
    root = await registry.create(root_options)
    rendered = []
    root._agent._render_message = lambda text, **_kw: rendered.append(text)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="ok")
    assert rendered == ["hello host"]
    assert _delegates(registry, root.id) == []
    assert root._agent.queue_manager.get_channel("delegates").qsize() == 0


async def test_a_closing_or_closed_parent_drops_messages(registry, root_options):
    root = await registry.create(root_options)
    child = await registry.create(
        root_options.model_copy(update={"name": "Kid", "kept": True}), parent_id=root.id
    )
    delegates = root._agent.queue_manager.get_channel("delegates")
    root._closing = True
    try:
        child._agent.message("while closing")
    finally:
        root._closing = False
    assert delegates.qsize() == 0
    # A parent no longer live in the registry (closed or detached).
    del registry.sessions[root.id]
    try:
        child._agent.message("after close")
    finally:
        registry.sessions[root.id] = root
    assert delegates.qsize() == 0
    assert _delegates(registry, root.id) == []
    # A live parent does get it.
    child._agent.message("live")
    await until(lambda: delegates.qsize() == 1)
    assert _delegates(registry, root.id) == [("ChildMessage", "live")]


def test_child_message_is_exported():
    import nooa_atom
    from nooa_atom.agent import agent

    assert nooa_atom.ChildMessage is ChildMessage
    assert "ChildMessage" in nooa_atom.__all__ and "ChildMessage" in agent.__all__
    ref = ChildRef(id="x", name="x", depth=1, status="running")
    assert ChildMessage.model_validate_json(ChildMessage(child=ref, text="t").model_dump_json())
