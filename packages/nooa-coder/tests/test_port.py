# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent-side port (self.session), ChildRef and delivery of child results."""

import asyncio
import logging

import pytest
from coder_test_agents import (
    BLOCKING_CELL,
    ScriptedModels,
    cell,
    done,
    fresh_events,
    wait_on,
)
from nooa_coder.session.items import ChildRef, TaskResult
from nooa_coder.session.port import SessionPort, current_port, install_port
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore

from nooa.agentdoc import doc
from nooa.interactive import Done
from nooa.llm_types import LLMUsage
from nooa.storage.json_snapshot import snapshot_to_json

TIMEOUT = 20


def _turns(registry, session_id):
    return registry.store.load_rows(session_id, frozenset({"TurnStarted"}))


async def _until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), TIMEOUT)


async def test_the_port_is_installed_visible_and_not_snapshotted(registry, root_options, caplog):
    root = await registry.create(root_options)
    port = root.agent.session
    assert isinstance(port, SessionPort)
    assert (port.id, port.depth, port.max_depth) == (root.id, 0, root_options.max_depth)
    assert "delegates" in root.agent.queue_manager.channels()
    rendered = doc(root.agent)
    assert "delegate" in rendered and "Create a child session" in rendered
    with caplog.at_level(logging.WARNING):
        snapshot = snapshot_to_json(root.agent)
    assert "session" not in snapshot["attributes"]
    assert "session" not in caplog.text


async def test_delegate_and_wait_returns_the_childs_done(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "child = await self.session.delegate('Review auth', 'CHILD-PROMPT')\n"
            "done = await child.wait()\n"
            "return_result(Done(explanation=f'{child.name}: {done.explanation}'))"
        )
    ]
    models.scripts["Review auth"] = [done("login is fine")]
    root = await registry.create(root_options)
    outcome = await asyncio.wait_for(root.prompt("review"), TIMEOUT)
    assert outcome == Done(explanation="Review auth: login is fine")

    [child_options] = [o for o in models.built if o.name == "Review auth"]
    assert (child_options.turn_method, child_options.retain) == ("handle_batch", False)
    assert "CHILD-PROMPT" in str(models.llms["Review auth"].calls[0].messages)
    # The throwaway child is closed after its result, which went to the
    # waiter only: the parent ran no extra turn for it.
    [child_info] = registry.children(root.id)
    await _until(lambda: registry.get(child_info.id) is None)
    await asyncio.sleep(0.05)
    assert len(_turns(registry, root.id)) == 1
    assert root.agent.queue_manager.get_channel("delegates").qsize() == 0


async def test_a_pydantic_result_arrives_as_its_own_class(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "child = await self.session.delegate('Fix', 'fix it')\n"
            "done = await child.wait()\n"
            "self.v.result = done.result\n"
            "return_result(Done(explanation=type(done.result).__name__))"
        )
    ]
    models.scripts["Fix"] = [
        cell(
            "return_result(Done(explanation='fixed', result=TaskResult("
            "solution_description='patched', evidence='tests pass', how_to_verify='run tests')))"
        )
    ]
    root = await registry.create(root_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="TaskResult")
    assert root.agent.v.result == TaskResult(
        solution_description="patched", evidence="tests pass", how_to_verify="run tests"
    )


async def test_a_throwaway_child_cannot_ask_and_ends_done(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "child = await self.session.delegate('Push', 'push the branch')\n"
            "done = await child.wait()\n"
            "return_result(Done(explanation=done.explanation))"
        )
    ]
    models.scripts["Push"] = [
        cell("return_result(NeedInput(question='Which branch?'))"),
        done("blocked: branch not given"),
    ]
    root = await registry.create(root_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="blocked: branch not given")
    assert "return_result validation error" in str(models.llms["Push"].calls[1].messages)


async def test_a_background_result_wakes_the_parent(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "await self.session.delegate('Bg', 'work in the background')\n"
            "return_result(Waiting(explanation='child working', on=['delegates']))"
        ),
        cell(
            "[r] = notification['delegates']\n"
            "return_result(Done(explanation=f'{type(r).__name__}: {r.done.explanation}'))"
        ),
    ]
    models.scripts["Bg"] = [done("background done")]
    root = await registry.create(root_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="ChildResult: background done")


async def test_a_retained_childs_question_is_answered_with_send(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "await self.session.delegate('Helper', 'ask me', retain=True)\n"
            "return_result(Waiting(explanation='child working', on=['delegates']))"
        ),
        cell(
            "[q] = notification['delegates']\n"
            "self.v.schema_title = q.answer_schema['title']\n"
            "await q.answer({'branch': 'main'})\n"
            "return_result(Waiting(explanation='answered', on=['delegates']))"
        ),
        cell(
            "[r] = notification['delegates']\n"
            "await r.child.send('SECOND-MESSAGE')\n"
            "return_result(Waiting(explanation='sent again', on=['delegates']))"
        ),
        cell(
            "[r] = notification['delegates']\nreturn_result(Done(explanation=r.done.explanation))"
        ),
    ]
    models.scripts["Helper"] = [
        cell("return_result(NeedInput(question='Which branch?', answer_type=Answer))"),
        cell(
            "[a] = notification['user_messages']\n"
            "return_result(Done(explanation=f\"branch {a['branch']}\"))"
        ),
        cell(
            "[m] = notification['user_messages']\nreturn_result(Done(explanation=f'second: {m}'))"
        ),
    ]
    root = await registry.create(root_options)
    outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert outcome == Done(explanation="second: SECOND-MESSAGE")
    assert root.agent.v.schema_title == "Answer"
    [child_options] = [o for o in models.built if o.name == "Helper"]
    assert (child_options.turn_method, child_options.retain) == ("handle", True)
    [helper] = registry.children(root.id)
    assert registry.get(helper.id) is not None  # retained: still live
    assert helper.status == "retained"


async def test_a_child_ref_kept_in_vars_works_after_a_reload(
    registry, root_options, models, sessions_dir
):
    models.scripts[None] = [
        cell(
            "self.v.helper = await self.session.delegate('Keeper', 'get ready', retain=True)\n"
            "return_result(Waiting(explanation='child working', on=['delegates']))"
        ),
        done("child is ready"),
    ]
    models.scripts["Keeper"] = [done("ready")]
    root = await registry.create(root_options)
    assert await asyncio.wait_for(root.prompt("start"), TIMEOUT) == Done(
        explanation="child is ready"
    )
    await root.wait_for_checkpoint()
    await registry.close_all()

    later = ScriptedModels(
        {
            None: [
                cell(
                    "await self.v.helper.send('PING')\n"
                    "return_result(Waiting(explanation='asked the helper', on=['delegates']))"
                ),
                cell(
                    "[r] = notification['delegates']\n"
                    "return_result(Done(explanation=r.done.explanation))"
                ),
            ],
            "Keeper": [cell("return_result(Done(explanation=f'got {notification}'))")],
        }
    )
    fresh = SessionRegistry(SessionStore(sessions_dir), agent_factory=later)
    try:
        loaded = await fresh.load(root.id)
        assert isinstance(loaded.agent.v.helper, ChildRef)
        outcome = await asyncio.wait_for(loaded.prompt("ping the helper"), TIMEOUT)
        assert isinstance(outcome, Done) and "PING" in outcome.explanation
    finally:
        await fresh.close_all()


async def test_closing_a_child_fails_the_parents_wait(registry, root_options, models):
    started, _block = fresh_events()
    models.scripts[None] = [
        cell(
            "child = await self.session.delegate('Slow', 'take forever')\n"
            "self.v.child_id = child.id\n"
            "try:\n"
            "    await child.wait()\n"
            "except ChildFailedError as exc:\n"
            "    return_result(Done(explanation=f'failed: {exc}'))"
        )
    ]
    models.scripts["Slow"] = [cell(BLOCKING_CELL)]
    root = await registry.create(root_options)
    pending = asyncio.ensure_future(root.prompt("go"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await registry.close(root.agent.v.child_id)
    outcome = await asyncio.wait_for(pending, TIMEOUT)
    assert isinstance(outcome, Done) and outcome.explanation.startswith("failed:")


async def test_children_rename_and_usage(registry, root_options, models):
    usage = LLMUsage(input_tokens=7, output_tokens=3, cost_usd=0.25)
    models.scripts[None] = [
        cell(
            "a = await self.session.delegate('A', 'a', retain=True)\n"
            "await a.wait()\n"
            "await self.session.rename('Renamed by the agent')\n"
            "names = sorted(c.name for c in self.session.children())\n"
            "return_result(Done(explanation=','.join(names)))",
            usage=LLMUsage(input_tokens=1, output_tokens=1),
        )
    ]
    models.scripts["A"] = [done("a done", usage=usage)]
    root = await registry.create(root_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="A")
    assert root.info.title == "Renamed by the agent"
    totals = root.agent.session.usage()
    assert (totals.input_tokens, totals.attributed_input_tokens) == (1, 7)
    assert (totals.attributed_output_tokens, totals.attributed_cost_usd) == (3, 0.25)


async def test_usage_rolls_up_to_every_ancestor(registry, root_options, models):
    usage = LLMUsage(input_tokens=5, output_tokens=2)
    models.scripts[None] = [
        cell(
            "c = await self.session.delegate('Mid', 'm')\n"
            "await c.wait()\n"
            "return_result(Done(explanation='ok'))"
        )
    ]
    models.scripts["Mid"] = [
        cell(
            "g = await self.session.delegate('Leaf', 'l')\n"
            "await g.wait()\n"
            "return_result(Done(explanation='mid ok'))",
            usage=usage,
        )
    ]
    models.scripts["Leaf"] = [done("leaf ok", usage=usage)]
    root = await registry.create(root_options)
    changes = []
    root.subscribe(lambda e: changes.append(e.usage) if e.kind == "usage_changed" else None)
    await asyncio.wait_for(root.prompt("go"), TIMEOUT)
    assert root.info.usage.attributed_input_tokens == 10
    assert root.info.usage.input_tokens == 0
    # One update per child turn that carried usage: the leaf's, then Mid's.
    assert [u.attributed_input_tokens for u in changes] == [5, 10]


async def test_child_ref_methods_need_a_turn():
    ref = ChildRef(id="x", name="x", depth=1, status="running")
    assert current_port.get() is None
    with pytest.raises(RuntimeError, match="turn"):
        await ref.send("hello")


async def test_waiting_turn_does_not_resolve_the_prompt(registry, root_options, models):
    models.scripts[None] = [wait_on("jobs")]
    root = await registry.create(root_options)
    pending = asyncio.ensure_future(root.prompt("go"))
    await _until(lambda: len(_turns(registry, root.id)) == 1)
    await asyncio.sleep(0.05)
    assert not pending.done()
    await root.cancel()
    await pending


async def test_wait_takes_a_result_that_is_already_queued(registry, root_options, models):
    models.scripts[None] = [
        cell(
            "child = await self.session.delegate('Quick', 'q')\n"
            "delegates = self.queue_manager.get_channel('delegates')\n"
            "while delegates.qsize() == 0:\n"
            "    await asyncio.sleep(0.01)\n"
            "done = await child.wait()\n"
            "return_result(Done(explanation=done.explanation))"
        )
    ]
    models.scripts["Quick"] = [done("quick result")]
    root = await registry.create(root_options)
    assert await asyncio.wait_for(root.prompt("go"), TIMEOUT) == Done(explanation="quick result")
    await asyncio.sleep(0.05)
    assert len(_turns(registry, root.id)) == 1
    assert root.agent.queue_manager.get_channel("delegates").qsize() == 0
    assert len(registry.store.load_rows(root.id, frozenset({"ItemWithdrawn"}))) == 1


async def test_a_result_for_a_cancelled_wait_arrives_on_delegates(registry, root_options, models):
    started, block = fresh_events()
    models.scripts[None] = [
        cell(
            "c = await self.session.delegate('Kid', 'k', retain=True)\n"
            "await c.wait()\n"
            "return_result(Done(explanation='not reached'))"
        ),
        cell(
            "[r] = notification['delegates']\n"
            "return_result(Done(explanation=f'{type(r).__name__}: {r.done.explanation}'))"
        ),
    ]
    models.scripts["Kid"] = [cell(BLOCKING_CELL + "return_result(Done(explanation='kid done'))")]
    root = await registry.create(root_options)
    first = asyncio.ensure_future(root.prompt("go"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    ended = []
    root.subscribe(lambda e: ended.append(e) if e.kind == "turn_ended" else None)
    assert await asyncio.wait_for(root.cancel(), TIMEOUT) is True
    await asyncio.wait_for(first, TIMEOUT)
    assert registry._waiters == {}
    block.set()
    await _until(lambda: len(ended) == 2)
    assert ended[1].outcome == {"explanation": "ChildResult: kid done", "result": None}


async def test_a_child_cancelled_during_wait_fails_the_wait(registry, root_options, models):
    started, _block = fresh_events()
    models.scripts[None] = [
        cell(
            "c = await self.session.delegate('Kid', 'k')\n"
            "self.v.cid = c.id\n"
            "try:\n"
            "    await c.wait()\n"
            "except ChildFailedError as exc:\n"
            "    return_result(Done(explanation=f'failed: {exc}'))"
        )
    ]
    models.scripts["Kid"] = [cell(BLOCKING_CELL)]
    root = await registry.create(root_options)
    pending = asyncio.ensure_future(root.prompt("go"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    assert await registry.get(root.agent.v.cid).cancel(by="user") is True
    assert await asyncio.wait_for(pending, TIMEOUT) == Done(explanation="failed: cancelled by user")


async def test_a_background_child_closed_mid_turn_wakes_the_parent(registry, root_options, models):
    started, _block = fresh_events()
    models.scripts[None] = [
        cell(
            "c = await self.session.delegate('Bg', 'b')\n"
            "self.v.cid = c.id\n"
            "return_result(Waiting(explanation='child working', on=['delegates']))"
        ),
        cell(
            "[f] = notification['delegates']\n"
            "return_result(Done(explanation=f'{type(f).__name__}: {f.error}'))"
        ),
    ]
    models.scripts["Bg"] = [cell(BLOCKING_CELL)]
    root = await registry.create(root_options)
    pending = asyncio.ensure_future(root.prompt("go"))
    await asyncio.wait_for(started.wait(), TIMEOUT)
    await registry.close(root.agent.v.cid)
    assert await asyncio.wait_for(pending, TIMEOUT) == Done(
        explanation="ChildFailed: cancelled by host"
    )


async def test_the_port_can_be_hidden_from_the_model(registry, root_options, make_session):
    hidden = await registry.create(
        root_options.model_copy(update={"agent_spec": "coder_test_agents:HiddenPortAgent"})
    )
    assert isinstance(hidden.agent.session, SessionPort)
    assert "Create a child session" not in doc(hidden.agent)

    session, _ = make_session(start=False)
    install_port(session.agent, session, registry, visible=False)
    assert "Create a child session" not in doc(session.agent)
    assert isinstance(session.agent.session, SessionPort)
