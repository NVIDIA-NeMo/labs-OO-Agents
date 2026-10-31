# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A kept child that needs its parent asks with NeedInput, not by waiting on user_messages.

Real parent and kept child sessions, fake models. The parent awaits the
child's request, which blocks until the child ends its turn; nothing
arrives on the child's user_messages meanwhile. Waiting on it is rejected
with what to do instead, and a mid-cell read's timeout says the same.
"""

import asyncio

import pytest
from atom_test_agents import AtomModels, ask, cell
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.runtime import harness_metrics

UNIFIED = "nooa_atom.agent:AtomAgent"
TIMEOUT = 60
HINT = (
    "If you need input, return NeedInput(question=...): your parent answers with send "
    "(a one-shot task cannot ask: decide and say so in the report). If you are done, "
    "return Done(explanation=..., result=TaskResult(...))."
)
WORK_RESULT = (
    "return Done(explanation='e', result=TaskResult(solution_description='s', "
    "evidence='e', how_to_verify='v', report={r!r}))"
)


@pytest.fixture
def models():
    return AtomModels()


@pytest.fixture
async def registry(sessions_dir, models):
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    yield registry
    await registry.close_all()


@pytest.fixture
def options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionOptions(workspace=workspace, agent_spec=UNIFIED, sessions_dir=sessions_dir)


@pytest.fixture
def span_attributes(monkeypatch):
    flushed: list[dict] = []
    flush = harness_metrics.HarnessMetrics.flush_to_span

    def spy(self):
        flushed.append(self.to_span_attributes())
        return flush(self)

    monkeypatch.setattr(harness_metrics.HarnessMetrics, "flush_to_span", spy)
    return flushed


def _rendered(models, name, call):
    messages = models.llms[name].calls[call].messages
    return "\n".join(str(message.get("content", "")) for message in messages)


async def test_a_kept_child_waiting_on_its_parent_is_told_to_ask(
    registry, options, models, span_attributes
):
    models.scripts[None] = [
        cell(
            "rev = self.jobs.launch(Subagent('review'), name='review')\n"
            "q = await rev.send('REVIEW-TASK')\n"
            "assert isinstance(q, Question) and q.text == 'Which file?', q\n"
            "r = await rev.send('ANSWER-42')\n"
            "return Done(explanation=r.report)"
        )
    ]
    models.scripts["review"] = [
        # The F03 deadlock shape: wait for the parent through user_messages.
        cell("return Waiting(explanation='need the file', on=['user_messages'])"),
        cell(
            "try:\n"
            "    await self.user_messages.get(timeout=0.05)\n"
            "except TimeoutError as e:\n"
            "    print('TIMEOUT:', e)"
        ),
        ask("Which file?"),
        cell("assert 'ANSWER-42' in str(notification)\n" + WORK_RESULT.format(r="R-42")),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("review"), TIMEOUT) == Done(explanation="R-42")

    rejected = _rendered(models, "review", 1)
    assert "Waiting on 'user_messages', but your parent is not a job." in rejected
    assert HINT in rejected
    timeout = _rendered(models, "review", 2)
    assert "An answer from your parent comes only after this turn ends (as a new turn)" in timeout
    assert "a note it sends with message() can still arrive now" in timeout
    assert HINT in timeout

    merged: dict = {}
    for attrs in span_attributes:
        for key in (
            "harness.waiting_on_person_rejected.roles",
            "harness.queue_read_timeout.channels",
        ):
            merged.setdefault(key, []).extend(attrs.get(key, []))
    assert merged["harness.waiting_on_person_rejected.roles"] == ["child"]
    assert merged["harness.queue_read_timeout.channels"] == ["user_messages"]


async def test_the_root_keeps_its_own_hint(registry, options, models):
    models.scripts[None] = [
        cell("return Waiting(explanation='w', on=['user_messages'])"),
        cell("return Done(explanation='ok')"),
    ]
    root = await registry.create(options)
    assert await asyncio.wait_for(root.prompt("hi"), TIMEOUT) == Done(explanation="ok")
    rejected = _rendered(models, None, 1)
    assert "Waiting on 'user_messages', but the person is not a job." in rejected
    assert "your parent is not a job" not in rejected and HINT not in rejected
