# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``handle_orchestrate`` may only wait on something that is running.

``require_running_wait_unattended`` checks each name in ``Waiting.on`` against the
registry's children and the queue manager's jobs; a name that cannot
deliver is rejected and the turn continues.
"""

import asyncio

import pytest
from atom_test_agents import ATOM_SPEC, AtomModels, cell
from nooa_atom import SessionOptions, SessionStore
from nooa_atom.agent.conditions import WaitingTargetNotRunning
from nooa_atom.session.registry import SessionRegistry

from nooa.strategy_validation import InvariantError

TIMEOUT = 30

RESULT = (
    "return_result(Done(explanation='verified', result=TaskResult("
    "solution_description='s', evidence='e', how_to_verify='h', report='VERIFIED')))"
)
REJECTED = "WaitingTargetNotRunning"
FIX = "Send a child work with `await child.send(...)`, spawn one, or finish with Done"


def _options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return SessionOptions(
        workspace=workspace,
        agent_spec=ATOM_SPEC,
        sessions_dir=sessions_dir,
        turn_method="handle_orchestrate",
    )


async def _run(tmp_path, sessions_dir, root_cells, workers=None):
    """Run the root to its Done; return the root's model calls as text, one per call."""
    models = AtomModels()
    models.scripts[None] = root_cells
    for name, cells in (workers or {}).items():
        models.scripts[name] = cells
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(_options(tmp_path, sessions_dir))
        outcome = await asyncio.wait_for(root.prompt("fix it"), TIMEOUT)
        assert outcome.result.report == "VERIFIED"
    finally:
        await registry.close_all()
    calls = models.llms[None].calls
    # What each call added: the previous cell's output and the context.
    return [
        "\n".join(
            str(message.get("content") or "")
            for message in call.messages[len(calls[index - 1].messages) if index else 0 :]
        )
        for index, call in enumerate(calls)
    ]


def _rejected(call_text):
    return REJECTED in call_text or "Running right now:" in call_text


def test_the_error_is_the_postcondition_error_type():
    assert issubclass(WaitingTargetNotRunning, InvariantError)


def test_every_turn_method_checks_what_it_waits_on():
    from nooa_atom.agent import AtomAgent

    def postconditions(method):
        config = getattr(AtomAgent, method)._plan_strategy.config
        return [fn.__name__ for fn in config.postconditions]

    assert postconditions("handle") == ["require_running_wait"]
    assert postconditions("handle_batch") == ["require_result", "require_running_wait_unattended"]
    assert postconditions("handle_orchestrate") == [
        "require_result",
        "require_running_wait_unattended",
    ]


async def test_waiting_on_delegates_without_a_child_is_rejected(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell("return_result(Waiting(explanation='w', on=['delegates']))"),
            cell(RESULT),
        ],
    )
    assert len(calls) == 2
    assert "Waiting on 'delegates', but no child is running" in calls[1]
    assert FIX in calls[1]
    assert "Running right now: nothing." in calls[1]


async def test_waiting_when_every_child_is_idle_with_nothing_pending_is_rejected(
    tmp_path, sessions_dir
):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'fix it')\n"
                "await self.v.child.wait()\n"
                "self.delegates.flush()\n"
                "assert self.children()[0].status == 'retained'\n"
                "return_result(Waiting(explanation='w', on=['delegates']))"
            ),
            cell("return_result(Waiting(explanation='w', on=['Worker']))"),
            cell(RESULT),
        ],
        {
            "Worker": [
                cell(
                    "return_result(Done(explanation='did it', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did it')))"
                )
            ]
        },
    )
    assert len(calls) == 3
    assert "Waiting on 'delegates', but no child is running" in calls[1]
    assert "Waiting on child 'Worker', but it is not running" in calls[2]


async def test_waiting_while_a_child_is_running_is_accepted(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'fix it')\n"
                "for _ in range(200):\n"
                "    if self.children()[0].status == 'running':\n"
                "        break\n"
                "    await asyncio.sleep(0.01)\n"
                "assert self.children()[0].status == 'running'\n"
                "return_result(Waiting(explanation='w', on=['delegates', 'Worker']))"
            ),
            cell("assert notification['delegates']\n" + RESULT),
        ],
        {
            "Worker": [
                cell(
                    "await asyncio.sleep(0.3)\nreturn_result(Done(explanation='did it', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did it')))"
                )
            ]
        },
    )
    assert len(calls) == 2
    assert not _rejected(calls[1])


async def test_waiting_on_a_buffered_result_is_accepted(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "child = await self.spawn('Worker', 'fix it')\n"
                "for _ in range(500):\n"
                "    if self.delegates.qsize():\n"
                "        break\n"
                "    await asyncio.sleep(0.01)\n"
                "assert self.delegates.qsize() == 1\n"
                "assert self.children()[0].status == 'retained'\n"
                "return_result(Waiting(explanation='w', on=['delegates']))"
            ),
            cell("[item] = notification['delegates']\n" + RESULT),
        ],
        {
            "Worker": [
                cell(
                    "return_result(Done(explanation='did it', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did it')))"
                )
            ]
        },
    )
    assert len(calls) == 2
    assert not _rejected(calls[1])


async def test_waiting_on_a_working_child_is_accepted(tmp_path, sessions_dir):
    """A kept child whose turn ended Waiting on its own job reads ``working``: it will deliver."""
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'fix it')\n"
                "for _ in range(500):\n"
                "    if self.children()[0].status == 'working':\n"
                "        break\n"
                "    await asyncio.sleep(0.01)\n"
                "assert self.children()[0].status == 'working', self.children()[0].status\n"
                "return_result(Waiting(explanation='w', on=['delegates', 'Worker']))"
            ),
            cell("[item] = notification['delegates']\n" + RESULT),
        ],
        {
            "Worker": [
                cell(
                    "self.queue_manager.queue('ci')\n"
                    "self.queue_manager.spawn(asyncio.sleep(0.5, result='green'), "
                    "channel='ci', label='ci-1')\n"
                    "return_result(Waiting(explanation='ci running', on=['ci']))"
                ),
                cell(
                    "assert notification['ci'] == ['green']\n"
                    "return_result(Done(explanation='did it', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did it')))"
                ),
            ]
        },
    )
    assert len(calls) == 2
    assert not _rejected(calls[1])


@pytest.mark.parametrize("target", ["delegates", "Worker"])
async def test_waiting_right_after_sending_to_an_idle_retained_child_is_accepted(
    tmp_path, sessions_dir, target
):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'fix it')\n"
                "await self.v.child.wait()\n"
                "self.delegates.flush()\n"
                "await self.v.child.send('FEEDBACK: one more case')\n"
                f"return_result(Waiting(explanation='w', on=[{target!r}]))"
            ),
            cell("[item] = notification['delegates']\n" + RESULT),
        ],
        {
            "Worker": [
                cell(
                    "return_result(Done(explanation='did it', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did it')))"
                ),
                cell(
                    "return_result(Done(explanation='did the case', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did the case')))"
                ),
            ]
        },
    )
    assert len(calls) == 2
    assert not _rejected(calls[1])


async def test_waiting_on_a_child_by_id_is_checked(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'fix it')\n"
                "await self.v.child.wait()\n"
                "self.delegates.flush()\n"
                "return_result(Waiting(explanation='w', on=[self.v.child.id]))"
            ),
            cell(
                "await self.v.child.send('more')\n"
                "return_result(Waiting(explanation='w', on=[self.v.child.id]))"
            ),
            cell("[item] = notification['delegates']\n" + RESULT),
        ],
        {
            "Worker": [
                cell(
                    "return_result(Done(explanation='did it', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did it')))"
                ),
                cell(
                    "return_result(Done(explanation='did more', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did more')))"
                ),
            ]
        },
    )
    assert len(calls) == 3
    assert "Waiting on child '" in calls[1] and "but it is not running" in calls[1]
    assert not _rejected(calls[2])


async def test_waiting_on_jobs_needs_a_running_job(tmp_path, sessions_dir):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell("return_result(Waiting(explanation='w', on=['jobs']))"),
            cell("return_result(Waiting(explanation='w', on=['jobs:ci-42']))"),
            cell(
                "self.queue_manager.queue('ci')\n"
                "self.queue_manager.spawn(asyncio.sleep(0.2, result='green'), "
                "channel='ci', label='ci-42')\n"
                "return_result(Waiting(explanation='w', on=['jobs', 'jobs:ci-42', 'ci']))"
            ),
            cell("assert notification['ci'] == ['green']\n" + RESULT),
        ],
    )
    assert len(calls) == 4
    assert "Waiting on 'jobs', but no such job is running." in calls[1]
    assert "Waiting on 'jobs:ci-42', but no such job is running." in calls[2]
    assert not _rejected(calls[3])


async def test_waiting_on_an_unknown_name_is_rejected_and_lists_what_is_running(
    tmp_path, sessions_dir
):
    calls = await _run(
        tmp_path,
        sessions_dir,
        [
            cell(
                "self.v.child = await self.spawn('Worker', 'fix it')\n"
                "return_result(Waiting(explanation='w', on=['user_messages', 'ci', 'delegates']))"
            ),
            cell("return_result(Waiting(explanation='w', on=['delegates']))"),
            cell("[item] = notification['delegates']\n" + RESULT),
        ],
        {
            "Worker": [
                cell(
                    "await asyncio.sleep(0.3)\nreturn_result(Done(explanation='did it', result=TaskResult("
                    "solution_description='s', evidence='e', how_to_verify='h', report='did it')))"
                )
            ]
        },
    )
    assert len(calls) == 3
    assert "Waiting on 'user_messages', but the person is not a job." in calls[1]
    assert "Waiting on 'ci', but it is not a running job, 'delegates', or a child's" in calls[1]
    assert "Waiting is only for running jobs or background agents." in calls[1]
    # handle_orchestrate cannot return NeedInput, so the hint does not offer it.
    assert "NeedInput" not in calls[1]
    assert FIX in calls[1]
    assert "Waiting on 'delegates'" not in calls[1]
    assert "Running right now: " in calls[1] and "'Worker'" in calls[1]
    assert "'delegates'" in calls[1]
    assert not _rejected(calls[2])
