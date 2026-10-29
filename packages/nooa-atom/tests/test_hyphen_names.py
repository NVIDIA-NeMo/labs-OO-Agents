# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyphenated job and child names (``grep-tests``, ``review-pr``) work end to end.

A name starts with a letter or ``_``, then letters, digits, ``_`` or ``-``; it is
never rewritten. ``/``, ``#``, ``.``, spaces, a leading digit or hyphen,
keywords and reserved channel names stay refused.
"""

import asyncio

import pytest
from atom_test_agents import AtomModels, cell
from nooa_atom.agent import work_manager, work_subagent
from nooa_atom.agent.work_manager import JobEvent
from nooa_atom.session.names import RESERVED_NAMES, check_name, is_valid_name
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from nooa.interactive import Done
from nooa.runtime.channels import QueueManager

UNIFIED = "nooa_atom.agent:AtomAgent"
TIMEOUT = 60

# (name, valid): the one table every validator must agree on.
NAMES = [
    ("tests", True),
    ("grep-tests", True),
    ("read-erc", True),
    ("review-pr", True),
    ("sh-1", True),
    ("build-x", True),
    ("_x", True),
    ("_-a", True),
    ("a-", True),
    ("ci_check", True),
    ("CI", True),
    ("x9-2", True),
    ("", False),
    (" ", False),
    ("ci check", False),
    ("a/b", False),
    ("/root/a", False),
    ("a#1", False),
    ("sh-1#2", False),
    ("a.b", False),
    ("1st", False),
    ("1-a", False),
    ("-x", False),
    ("-", False),
    ("class", False),
    ("None", False),
    ("a+b", False),
    ("a\nb", False),
    (None, False),
    (3, False),
]


@pytest.mark.parametrize(("name", "valid"), NAMES)
def test_all_validators_agree_on_the_name_rule(name, valid):
    assert is_valid_name(name) is valid
    assert work_manager._valid_name(name) is valid

    class Port:  # _child_name keeps a valid job name as the child's base name
        def check_child_name(self, candidate):
            check_name(candidate, "child")

    if isinstance(name, str) and "#" not in name and name not in RESERVED_NAMES:
        expected = name if valid else "subagent"
        assert work_subagent._child_name(Port(), name) == expected
    if valid:
        assert check_name(name, "job") == name  # never rewritten


@pytest.mark.parametrize(("name", "valid"), [n for n in NAMES if not n[1]])
def test_invalid_names_get_a_valid_hint(name, valid):
    with pytest.raises(ValueError, match="is not valid: use a name such as 'tests', 'review-pr'"):
        check_name(name, "child")
    hint = work_manager._name_hint(name)
    assert work_manager._valid_name(hint), hint


async def test_launch_keeps_a_hyphenated_name_and_hints_for_others(tmp_path):
    manager = work_manager.WorkManager(QueueManager().queue("jobs"), scratch=tmp_path)
    try:
        job = manager.launch(asyncio.sleep(0, result="ok"), name="grep-tests")
        assert job.name == "grep-tests" and await job == "ok"
        assert manager.get("grep-tests") is job
        assert manager.records() == []  # finished: nothing to report lost
        for bad, hint in [
            ("grep tests", "'grep_tests'"),
            ("a/b", "'a_b'"),
            ("x#1", "'x_1'"),
            ("a.b", "'a_b'"),
            ("-x", "'x'"),
            ("2-x", "'job_2-x'"),
            ("class", "'class_'"),
        ]:
            with pytest.raises(ValueError, match=f"is not valid.*use e.g. {hint}"):
                manager.launch(asyncio.sleep(0), name=bad)
    finally:
        await manager.aclose()


@pytest.fixture
def models():
    return AtomModels()


@pytest.fixture
def options(tmp_path, sessions_dir):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionOptions(workspace=workspace, agent_spec=UNIFIED, sessions_dir=sessions_dir)


WORK_RESULT = (
    "return_result(Done(explanation='reviewed', result=TaskResult(solution_description='s', "
    "evidence='e', how_to_verify='v', report='R1')))"
)


async def test_hyphenated_shell_jobs_resources_children_and_waiting(sessions_dir, options, models):
    models.scripts[None] = [
        cell(
            "r = await self.jobs.launch(Shell().run('echo hi'), name='grep-tests')\n"
            "assert r.stdout == 'hi\\n', r\n"
            "assert self.jobs.get('grep-tests').status == 'done'\n"
            "sh = self.jobs.launch(Shell(), name='sh-1')\n"
            "a = await sh.send('X=1')\n"
            "b = sh.send('echo $X')\n"
            "assert b.name == 'sh-1#2', b.name\n"
            "assert (await b).stdout == '1\\n'\n"
            "assert self.jobs.get('sh-1') is sh and self.jobs.get('sh-1#2') is b\n"
            "status = self._status()\n"
            "assert 'sh-1' in status, status\n"
            "rev = self.jobs.launch(Subagent('review'), name='review-pr')\n"
            "res = await rev.send('REVIEW')\n"
            "assert res.report == 'R1', res\n"
            "self.v.names = [j.name for j in self.jobs.list()]\n"
            "self.jobs.launch(Shell().run('sleep 0.3; echo built'), name='build-x')\n"
            "return_result(Waiting(explanation='w', on=['build-x']))"
        ),
        cell(
            "[event] = [e for e in notification['jobs'] if e.name == 'build-x']\n"
            "assert event.kind == 'done' and event.value.stdout == 'built\\n', event\n"
            "self.v.long = self.jobs.launch(asyncio.sleep(60), name='read-erc')\n"
            "return_result(Done(explanation='done'))"
        ),
    ]
    models.scripts["review-pr"] = [cell("assert 'REVIEW' in str(notification)\n" + WORK_RESULT)]
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    try:
        root = await registry.create(options)
        outcome = await asyncio.wait_for(root.prompt("go"), TIMEOUT)
        assert outcome == Done(explanation="done")
        agent = root._agent
        assert agent.v.names == [
            "grep-tests",
            "sh-1",
            "sh-1#1",
            "sh-1#2",
            "review-pr",
            "review-pr#1",
        ]
        [child] = registry.children(root.id)
        assert (child.name, registry.path(child.id)) == ("review-pr", "/root/review-pr")
        assert "WaitingTargetNotRunning" not in str(models.llms[None].calls[1].messages)
        root._checkpoint()
        await root.wait_for_checkpoint()
        assert [r["name"] for r in agent._work_records] == ["sh-1", "review-pr", "read-erc"]
        session_id = root.id
    finally:
        await registry.close_all()

    # Restart: the persisted records reload, and each running job is reported lost by name.
    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=AtomModels())
    try:
        loaded = await registry.load(session_id)
        agent = loaded._agent
        events = [e for e in agent._jobs_in.snapshot() if isinstance(e, JobEvent)]
        assert {e.name for e in events} == {"sh-1", "review-pr", "read-erc"}
        assert all(e.kind == "failed" and e.value.startswith("lost:") for e in events)
        [child] = registry.children(session_id)
        assert child.name == "review-pr"
        assert loaded._agent.session.descendant("/root/review-pr").name == "review-pr"
    finally:
        await registry.close_all()
