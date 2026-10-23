# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import functools
import inspect
import io
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig, validate_invocation
from responses_api_agents.nooa_agent.runner import InProcessNOOARunner, NOOARunRequest
from responses_api_agents.nooa_agent.sandbox_runtime import _runtime_archive
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response


def test_archive_does_not_distribute_external_artifacts(tmp_path):
    # Supply a pristine checkout of the documented Gym commit via GYM_ROOT.
    gym_root = Path(os.environ["GYM_ROOT"]).resolve()
    external_wheel = tmp_path / "gym_atom_bridge-0.0.1-py3-none-any.whl"
    external_wheel.write_bytes(b"external artifact fixture")
    requirements_path = tmp_path / "requirements.txt"
    requirements_path.write_text(f"jsonschema>=4.25.1,<5\n{external_wheel}\n")
    archive, fingerprint = _runtime_archive(gym_root, requirements_path=requirements_path)
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        names = tar.getnames()
        requirements = tar.extractfile("runtime-requirements.txt").read().decode()
    assert not any("gym_atom_bridge" in name or name.endswith(".whl") for name in names)
    assert str(external_wheel) in requirements
    assert not any(name.startswith("nooa_atom/") for name in names)
    assert "responses_api_agents/nooa_agent/runner.py" in names
    print("archive", fingerprint, "members", len(names), "external requirements preserved but artifacts absent")


@pytest.mark.asyncio
async def test_regular_runner_native_root_and_child(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    code = [
        "child = await self.delegate('offline child', 'return child evidence')\nreturn_result(Done(explanation='root', result=TaskResult(solution_description='root', evidence=child.result.evidence, how_to_verify='offline', report='native root and child')))",
        "return_result(Done(explanation='child', result=TaskResult(solution_description='child', evidence='native child evidence', how_to_verify='offline', report='child')))",
    ]
    calls = []
    async def post(**kwargs):
        calls.append(kwargs)
        assert len(calls) <= 2
        out = NeMoGymResponseFunctionToolCall(id=f"f{len(calls)}", call_id=f"c{len(calls)}", name="python_cell", arguments=json.dumps({"code": code[len(calls)-1]}))
        return FakeHTTPResponse(model_response(out, response_id=f"r{len(calls)}"))
    runner = InProcessNOOARunner(
        invocation=NOOAInvocationConfig(agent_class="gym_atom_bridge:GymAtomAgent", invocation_adapter="gym_atom_bridge:invoke_atom"),
        server_client=SimpleNamespace(post=post), model_server_name="policy", max_policy_calls=2, context_window=1048576,
    )
    result = await runner.run(NOOARunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="offline native fixture", max_output_tokens=131072), model_url_path="/fixture/v1/responses", model_cookies={"session":"fixture"}))
    assert result.return_value.evidence == "native child evidence"
    assert result.termination_reason is None
    assert len(calls) == len(result.trajectory.turns) == 2
    assert len(result.trajectory.invocations) >= 2
    assert all(call["server_name"] == "policy" and call["url_path"] == "/fixture/v1/responses" for call in calls)
    assert all(call["json"].max_output_tokens == 131072 for call in calls)
    assert all(call["cookies"]["session"] == "fixture" for call in calls)
    print("native result", result.return_value.model_dump(), "turns", len(result.trajectory.turns), "invocations", len(result.trajectory.invocations))
    (tmp_path / "offline-trajectory.json").write_text(result.trajectory.model_dump_json(indent=2))


ROOT_RESULT = "return_result(Done(explanation='root', result=TaskResult(solution_description='root', evidence='root evidence', how_to_verify='offline', report='root')))"


def _strings(value):
    """Every string in a request dump, so prompt text is matched without JSON escapes."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item)


def _role_markers(turn_method):
    """Prompt text that must and must not appear when the root ran ``turn_method``.

    The single-handle Atom runs every root turn with ``handle()``; the
    ``turn_method`` selects a role whose instructions (``nooa_atom.agent.roles``)
    form the ``role`` block. Earlier Atoms run one method per role, and that
    method's name and docstring are the task text the policy sees.
    """
    from gym_atom_bridge import TURN_METHODS

    try:
        from nooa_atom.agent.roles import POLICY_BY_TURN_METHOD, ROLE_INSTRUCTIONS
    except ImportError:
        docs = {
            "handle_batch": "Work on one task unattended",
            "handle_orchestrate": "You work on one task on your own.",
        }
        present = [docs[turn_method], f"Task: {turn_method}"]
        absent = [docs[m] for m in TURN_METHODS if m != turn_method]
        return present, absent
    policy = POLICY_BY_TURN_METHOD[turn_method]
    others = [POLICY_BY_TURN_METHOD[m] for m in TURN_METHODS if m != turn_method]
    present = [ROLE_INSTRUCTIONS[policy]]
    absent = [ROLE_INSTRUCTIONS[p] for p in others] + [f"Task: {m}" for m in TURN_METHODS]
    return present, absent


def _assert_root_role(text, turn_method):
    present, absent = _role_markers(turn_method)
    for marker in present:
        assert marker in text
    for marker in absent:
        assert marker not in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("init_kwargs", "turn_method"),
    [
        ({}, "handle_batch"),
        ({"turn_method": "handle_batch"}, "handle_batch"),
        ({"turn_method": "handle_orchestrate"}, "handle_orchestrate"),
    ],
)
async def test_init_kwargs_choose_the_root_turn_method(
    tmp_path, monkeypatch, init_kwargs, turn_method
):
    monkeypatch.chdir(tmp_path)
    _assert_root_role(await _run_root(init_kwargs), turn_method)


def test_unknown_turn_method_is_rejected():
    from gym_atom_bridge import GymAtomAgent

    with pytest.raises(ValueError, match="turn_method must be one of"):
        GymAtomAgent(turn_method="handle")


# A file spec (``path.py:Class``) loads like any module spec, so this alternate
# agent class works with every Atom that has ``load_agent_class``.
MARKED_AGENT = """
import os
from pathlib import Path

from nooa_atom.agent import AtomAgent


class MarkedAtomAgent(AtomAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        Path(os.environ["MARKED_AGENT_LOG"]).open("a").write(type(self).__name__ + "\\n")
"""


async def _run_root(init_kwargs):
    calls = []

    async def post(**kwargs):
        calls.append(kwargs)
        out = NeMoGymResponseFunctionToolCall(
            id="f1", call_id="c1", name="python_cell", arguments=json.dumps({"code": ROOT_RESULT})
        )
        return FakeHTTPResponse(model_response(out, response_id="r1"))

    invocation = NOOAInvocationConfig(
        agent_class="gym_atom_bridge:GymAtomAgent",
        invocation_adapter="gym_atom_bridge:invoke_atom",
        init_kwargs=init_kwargs,
    )
    validate_invocation(invocation)
    runner = InProcessNOOARunner(
        invocation=invocation,
        server_client=SimpleNamespace(post=post),
        model_server_name="policy",
        max_policy_calls=1,
        context_window=1048576,
    )
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(
                input="offline agent spec fixture"
            ),
            model_url_path="/fixture/v1/responses",
            model_cookies={},
        )
    )
    assert result.return_value.evidence == "root evidence"
    assert len(calls) == 1
    return "\n".join(_strings(calls[0]["json"].model_dump()))


def test_default_agent_spec_is_atom_agent():
    from gym_atom_bridge import ATOM_AGENT, GymAtomAgent

    assert ATOM_AGENT == "nooa_atom.agent:AtomAgent"
    assert inspect.signature(GymAtomAgent.__init__).parameters["agent_spec"].default == ATOM_AGENT


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_method", ["handle_batch", "handle_orchestrate"])
async def test_init_kwargs_agent_spec_selects_the_session_agent_class(
    tmp_path, monkeypatch, turn_method
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    monkeypatch.chdir(workspace)
    agent_file = tmp_path / "marked_agent.py"
    agent_file.write_text(MARKED_AGENT)
    log = tmp_path / "constructed.txt"
    monkeypatch.setenv("MARKED_AGENT_LOG", str(log))
    text = await _run_root(
        {"turn_method": turn_method, "agent_spec": f"{agent_file}:MarkedAtomAgent"}
    )
    # Only the session's own agent is the selected class (Gym's construction
    # shell is GymAtomAgent), so exactly one root was built from the spec.
    assert log.read_text().splitlines() == ["MarkedAtomAgent"]
    # A subclass of AtomAgent gets the same root role as AtomAgent itself.
    _assert_root_role(text, turn_method)


@pytest.mark.asyncio
async def test_canonical_atom_agent_is_the_single_handle_agent(tmp_path, monkeypatch):
    # Needs an Atom with the single-handle rewrite (one handle(), roles, self.jobs as a
    # WorkManager); set GYM_ATOM_BRIDGE_REQUIRE_SINGLE_HANDLE=1 to make its absence a
    # failure instead of a skip.
    from nooa_atom.agent import AtomAgent

    try:
        from nooa_atom.agent.roles import ROLE_INSTRUCTIONS
        from nooa_atom.agent.work_manager import WorkManager
    except ImportError:
        if os.environ.get("GYM_ATOM_BRIDGE_REQUIRE_SINGLE_HANDLE") == "1":
            raise
        pytest.skip("installed Atom predates the single-handle AtomAgent")
    assert AtomAgent.turn_entry == "handle"
    monkeypatch.chdir(tmp_path)
    built = []
    original = AtomAgent.__init__

    # wraps() keeps the signature the agent factory inspects to pick its arguments.
    @functools.wraps(original)
    def record(self, *args, **kwargs):
        original(self, *args, **kwargs)
        built.append(self)

    monkeypatch.setattr(AtomAgent, "__init__", record)
    # The default agent_spec, as the Gym recipes configure it.
    text = await _run_root({"turn_method": "handle_orchestrate"})
    assert [type(agent) for agent in built] == [AtomAgent]
    assert isinstance(built[0].jobs, WorkManager)
    assert ROLE_INSTRUCTIONS["orchestrator"] in text


@pytest.mark.parametrize(
    "agent_spec",
    [
        "nooa_atom.agent.AtomAgent",  # no ':Class' part
        "nooa_atom.no_such_module:AtomAgent",
        "nooa_atom.agent:NoSuchAgent",
        "nooa:Agent",  # importable, but not an InteractiveAgent
    ],
)
def test_invalid_agent_spec_is_rejected_at_construction(agent_spec):
    from gym_atom_bridge import GymAtomAgent
    from nooa_atom.session.loader import AgentSpecError

    with pytest.raises(AgentSpecError):
        GymAtomAgent(agent_spec=agent_spec)
