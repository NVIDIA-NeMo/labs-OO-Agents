# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import os
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from responses_api_agents.nooa_agent.invocation import NOOAInvocationConfig
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
