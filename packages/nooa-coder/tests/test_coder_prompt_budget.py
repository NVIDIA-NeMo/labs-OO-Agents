# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prompt-size guards for the coding agents' first model call.

Installed ``nooa.skills`` entry points are loaded into the agent, and the
user's skill directories under the home directory are discovered, so both
change the prompt with whatever this machine has. Discovery is patched to
nothing and ``HOME`` points at an empty directory, which leaves the agent's
own tools only.

The limits are the sizes measured when this guard was added (2026-09-24),
rounded up to the next 100 characters, with the workspace path (which
varies by machine) counted as a fixed placeholder. A change that grows the
prompt must raise a limit here on purpose.
"""

import asyncio

import pytest
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.store import SessionStore
from test_experimental_agent import python_cell

from nooa.unifiedllm import FakeLLMClient

CODER = "nooa_coder.coding.agent:CodingAgent"
EXPERIMENTAL = "nooa_coder.coding.experimental_agent:ExperimentalCodingAgent"
RESULT = (
    "return_result(Done(explanation='x', result=TaskResult("
    "solution_description='a', evidence='b', how_to_verify='c')))"
)

# (spec, turn method): (system prompt chars, all message chars)
# Measured after PR 0, then raised on tree/4-router: the state block names the
# repository root, the class says ``cd`` moves shell and repo tools, the turn
# methods say locals last one method call, and the repo tools take ``cwd``.
# Lowered when CodingAgent moved to CodeActV2 (2026-09-25; was 19,827 / 23,935
# for handle): measured values plus about 2% headroom.
LIMITS = {
    (CODER, "handle"): (11_500, 15_200),  # measured 11,279 / 14,879
    (CODER, "handle_batch"): (11_500, 14_800),  # measured 11,279 / 14,523
    (EXPERIMENTAL, "handle"): (11_200, 13_700),  # measured 11,162 / 13,627
    (EXPERIMENTAL, "handle_batch"): (11_200, 14_300),  # measured 11,162 / 14,006 (P2a port docs)
}
# The bench guard's system-prompt ceiling; every agent stays under it.
BENCH_SYSTEM_LIMIT = 20_000


@pytest.fixture(autouse=True)
def _no_installed_skills(monkeypatch, tmp_path):
    monkeypatch.setattr("nooa.skill_registry.entry_points", lambda *, group: [])
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    # User skill directories (~/.claude/skills and the like) join the prompt.
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))


@pytest.mark.parametrize(("spec", "method"), list(LIMITS))
async def test_first_call_prompt_stays_within_budget(spec, method, tmp_path, sessions_dir):
    code = RESULT if method == "handle_batch" else "return_result(Done(explanation='x'))"
    response = python_cell(code, "call_1")
    llm = FakeLLMClient([response], strict_exhaustion=True)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    registry = SessionRegistry(SessionStore(sessions_dir))
    try:
        root = await registry.create(
            SessionOptions(
                workspace=workspace,
                agent_spec=spec,
                llm=llm,
                turn_method=method,
                sessions_dir=sessions_dir,
            )
        )
        await asyncio.wait_for(root.prompt("hello"), 30)
    finally:
        await registry.close_all()
    messages = llm.calls[0].messages
    system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
    rendered = "\n".join(str(m.get("content", "")) for m in messages)
    # The workspace path appears in the prompt; count it as a fixed token.
    rendered = rendered.replace(str(workspace.resolve()), "<workspace>")
    system_limit, total_limit = LIMITS[(spec, method)]
    print(f"{spec} {method}: system={len(system)} total={len(rendered)}")
    assert len(system) <= system_limit
    assert len(rendered) <= total_limit
    assert len(system) < BENCH_SYSTEM_LIMIT
