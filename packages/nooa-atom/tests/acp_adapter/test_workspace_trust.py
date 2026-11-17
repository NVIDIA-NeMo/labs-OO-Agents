# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Atom over ACP in an untrusted workspace: the notice, the trust question, /trust."""

import asyncio

import pytest
from acp import text_block
from acp.schema import (
    AcceptElicitationResponse,
    AgentMessageChunk,
    ClientCapabilities,
    ElicitationCapabilities,
    ElicitationFormCapabilities,
)
from atom_test_agents import ATOM_SPEC, AtomModels, reply

from nooa.workspace_trust import is_trusted

FORMS = ClientCapabilities(elicitation=ElicitationCapabilities(form=ElicitationFormCapabilities()))


@pytest.fixture
def untrusted(workspace, tmp_path, monkeypatch, untrusted_folders):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    skills = workspace / ".claude" / "skills"
    skills.mkdir(parents=True)
    (skills / "probe.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(tmp_path / 'marker')!r}).write_text('x')\n"
        'from nooa.skill import Skill\n\n\nclass Probe(Skill):\n    """Probe."""\n'
    )
    return tmp_path / "marker"


def _models():
    return AtomModels({None: [reply("ok"), reply("ok")]})


async def test_untrusted_session_gets_the_notice_and_no_question_without_forms(
    make_adapter, workspace, client, untrusted
):
    adapter = await make_adapter(_models(), agent_spec=ATOM_SPEC)
    session_id = (await adapter.new_session(str(workspace))).session_id
    await client.wait_for(lambda: client.texts(AgentMessageChunk, session_id))
    notice = client.texts(AgentMessageChunk, session_id)[0]
    assert "Code skills and model settings (.nooa/llm_config.yaml) in this folder are off" in notice
    assert "Type /trust to allow." in notice
    await adapter.prompt(session_id, [text_block("hello")])
    assert not [entry for entry in client.log if entry[0] in ("elicitation", "ext")]
    assert not untrusted.exists()
    assert not is_trusted(workspace)


@pytest.mark.parametrize("answer", ["Yes", "No"])
async def test_first_prompt_asks_whether_to_trust_the_folder(
    make_adapter, workspace, client, untrusted, answer
):
    adapter = await make_adapter(_models(), agent_spec=ATOM_SPEC, capabilities=FORMS)
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.elicitation_answers = [
        AcceptElicitationResponse(action="accept", content={"trust": answer})
    ]
    await adapter.prompt(session_id, [text_block("hello")])
    [request] = [entry for entry in client.log if entry[0] == "elicitation"]
    assert "Do you trust this folder?" in request[1]
    assert is_trusted(workspace) is (answer == "Yes")
    assert untrusted.exists() is (answer == "Yes")
    # Asked once per session.
    await adapter.prompt(session_id, [text_block("again")])
    assert len([entry for entry in client.log if entry[0] == "elicitation"]) == 1


async def test_trust_command_enables_the_folder(make_adapter, workspace, client, untrusted):
    adapter = await make_adapter(_models(), agent_spec=ATOM_SPEC)
    session_id = (await adapter.new_session(str(workspace))).session_id
    await adapter.prompt(session_id, [text_block("/trust")])
    assert is_trusted(workspace)
    assert untrusted.exists()
    assert "Trusted" in client.texts(AgentMessageChunk, session_id)[-1]


async def test_stop_during_the_trust_form_cancels_the_prompt(
    make_adapter, workspace, client, untrusted
):
    """Quality m3: Stop while the form is open does not run the prompt."""
    adapter = await make_adapter(_models(), agent_spec=ATOM_SPEC, capabilities=FORMS)
    session_id = (await adapter.new_session(str(workspace))).session_id
    client.elicitation_gate = asyncio.Event()
    client.elicitation_answers = [
        AcceptElicitationResponse(action="accept", content={"trust": "Yes"})
    ]
    task = asyncio.create_task(adapter.prompt(session_id, [text_block("hello")]))
    await client.elicitation_started.wait()
    await adapter.cancel(session_id)
    client.elicitation_gate.set()
    response = await task
    assert response.stop_reason == "cancelled"
    assert not is_trusted(workspace)
    assert not untrusted.exists()


async def test_no_trust_form_in_the_home_folder(
    make_adapter, client, tmp_path, monkeypatch, untrusted_folders
):
    """Quality M2: the form must not offer to trust the whole home folder."""
    home = tmp_path / "home"
    (home / ".nooa" / "skills" / "x").mkdir(parents=True)
    (home / ".nooa" / "skills" / "x" / "pyproject.toml").write_text('[project]\nname = "x"\n')
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    adapter = await make_adapter(_models(), agent_spec=ATOM_SPEC, capabilities=FORMS)
    session_id = (await adapter.new_session(str(home))).session_id
    await adapter.prompt(session_id, [text_block("hello")])
    assert not [entry for entry in client.log if entry[0] == "elicitation"]
    notice = client.texts(AgentMessageChunk, session_id)[0]
    assert "Open a project folder instead" in notice
