# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACP advertises and dispatches setup without passing credentials to an agent."""

from types import SimpleNamespace

import yaml
from acp import text_block
from acp.schema import AvailableCommandsUpdate
from nooa_acp.server import CodingACPAdapter
from nooa_cli.interactive.options import SessionOptions

from nooa.unifiedllm import FakeLLMClient, connect


async def test_connect_runs_as_a_session_local_control(tmp_path, monkeypatch):
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "wrong-project"))
    monkeypatch.delenv("NOOA_SESSIONS_DIR", raising=False)
    monkeypatch.setenv("CONNECT_ACP_KEY", "private-key")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    calls = []

    async def discover(endpoint, **kwargs):
        assert kwargs == {"api_style": "chat", "api_key": "private-key"}
        return connect.Discovery(endpoint, ({"id": "model"},))

    async def run(proposal, **kwargs):
        calls.append(kwargs)
        assert kwargs == {"approved": "minimal", "api_key": "private-key"}
        return connect.ConnectResult(proposal.alias, proposal.entry)

    monkeypatch.setattr(connect, "discover", discover)
    monkeypatch.setattr(connect, "run", run)
    updates = []

    async def update(session_id, payload, **kwargs):
        updates.append(payload)

    llm = FakeLLMClient()
    adapter = CodingACPAdapter(
        lambda: llm, options_factory=lambda root: SessionOptions(working_dir=str(root))
    )
    adapter.on_connect(SimpleNamespace(session_update=update))
    try:
        session_id = (await adapter.new_session(str(workspace))).session_id
        for command in [
            "/connect https://gateway.example/v1 --api-key-env CONNECT_ACP_KEY",
            "/connect model model --as work --max-tokens 256",
        ]:
            response = await adapter.prompt(session_id, [text_block(command)])
            assert response.stop_reason == "end_turn"
        assert not calls
        target = workspace / ".nooa" / "llm_config.yaml"
        assert not target.exists()
        await adapter.prompt(session_id, [text_block("/connect check minimal")])
        assert len(calls) == 1 and not target.exists()
        await adapter.prompt(session_id, [text_block("/connect save")])
        assert yaml.safe_load(target.read_text())["models"]["work"]["api_style"] == "chat"
        assert not (tmp_path / "wrong-project" / "llm_config.yaml").exists()
        advertised = [u for u in updates if isinstance(u, AvailableCommandsUpdate)]
        assert any(c.name == "connect" for u in advertised for c in u.available_commands)
        assert "private-key" not in "\n".join(u.model_dump_json() for u in updates)
        assert llm.call_count == 0
        session = (await adapter._sessions.get(session_id)).value
        assert session.handle.turns() == []
        assert session.agent.llm is llm
    finally:
        await adapter.close()
