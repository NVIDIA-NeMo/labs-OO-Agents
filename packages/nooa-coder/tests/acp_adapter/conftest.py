# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the ACP adapter tests: an in-process fake client and adapter."""

import asyncio
from typing import Any

import pytest


class FakeClient:
    """An ACP client that records everything the adapter sends, in one ordered log.

    ``log`` holds ``("update", session_id, update)``, ``("elicitation", ...)``,
    ``("permission", ...)`` and ``("response", label, value)`` entries (the
    tests add the last kind), so tests can assert ordering across kinds.
    ``elicitation_answers`` and ``permission_answers`` are scripted replies,
    used in order; ``elicitation_gate`` (an Event) holds an elicitation open.
    """

    def __init__(self) -> None:
        self.log: list[tuple[Any, ...]] = []
        self.elicitation_answers: list[Any] = []
        self.permission_answers: list[Any] = []
        self.elicitation_gate: asyncio.Event | None = None
        self.elicitation_started = asyncio.Event()
        self.changed = asyncio.Event()

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.log.append(("update", session_id, update))
        self.changed.set()

    async def create_elicitation(self, message: str, mode: Any, **kwargs: Any) -> Any:
        self.log.append(("elicitation", message, mode))
        self.elicitation_started.set()
        if self.elicitation_gate is not None:
            await self.elicitation_gate.wait()
        return self.elicitation_answers.pop(0)

    async def request_permission(
        self, session_id: str, tool_call: Any, options: list[Any], **kwargs: Any
    ) -> Any:
        self.log.append(("permission", session_id, tool_call, options))
        return self.permission_answers.pop(0)

    def updates(self, session_id: str | None = None, kind: type | None = None) -> list[Any]:
        return [
            entry[2]
            for entry in self.log
            if entry[0] == "update"
            and (session_id is None or entry[1] == session_id)
            and (kind is None or isinstance(entry[2], kind))
        ]

    def texts(self, kind: type, session_id: str | None = None) -> list[str]:
        return [getattr(update.content, "text", "") for update in self.updates(session_id, kind)]

    async def wait_for(self, predicate: Any, timeout: float = 10) -> None:
        async def poll() -> None:
            while not predicate():
                self.changed.clear()
                await self.changed.wait()

        await asyncio.wait_for(poll(), timeout)


@pytest.fixture
def workspace(tmp_path):
    path = tmp_path / "workspace"
    path.mkdir()
    return path


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
async def make_adapter(sessions_dir, client):
    """Build a ``CoderACPAgent`` over a registry of scripted-model agents.

    ``make_adapter(models, agent_spec=..., capabilities=...)`` returns the
    adapter, connected to ``client`` and initialized. Everything it built
    is closed after the test.
    """
    from acp import PROTOCOL_VERSION
    from nooa_coder.acp.server import CoderACPAgent
    from nooa_coder.session.registry import SessionRegistry
    from nooa_coder.session.store import SessionStore

    built: list[Any] = []

    async def make(
        models: Any,
        *,
        agent_spec: str = "coder_test_agents:EchoAgent",
        capabilities: Any = None,
        llm_factory: Any = None,
        model: str | None = None,
        client_: Any = None,
    ) -> Any:
        registry = SessionRegistry(
            SessionStore(sessions_dir), agent_factory=models, llm_factory=llm_factory
        )
        adapter = CoderACPAgent(registry, agent_spec=agent_spec, model=model)
        adapter.on_connect(client_ or client)
        await adapter.initialize(PROTOCOL_VERSION, client_capabilities=capabilities)
        built.append(adapter)
        return adapter

    yield make
    for adapter in built:
        await adapter.close()
