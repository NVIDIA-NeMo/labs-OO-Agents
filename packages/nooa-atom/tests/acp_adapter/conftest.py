# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the ACP adapter tests: an in-process fake client and adapter."""

import asyncio
from pathlib import Path
from typing import Any

import pytest


class FakeClient:
    """An ACP client that records everything the adapter sends, in one ordered log.

    ``log`` holds ``("update", session_id, update)``, ``("elicitation", ...)``,
    ``("permission", ...)`` and ``("response", label, value)`` entries (the
    tests add the last kind), so tests can assert ordering across kinds.
    ``elicitation_answers`` and ``permission_answers`` are scripted replies,
    used in order; ``elicitation_gate`` (an Event) holds an elicitation open.
    Extension requests are logged as ``("ext", method, params)`` and answered
    from ``ext_answers`` (an exception there is raised); ``ext_gate`` holds
    them open.
    """

    def __init__(self) -> None:
        self.log: list[tuple[Any, ...]] = []
        self.elicitation_answers: list[Any] = []
        self.permission_answers: list[Any] = []
        self.ext_answers: list[Any] = []
        self.elicitation_gate: asyncio.Event | None = None
        self.ext_gate: asyncio.Event | None = None
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

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.log.append(("ext", method, params))
        self.changed.set()
        if self.ext_gate is not None:
            await self.ext_gate.wait()
        answer = self.ext_answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer

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
def sessions_dir(workspace):
    """Where the ``workspace`` fixture's sessions live by default."""
    return workspace / ".nooa" / "sessions"


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
async def make_adapter(client):
    """Build an ``AtomACPAgent`` over a registry of scripted-model agents.

    ``make_adapter(models, agent_spec=..., capabilities=..., client_info=...)`` returns the
    adapter, connected to ``client`` and initialized. Everything it built
    is closed after the test.
    """
    from acp import PROTOCOL_VERSION
    from nooa_atom.acp.server import AtomACPAgent
    from nooa_atom.session.registry import SessionRegistry

    built: list[Any] = []

    async def make(
        models: Any,
        *,
        agent_spec: str = "atom_test_agents:EchoAgent",
        capabilities: Any = None,
        llm_factory: Any = None,
        model: str | None = None,
        client_: Any = None,
        client_info: Any = None,
    ) -> Any:
        def new_registry(store: Any) -> SessionRegistry:
            return SessionRegistry(store, agent_factory=models, llm_factory=llm_factory)

        adapter = AtomACPAgent(new_registry, agent_spec=agent_spec, model=model)
        adapter.on_connect(client_ or client)
        await adapter.initialize(
            PROTOCOL_VERSION, client_capabilities=capabilities, client_info=client_info
        )
        built.append(adapter)
        return adapter

    yield make
    for adapter in built:
        await adapter.close()


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path_factory, monkeypatch):
    """Keep sessions out of the developer's skills and settings (~/.agents/skills etc.)."""
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("NEMO_OO_SETTINGS", raising=False)
    monkeypatch.delenv("NEMO_OO_PROJECT_DIR", raising=False)
    return home


FIXTURES = Path(__file__).parent / "fixtures"


def agent_file_spec(name: str) -> str:
    """The file spec of a class in ``fixtures/acp_test_agents.py``."""
    return f"{FIXTURES / 'acp_test_agents.py'}:{name}"


@pytest.fixture
def file_spec():
    return agent_file_spec


@pytest.fixture
async def atom_adapter(make_adapter):
    """``atom_adapter(*responses)``: an adapter building real Atom agents.

    Sessions are built by ``create_session_agent``, so workspace settings, skills and slash
    commands apply; each session's model is a strict fake scripted with
    ``responses`` (one list per session, in creation order).
    """
    from atom_test_agents import ATOM_SPEC, ModelFactory
    from nooa_atom.agent.factory import create_session_agent

    async def make(*scripts: list[Any], capabilities: Any = None) -> Any:
        factory = ModelFactory({"fake": [list(script) for script in scripts]})
        adapter = await make_adapter(
            create_session_agent,
            agent_spec=ATOM_SPEC,
            llm_factory=factory,
            model="fake",
            capabilities=capabilities,
        )
        adapter.test_models = factory
        return adapter

    return make


@pytest.fixture(autouse=True)
def _protocol_subprocess_environment(monkeypatch):
    """Keep this checkout's sources and test configuration in ACP subprocesses.

    The library's launcher builds a sanitised environment; pass through the
    variables that point at test directories.
    """
    import os

    import acp.transports

    original = acp.transports.default_environment
    root = Path(__file__).resolve().parents[4]
    sources = [root / "src", root / "packages" / "nooa-atom" / "src"]

    def environment():
        values = original()
        values["PYTHONPATH"] = os.pathsep.join(str(path) for path in sources)
        for name in (
            "HOME",
            "NEMO_OO_USER_DIR",
            "NEMO_OO_PROJECT_DIR",
            "NEMO_OO_SETTINGS",
            "NOOA_SESSIONS_DIR",
            "NOOA_ACP_MCP_TRACE",
        ):
            if name in os.environ:
                values[name] = os.environ[name]
        return values

    monkeypatch.setattr(acp.transports, "default_environment", environment)
