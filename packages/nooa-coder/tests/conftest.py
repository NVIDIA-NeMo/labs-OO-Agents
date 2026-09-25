# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the nooa-coder tests."""

import pytest


@pytest.fixture(autouse=True)
def _user_dir(tmp_path, monkeypatch):
    """Keep user-level settings in tmp_path and sessions in their workspace.

    Sessions live in ``<workspace>/.nooa/sessions`` unless ``NOOA_SESSIONS_DIR``
    names one shared directory; a developer's setting must not leak in.
    """
    user_dir = tmp_path / "user"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))
    monkeypatch.delenv("NOOA_SESSIONS_DIR", raising=False)
    return user_dir


@pytest.fixture
def sessions_dir(tmp_path):
    """Where sessions of the ``tmp_path`` workspace live by default."""
    return tmp_path / ".nooa" / "sessions"


@pytest.fixture
async def make_session(sessions_dir, tmp_path):
    """Build a Session directly (no registry) around a scripted fake model.

    Returns ``(session, llm)``. The fake model is strict: a model call with
    no scripted response left fails the turn, so tests see extra calls.
    """
    from nooa_coder.session.loader import default_agent_factory
    from nooa_coder.session.options import SessionOptions
    from nooa_coder.session.session import Session
    from nooa_coder.session.store import SessionStore

    from nooa.unifiedllm import FakeLLMClient

    made = []

    def factory(
        *responses,
        agent_spec: str = "coder_test_agents:EchoAgent",
        turn_method: str = "handle",
        start: bool = True,
        llm=None,
    ):
        llm = llm if llm is not None else FakeLLMClient(list(responses), strict_exhaustion=True)
        store = SessionStore(sessions_dir)
        options = SessionOptions(
            workspace=tmp_path,
            agent_spec=agent_spec,
            llm=llm,
            turn_method=turn_method,
        )
        handle = store.create(agent=agent_spec, workspace=str(tmp_path), host=options.host)
        agent = default_agent_factory(options, handle.storage)
        session = Session(options=options, agent=agent, handle=handle)
        if start:
            session.start()
        made.append(session)
        return session, llm

    yield factory
    for session in made:
        await session.close()


@pytest.fixture
def models():
    from coder_test_agents import ScriptedModels

    return ScriptedModels()


@pytest.fixture
async def registry(sessions_dir, models):
    from nooa_coder.session.registry import SessionRegistry
    from nooa_coder.session.store import SessionStore

    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    yield registry
    await registry.close_all()


@pytest.fixture
def root_options(tmp_path):
    from nooa_coder.session.options import SessionOptions

    return SessionOptions(workspace=tmp_path, agent_spec="coder_test_agents:EchoAgent")
