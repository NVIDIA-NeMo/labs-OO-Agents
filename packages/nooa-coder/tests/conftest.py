# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the nooa-coder tests."""

import pytest


@pytest.fixture(autouse=True)
def _user_dir(tmp_path, monkeypatch):
    """Point the user-level NOOA directory (and so the default sessions dir) at tmp_path."""
    user_dir = tmp_path / "user"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))
    return user_dir


@pytest.fixture
def sessions_dir(tmp_path):
    return tmp_path / "sessions"


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
            sessions_dir=sessions_dir,
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


class ScriptedModels:
    """An agent factory that gives each session its own strict fake model.

    Scripts are keyed by session name (``None`` for an unnamed root); a
    session whose name has no script gets an empty strict model.
    """

    def __init__(self, scripts=None):
        self.scripts = dict(scripts or {})
        self.llms = {}
        self.built = []

    def __call__(self, options, storage):
        from nooa_coder.session.loader import default_agent_factory

        from nooa.unifiedllm import FakeLLMClient

        llm = FakeLLMClient(list(self.scripts.get(options.name, [])), strict_exhaustion=True)
        self.llms[options.name] = llm
        self.built.append(options)
        return default_agent_factory(options.model_copy(update={"llm": llm}), storage)


@pytest.fixture
def models():
    return ScriptedModels()


@pytest.fixture
async def registry(sessions_dir, models):
    from nooa_coder.session.registry import SessionRegistry
    from nooa_coder.session.store import SessionStore

    registry = SessionRegistry(SessionStore(sessions_dir), agent_factory=models)
    yield registry
    await registry.close_all()


@pytest.fixture
def root_options(tmp_path, sessions_dir):
    from nooa_coder.session.options import SessionOptions

    return SessionOptions(
        workspace=tmp_path, agent_spec="coder_test_agents:EchoAgent", sessions_dir=sessions_dir
    )
