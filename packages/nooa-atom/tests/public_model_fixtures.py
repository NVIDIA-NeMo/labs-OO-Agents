# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actual registry/agents backed by borrowed real SDK HTTP fixtures."""

import pytest
from atom_test_agents import EchoAgent
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore

from tests.runtime.public_llm_fixtures import native_models as native_models


class PublicAgent(EchoAgent):
    _set_llm_failure = None

    def set_llm(self, client, **options):
        if self._set_llm_failure is not None:
            raise self._set_llm_failure
        return super().set_llm(client, **options)


def public_agent_factory(options, storage):
    return PublicAgent(storage=storage, llm=options.model_binding or options.llm)


@pytest.fixture
async def public_registry(native_models, tmp_path):
    made = []

    def factory(alias, workspace):
        model = native_models.make(
            model=alias or "initial",
            reasoning_levels={
                "low": {"reasoning_effort": "low"},
                "high": {"reasoning_effort": "high"},
            },
            reasoning_default="low",
        )
        made.append(model)
        return model

    registry = SessionRegistry(
        SessionStore(tmp_path / ".nooa" / "sessions"),
        agent_factory=public_agent_factory,
        llm_factory=factory,
    )
    options = SessionOptions(
        workspace=tmp_path, agent_spec="atom_test_agents:EchoAgent", model="initial"
    )
    yield registry, options, made
    await registry.close_all()
