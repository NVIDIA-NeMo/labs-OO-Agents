# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cell names come from one provider registry, shared by the in-process and sandbox backends."""

import pytest

from nooa import Agent
from nooa.agentdoc.visibility import filter_mro_module_globals
from nooa.unifiedllm import FakeLLMClient

# The framework names a default agent's cell saw before the registry (#558 L9).
FRAMEWORK_NAMES = frozenset(
    {
        "self", "asyncio", "typing",
        "Annotated", "Any", "Literal", "Optional", "Union",
        "doc", "methods", "variables", "help", "pprint",
        "show", "Media", "Image", "Audio", "Video", "File",
        "strategy", "PurePythonStrategy", "CodeActStrategy", "ReflexionStrategy",
        "PredictStrategy", "TemplateStrategy", "CompositeStrategy",
    }
)  # fmt: skip


class Plain(Agent, llm=FakeLLMClient()):
    pass


def _spy(monkeypatch) -> list[set[str]]:
    import nooa.runtime.actor as actor

    seen: list[set[str]] = []
    original = actor._strip_blocked_modules

    def spy(ns, blocked):
        seen.append(set(ns))
        return original(ns, blocked)

    monkeypatch.setattr(actor, "_strip_blocked_modules", spy)
    return seen


@pytest.mark.asyncio
async def test_a_default_agent_cell_sees_the_same_framework_names(monkeypatch):
    seen = _spy(monkeypatch)
    await Plain().runtime.execute_code("x = 1")
    names = seen[0] - set(filter_mro_module_globals(Plain))
    assert names == FRAMEWORK_NAMES


def test_the_sandbox_worker_reads_the_same_registry():
    from nooa.runtime.sandbox.worker import build_namespace

    agent = Plain()
    names = set(build_namespace(agent, {}, object(), None)) - set(filter_mro_module_globals(Plain))
    assert names == FRAMEWORK_NAMES


def test_the_media_provider_supplies_the_show_note():
    from nooa.runtime.namespace_providers import namespace_prompt_notes
    from nooa.strategies import CodeActV2

    notes = namespace_prompt_notes()
    assert any(line.startswith("show(obj) attaches a nooa.media.Image") for line in notes)
    assert " ".join(notes) in CodeActV2()._always_available_text()
