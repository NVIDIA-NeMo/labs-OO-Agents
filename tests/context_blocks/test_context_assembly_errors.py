# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Context assembly failures are deterministic: strategies must not retry them."""

import pytest

from nooa import Agent, CacheBoundary, ContextAssemblyError, strategy
from nooa.context_blocks import UnsupportedContextLayout
from nooa.default_context_view import CodeActView, PredictView, PurePythonView
from nooa.events import Error, Feedback
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient


class CountingClient(FakeLLMClient):
    calls = 0

    async def acall(self, messages, **kwargs):
        self.calls += 1
        return await super().acall(messages, **kwargs)


def broken_view(base: type, failure: str):
    class BrokenView(base):
        assembled = 0

        async def assemble(self, owner, call):
            type(self).assembled += 1
            async for item in super().assemble(owner, call):
                yield item
            if failure == "raw_item":
                yield {"role": "user", "content": "not a context item"}
            else:
                raise KeyError("missing_state")

    return BrokenView()


@pytest.mark.parametrize("failure", ["raw_item", "view_raises"])
@pytest.mark.parametrize(
    ("make_strategy", "view_base"),
    [
        (CodeActStrategy, CodeActView),
        (PurePythonStrategy, PurePythonView),
        (PredictStrategy, PredictView),
    ],
    ids=["codeact", "pure_python", "predict"],
)
async def test_view_failure_fails_once_without_model_calls_or_feedback(
    make_strategy, view_base, failure
):
    view = broken_view(view_base, failure)
    client = CountingClient()

    class Example(Agent, llm=client):
        @strategy(make_strategy(), context_view=view)
        async def run(self) -> int:
            """Return a number."""
            ...

    agent = Example()
    with pytest.raises(ContextAssemblyError, match="BrokenView.assemble\\(\\) failed") as info:
        await agent.run()
    assert isinstance(info.value.__cause__, TypeError if failure == "raw_item" else KeyError)
    assert type(view).assembled == 1
    assert client.calls == 0
    assert not [e for e in agent.event_manager.values() if isinstance(e, Error | Feedback)]


@pytest.mark.parametrize(
    ("make_strategy", "view_base"),
    [
        (CodeActStrategy, CodeActView),
        (PurePythonStrategy, PurePythonView),
        (PredictStrategy, PredictView),
    ],
    ids=["codeact", "pure_python", "predict"],
)
async def test_second_cache_boundary_fails_once_without_model_calls_or_feedback(
    make_strategy, view_base
):
    class TwoBoundaries(view_base):
        async def assemble(self, owner, call):
            yield CacheBoundary()
            async for item in super().assemble(owner, call):
                yield item
            yield CacheBoundary()

    client = CountingClient()

    class Example(Agent, llm=client):
        @strategy(make_strategy(), context_view=TwoBoundaries())
        async def run(self) -> int:
            """Return a number."""
            ...

    agent = Example()
    with pytest.raises(UnsupportedContextLayout, match="at most one CacheBoundary"):
        await agent.run()
    assert client.calls == 0
    assert not [e for e in agent.event_manager.values() if isinstance(e, Error | Feedback)]


def test_unsupported_layout_is_an_assembly_error():
    assert issubclass(UnsupportedContextLayout, ContextAssemblyError)
