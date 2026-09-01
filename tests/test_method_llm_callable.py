# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for callable ``@strategy(llm=...)`` overrides.

A callable defers the per-method LLM choice from import time to generation
time, where the agent instance is available — so two instances of the same
class can run the same method on different models.
"""

import json

import pytest

from nooa import Agent, strategy
from nooa.method_llm import resolve_method_llm, validate_method_llm_spec
from nooa.strategies.predict import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _fake(answer: str = "") -> FakeLLMClient:
    """A FakeLLMClient whose single PredictStrategy answer is *answer*.

    PredictStrategy wraps a plain ``-> str`` return in ``{"value": ...}``.
    """
    content = json.dumps({"value": answer})
    return FakeLLMClient(
        scripted_responses=[
            LLMResponse(
                raw_response=None,
                content=content,
                tool_calls=[],
                finish_reason="stop",
                assistant_message={"role": "assistant", "content": content},
            )
        ]
    )


class TestValidation:
    """Decoration-time validation of the llm= value."""

    def test_client_accepted(self) -> None:
        client = _fake("ok")

        @strategy(PredictStrategy(), llm=client)
        async def method(self) -> str: ...

        assert method._plan_llm is client  # type: ignore[attr-defined]

    def test_callable_accepted_and_stored_unresolved(self) -> None:
        """The callable is stored as-is — not invoked at decoration time."""
        calls = []

        def resolver(agent):
            calls.append(agent)
            return _fake("ok")

        @strategy(PredictStrategy(), llm=resolver)
        async def method(self) -> str: ...

        assert method._plan_llm is resolver  # type: ignore[attr-defined]
        assert calls == [], "resolver must not run until generation time"

    def test_string_rejected_at_decoration_time(self) -> None:
        """There is no alias registry — a model name string is a mistake."""
        with pytest.raises(TypeError, match="must be a UnifiedLLM instance or a callable"):

            @strategy(PredictStrategy(), llm="gpt-4")  # type: ignore[arg-type]
            async def method(self) -> str: ...

    def test_arbitrary_object_rejected(self) -> None:
        with pytest.raises(TypeError, match="got int"):

            @strategy(PredictStrategy(), llm=42)  # type: ignore[arg-type]
            async def method(self) -> str: ...

    def test_callable_rejected_on_standalone_function(self) -> None:
        """No 'self' parameter means no instance to resolve against."""
        with pytest.raises(TypeError, match="standalone function"):

            @strategy(PredictStrategy(), llm=lambda agent: _fake("ok"))
            async def standalone(x: int) -> str: ...

    def test_client_still_allowed_on_standalone_function(self) -> None:
        client = _fake("ok")

        @strategy(PredictStrategy(), llm=client)
        async def standalone(x: int) -> str: ...

        assert standalone._plan_llm is client  # type: ignore[attr-defined]

    def test_duck_typed_client_accepted(self) -> None:
        """Clients have never been required to subclass UnifiedLLM."""

        class DuckLLM:
            async def acall(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
                raise NotImplementedError

        duck = DuckLLM()
        validate_method_llm_spec(duck, "method")  # must not raise
        assert resolve_method_llm(duck, object(), "method") is duck

    def test_client_subclass_defining_call_is_not_treated_as_resolver(self) -> None:
        """isinstance runs before the callable check, so a client stays a client."""

        class CallableClient(FakeLLMClient):
            def __call__(self, agent):  # noqa: ANN001, ANN204
                raise AssertionError("must not be invoked as a resolver")

        client = CallableClient()
        assert resolve_method_llm(client, object(), "method") is client

    def test_class_object_is_treated_as_factory_not_client(self) -> None:
        """``hasattr(SomeClass, 'acall')`` is True for the unbound function.

        Without the isclass guard the class would pass validation as a client
        and then fail on an unbound-method call inside generation. It is a
        callable, so it is treated as a resolver and constructed instead.
        """
        built = _fake("from-factory")

        class Factory:
            async def acall(self, messages, tools=None, **kwargs):  # noqa: ANN001, ANN003
                raise NotImplementedError

            def __new__(cls, agent):  # noqa: ANN001, ANN204
                return built

        validate_method_llm_spec(Factory, "method")  # must not raise
        assert resolve_method_llm(Factory, object(), "method") is built


class TestResolution:
    """resolve_method_llm behaviour against an agent instance."""

    def test_callable_receives_agent_and_returns_client(self) -> None:
        client = _fake("ok")
        sentinel = object()
        seen = []

        def resolver(agent):
            seen.append(agent)
            return client

        assert resolve_method_llm(resolver, sentinel, "method") is client
        assert seen == [sentinel]

    def test_callable_exception_is_wrapped_with_method_name(self) -> None:
        """A typo'd attribute must not surface as a bare AttributeError."""

        def resolver(agent):
            return agent.does_not_exist

        with pytest.raises(RuntimeError, match=r"callable for 'analyze' raised AttributeError"):
            resolve_method_llm(resolver, object(), "analyze")

    def test_callable_exception_chains_original(self) -> None:
        def resolver(agent):
            raise ValueError("boom")

        with pytest.raises(RuntimeError) as exc_info:
            resolve_method_llm(resolver, object(), "analyze")
        assert isinstance(exc_info.value.__cause__, ValueError)

    def test_callable_returning_non_client_rejected(self) -> None:
        with pytest.raises(TypeError, match="must return a UnifiedLLM instance, got str"):
            resolve_method_llm(lambda agent: "not-a-client", object(), "analyze")

    def test_client_passed_through_untouched(self) -> None:
        client = _fake("ok")
        assert resolve_method_llm(client, object(), "method") is client


class TestStrategyHelperPath:
    """Framework kwargs are only stripped where something pops them back."""

    @pytest.mark.asyncio
    async def test_llm_kwarg_still_rejected_on_strategy_helper(self) -> None:
        """The strategy-helper path routes to execute_nested(), which pops nothing.

        Stripping ``llm`` there would let it slip past validation and into
        CurrentCall's prompt arguments instead of being rejected.
        """
        from unittest.mock import MagicMock

        from nooa.runtime.method_wrapper import create_agent_method_wrapper
        from nooa.strategies.base import RuntimeServices

        async def helper(self, runtime: RuntimeServices, x: int) -> int:
            """Do something with {x}."""
            ...

        wrapper = create_agent_method_wrapper(
            helper, needs_generation=True, needs_tracing=False, strategy=PredictStrategy()
        )

        class _Bare:  # no 'runtime' attribute → strategy-helper branch
            pass

        with pytest.raises(TypeError, match="unexpected keyword argument 'llm'"):
            await wrapper(_Bare(), MagicMock(spec=RuntimeServices), 42, llm=_fake("x"))


class TestEndToEnd:
    """The callable actually drives which model a generation call uses."""

    @pytest.mark.asyncio
    async def test_two_instances_same_method_different_llms(self) -> None:
        """The whole point: per-method LLM chosen at object creation time."""
        default = _fake("default")

        class Researcher(Agent, llm=default):
            """A researcher."""

            def __init__(self, big, **kwargs):  # noqa: ANN001, ANN003
                super().__init__(**kwargs)
                self.big = big

            @strategy(PredictStrategy(), llm=lambda self: self.big)
            async def analyze(self, doc: str) -> str:
                """Return the analysis of {doc}."""
                ...

        alice = Researcher(big=_fake("alice-model"))
        bob = Researcher(big=_fake("bob-model"))

        assert await alice.analyze("x") == "alice-model"
        assert await bob.analyze("x") == "bob-model"

    @pytest.mark.asyncio
    async def test_resolver_runs_per_call_not_once(self) -> None:
        """Re-resolving each call is what lets the choice depend on state."""
        first = _fake("first")
        second = _fake("second")

        # Class default is a third client, so *both* assertions below fail if
        # the resolver is ignored — not just the second.
        class Switcher(Agent, llm=_fake("class-default")):
            """A switcher."""

            def __init__(self, **kwargs):  # noqa: ANN003
                super().__init__(**kwargs)
                self.calls = 0

            @strategy(PredictStrategy(), llm=lambda self: first if self.calls == 0 else second)
            async def run(self) -> str:
                """Return a word."""
                ...

        agent = Switcher()
        assert await agent.run() == "first"
        agent.calls = 1
        assert await agent.run() == "second"

    @pytest.mark.asyncio
    async def test_call_level_llm_wins_and_skips_resolver(self) -> None:
        """A call-level override must not invoke the resolver at all."""
        invocations = []

        def resolver(agent):
            invocations.append(agent)
            return _fake("from-resolver")

        class Agent1(Agent, llm=_fake("from-agent")):
            """An agent."""

            @strategy(PredictStrategy(), llm=resolver)
            async def run(self) -> str:
                """Return a word."""
                ...

        agent = Agent1()
        assert await agent.run(llm=_fake("from-call")) == "from-call"
        assert invocations == [], "resolver ran despite a call-level override"

    @pytest.mark.asyncio
    async def test_call_level_llm_survives_argument_validation(self) -> None:
        """``llm=`` is a framework kwarg, not part of the method signature.

        Regression: the argument validator stripped ``_session_locals`` but
        not ``llm``, so a call-level override was rejected as an unexpected
        keyword argument before ever reaching the LLM-resolution code.
        """

        class Sig(Agent, llm=_fake("from-agent")):
            """An agent."""

            @strategy(PredictStrategy())
            async def run(self, topic: str, depth: int = 1) -> str:
                """Return a word about {topic} at depth {depth}."""
                ...

        agent = Sig()
        assert await agent.run("physics", depth=2, llm=_fake("from-call")) == "from-call"

    @pytest.mark.asyncio
    async def test_unexpected_kwarg_still_rejected(self) -> None:
        """Stripping framework kwargs must not weaken signature validation."""

        class Sig(Agent, llm=_fake("x")):
            """An agent."""

            @strategy(PredictStrategy())
            async def run(self, topic: str) -> str:
                """Return a word about {topic}."""
                ...

        with pytest.raises(TypeError, match="unexpected keyword argument 'bogus'"):
            await Sig().run("physics", bogus=1)

    @pytest.mark.asyncio
    async def test_falls_back_to_agent_llm_without_decorator_override(self) -> None:
        class Plain(Agent, llm=_fake("from-agent")):
            """An agent."""

            @strategy(PredictStrategy())
            async def run(self) -> str:
                """Return a word."""
                ...

        assert await Plain().run() == "from-agent"

    @pytest.mark.asyncio
    async def test_resolver_beats_instance_level_llm(self) -> None:
        """A per-method resolver outranks the agent-wide instance override."""

        class Researcher(Agent, llm=_fake("class-default")):
            """A researcher."""

            @strategy(PredictStrategy(), llm=lambda self: self.big)
            async def analyze(self) -> str:
                """Return a word."""
                ...

        agent = Researcher(llm=_fake("instance-level"))
        agent.big = _fake("from-resolver")
        assert await agent.analyze() == "from-resolver"

    @pytest.mark.asyncio
    async def test_call_level_strategy_survives_argument_validation(self) -> None:
        """Same regression as ``llm=``, for the ``_strategy`` framework kwarg."""

        class Sig(Agent, llm=_fake("ok")):
            """An agent."""

            @strategy(PredictStrategy())
            async def run(self, topic: str) -> str:
                """Return a word about {topic}."""
                ...

        assert await Sig().run("physics", _strategy=PredictStrategy()) == "ok"

    @pytest.mark.asyncio
    async def test_resolver_failure_surfaces_at_call_time(self) -> None:
        class Broken(Agent, llm=_fake("unused")):
            """An agent."""

            @strategy(PredictStrategy(), llm=lambda self: self.missing_attr)
            async def run(self) -> str:
                """Return a word."""
                ...

        with pytest.raises(RuntimeError, match=r"callable for 'run' raised AttributeError"):
            await Broken().run()
