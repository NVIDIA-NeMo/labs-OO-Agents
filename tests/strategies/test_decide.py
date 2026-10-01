# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from enum import Enum
from typing import Annotated

import pytest
from pydantic import BaseModel

from nooa import (
    Agent,
    BooleanDecision,
    Context,
    Criteria,
    DecideStrategy,
    DecisionModelRequiredError,
    EventQuery,
    Instructions,
    PredictStrategy,
    Threshold,
    strategy,
)
from nooa.context_blocks import ScopedContext
from nooa.decisions.client import (
    BooleanAnswer,
    BooleanQuestion,
    DecisionRequest,
    DecisionResponse,
)
from nooa.decisions.provenance import LLM_FALLBACK_SCHEMA_VERSION, question_digest
from nooa.events import DebugTrace, DecisionRecord, Error, EventBase, Message
from nooa.runtime.middleware import DecisionCallContext
from nooa.unifiedllm import AssistantText, FakeLLMClient, LLMResponse


class FakeDecisionClient:
    model = "fake-decisions"

    def __init__(self, probability: float = 0.9) -> None:
        self.probability = probability
        self.requests: list[DecisionRequest] = []

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        self.requests.append(request)
        return DecisionResponse(
            model=self.model,
            answers={"result": BooleanAnswer(self.probability)},
            id="decision-response-1",
            usage={"prompt_tokens": 12, "completion_tokens": 3},
        )


def _chat_response(content: str) -> LLMResponse:
    """Build one deterministic response for a Predict fallback test."""
    return LLMResponse(
        parts=(AssistantText(text=content),),
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_decide_strategy_agent_method_end_to_end() -> None:
    client = FakeDecisionClient()

    class TriageAgent(
        Agent,
        llm=FakeLLMClient(),
        decision_model=client,
    ):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> Annotated[bool, Threshold(0.8)]:
            """Does the message require immediate action?"""
            ...

    agent = TriageAgent()
    events: list[str] = []
    turn_events: list[EventBase] = []
    agent.event_manager.on("DecisionCallStart", lambda event: events.append(type(event).__name__))
    agent.event_manager.on("DecisionCallEnd", lambda event: events.append(type(event).__name__))
    agent.event_manager.on("BeforeTurn", turn_events.append)
    agent.event_manager.on("AfterTurn", turn_events.append)

    assert await agent.urgent("Payouts have failed for three days") is True
    assert client.requests[0].state == {"inputs": {"message": "Payouts have failed for three days"}}
    assert events == ["DecisionCallStart", "DecisionCallEnd"]
    assert turn_events == []


@pytest.mark.asyncio
async def test_decide_strategy_adds_opted_in_context_and_events_to_state() -> None:
    client = FakeDecisionClient()

    class TriageAgent(
        Agent,
        llm=FakeLLMClient(),
        decision_model=client,
    ):
        policy = "Refunds over $500 require manual review."

        @strategy(
            DecideStrategy(),
            ScopedContext(
                context={
                    "support_policy": Context(expr="self.policy"),
                    "customer_tier": "enterprise",
                },
                events=EventQuery.last_n(3),
            ),
        )
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    agent = TriageAgent()
    agent.event_manager.add(Message(content="Customer requested a refund."))
    agent.event_manager.add(DebugTrace(content="internal trace"))
    agent.event_manager.add(Error(content="Previous refund attempt failed."))

    assert await agent.urgent("Please refund my $900 purchase") is True
    assert client.requests[0].state == {
        "inputs": {"message": "Please refund my $900 purchase"},
        "context": {
            "support_policy": "Refunds over $500 require manual review.",
            "customer_tier": "enterprise",
        },
        "events": [
            {
                "type": "Message",
                "role": "assistant",
                "data": {"content": "Customer requested a refund."},
            },
            {
                "type": "Error",
                "role": "user",
                "data": {"content": "Previous refund attempt failed."},
            },
        ],
    }


@pytest.mark.asyncio
async def test_decide_strategy_honors_active_scoped_context() -> None:
    client = FakeDecisionClient()

    class TriageAgent(Agent, llm=FakeLLMClient(), decision_model=client):
        @strategy(DecideStrategy(), context={"policy": "default"})
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    agent = TriageAgent()
    with ScopedContext(context={"policy": "temporary", "region": "EU"}):
        assert await agent.urgent("Production is down") is True

    assert client.requests[0].state == {
        "inputs": {"message": "Production is down"},
        "context": {"policy": "temporary", "region": "EU"},
    }


@pytest.mark.asyncio
async def test_decide_strategy_expands_method_docstring_from_agent_state() -> None:
    client = FakeDecisionClient()

    class RefundAgent(Agent, llm=FakeLLMClient(), decision_model=client):
        region = "EU"

        def render_policy(self) -> str:
            """Return the policy text used by the decision instructions."""
            return "Refunds are allowed within 14 days."

        @strategy(DecideStrategy())
        async def should_refund(self, request: str) -> bool:
            """Apply the {self.region} policy: {self.render_policy()}"""
            ...

    assert await RefundAgent().should_refund("Refund my purchase") is True
    assert (
        client.requests[0].questions["result"].instructions
        == "Apply the EU policy: Refunds are allowed within 14 days."
    )


@pytest.mark.asyncio
async def test_decide_strategy_standalone_function() -> None:
    llm = FakeLLMClient([_chat_response('{"value": false}')])

    @strategy(DecideStrategy(), llm=llm)
    async def urgent(
        message: str,
    ) -> Annotated[
        bool,
        Criteria(by_value={True: "urgent", False: "not urgent"}),
    ]:
        """Is this urgent?"""
        ...

    assert await urgent("Routine request") is False
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_standalone_function_uses_explicit_native_decision_model_without_llm() -> None:
    client = FakeDecisionClient(probability=0.91)

    @strategy(DecideStrategy(), decision_model=client)
    async def urgent(message: str) -> Annotated[bool, Threshold(0.8)]:
        """Is this urgent?"""
        ...

    assert await urgent("Production is down") is True
    assert client.requests[0].state == {"inputs": {"message": "Production is down"}}


@pytest.mark.asyncio
async def test_standalone_function_inherits_parent_decision_model() -> None:
    client = FakeDecisionClient()

    @strategy(DecideStrategy())
    async def urgent(message: str) -> bool:
        """Is this urgent?"""
        ...

    class TriageAgent(Agent, llm=FakeLLMClient(), decision_model=client):
        async def route(self, message: str) -> bool:
            """Call the standalone decision from an agent method."""
            return await urgent(message)

    assert await TriageAgent().route("Production is down") is True
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_standalone_explicit_decision_model_overrides_parent() -> None:
    parent_client = FakeDecisionClient(probability=0.1)
    explicit_client = FakeDecisionClient(probability=0.9)

    @strategy(DecideStrategy(), decision_model=explicit_client)
    async def urgent(message: str) -> bool:
        """Is this urgent?"""
        ...

    class TriageAgent(Agent, llm=FakeLLMClient(), decision_model=parent_client):
        async def route(self, message: str) -> bool:
            """Call the standalone decision from an agent method."""
            return await urgent(message)

    assert await TriageAgent().route("Production is down") is True
    assert len(explicit_client.requests) == 1
    assert parent_client.requests == []


@pytest.mark.asyncio
async def test_standalone_decision_alias_is_resolved_once(monkeypatch) -> None:
    client = FakeDecisionClient()
    resolutions: list[str] = []

    def resolve(alias: str):
        resolutions.append(alias)
        return client

    monkeypatch.setattr("nooa.unifiedllm.get_decision_model", resolve)

    @strategy(DecideStrategy(), decision_model="decisions")
    async def urgent(message: str) -> bool:
        """Is this urgent?"""
        ...

    assert await urgent("First") is True
    assert await urgent("Second") is True
    assert resolutions == ["decisions"]
    assert len(client.requests) == 2


@pytest.mark.asyncio
async def test_method_decision_model_overrides_agent_default() -> None:
    default_client = FakeDecisionClient(probability=0.1)
    method_client = FakeDecisionClient(probability=0.9)

    class TriageAgent(
        Agent,
        llm=FakeLLMClient(),
        decision_model=default_client,
    ):
        @strategy(DecideStrategy(), decision_model=method_client)
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

        @strategy(DecideStrategy())
        async def important(self, message: str) -> bool:
            """Is this important?"""
            ...

    agent = TriageAgent()
    assert await agent.urgent("Production is down") is True
    assert await agent.important("Routine request") is False
    assert len(method_client.requests) == 1
    assert len(default_client.requests) == 1


@pytest.mark.asyncio
async def test_method_decision_model_works_without_agent_default() -> None:
    method_client = FakeDecisionClient()

    class TriageAgent(Agent, llm=FakeLLMClient()):
        @strategy(DecideStrategy(), decision_model=method_client)
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    assert await TriageAgent().urgent("Production is down") is True
    assert len(method_client.requests) == 1


@pytest.mark.asyncio
async def test_agent_resolves_configured_decision_model_alias(monkeypatch) -> None:
    client = FakeDecisionClient()
    resolutions: list[str] = []

    def resolve(alias: str):
        resolutions.append(alias)
        return client

    monkeypatch.setattr("nooa.unifiedllm.get_decision_model", resolve)

    class TriageAgent(Agent, llm=FakeLLMClient(), decision_model="decisions"):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    assert await TriageAgent().urgent("Production is down") is True
    assert resolutions == ["decisions"]
    assert len(client.requests) == 1


@pytest.mark.asyncio
async def test_method_decision_alias_is_cached_per_agent(monkeypatch) -> None:
    client = FakeDecisionClient()
    resolutions: list[str] = []

    def resolve(alias: str):
        resolutions.append(alias)
        return client

    monkeypatch.setattr("nooa.unifiedllm.get_decision_model", resolve)

    class TriageAgent(Agent, llm=FakeLLMClient()):
        @strategy(DecideStrategy(), decision_model="decisions")
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    agent = TriageAgent()
    assert await agent.urgent("First") is True
    assert await agent.urgent("Second") is True
    assert resolutions == ["decisions"]


def test_method_rejects_incompatible_decision_model_at_decoration_time() -> None:
    with pytest.raises(TypeError, match="decision_model for method 'urgent'.*adecide"):

        class TriageAgent(Agent, llm=FakeLLMClient()):
            @strategy(DecideStrategy(), decision_model=object())  # type: ignore[arg-type]
            async def urgent(self, message: str) -> bool:
                """Is this urgent?"""
                ...


@pytest.mark.asyncio
async def test_decision_middleware_mutation_and_durable_record() -> None:
    client = FakeDecisionClient()

    class TriageAgent(Agent, llm=FakeLLMClient(), decision_model=client):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    agent = TriageAgent()

    async def replace_state(ctx: DecisionCallContext, nxt):
        ctx.request = DecisionRequest(
            state={"normalized": True},
            questions=ctx.request.questions,
        )
        return await nxt(ctx)

    agent.event_manager.intercept("decision_call", replace_state)

    assert await agent.urgent("Production is down") is True
    assert client.requests[0].state == {"normalized": True}
    records = agent.event_manager.filter(type="DecisionRecord")
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, DecisionRecord)
    assert record.state == {"normalized": True}
    assert record.questions["result"]["type"] == "noul"
    assert record.answers == {"result": {"probability_true": 0.9}}
    assert record.requested_model == "fake-decisions"
    assert record.resolved_model == "fake-decisions"
    assert record.response_id == "decision-response-1"
    assert record.usage is not None
    assert record.usage.input_tokens == 12
    assert record.usage.output_tokens == 3
    assert record.success is True
    assert record.decision_source == "native"
    assert record.question_digest == question_digest(client.requests[0].questions)
    assert record.fallback_schema_version is None


@pytest.mark.asyncio
async def test_native_question_digest_reflects_middleware_question_changes() -> None:
    client = FakeDecisionClient()

    class TriageAgent(Agent, llm=FakeLLMClient(), decision_model=client):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    agent = TriageAgent()
    original: list[DecisionRequest] = []

    async def replace_question(ctx: DecisionCallContext, nxt):
        original.append(ctx.request)
        ctx.request = DecisionRequest(
            state=ctx.request.state,
            questions={"result": BooleanQuestion(instructions="Is production unavailable?")},
        )
        return await nxt(ctx)

    agent.event_manager.intercept("decision_call", replace_question)

    assert await agent.urgent("Production is down") is True
    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    assert record.question_digest == question_digest(client.requests[0].questions)
    assert record.question_digest != question_digest(original[0].questions)


@pytest.mark.asyncio
async def test_decision_middleware_can_short_circuit() -> None:
    client = FakeDecisionClient()

    class TriageAgent(Agent, llm=FakeLLMClient(), decision_model=client):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    agent = TriageAgent()

    async def fixed_response(ctx: DecisionCallContext, nxt):
        ctx.response = DecisionResponse(
            model="policy-cache",
            answers={"result": BooleanAnswer(0.1)},
        )
        return ctx

    agent.event_manager.intercept("decision_call", fixed_response)

    assert await agent.urgent("Routine request") is False
    assert client.requests == []
    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    assert record.resolved_model == "policy-cache"


@pytest.mark.asyncio
async def test_short_circuit_without_decision_response_is_recorded_as_failure() -> None:
    client = FakeDecisionClient()

    class TriageAgent(Agent, llm=FakeLLMClient(), decision_model=client):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    agent = TriageAgent()

    async def broken_policy(ctx: DecisionCallContext, nxt):
        return ctx

    agent.event_manager.intercept("decision_call", broken_policy)

    with pytest.raises(RuntimeError, match="must set ctx.response"):
        await agent.urgent("Routine request")
    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    assert record.success is False
    assert record.exception_type == "RuntimeError"


async def test_decide_strategy_rejects_incompatible_decision_model() -> None:
    class ChatOnly:
        async def acall(self, messages):
            return messages

    class TriageAgent(
        Agent,
        llm=FakeLLMClient(),
        decision_model=ChatOnly(),  # type: ignore[arg-type]
    ):
        @strategy(DecideStrategy())
        async def invalid(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    with pytest.raises(TypeError, match="decision_model.*adecide"):
        await TriageAgent().invalid("hello")


@pytest.mark.asyncio
async def test_decision_lifecycle_is_balanced_on_cancellation() -> None:
    started = asyncio.Event()

    class BlockingDecisionClient(FakeDecisionClient):
        async def adecide(self, request: DecisionRequest) -> DecisionResponse:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

    client = BlockingDecisionClient()

    class TriageAgent(
        Agent,
        llm=FakeLLMClient(),
        decision_model=client,
    ):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Is this urgent?"""
            ...

    agent = TriageAgent()
    ended = []
    agent.event_manager.on("DecisionCallEnd", ended.append)
    task = asyncio.create_task(agent.urgent("Wait"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(ended) == 1
    assert ended[0].success is False
    assert ended[0].exception_type == "CancelledError"
    records = agent.event_manager.filter(type="DecisionRecord")
    assert len(records) == 1
    assert isinstance(records[0], DecisionRecord)
    assert records[0].success is False
    assert records[0].exception_type == "CancelledError"


@pytest.mark.asyncio
async def test_decide_strategy_falls_back_to_llm_for_primitive_result() -> None:
    llm = FakeLLMClient([_chat_response('{"value": true}')])

    class TriageAgent(Agent, llm=llm):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    assert await TriageAgent().urgent("Production is down") is True
    assert llm.call_count == 1


@pytest.mark.asyncio
async def test_llm_fallback_persists_decision_provenance() -> None:
    llm = FakeLLMClient([_chat_response('{"value": true}')])
    llm.model = "fallback-chat"

    class TriageAgent(Agent, llm=llm):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    agent = TriageAgent()
    assert await agent.urgent("Production is down") is True

    records = agent.event_manager.filter(type="DecisionRecord")
    assert len(records) == 1
    record = records[0]
    assert isinstance(record, DecisionRecord)
    assert record.method_name == "urgent"
    assert record.decision_source == "llm_fallback"
    assert record.fallback_schema_version == LLM_FALLBACK_SCHEMA_VERSION
    assert record.requested_model == "fallback-chat"
    assert record.resolved_model is None
    assert record.usage is None
    assert record.questions["result"]["type"] == "noul"
    assert record.answers == {"result": {"value": True}}
    assert record.success is True
    assert record.exception_type is None


@pytest.mark.asyncio
async def test_llm_fallback_and_native_share_question_digest() -> None:
    client = FakeDecisionClient()

    class NativeAgent(Agent, llm=FakeLLMClient(), decision_model=client):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    class FallbackAgent(Agent, llm=FakeLLMClient([_chat_response('{"value": false}')])):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    native = NativeAgent()
    fallback = FallbackAgent()
    await native.urgent("Production is down")
    await fallback.urgent("A routine question")

    native_record = native.event_manager.filter(type="DecisionRecord")[0]
    fallback_record = fallback.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(native_record, DecisionRecord)
    assert isinstance(fallback_record, DecisionRecord)
    assert native_record.question_digest == fallback_record.question_digest
    assert native_record.decision_source != fallback_record.decision_source


@pytest.mark.asyncio
async def test_llm_fallback_serializes_composite_values_without_probabilities() -> None:
    class Priority(Enum):
        LOW = "low"
        HIGH = "high"

    class Triage(BaseModel):
        urgent: Annotated[bool, Instructions("Does it need immediate action?")]
        priority: Annotated[
            Priority, Instructions("How important is it?"), Criteria("Minor", "Major")
        ]

    llm = FakeLLMClient([_chat_response('{"urgent": true, "priority": "high"}')])

    class TriageAgent(Agent, llm=llm):
        @strategy(DecideStrategy())
        async def triage(self, message: str) -> Triage:
            """Triage the message."""
            ...

    agent = TriageAgent()
    assert await agent.triage("Production is down") == Triage(urgent=True, priority=Priority.HIGH)
    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    assert record.answers == {"urgent": {"value": True}, "priority": {"value": "high"}}
    assert set(record.questions) == {"urgent", "priority"}


@pytest.mark.asyncio
async def test_failed_llm_fallback_is_recorded() -> None:
    class UnavailableLLM(FakeLLMClient):
        async def acall(self, *args, **kwargs):  # type: ignore[override]
            raise ConnectionError("chat backend unavailable")

    class TriageAgent(Agent, llm=UnavailableLLM()):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    agent = TriageAgent()
    with pytest.raises(ConnectionError):
        await agent.urgent("Production is down")
    record = agent.event_manager.filter(type="DecisionRecord")[0]
    assert isinstance(record, DecisionRecord)
    assert record.decision_source == "llm_fallback"
    assert record.success is False
    assert record.answers is None
    assert record.exception_type == "ConnectionError"


@pytest.mark.asyncio
async def test_llm_fallback_rejects_evidence_dependent_results() -> None:
    llm = FakeLLMClient([_chat_response('{"value": true}')])

    class DetailedAgent(Agent, llm=llm):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> BooleanDecision:
            """Does the message require immediate action?"""
            ...

    class ThresholdAgent(Agent, llm=llm):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> Annotated[bool, Threshold(0.8)]:
            """Does the message require immediate action?"""
            ...

    for agent in (DetailedAgent(), ThresholdAgent()):
        with pytest.raises(DecisionModelRequiredError, match="configured decision_model"):
            await agent.urgent("Production is down")
    assert llm.call_count == 0


@pytest.mark.asyncio
async def test_instance_can_disable_inherited_decision_model() -> None:
    decision_model = FakeDecisionClient()
    llm = FakeLLMClient([_chat_response('{"value": false}')])

    class TriageAgent(Agent, llm=llm, decision_model=decision_model):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    assert await TriageAgent(decision_model=None).urgent("Routine request") is False
    assert decision_model.requests == []


@pytest.mark.asyncio
async def test_decision_only_agent_needs_no_chat_llm() -> None:
    client = FakeDecisionClient()

    class Router(Agent, decision_model=client):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    agent = Router()

    assert await agent.urgent("Production is down") is True
    assert len(client.requests) == 1
    with pytest.raises(RuntimeError, match="decision model but no chat LLM"):
        _ = agent.llm


@pytest.mark.asyncio
async def test_decision_only_agent_rejects_chat_strategy_methods() -> None:
    class Router(Agent, decision_model=FakeDecisionClient()):
        async def summarize(self, message: str) -> str:
            """Summarize the message."""
            ...

    with pytest.raises(RuntimeError, match="decision-only agent"):
        await Router().summarize("Production is down")


def test_agent_without_llm_or_decision_model_still_fails() -> None:
    class Router(Agent, decision_model=None):
        pass

    with pytest.raises(ValueError, match="No LLM available"):
        Router()


def test_child_of_decision_only_parent_needs_its_own_llm_without_decisions() -> None:
    from nooa.runtime.context_vars import _parent_agent_var

    class Router(Agent, decision_model=FakeDecisionClient()):
        pass

    class Writer(Agent):
        pass

    parent = Router()
    token = _parent_agent_var.set(parent)
    try:
        child = Writer()
        assert child.decision_model is parent.decision_model
        with pytest.raises(RuntimeError, match="no chat LLM"):
            _ = child.llm
        with pytest.raises(ValueError, match="No LLM available"):
            Writer(decision_model=None)
    finally:
        _parent_agent_var.reset(token)


@pytest.mark.asyncio
async def test_agent_decision_alias_is_shared_per_class(monkeypatch) -> None:
    resolutions: list[str] = []

    def resolve(alias: str):
        resolutions.append(alias)
        return FakeDecisionClient()

    monkeypatch.setattr("nooa.unifiedllm.get_decision_model", resolve)

    class TriageAgent(Agent, decision_model="decisions"):
        pass

    class OtherAgent(Agent, decision_model="decisions"):
        pass

    first, second = TriageAgent(), TriageAgent()
    other = OtherAgent()

    assert first.decision_model is second.decision_model
    assert other.decision_model is not first.decision_model
    assert resolutions == ["decisions", "decisions"]


@pytest.mark.asyncio
async def test_standalone_chat_function_keeps_llm_inside_decision_agent() -> None:
    @strategy(PredictStrategy(), llm=FakeLLMClient([_chat_response('{"value": "hello"}')]))
    async def greet(name: str) -> str:
        """Greet the person."""
        ...

    class Router(Agent, llm=FakeLLMClient(), decision_model=FakeDecisionClient()):
        async def run(self) -> str:
            return await greet("Ada")

    assert await Router().run() == "hello"


@pytest.mark.asyncio
async def test_llm_fallback_holds_the_generation_lock() -> None:
    lock_states: list[bool] = []

    class RecordingLLM(FakeLLMClient):
        async def acall(self, *args, **kwargs):
            lock_states.append(agent.runtime._generation_lock.locked())
            return await super().acall(*args, **kwargs)

    class TriageAgent(
        Agent,
        llm=RecordingLLM([_chat_response('{"value": true}'), _chat_response('{"value": false}')]),
    ):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    agent = TriageAgent()
    await asyncio.gather(agent.urgent("a"), agent.urgent("b"))

    assert lock_states == [True, True]


@pytest.mark.asyncio
async def test_native_decisions_do_not_take_the_generation_lock() -> None:
    lock_states: list[bool] = []

    class RecordingDecisionClient(FakeDecisionClient):
        async def adecide(self, request: DecisionRequest) -> DecisionResponse:
            lock_states.append(agent.runtime._generation_lock.locked())
            return await super().adecide(request)

    class TriageAgent(Agent, decision_model=RecordingDecisionClient()):
        @strategy(DecideStrategy())
        async def urgent(self, message: str) -> bool:
            """Does the message require immediate action?"""
            ...

    agent = TriageAgent()
    await agent.urgent("a")

    assert lock_states == [False]


def _prompt_text(call) -> str:
    return "\n".join(str(message.get("content", "")) for message in call.messages)


@pytest.mark.asyncio
async def test_llm_fallback_prompt_contains_compiled_questions() -> None:
    class Priority(Enum):
        LOW = "low"
        HIGH = "high"

    class Triage(BaseModel):
        urgent: Annotated[
            bool,
            Instructions("Does it need immediate action?"),
            Criteria(by_value={True: "Customers are blocked.", False: "It can wait."}),
        ]
        priority: Annotated[
            Priority, Instructions("How important is it?"), Criteria("Minor", "Major")
        ]

    llm = FakeLLMClient([_chat_response('{"urgent": true, "priority": "high"}')])

    class TriageAgent(Agent, llm=llm):
        @strategy(DecideStrategy())
        async def triage(self, message: str) -> Triage:
            """Triage the message."""
            ...

    await TriageAgent().triage("Production is down")

    prompt = _prompt_text(llm.calls[0])
    for text in (
        "Decision questions",
        "Does it need immediate action?",
        "Customers are blocked.",
        "How important is it?",
        '"value": "high"',
        "Major",
    ):
        assert text in prompt


@pytest.mark.asyncio
async def test_llm_fallback_score_is_bounded_to_its_levels() -> None:
    Severity = Annotated[float, Criteria("Cosmetic", "Degraded", "Outage")]
    llm = FakeLLMClient([_chat_response('{"value": 7}'), _chat_response('{"value": 1.5}')])

    class TriageAgent(Agent, llm=llm):
        @strategy(DecideStrategy())
        async def severity(self, message: str) -> Severity:
            """Rate the severity of the report."""
            ...

    agent = TriageAgent()

    assert await agent.severity("Checkout is slow") == 1.5
    assert len(llm.calls) == 2
    assert "Outage" in _prompt_text(llm.calls[0])


@pytest.mark.asyncio
async def test_llm_fallback_composite_score_is_bounded_and_restored() -> None:
    class Assessment(BaseModel):
        urgent: Annotated[bool, Instructions("Does it need immediate action?")]
        severity: Annotated[
            float,
            Instructions("How severe is it?"),
            Criteria("Cosmetic", "Degraded", "Outage"),
        ]

    llm = FakeLLMClient(
        [
            _chat_response('{"urgent": true, "severity": 3}'),
            _chat_response('{"urgent": true, "severity": 2}'),
        ]
    )

    class TriageAgent(Agent, llm=llm):
        @strategy(DecideStrategy())
        async def assess(self, message: str) -> Assessment:
            """Assess the report."""
            ...

    result = await TriageAgent().assess("Production is down")

    assert type(result) is Assessment
    assert result == Assessment(urgent=True, severity=2)
    assert len(llm.calls) == 2
