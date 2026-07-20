# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct unit tests for pure functions and runtime methods (gl-105).

Tests:
1. agent._try_auto_enable_tracing — second call is a no-op
2. agent._validate_llm_param — raises ValueError for None
3. Agent.__init_subclass__ event_query path — sets _agent_event_query
4. Agent._resolve_event_query — instance and class level resolution
5. Agent.__type_info__ — classmethods and hidden fields
6. Agent.__instance_values__ — exception swallowing
7. strategies/prefill._get_complex_type — all branches
8. strategies/current_call.CurrentCall.format_parameters_as_code — fallback
9. runtime/actor evaluate_expression subprocess path
10. runtime/actor expand_variables format conversions
"""

import subprocess
import typing
from typing import Annotated, Optional, Union

import pytest

from nooa.agent import Agent
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


# ---------------------------------------------------------------------------
# 1. _try_auto_enable_tracing — second call is a no-op
# ---------------------------------------------------------------------------


def test_try_auto_enable_tracing_noop_on_second_call():
    """Second call to _try_auto_enable_tracing returns early (flag already set)."""
    import nooa.agent as agent_mod

    # Reset to ensure clean state
    original = agent_mod._auto_tracing_attempted
    agent_mod._auto_tracing_attempted = False
    try:
        agent_mod._try_auto_enable_tracing()
        assert agent_mod._auto_tracing_attempted is True, "Flag should be set after first call"
        # Call again — should be a no-op (flag remains True, no exception)
        agent_mod._try_auto_enable_tracing()
        assert agent_mod._auto_tracing_attempted is True
    finally:
        agent_mod._auto_tracing_attempted = original


def test_try_auto_enable_tracing_sets_flag():
    """_try_auto_enable_tracing sets _auto_tracing_attempted on first call."""
    import nooa.agent as agent_mod

    original = agent_mod._auto_tracing_attempted
    agent_mod._auto_tracing_attempted = False
    try:
        assert agent_mod._auto_tracing_attempted is False
        agent_mod._try_auto_enable_tracing()
        assert agent_mod._auto_tracing_attempted is True
    finally:
        agent_mod._auto_tracing_attempted = original


# ---------------------------------------------------------------------------
# 2. _validate_llm_param — raises ValueError for None
# ---------------------------------------------------------------------------


def test_validate_llm_param_raises_for_none():
    """_validate_llm_param raises ValueError when llm=None."""
    from nooa.agent import _validate_llm_param

    with pytest.raises(ValueError, match="llm=None is not allowed"):
        _validate_llm_param(None, "MyAgent")


def test_validate_llm_param_ok_for_fake_llm():
    """_validate_llm_param does not raise for a valid LLM client."""
    from nooa.agent import _validate_llm_param

    _validate_llm_param(FakeLLMClient(), "MyAgent")  # should not raise


def test_validate_llm_param_ok_for_inherit():
    """_validate_llm_param does not raise for the INHERIT sentinel."""
    from nooa.agent import INHERIT, _validate_llm_param

    _validate_llm_param(INHERIT, "MyAgent")  # should not raise


# ---------------------------------------------------------------------------
# 3. __init_subclass__ event_query path — _agent_event_query is set
# ---------------------------------------------------------------------------


def test_init_subclass_sets_agent_event_query():
    """Agent subclass with event_query= gets _agent_event_query set."""
    from nooa.runtime.event_query import EventQuery

    class EQAgent(Agent, llm=_TEST_LLM, event_query=EventQuery(call_id="current")):
        async def work(self) -> str: ...

    assert hasattr(EQAgent, "_agent_event_query")
    assert isinstance(EQAgent._agent_event_query, EventQuery)
    assert EQAgent._agent_event_query.call_id == "current"


# ---------------------------------------------------------------------------
# 4. _resolve_event_query — instance and class level
# ---------------------------------------------------------------------------


def test_resolve_event_query_instance_overrides_class():
    """Instance-level event_query overrides the class-level one."""
    from nooa.runtime.event_query import EventQuery

    class ClassEQAgent(Agent, llm=_TEST_LLM, event_query=EventQuery(call_id="current")):
        async def work(self) -> str: ...

    instance_eq = EventQuery(type="Error")
    agent = ClassEQAgent(event_query=instance_eq)

    assert agent.event_query is instance_eq


def test_resolve_event_query_class_level_used_when_no_instance():
    """Class-level event_query is used when none is given at instantiation."""
    from nooa.runtime.event_query import EventQuery

    class ClassOnlyEQAgent(Agent, llm=_TEST_LLM, event_query=EventQuery(type="Task")):
        async def work(self) -> str: ...

    agent = ClassOnlyEQAgent()
    assert agent.event_query is ClassOnlyEQAgent._agent_event_query


def test_resolve_event_query_none_when_not_specified():
    """When no event_query at class or instance level, resolve returns None."""

    class NoEQAgent(Agent, llm=_TEST_LLM):
        async def work(self) -> str: ...

    agent = NoEQAgent()
    assert agent.event_query is None


# ---------------------------------------------------------------------------
# 5. __type_info__ — classmethods are included, hidden fields are excluded
# ---------------------------------------------------------------------------


def test_type_info_excludes_hidden_fields():
    """__type_info__ filters out Annotated[T, hidden] fields."""
    from nooa.agentdoc import doc, hidden

    class TypeInfoAgent(Agent, llm=_TEST_LLM):
        hidden_field: Annotated[str, hidden] = "secret"
        visible_field: str = "visible"

        async def generate(self) -> str: ...

    agent = TypeInfoAgent()
    result = doc(agent)
    assert "visible_field" in result
    assert "hidden_field" not in result


def test_type_info_includes_classmethods():
    """__type_info__ includes @classmethod entries."""
    from nooa.agentdoc import doc

    class CMAgent(Agent, llm=_TEST_LLM):
        @classmethod
        def class_method(cls) -> str:
            """A classmethod."""
            return "hi"

        async def generate(self) -> str: ...

    agent = CMAgent()
    result = doc(agent)
    assert "class_method" in result


# ---------------------------------------------------------------------------
# 6. __instance_values__ — exception swallowing (non-AttributeError/TypeError)
# ---------------------------------------------------------------------------


def test_instance_values_swallows_unexpected_exception():
    """__instance_values__ swallows RuntimeError from a broken property."""
    from nooa.agentdoc import doc

    class BrokenAgent(Agent, llm=_TEST_LLM):
        @property
        def broken(self) -> str:
            raise RuntimeError("unexpected!")

        async def work(self) -> str: ...

    agent = BrokenAgent()
    # doc() calls __instance_values__ internally; must not raise
    result = doc(agent)
    assert isinstance(result, str)
    # The broken property should not appear in the output at all
    assert "broken" not in result
    assert "RuntimeError" not in result


def test_instance_values_directly_does_not_propagate_exception():
    """Calling __instance_values__() directly does not raise on broken property."""

    class BrokenDirectAgent(Agent, llm=_TEST_LLM):
        @property
        def bad_prop(self) -> int:
            raise RuntimeError("boom")

        async def work(self) -> str: ...

    agent = BrokenDirectAgent()
    values = agent.__instance_values__()
    # bad_prop must not be present (exception was swallowed)
    assert "bad_prop" not in values


# ---------------------------------------------------------------------------
# 7. strategies/prefill._get_complex_type — all branches
# ---------------------------------------------------------------------------


class _MyModel:
    """Minimal concrete class for _get_complex_type branch tests."""


T = typing.TypeVar("T")


class _MyGeneric(typing.Generic[T]):
    """Minimal generic class for _get_complex_type generic-origin branch tests."""


def test_get_complex_type_unwraps_optional():
    """Optional[MyModel] -> MyModel."""
    from nooa.strategies.prefill import _get_complex_type

    result = _get_complex_type(Optional[_MyModel])
    assert result is _MyModel


def test_get_complex_type_returns_none_for_multiple_union():
    """Union[MyModel, str] (multiple non-None) -> None."""
    from nooa.strategies.prefill import _get_complex_type

    result = _get_complex_type(Union[_MyModel, str])
    assert result is None


def test_get_complex_type_complex_generic_origin():
    """MyGeneric[int] -> MyGeneric (the origin class)."""
    from nooa.strategies.prefill import _get_complex_type

    result = _get_complex_type(_MyGeneric[int])
    assert result is _MyGeneric


def test_get_complex_type_returns_none_for_non_type():
    """A non-type annotation (lambda) -> None."""
    from nooa.strategies.prefill import _get_complex_type

    result = _get_complex_type(lambda: None)
    assert result is None


def test_get_complex_type_returns_none_for_simple_generic():
    """list[str] -> None (simple generic origin)."""
    from nooa.strategies.prefill import _get_complex_type

    result = _get_complex_type(list[str])
    assert result is None


def test_get_complex_type_plain_model():
    """A plain class (non-simple) -> that class."""
    from nooa.strategies.prefill import _get_complex_type

    result = _get_complex_type(_MyModel)
    assert result is _MyModel


def test_get_complex_type_plain_simple_type():
    """str -> None (simple type)."""
    from nooa.strategies.prefill import _get_complex_type

    result = _get_complex_type(str)
    assert result is None


def test_get_complex_type_none_annotation():
    """None annotation -> None."""
    from nooa.strategies.prefill import _get_complex_type

    assert _get_complex_type(None) is None
    assert _get_complex_type(type(None)) is None


# ---------------------------------------------------------------------------
# 8. CurrentCall.format_parameters_as_code — fallback path
# ---------------------------------------------------------------------------


def test_format_parameters_as_code_fallback_uses_kwargs():
    """When signature is absent, format_parameters_as_code uses kwargs only."""
    from nooa.strategies.current_call import CurrentCall

    call = CurrentCall(
        id="test-id",
        method_name="my_method",
        decorator="agent",
        signature=None,  # no signature — triggers fallback
        kwargs={"alpha": 1, "beta": "hello"},
    )
    result = call.format_parameters_as_code()
    assert "alpha = 1" in result
    assert "beta = 'hello'" in result


def test_format_parameters_as_code_no_signature_no_kwargs():
    """No signature and no kwargs returns empty string."""
    from nooa.strategies.current_call import CurrentCall

    call = CurrentCall(
        id="test-id",
        method_name="my_method",
        decorator="agent",
        signature=None,
        kwargs={},
    )
    result = call.format_parameters_as_code()
    assert result == ""


def test_format_parameters_as_code_with_signature():
    """With a valid signature, positional args are mapped by name."""
    from nooa.strategies.current_call import CurrentCall

    def my_method(self, x: int, y: str): ...

    call = CurrentCall.from_method(my_method, args=(42,), kwargs={"y": "world"})
    result = call.format_parameters_as_code()
    assert "x = 42" in result
    assert "y = 'world'" in result


# ---------------------------------------------------------------------------
# 9 & 10. runtime/actor — subprocess path + format conversions
# (using the evaluate_expression / expand_variables helpers, no full session needed)
# ---------------------------------------------------------------------------


class _EvalAgent(Agent, llm=_TEST_LLM):
    """Minimal agent for evaluate_expression / expand_variables tests."""

    async def work(self) -> str: ...


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "proc,expected",
    [
        pytest.param(
            subprocess.CompletedProcess(
                args=["echo", "hi"], returncode=0, stdout="hello world\n", stderr=""
            ),
            "hello world\n",
            id="stdout",
        ),
        pytest.param(
            subprocess.CompletedProcess(args=["cmd"], returncode=1, stdout="", stderr="error text"),
            "error text",
            id="stderr_fallback",
        ),
        pytest.param(
            subprocess.CompletedProcess(args=["cmd"], returncode=2, stdout="", stderr=""),
            "[exit code: 2]",
            id="returncode_fallback",
        ),
    ],
)
async def test_evaluate_expression_subprocess(proc, expected):
    """evaluate_expression extracts the right field from CompletedProcess (stdout, stderr, exit-code fallback)."""
    agent = _EvalAgent()
    agent._last_execution_result = proc
    result = await agent.runtime.evaluate_expression("result")
    assert result == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "template,value,expected",
    [
        pytest.param("{value!r}", "hi\n", repr("hi\n"), id="repr_conversion"),
        pytest.param("{value!s}", 42, "42", id="str_conversion"),
        pytest.param("{value!a}", "café", ascii("café"), id="ascii_conversion"),
        pytest.param("{value:.2f}", 3.14159, "3.14", id="format_spec"),
    ],
)
async def test_expand_variables_conversions(template, value, expected):
    """expand_variables correctly applies !r/!s/!a conversions and format specs."""
    agent = _EvalAgent()
    result = await agent.runtime.expand_variables(template, extra_context={"value": value})
    assert result == expected
