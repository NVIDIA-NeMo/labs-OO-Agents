# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Issue #139: create_model fails for non-Pydantic return types.

When an agent method has a return type that isn't Pydantic-serializable
(e.g. pd.DataFrame, np.ndarray, custom classes), the codeact strategy
should gracefully handle it instead of crashing.

Tests cover:
- _is_pydantic_compatible() correctly classifies types
- _create_return_model() returns (model, is_validated) correctly
- _build_return_result_tool() doesn't crash, schema falls back to Any, description guides LLM
- _handle_return_result() does isinstance validation for non-Pydantic types
- _try_validate_return_value() does isinstance validation for non-Pydantic types
- Annotated[NonPydanticType, "description"] works across all paths
- Full agent integration: CodeAct with non-Pydantic return type works end-to-end
- Error messages guide the LLM when type doesn't match
"""

import json
from typing import Annotated, Any

import pytest
from pydantic import BaseModel

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

# --- Non-Pydantic types for testing ---


class CustomResult:
    """A simple non-Pydantic class that create_model() can't handle."""

    def __init__(self, value: int, label: str):
        self.value = value
        self.label = label

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, CustomResult)
            and self.value == other.value
            and self.label == other.label
        )


class DataContainer:
    """Another non-Pydantic class with no __init__ type hints."""

    def __init__(self, data):
        self.data = data


# --- Test helpers ---

_TEST_LLM = FakeLLMClient(scripted_responses=[])


def _resp(content: str, tool_calls: list | None = None) -> LLMResponse:
    finish_reason = "tool_calls" if tool_calls else "stop"
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls or [],
        finish_reason=finish_reason,
        assistant_message={"role": "assistant", "content": content},
    )


def _tool_call(code: str, call_id: str = "call_1") -> ToolCall:
    return ToolCall(
        id=call_id,
        name="execute_python",
        arguments=json.dumps({"code": code}),
    )


def _return_result(call_id: str = "call_return", result: Any = None) -> ToolCall:
    return ToolCall(
        id=call_id,
        name="return_result",
        arguments=json.dumps({"result": result}),
    )


# === Unit tests for _is_pydantic_compatible ===


class TestIsPydanticCompatible:
    """_is_pydantic_compatible should correctly classify types."""

    def test_custom_class_is_not_compatible(self):
        strat = CodeActStrategy(config=CodeActConfig())
        assert strat._is_pydantic_compatible(CustomResult) is False

    def test_data_container_is_not_compatible(self):
        strat = CodeActStrategy(config=CodeActConfig())
        assert strat._is_pydantic_compatible(DataContainer) is False

    def test_basic_types_are_compatible(self):
        strat = CodeActStrategy(config=CodeActConfig())
        for t in (str, int, float, bool, list[str], dict[str, int]):
            assert strat._is_pydantic_compatible(t) is True, f"{t} should be compatible"

    def test_pydantic_model_is_compatible(self):
        class MyModel(BaseModel):
            x: int

        strat = CodeActStrategy(config=CodeActConfig())
        assert strat._is_pydantic_compatible(MyModel) is True

    def test_dataclass_is_compatible(self):
        """Already existed — kept for completeness alongside the new class."""
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: float
            y: float

        strat = CodeActStrategy(config=CodeActConfig())
        assert strat._is_pydantic_compatible(Point) is True


# === Unit tests for _create_return_model ===


class TestCreateReturnModel:
    """_create_return_model should return (model, is_validated) correctly."""

    def test_pydantic_type_returns_validated_true(self):
        strat = CodeActStrategy(config=CodeActConfig())
        model, is_validated = strat._create_return_model(str, "my_method")
        assert is_validated is True

    def test_non_pydantic_type_returns_validated_false(self):
        strat = CodeActStrategy(config=CodeActConfig())
        model, is_validated = strat._create_return_model(CustomResult, "my_method")
        assert is_validated is False

    def test_non_pydantic_model_accepts_any_value(self):
        """When is_validated=False the model field uses Any, so any value passes Pydantic."""
        strat = CodeActStrategy(config=CodeActConfig())
        model, _ = strat._create_return_model(CustomResult, "my_method")
        # Should not raise — the model accepts anything
        instance = model(result="literally anything")
        assert instance.result == "literally anything"


# === Unit tests for _build_return_result_tool ===


class TestBuildReturnResultToolNonPydantic:
    """_build_return_result_tool should not crash for non-Pydantic return types."""

    def test_custom_class_does_not_crash(self):
        """Building return_result tool with a custom class return type should not raise."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(CustomResult, "my_method")
        assert tool.name == "return_result"

    def test_data_container_does_not_crash(self):
        """Building return_result tool with DataContainer return type should not raise."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(DataContainer, "my_method")
        assert tool.name == "return_result"

    def test_tool_description_mentions_type(self):
        """Even when type is non-Pydantic, tool description should mention expected type."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(CustomResult, "my_method")
        assert "CustomResult" in tool.description

    def test_non_pydantic_schema_uses_any(self):
        """For non-Pydantic types, the tool schema should fall back to Any (not the type)."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(CustomResult, "my_method")
        schema = tool.parameters_model.model_json_schema()
        result_field = schema["properties"]["result"]
        # Any has no "type" constraint — should not have a restrictive type key
        assert "type" not in result_field or result_field.get("type") != "object"

    def test_non_pydantic_description_warns_about_tool_limitation(self):
        """For non-Pydantic types, description should tell LLM to use execute_python."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(CustomResult, "my_method")
        assert (
            "cannot be passed directly" in tool.description.lower()
            or "execute_python" in tool.description
        )

    def test_pydantic_type_description_says_tip(self):
        """For Pydantic-compatible types, description should use the 'Tip' form."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(str, "my_method")
        assert "Tip" in tool.description or "tip" in tool.description.lower()

    def test_pydantic_model_schema_uses_actual_type(self):
        """For a Pydantic BaseModel, the schema should reference the model (not Any)."""

        class MyModel(BaseModel):
            x: int
            y: str

        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(MyModel, "my_method")
        schema = tool.parameters_model.model_json_schema()
        # The result field should reference MyModel — check for its properties
        # (Pydantic nests it in $defs or inline)
        schema_str = json.dumps(schema)
        assert "x" in schema_str and "y" in schema_str


# === Edge case: dataclasses are Pydantic-compatible ===


class TestDataclassStillWorksThroughPydantic:
    """Dataclasses should still be validated via Pydantic (not the isinstance fallback)."""

    def test_dataclass_validated_through_pydantic(self):
        """Dataclass return values should be validated via Pydantic, not isinstance."""
        from dataclasses import dataclass

        @dataclass
        class Point:
            x: float
            y: float

        strat = CodeActStrategy(config=CodeActConfig())
        # Dict should be coerced to Point by Pydantic
        success, validated = strat._try_validate_return_value(
            {"x": 1.0, "y": 2.0}, Point, "my_method"
        )
        assert success is True
        assert isinstance(validated, Point)
        assert validated.x == 1.0


# === Unit tests for _try_validate_return_value ===


class TestTryValidateNonPydantic:
    """_try_validate_return_value should handle non-Pydantic types via isinstance."""

    def test_accepts_correct_instance(self):
        """Should accept a value that is an instance of the expected non-Pydantic type."""
        strat = CodeActStrategy(config=CodeActConfig())
        obj = CustomResult(value=42, label="test")
        success, validated = strat._try_validate_return_value(obj, CustomResult, "my_method")
        assert success is True
        assert validated is obj

    def test_rejects_wrong_type(self):
        """Should reject a value that is NOT an instance of the expected non-Pydantic type."""
        strat = CodeActStrategy(config=CodeActConfig())
        success, validated = strat._try_validate_return_value("wrong", CustomResult, "my_method")
        assert success is False

    def test_annotated_non_pydantic_accepts_correct_instance(self):
        """Annotated[NonPydantic, desc] should accept a correct instance."""
        strat = CodeActStrategy(config=CodeActConfig())
        obj = CustomResult(value=1, label="ok")
        success, validated = strat._try_validate_return_value(
            obj, Annotated[CustomResult, "The result"], "my_method"
        )
        assert success is True
        assert validated is obj

    def test_annotated_non_pydantic_rejects_wrong_type(self):
        """Annotated[NonPydantic, desc] should reject wrong types (not crash)."""
        strat = CodeActStrategy(config=CodeActConfig())
        success, validated = strat._try_validate_return_value(
            "wrong", Annotated[CustomResult, "The result"], "my_method"
        )
        assert success is False


# === Integration tests: full agent run with CodeAct ===


class TestCodeActNonPydanticReturnType:
    """Full integration: agent methods with non-Pydantic return types should work."""

    @pytest.mark.asyncio
    async def test_custom_class_via_execute_python(self):
        """Agent method returning CustomResult via execute_python should work.

        The LLM constructs the object in code, then returns it via return_result().
        This is the expected workflow for non-Pydantic types.
        """
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call(
                            "obj = CustomResult(value=42, label='answer')\nreturn_result(obj)"
                        )
                    ],
                ),
            ]
        )

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> CustomResult:
                """Compute and return a CustomResult."""
                ...

        agent = TestAgent(llm=fake_llm)
        result = await agent.compute()
        assert isinstance(result, CustomResult)
        assert result.value == 42
        assert result.label == "answer"


# === Tests for Annotated[NonPydanticType, "description"] ===


class TestAnnotatedNonPydanticReturnType:
    """Annotated[NonPydanticType, "description"] should work like bare NonPydanticType.

    The fix in _build_return_result_tool extracts the base type from Annotated before
    calling _is_pydantic_compatible, so Annotated[CustomResult, "desc"] uses Any as the
    schema type (not crash) and preserves the description in the Field.
    """

    def test_annotated_non_pydantic_does_not_crash(self):
        """_build_return_result_tool should not crash for Annotated[NonPydantic, desc]."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(
            Annotated[CustomResult, "The computed result"], "my_method"
        )
        assert tool.name == "return_result"

    def test_annotated_non_pydantic_description_in_tool(self):
        """Tool description should mention the actual (non-Pydantic) type name."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(
            Annotated[CustomResult, "The computed result"], "my_method"
        )
        assert "CustomResult" in tool.description

    def test_annotated_non_pydantic_field_description_preserved(self):
        """The string annotation in Annotated[T, 'desc'] should become the field description."""
        strat = CodeActStrategy(config=CodeActConfig())
        tool = strat._build_return_result_tool(
            Annotated[CustomResult, "The computed result"], "my_method"
        )
        # The field description from Annotated should appear in the tool schema
        schema = tool.parameters_model.model_json_schema()
        result_field = schema.get("properties", {}).get("result", {})
        assert result_field.get("description") == "The computed result"

    def test_annotated_pydantic_type_still_uses_pydantic(self):
        """Annotated[str, 'description'] should still use Pydantic validation (not fallback)."""
        strat = CodeActStrategy(config=CodeActConfig())
        # str is Pydantic-compatible — even when wrapped in Annotated it should stay validated
        tool = strat._build_return_result_tool(Annotated[str, "A string result"], "my_method")
        assert tool.name == "return_result"
        # Schema should use str, not Any
        schema = tool.parameters_model.model_json_schema()
        result_field = schema.get("properties", {}).get("result", {})
        assert result_field.get("type") == "string"

    @pytest.mark.asyncio
    async def test_annotated_non_pydantic_integration(self):
        """Full agent run: Annotated[CustomResult, desc] return type should work end-to-end."""
        fake_llm = FakeLLMClient(
            scripted_responses=[
                _resp(
                    "",
                    tool_calls=[
                        _tool_call(
                            "obj = CustomResult(value=7, label='annotated')\nreturn_result(obj)"
                        )
                    ],
                ),
            ]
        )

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> Annotated[CustomResult, "The computed result"]:
                """Compute and return an annotated CustomResult."""
                ...

        agent = TestAgent(llm=fake_llm)
        result = await agent.compute()
        assert isinstance(result, CustomResult)
        assert result.value == 7
        assert result.label == "annotated"

    @pytest.mark.asyncio
    async def test_return_result_tool_call_with_non_pydantic(self):
        """When LLM calls return_result as a tool with a dict for a non-Pydantic type,
        it should get a helpful error message (not a crash).

        The error should guide the LLM to construct the object in execute_python instead.
        """
        # LLM tries to call return_result as a tool with a dict
        # This can't work because we can't construct a CustomResult from a dict
        # The LLM should get an error and then use execute_python instead
        fake_llm = FakeLLMClient(
            scripted_responses=[
                # First attempt: LLM tries return_result as tool (will fail gracefully)
                _resp(
                    "",
                    tool_calls=[_return_result(result={"value": 42, "label": "answer"})],
                ),
                # Second attempt: LLM uses execute_python (will succeed)
                _resp(
                    "",
                    tool_calls=[
                        _tool_call(
                            "obj = CustomResult(value=42, label='answer')\nreturn_result(obj)"
                        )
                    ],
                ),
            ]
        )

        class TestAgent(Agent, llm=_TEST_LLM):
            @strategy(CodeActStrategy(config=CodeActConfig()))
            async def compute(self) -> CustomResult:
                """Compute and return a CustomResult."""
                ...

        agent = TestAgent(llm=fake_llm)
        result = await agent.compute()
        assert isinstance(result, CustomResult)
        assert result.value == 42
