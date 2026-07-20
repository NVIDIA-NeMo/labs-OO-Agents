# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PredictStrategy."""

import pytest
from pydantic import BaseModel

from nooa import strategy
from nooa.agent import Agent
from nooa.config.strategy_config import PredictConfig
from nooa.errors import GenerationError
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


class TestCreateResponseModel:
    """Tests for _create_response_model method."""

    def _create_strategy(self) -> PredictStrategy:
        """Create strategy instance for testing."""
        return PredictStrategy()

    def test_pydantic_model_used_directly(self):
        """Pydantic models are used directly without wrapping."""
        strategy = self._create_strategy()

        class Person(BaseModel):
            name: str
            age: int

        model = strategy._create_response_model(Person, "test")
        assert model is Person

    def test_basic_dict_uses_root_model(self):
        """Basic dict type uses RootModel for direct value (not wrapped in 'value')."""
        from pydantic import RootModel

        strategy = self._create_strategy()
        model = strategy._create_response_model(dict, "test")

        # Should be a RootModel subclass
        assert issubclass(model, RootModel)
        # Should have 'root' field, not 'value'
        assert "root" in model.model_fields

    def test_basic_list_wrapped_in_object(self):
        """Basic list type is wrapped under a 'value' field, not a RootModel.

        A top-level array schema (RootModel[list]) is rejected by the Responses API,
        so list is wrapped so the root schema stays type: object (issue 232).
        """
        from pydantic import RootModel

        strategy = self._create_strategy()
        model = strategy._create_response_model(list, "test")

        # Should NOT be a RootModel subclass (that would yield an array-rooted schema)
        assert not issubclass(model, RootModel)
        # Should have 'value' field, and an object-rooted schema
        assert "value" in model.model_fields
        assert model.model_json_schema()["type"] == "object"

    def test_basic_str_wrapped(self):
        """Basic str type gets wrapped in model with 'value' field."""
        strategy = self._create_strategy()
        model = strategy._create_response_model(str, "test")

        assert hasattr(model, "model_fields")
        assert "value" in model.model_fields

    def test_none_type_raises_error(self):
        """None return type raises GenerationError."""
        strategy = self._create_strategy()

        with pytest.raises(GenerationError) as exc_info:
            strategy._create_response_model(type(None), "test_method")

        assert "return type None" in str(exc_info.value)

    def test_optional_unwraps_inner_type(self):
        """Optional[X] unwraps to use X."""
        strategy = self._create_strategy()

        class Person(BaseModel):
            name: str

        model = strategy._create_response_model(Person | None, "test")
        # Should unwrap to Person
        assert model is Person

    def test_optional_basic_type_wrapped(self):
        """Optional[str] unwraps and wraps str."""
        strategy = self._create_strategy()
        model = strategy._create_response_model(str | None, "test")

        # Should create wrapper model for str
        assert hasattr(model, "model_fields")
        assert "value" in model.model_fields

    def test_model_name_generated(self):
        """Generated model name is based on method name."""
        strategy = self._create_strategy()
        model = strategy._create_response_model(dict, "analyze_data")

        # Model name should be title-cased method name + "Response"
        # title() capitalizes each word, then underscores are removed
        assert model.__name__ == "AnalyzeDataResponse"


class TestValidateResponse:
    """Tests for _validate_response method."""

    def _create_strategy(self) -> PredictStrategy:
        """Create strategy instance for testing."""
        return PredictStrategy()

    def test_validate_pydantic_returns_model(self):
        """Pydantic model validation returns validated instance."""
        strategy = self._create_strategy()

        class Person(BaseModel):
            name: str
            age: int

        data = {"name": "Alice", "age": 30}
        result = strategy._validate_response(data, Person, Person)

        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_validate_wrapped_unwraps_value(self):
        """Wrapped basic types are unwrapped to return original type."""
        strategy = self._create_strategy()

        # Create wrapper model like _create_response_model does
        from pydantic import create_model

        wrapper = create_model("TestResponse", value=(str, ...))

        data = {"value": "hello"}
        result = strategy._validate_response(data, wrapper, str)

        assert result == "hello"
        assert isinstance(result, str)

    def test_validate_dict_root_model_unwraps(self):
        """Dict RootModel is unwrapped correctly to raw dict."""
        strategy = self._create_strategy()

        from pydantic import RootModel

        class DictModel(RootModel[dict]):
            pass

        # Data is the dict directly (not wrapped in "value")
        data = {"key": "val"}
        result = strategy._validate_response(data, DictModel, dict)

        assert result == {"key": "val"}


class TestPredictStrategyIntegration:
    """Integration tests for PredictStrategy."""

    @pytest.mark.asyncio
    async def test_basic_pydantic_model_generation(self):
        """Test basic structured output generation with Pydantic model."""

        class Person(BaseModel):
            name: str
            age: int

        # LLM returns JSON matching Person schema
        content = '{"name": "Alice", "age": 30}'
        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=content,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content},
                    reasoning=None,
                    usage=None,
                ),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PredictStrategy())
            async def get_person(self) -> Person:
                """Return a person."""
                ...

        agent_instance = TestAgent()
        result = await agent_instance.get_person()

        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_basic_type_generation(self):
        """Test structured output with basic return type (str)."""

        # LLM returns JSON with "value" field (wrapped by strategy)
        content = '{"value": "Hello, World!"}'
        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=content,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content},
                    reasoning=None,
                    usage=None,
                ),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PredictStrategy())
            async def get_greeting(self) -> str:
                """Return a greeting."""
                ...

        agent_instance = TestAgent()
        result = await agent_instance.get_greeting()

        assert result == "Hello, World!"
        assert isinstance(result, str)
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_validation_retry(self):
        """Test that validation errors trigger retry."""

        class Person(BaseModel):
            name: str
            age: int

        # First response: missing 'age' field (validation error)
        # Second response: complete and valid
        content1 = '{"name": "Bob"}'  # Missing age
        content2 = '{"name": "Bob", "age": 25}'  # Complete
        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=content1,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content1},
                    reasoning=None,
                    usage=None,
                ),
                LLMResponse(
                    raw_response=None,
                    content=content2,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content2},
                    reasoning=None,
                    usage=None,
                ),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PredictStrategy())
            async def get_person(self) -> Person:
                """Return a person."""
                ...

        agent_instance = TestAgent()
        result = await agent_instance.get_person()

        assert isinstance(result, Person)
        assert result.name == "Bob"
        assert result.age == 25
        assert fake_llm.call_count == 2  # Retry after validation error

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self):
        """Test that max retries raises GenerationError."""

        class Person(BaseModel):
            name: str
            age: int

        # Always return invalid data (missing age)
        content = '{"name": "Invalid"}'
        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=content,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content},
                    reasoning=None,
                    usage=None,
                )
                for _ in range(5)  # More than max_retries (default 3)
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
            async def get_person(self) -> Person:
                """Return a person."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.get_person()

        assert "failed after 3 attempts" in str(exc_info.value)
        assert fake_llm.call_count == 3  # max_retries = 3

    @pytest.mark.asyncio
    async def test_method_arguments_available(self):
        """Test that method arguments are passed in prompt context."""

        class Greeting(BaseModel):
            message: str

        # LLM should use the 'name' argument in context
        content = '{"message": "Hello, World!"}'
        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=content,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content},
                    reasoning=None,
                    usage=None,
                ),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PredictStrategy())
            async def greet(self, name: str) -> Greeting:
                """Generate a greeting for the given name."""
                ...

        agent_instance = TestAgent()
        result = await agent_instance.greet("World")

        assert isinstance(result, Greeting)
        assert "World" in result.message or "Hello" in result.message
        assert fake_llm.call_count == 1

    @pytest.mark.asyncio
    async def test_no_return_type_raises_error(self):
        """Test that missing return type annotation raises GenerationError."""

        fake_llm = FakeLLMClient(scripted_responses=[])

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PredictStrategy())
            async def get_data(self):
                """Get some data."""
                ...

        agent_instance = TestAgent()

        with pytest.raises(GenerationError) as exc_info:
            await agent_instance.get_data()

        assert "no return type annotation" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_optional_type(self):
        """Test structured output with Optional return type."""

        class Person(BaseModel):
            name: str
            age: int

        # LLM returns valid Person (Optional unwraps to Person)
        content = '{"name": "Diana", "age": 28}'
        fake_llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=content,
                    tool_calls=[],
                    finish_reason="stop",
                    assistant_message={"role": "assistant", "content": content},
                    reasoning=None,
                    usage=None,
                ),
            ]
        )

        class TestAgent(Agent, llm=fake_llm):
            @strategy(PredictStrategy())
            async def find_person(self) -> Person | None:
                """Find a person (might return None)."""
                ...

        agent_instance = TestAgent()
        result = await agent_instance.find_person()

        assert isinstance(result, Person)
        assert result.name == "Diana"
        assert result.age == 28


class TestStripXMLWrapper:
    """Tests for _strip_xml_wrapper method."""

    def _create_strategy(self) -> PredictStrategy:
        """Create strategy instance for testing."""
        return PredictStrategy()

    def test_strip_xml_with_attributes(self):
        """Strip XML wrapper with attributes."""
        strategy = self._create_strategy()

        content = '<assistant_message expr="self.event_manager.values()[1].content">{"name":"Alice","age":28}</assistant_message>'
        cleaned = strategy._strip_xml_wrapper(content)

        assert cleaned == '{"name":"Alice","age":28}'

    def test_strip_simple_xml(self):
        """Strip simple XML wrapper without attributes."""
        strategy = self._create_strategy()

        content = '<result>{"status":"ok"}</result>'
        cleaned = strategy._strip_xml_wrapper(content)

        assert cleaned == '{"status":"ok"}'

    def test_no_xml_wrapper_returns_original(self):
        """Content without XML wrapper is returned as-is."""
        strategy = self._create_strategy()

        content = '{"name":"Bob","age":35}'
        cleaned = strategy._strip_xml_wrapper(content)

        assert cleaned == content

    def test_strip_xml_with_multiline_json(self):
        """Strip XML wrapper from multiline JSON content."""
        strategy = self._create_strategy()

        content = """<assistant_message expr="test">
{
  "name": "Carol",
  "age": 42,
  "email": "carol@example.com"
}
</assistant_message>"""

        cleaned = strategy._strip_xml_wrapper(content)

        expected = """
{
  "name": "Carol",
  "age": 42,
  "email": "carol@example.com"
}
""".strip()

        assert cleaned == expected

    def test_strip_xml_with_whitespace(self):
        """Strip XML wrapper handles leading/trailing whitespace."""
        strategy = self._create_strategy()

        content = '  <data attr="value">  {"key":"value"}  </data>  '
        cleaned = strategy._strip_xml_wrapper(content)

        assert cleaned == '{"key":"value"}'

    def test_strip_xml_handles_none_content(self):
        """Strip XML wrapper handles None content (e.g. from API tool-call-only response)."""
        strategy = self._create_strategy()

        # When LLM returns None for message.content, downstream can pass None into _strip_xml_wrapper.
        # Must not raise AttributeError: 'NoneType' object has no attribute 'strip'.
        cleaned = strategy._strip_xml_wrapper(None)

        assert cleaned == ""
