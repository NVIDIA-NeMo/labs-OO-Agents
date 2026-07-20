# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for agentdoc integration with nooa.

These tests verify that agentdoc functions (doc, brief, methods, variables)
work correctly with nooa Agent instances.

Note: The output format uses Python class syntax per the agentdoc design doc.
"""

from nooa import Agent
from nooa.agentdoc import doc
from nooa.agentdoc.introspect import methods, variables
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()

# Test fixtures - simple agent classes for testing


class SimpleAgent(Agent, llm=_TEST_LLM):
    """A simple test agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Type annotations required for agentdoc extraction
        self.counter: int = 0
        self.items: list[str] = ["a", "b", "c"]
        self.data: dict[str, str] = {"key": "value"}

    async def process(self, x: int) -> int:
        """Process a number and return doubled value."""
        ...

    def get_count(self) -> int:
        """Return the current counter value."""
        return self.counter


class Calculator:
    """A calculator tool."""

    def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


class AgentWithTool(Agent, llm=_TEST_LLM):
    """Agent with a tool."""

    calculator = Calculator()

    async def calculate(self, x: int, y: int) -> int:
        """Do a calculation."""
        ...


class TestAgentDocProtocol:
    """Tests for Agent documentation via extractor registry."""

    def test_agent_doc_is_callable(self):
        """Test that doc() works on Agent instances."""
        from nooa.agentdoc import doc

        agent_instance = SimpleAgent()
        # doc() should return useful documentation for agents
        result = doc(agent_instance)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "SimpleAgent" in result

    def test_agent_uses_type_info_protocol(self):
        """Test that Agent class uses __type_info__ protocol for documentation."""
        from nooa.agentdoc.ext import has_type_info

        # Agent implements __type_info__ protocol directly
        assert has_type_info(SimpleAgent)

        agent_instance = SimpleAgent()
        # doc() should work via the protocol
        result = doc(agent_instance)
        assert "SimpleAgent" in result

    def test_doc_returns_python_class_syntax(self):
        """Test that doc() returns Python class syntax."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        # Should have Python class syntax (per design doc)
        assert "class SimpleAgent" in output

    def test_agent_doc_includes_methods(self):
        """Test agent documentation includes methods."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        assert "process" in output
        assert "get_count" in output

    def test_agent_doc_includes_variables(self):
        """Test agent documentation includes variables."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        assert "counter" in output
        assert "items" in output
        assert "data" in output

    def test_agent_doc_includes_class_docstring(self):
        """Test agent documentation includes class docstring."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        assert "simple test agent" in output.lower()


class TestChildAgents:
    """Tests for child agent rendering."""

    def test_child_agent_in_doc(self):
        """Test that child agents as class attributes can be discovered."""

        class WorkerAgent(Agent, llm=_TEST_LLM):
            """A worker that processes items."""

            async def process(self, item: str) -> str:
                """Process an item."""
                ...

        class CoordinatorAgent(Agent, llm=_TEST_LLM):
            """Coordinates multiple workers."""

            async def coordinate(self, items: list[str]) -> list[str]:
                """Coordinate processing."""
                ...

        # Assign child agent as class attribute after definition
        CoordinatorAgent.WorkerAgent = WorkerAgent  # type: ignore[attr-defined]

        coordinator = CoordinatorAgent()

        # Child agent class is accessible as attribute
        assert hasattr(coordinator, "WorkerAgent")
        assert coordinator.WorkerAgent is WorkerAgent  # type: ignore[attr-defined]

        # Note: Class-level type attributes may not show in instance doc()
        # but are accessible for LLM to use. This is acceptable behavior
        # per the design - doc(self) shows instance state, not class structure.
        output = doc(coordinator)
        assert "CoordinatorAgent" in output
        assert "coordinate" in output

    def test_no_child_agents_section_when_none(self):
        """Test that Child Agents section is omitted when there are none."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        # Should NOT have Child Agents section (old markdown format)
        assert "## Child Agents" not in output


class TestAgentDocFunctions:
    """Tests for agentdoc functions with agent instances."""

    def test_doc_function(self):
        """Test doc() function with agent."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        assert isinstance(output, str)
        assert len(output) > 0
        # Should have Python class syntax
        assert "class SimpleAgent" in output

    def test_methods_function(self):
        """Test methods() function with agent."""
        agent_instance = SimpleAgent()
        output = methods(agent_instance)

        assert "process" in output
        assert "get_count" in output

    def test_variables_function(self):
        """Test variables() function with agent."""
        agent_instance = SimpleAgent()
        output = variables(agent_instance)

        assert "counter" in output
        assert "items" in output
        assert "data" in output


class TestAgentWithTools:
    """Tests for agents with tool instances."""

    def test_agent_with_tool_shows_in_variables(self):
        """Test that tool instances appear in variables."""
        agent_instance = AgentWithTool()
        output = variables(agent_instance)

        # Calculator tool should be visible
        assert "calculator" in output.lower() or "Calculator" in output

    def test_tool_methods_accessible(self):
        """Test that we can drill down into tool methods."""
        agent_instance = AgentWithTool()

        # Get the calculator
        calc = agent_instance.calculator

        # Should be able to get methods
        output = methods(calc)
        assert "add" in output
        assert "multiply" in output


class TestAgentDocHidesInternals:
    """Tests that framework internals are hidden."""

    def test_hides_private_attributes(self):
        """Test that private attributes are hidden."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        # Should NOT show private attributes
        assert "_agent_id" not in output
        assert "_llm" not in output

    def test_hides_framework_attributes(self):
        """Test that framework attributes are hidden."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        # Should NOT show framework internals (they have _ prefix)
        assert "_instance_blocks" not in output
        assert "_instance_event_blocks" not in output


class TestDrillDownHints:
    """Tests for drill-down hints in documentation."""

    def test_hints_for_complex_objects(self):
        """Test that complex objects show drill-down hints."""
        agent_instance = AgentWithTool()
        output = variables(agent_instance)

        # Calculator should have a hint (type shown)
        assert "Calculator" in output

    def test_can_follow_hints(self):
        """Test that drill-down hints actually work."""
        agent_instance = AgentWithTool()

        # Get hint for calculator
        calc = agent_instance.calculator

        # Should be able to get methods as hinted
        output = methods(calc)
        assert "add" in output
        assert "multiply" in output


class TestAgentDocFrameworkAPIs:
    """Tests for framework API visibility in doc().

    context and events are always present on every Agent instance but hidden from
    the LLM by default. Subclasses opt in by calling spec(self, "context", hidden=False)
    in their __init__. Both doc(Agent) and doc(agent_instance) hide them unless opted in.
    """

    def test_context_hidden_in_doc_instance(self):
        """Test that self.context is hidden in doc(agent_instance) by default."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)
        assert "context" not in output, (
            "self.context should be hidden from doc(self) by default — "
            "subclasses must call spec(self, 'context', hidden=False) to expose it"
        )

    def test_context_hidden_in_doc_type(self):
        """Test that self.context is NOT in doc(AgentClass) — hidden at class level."""
        output = doc(SimpleAgent)
        assert "context" not in output, (
            "self.context should be hidden from doc(AgentClass) — "
            "the class annotation is Annotated[ContextApi, hidden]"
        )

    def test_context_always_present_on_agent(self):
        """Test that self.context is always present on agent."""
        from nooa.runtime.context import ContextApi

        agent_instance = SimpleAgent()
        assert hasattr(agent_instance, "context"), "self.context should always be present"
        assert isinstance(agent_instance.context, ContextApi)

    def test_events_hidden_in_doc_instance(self):
        """Test that self.events is hidden in doc(agent_instance) by default."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)
        assert "events" not in output, (
            "self.events should be hidden from doc(self) by default — "
            "subclasses must call spec(self, 'events', hidden=False) to expose it"
        )

    def test_events_hidden_in_doc_type(self):
        """Test that self.events is NOT in doc(AgentClass) — hidden at class level."""
        output = doc(SimpleAgent)
        assert "events" not in output, (
            "self.events should be hidden from doc(AgentClass) — "
            "the class annotation is Annotated[EventsApi, hidden]"
        )

    def test_events_always_present_on_agent(self):
        """Test that self.events is always present on agent."""
        from nooa.runtime.events import EventsApi

        agent_instance = SimpleAgent()
        assert hasattr(agent_instance, "events"), "self.events should always be present"
        assert isinstance(agent_instance.events, EventsApi)

    def test_runtime_hidden_in_doc(self):
        """Test that self.runtime is NOT visible in doc(agent_instance).

        runtime is an internal framework attribute, not a user-facing API.
        """
        agent_instance = SimpleAgent()
        output = doc(agent_instance)
        for line in output.split("\n"):
            field_name = line.strip().split(":")[0].split("=")[0].strip()
            assert field_name != "runtime", f"runtime field should be hidden, found in: {line}"

    def test_agent_id_hidden_in_doc(self):
        """Test that agent_id is NOT visible in doc(agent_instance).

        agent_id is a @hidden @property — public read-only for devs but hidden from LLM.
        """
        agent_instance = SimpleAgent()
        output = doc(agent_instance)
        assert "agent_id" not in output, (
            "agent_id should be hidden from doc(self) — it is a @hidden @property"
        )

    def test_event_manager_hidden_in_doc(self):
        """Test that self.event_manager is NOT visible in doc(agent_instance).

        event_manager is an internal framework attribute.
        """
        agent_instance = SimpleAgent()
        output = doc(agent_instance)
        assert "event_manager" not in output

    def test_context_alongside_user_fields(self):
        """Test that user-defined fields appear (context is hidden by default)."""
        agent_instance = SimpleAgent()
        output = doc(agent_instance)

        # User fields should be present
        assert "counter" in output
        assert "items" in output
        assert "data" in output


class TestAgentDocFieldExtraction:
    """Tests for field extraction from Agent classes.

    Verifies that _extract_fields / extract_type_info finds fields
    correctly for different Agent class patterns.
    """

    def test_extract_type_info_finds_subclass_init_fields(self):
        """Test that fields from subclass __init__ are found."""
        from nooa.agentdoc.ext import extract_type_info

        info = extract_type_info(SimpleAgent, _skip_protocol=True)
        field_names = {f.name for f in info.fields}

        assert "counter" in field_names
        assert "items" in field_names
        assert "data" in field_names

    def test_type_info_hides_true_internals(self):
        """Test that true framework internals remain hidden."""
        info = SimpleAgent.__type_info__()
        field_names = {f.name for f in info.fields}

        # These should remain hidden
        assert "runtime" not in field_names
        assert "event_manager" not in field_names
        assert "_llm" not in field_names
        assert "_agent_id" not in field_names

    def test_agent_with_tool_shows_tool(self):
        """Test that tool-bearing agents show the tool in doc output."""
        agent_instance = AgentWithTool()
        output = doc(agent_instance)

        # Tool should be visible
        assert "calculator" in output.lower() or "Calculator" in output


def test_hidden_method_excluded_from_type_info():
    """@hidden methods should not appear in __type_info__()."""
    from unittest.mock import MagicMock

    from nooa import Agent, hidden

    llm = MagicMock()
    llm.model = "test"

    class TestAgent(Agent, llm=llm):
        async def public_method(self):
            """Public."""
            ...

        @hidden
        async def secret_method(self):
            """Hidden."""
            ...

    info = TestAgent.__type_info__()
    method_names = [m.name for m in info.methods]
    assert any("public_method" in n for n in method_names)
    assert not any("secret_method" in n for n in method_names)


def test_underscore_method_hidden_by_default():
    """_private methods are hidden by default; @spec(hidden=False) opts them back in."""
    from unittest.mock import MagicMock

    from nooa import Agent
    from nooa.agentdoc import spec

    llm = MagicMock()
    llm.model = "test"

    class TestAgent(Agent, llm=llm):
        async def _private_method(self):
            """Hidden by default."""
            ...

        @spec(hidden=False)  # type: ignore[misc]
        async def _shown_method(self):
            """Explicitly shown."""
            ...

    info = TestAgent.__type_info__()
    method_names = [m.name for m in info.methods]
    assert not any("_private_method" in n for n in method_names), "_private_method should be hidden"
    assert any("_shown_method" in n for n in method_names), "_shown_method should be visible"


def test_hidden_field_excluded_from_instance_values():
    """Annotated[T, hidden] fields should not appear in __instance_values__()."""
    from typing import Annotated
    from unittest.mock import MagicMock

    from nooa import Agent, hidden

    llm = MagicMock()
    llm.model = "test"

    class TestAgent(Agent, llm=llm):
        public_val: str = "visible"
        secret_val: Annotated[str, hidden] = "hidden"

    agent = TestAgent()
    agent.public_val = "visible"
    agent.secret_val = "hidden"
    values = agent.__instance_values__()
    assert "public_val" in values
    assert "secret_val" not in values


def test_underscore_field_visible_without_hidden():
    """_private fields should now be VISIBLE (no underscore convention)."""
    from unittest.mock import MagicMock

    from nooa import Agent

    llm = MagicMock()
    llm.model = "test"

    class TestAgent(Agent, llm=llm):
        pass

    agent = TestAgent()
    agent._custom = "was private"  # type: ignore[reportAttributeAccessIssue]
    values = agent.__instance_values__()
    assert "_custom" in values


def test_framework_attrs_hidden_via_annotation():
    """runtime, _event_manager, event_query, render_config should be hidden via Annotated[T, hidden]."""
    from nooa import Agent
    from nooa.agentdoc.visibility import is_hidden_field

    assert is_hidden_field(Agent, "runtime") is True
    assert is_hidden_field(Agent, "event_manager") is True
    assert is_hidden_field(Agent, "event_query") is True
    assert is_hidden_field(Agent, "render_config") is True
    assert is_hidden_field(Agent, "context") is True
    assert is_hidden_field(Agent, "events") is True


class TestAgentDocInExpressions:
    """Tests for using agentdoc in expressions (context blocks)."""

    async def test_doc_available_in_runtime_expressions(self):
        """Test that doc() is available in runtime.evaluate_expression()."""
        agent_instance = SimpleAgent()

        # This is how context blocks evaluate expressions
        result = await agent_instance.runtime.evaluate_expression("doc(self)")

        assert isinstance(result, str)
        # Should have Python class syntax
        assert "class SimpleAgent" in result

    async def test_doc_concise_available_in_runtime_expressions(self):
        """Test that doc(concise=True) is available in runtime.evaluate_expression()."""
        agent_instance = SimpleAgent()

        result = await agent_instance.runtime.evaluate_expression("doc(self, concise=True)")

        assert isinstance(result, str)
        assert "SimpleAgent" in result

    async def test_methods_available_in_runtime_expressions(self):
        """Test that methods() is available in runtime.evaluate_expression()."""
        agent_instance = SimpleAgent()

        result = await agent_instance.runtime.evaluate_expression("methods(self)")

        assert isinstance(result, str)
        assert "process" in result

    async def test_variables_available_in_runtime_expressions(self):
        """Test that variables() is available in runtime.evaluate_expression()."""
        agent_instance = SimpleAgent()

        result = await agent_instance.runtime.evaluate_expression("variables(self)")

        assert isinstance(result, str)
        assert "counter" in result
