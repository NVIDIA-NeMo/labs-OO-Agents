# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for runtime expression evaluation and variable expansion."""

import pytest

from nooa import Agent, strategy
from nooa.unifiedllm import FakeLLMClient


class EvalAgent(Agent, llm=FakeLLMClient()):
    """Agent with state for evaluation tests."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tools = ["read", "write", "execute"]
        self.config = {"max_retries": 3, "timeout": 30}
        self.items = [1, 2, 3, 4, 5]


class TestEvaluateExpression:
    """Test runtime.evaluate_expression() method."""

    @pytest.mark.asyncio
    async def test_basic_expression(self):
        """Test evaluating a basic expression."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression("2 + 2")
        assert result == 4

    @pytest.mark.asyncio
    async def test_agent_state_access(self):
        """Test accessing agent state via self."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression("len(self.tools)")
        assert result == 3

    @pytest.mark.asyncio
    async def test_nested_attribute_access(self):
        """Test accessing nested agent attributes."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression("self.config['max_retries']")
        assert result == 3

    @pytest.mark.asyncio
    async def test_list_operations(self):
        """Test list operations on agent state."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression("sum(self.items)")
        assert result == 15

    @pytest.mark.asyncio
    async def test_extra_context(self):
        """Test evaluation with extra context (like method args)."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression(
            "filename.upper()", extra_context={"filename": "test.py"}
        )
        assert result == "TEST.PY"

    @pytest.mark.asyncio
    async def test_extra_context_with_agent_state(self):
        """Test combining extra context with agent state."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression(
            "len(self.tools) + count", extra_context={"count": 10}
        )
        assert result == 13

    @pytest.mark.asyncio
    async def test_error_mode_show(self):
        """Test error_mode='show' returns error string."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression(
            "undefined_variable", error_mode="show"
        )
        assert "ERROR" in result
        assert "undefined_variable" in result or "not defined" in result

    @pytest.mark.asyncio
    async def test_error_mode_silent(self):
        """Test error_mode='silent' returns None."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression(
            "undefined_variable", error_mode="silent"
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_error_mode_raise(self):
        """Test error_mode='raise' raises exception."""
        agent_inst = EvalAgent()
        with pytest.raises(NameError):
            await agent_inst.runtime.evaluate_expression("undefined_variable", error_mode="raise")

    @pytest.mark.asyncio
    async def test_safe_builtins_available(self):
        """Test that safe builtins are available."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.evaluate_expression("sum([1, 2, 3])")
        assert result == 6

        result = await agent_inst.runtime.evaluate_expression("len('hello')")
        assert result == 5

        result = await agent_inst.runtime.evaluate_expression("max([5, 2, 8, 1])")
        assert result == 8

    @pytest.mark.asyncio
    async def test_repl_locals_integration(self):
        """Test that REPL locals are included if available."""
        agent_inst = EvalAgent()

        # Simulate REPL with _data attribute
        class MockREPL:
            _data = {"result": "success", "count": 42}

        agent_inst.repl = MockREPL()

        result = await agent_inst.runtime.evaluate_expression("result")
        assert result == "success"

        result = await agent_inst.runtime.evaluate_expression("count * 2")
        assert result == 84

    @pytest.mark.asyncio
    async def test_last_execution_result(self):
        """Test that _last_execution_result is available as 'result'."""
        agent_inst = EvalAgent()
        agent_inst._last_execution_result = {"status": "done", "value": 100}

        result = await agent_inst.runtime.evaluate_expression("result['status']")
        assert result == "done"

        result = await agent_inst.runtime.evaluate_expression("result['value'] * 2")
        assert result == 200


class TestDocIncludesCurrentMethod:
    """Test that doc(self) includes the current method during generation."""

    @pytest.mark.asyncio
    async def test_doc_includes_current_method(self):
        """Test doc(self) includes the current method being generated.

        The current method is shown in the documentation to preserve stable
        prefixes for KV caching. Hiding it caused cache misses because the
        document changed on every call.
        """
        from nooa.strategies.pure_python import PurePythonStrategy

        captured_doc = None

        class CapturingStrategy(PurePythonStrategy):
            async def execute(self, runtime, call):
                nonlocal captured_doc
                result = await runtime.evaluate_expression("doc(self)")
                captured_doc = result
                return "done"

        class DocTestAgent(Agent, llm=FakeLLMClient()):
            """A test agent."""

            @strategy(CapturingStrategy())
            async def calculate(self, x: int) -> int:
                """Calculate something."""
                ...

            async def other_method(self) -> str:
                """Another method that should be visible."""
                ...

        agent_inst = DocTestAgent()
        await agent_inst.calculate(5)

        assert captured_doc is not None
        # The current method 'calculate' should be visible for KV cache stability
        assert "def calculate" in captured_doc
        # Other methods should also be visible
        assert "other_method" in captured_doc

    @pytest.mark.asyncio
    async def test_doc_is_stable_across_different_methods(self):
        """Test doc(self) produces identical output regardless of which method is running.

        This is the core KV cache property: the document must not change between
        calls just because a different method is executing.
        """
        from nooa.strategies.pure_python import PurePythonStrategy

        docs: list[str] = []

        class CapturingStrategy(PurePythonStrategy):
            async def execute(self, runtime, call):
                docs.append(await runtime.evaluate_expression("doc(self)"))
                return "done"

        capturing = CapturingStrategy()

        class StableDocAgent(Agent, llm=FakeLLMClient()):
            """A test agent."""

            @strategy(capturing)
            async def method_a(self) -> str:
                """First method."""
                ...

            @strategy(capturing)
            async def method_b(self) -> str:
                """Second method."""
                ...

        agent_inst = StableDocAgent()
        await agent_inst.method_a()
        await agent_inst.method_b()

        assert len(docs) == 2
        # Both calls must produce exactly the same document for KV cache stability
        assert docs[0] == docs[1]

    @pytest.mark.asyncio
    async def test_doc_shows_sole_method(self):
        """Test doc(self) shows the method even when it is the only one on the agent."""
        from nooa.strategies.pure_python import PurePythonStrategy

        captured_doc = None

        class CapturingStrategy(PurePythonStrategy):
            async def execute(self, runtime, call):
                nonlocal captured_doc
                captured_doc = await runtime.evaluate_expression("doc(self)")
                return "done"

        class SingleMethodAgent(Agent, llm=FakeLLMClient()):
            """An agent with only one method."""

            @strategy(CapturingStrategy())
            async def only_method(self) -> str:
                """The sole method."""
                ...

        agent_inst = SingleMethodAgent()
        await agent_inst.only_method()

        assert captured_doc is not None
        assert "def only_method" in captured_doc


class TestExpandVariables:
    """Test runtime.expand_variables() method."""

    @pytest.mark.asyncio
    async def test_basic_expansion(self):
        """Test basic variable expansion."""
        agent_inst = EvalAgent()
        text = "I have {len(self.tools)} tools"
        result = await agent_inst.runtime.expand_variables(text)
        assert result == "I have 3 tools"

    @pytest.mark.asyncio
    async def test_multiple_placeholders(self):
        """Test multiple placeholders in one string."""
        agent_inst = EvalAgent()
        text = "Tools: {len(self.tools)}, Items: {len(self.items)}"
        result = await agent_inst.runtime.expand_variables(text)
        assert result == "Tools: 3, Items: 5"

    @pytest.mark.asyncio
    async def test_complex_expressions(self):
        """Test complex expressions in placeholders."""
        agent_inst = EvalAgent()
        text = "Tool list: {', '.join(self.tools)}"
        result = await agent_inst.runtime.expand_variables(text)
        assert result == "Tool list: read, write, execute"

    @pytest.mark.asyncio
    async def test_format_specs(self):
        """Test format specifications in placeholders."""
        agent_inst = EvalAgent()
        text = "Average: {sum(self.items) / len(self.items):.2f}"
        result = await agent_inst.runtime.expand_variables(text)
        assert result == "Average: 3.00"

    @pytest.mark.asyncio
    async def test_escaped_braces(self):
        """Test that escaped braces are preserved."""
        agent_inst = EvalAgent()
        text = "Use {{variable}} syntax, got {len(self.tools)} tools"
        result = await agent_inst.runtime.expand_variables(text)
        assert result == "Use {variable} syntax, got 3 tools"

    @pytest.mark.asyncio
    async def test_nested_expressions_not_supported(self):
        """Test that nested braces don't break (just don't expand)."""
        agent_inst = EvalAgent()
        # This should either expand partially or show error
        text = "Data: {{'key': len(self.tools)}}"
        result = await agent_inst.runtime.expand_variables(text)
        # Should at least not crash
        assert "Data:" in result

    @pytest.mark.asyncio
    async def test_extra_context_in_expansion(self):
        """Test expand_variables with extra_context."""
        agent_inst = EvalAgent()
        text = "Processing {filename} with {len(self.tools)} tools"
        result = await agent_inst.runtime.expand_variables(
            text, extra_context={"filename": "test.py"}
        )
        assert result == "Processing test.py with 3 tools"

    @pytest.mark.asyncio
    async def test_error_mode_show_in_expansion(self):
        """Test that errors are shown in expansion."""
        agent_inst = EvalAgent()
        text = "Value: {undefined_var}"
        result = await agent_inst.runtime.expand_variables(text, error_mode="show")
        assert "ERROR" in result

    @pytest.mark.asyncio
    async def test_error_mode_silent_in_expansion(self):
        """Test that errors are silent in expansion."""
        agent_inst = EvalAgent()
        text = "Value: {undefined_var}"
        result = await agent_inst.runtime.expand_variables(text, error_mode="silent")
        # Should keep original placeholder when silent
        assert "{undefined_var}" in result

    @pytest.mark.asyncio
    async def test_no_placeholders(self):
        """Test that text without placeholders is unchanged."""
        agent_inst = EvalAgent()
        text = "This is plain text with no variables"
        result = await agent_inst.runtime.expand_variables(text)
        assert result == text

    @pytest.mark.asyncio
    async def test_empty_text(self):
        """Test that empty text is handled."""
        agent_inst = EvalAgent()
        result = await agent_inst.runtime.expand_variables("")
        assert result == ""

    @pytest.mark.asyncio
    async def test_subprocess_completed_process_handling(self):
        """Test that CompletedProcess objects are handled specially."""
        import subprocess

        agent_inst = EvalAgent()

        # Create a mock CompletedProcess
        proc = subprocess.CompletedProcess(
            args=["echo", "hello"], returncode=0, stdout="hello world\n", stderr=""
        )

        agent_inst._last_execution_result = proc

        # Should extract stdout
        text = "Output: {result}"
        result = await agent_inst.runtime.expand_variables(text)
        assert "Output: hello world" in result


class TestDocstringExpansion:
    """Test docstring expansion in prompts (via runtime)."""

    @pytest.mark.asyncio
    async def test_method_args_expansion(self):
        """Test that method arguments are expanded in docstrings."""
        agent_inst = EvalAgent()

        # Simulate what prompts.py does
        docstring = "Process file {filename} with options {options}"
        method_args = {"filename": "test.py", "options": {"verbose": True}}

        result = await agent_inst.runtime.expand_variables(
            docstring, extra_context=method_args, error_mode="silent"
        )

        assert "test.py" in result
        assert "verbose" in result.lower() or "true" in result.lower()

    @pytest.mark.asyncio
    async def test_silent_error_for_missing_args(self):
        """Test that missing method args don't break docstring (silent mode)."""
        agent_inst = EvalAgent()

        docstring = "Process file {missing_arg}"
        result = await agent_inst.runtime.expand_variables(docstring, error_mode="silent")

        # Should keep original placeholder
        assert "{missing_arg}" in result

    @pytest.mark.asyncio
    async def test_combining_self_and_method_args(self):
        """Test combining agent state and method args in docstring."""
        agent_inst = EvalAgent()

        docstring = "Use {len(self.tools)} tools to process {filename}"
        method_args = {"filename": "data.json"}

        result = await agent_inst.runtime.expand_variables(
            docstring, extra_context=method_args, error_mode="silent"
        )

        assert "3 tools" in result
        assert "data.json" in result


class TestExpandVariablesEdgeCases:
    """Test edge cases from old expand_docstring tests."""

    @pytest.mark.asyncio
    async def test_escape_braces_only(self):
        """Test only escaped braces with no placeholders."""
        agent_inst = EvalAgent()
        text = "Dict: {{a: 1, b: 2}}"
        result = await agent_inst.runtime.expand_variables(text)
        assert result == "Dict: {a: 1, b: 2}"

    @pytest.mark.asyncio
    async def test_slice_not_confused_with_format_spec(self):
        """Test that slices [0:2] aren't confused with format specs."""
        agent_inst = EvalAgent()
        text = "Subset: {items[0:2]}"
        result = await agent_inst.runtime.expand_variables(
            text, extra_context={"items": ["a", "b", "c"]}
        )
        assert result == "Subset: ['a', 'b']"

    @pytest.mark.asyncio
    async def test_none_value_expansion(self):
        """Test that None values become 'None' string."""
        agent_inst = EvalAgent()
        text = "Value: {value}"
        result = await agent_inst.runtime.expand_variables(text, extra_context={"value": None})
        assert result == "Value: None"

    @pytest.mark.asyncio
    async def test_repr_function(self):
        """Test that repr() function works."""
        agent_inst = EvalAgent()
        text = "Debug: {repr(value)}"
        result = await agent_inst.runtime.expand_variables(
            text, extra_context={"value": "test\nstring"}
        )
        assert result == "Debug: 'test\\nstring'"

    @pytest.mark.asyncio
    async def test_format_spec_with_self_attribute(self):
        """Test format specs work with self.* attributes."""
        agent_inst = EvalAgent()
        agent_inst.score = 0.98765
        text = "Score: {self.score:.1%}"
        result = await agent_inst.runtime.expand_variables(text)
        assert result == "Score: 98.8%"


class TestDocstringExpansionWithArgs:
    """Test that method docstrings are expanded with call arguments."""

    @pytest.mark.asyncio
    async def test_docstring_expands_with_args(self):
        """Test that {placeholders} in docstrings expand using call args."""
        from nooa.strategies.pure_python import PurePythonStrategy

        captured_docstring = None

        # Custom strategy that captures the docstring
        class CapturingStrategy(PurePythonStrategy):
            async def execute(self, runtime, call):
                nonlocal captured_docstring
                captured_docstring = call.docstring
                # Return a valid result to avoid generation
                return "done"

        class DocstringAgent(Agent, llm=FakeLLMClient()):
            @strategy(CapturingStrategy())
            async def analyze(self, data: list) -> str:
                """Analyze {len(data)} items from the dataset."""
                ...

        agent_inst = DocstringAgent()
        await agent_inst.analyze([1, 2, 3, 4, 5])

        # Docstring should have {len(data)} expanded to 5
        assert captured_docstring is not None
        assert "5 items" in captured_docstring
        assert "{len(data)}" not in captured_docstring

    @pytest.mark.asyncio
    async def test_docstring_expands_with_kwargs(self):
        """Test that docstrings expand with keyword arguments."""
        from nooa.strategies.pure_python import PurePythonStrategy

        captured_docstring = None

        class CapturingStrategy(PurePythonStrategy):
            async def execute(self, runtime, call):
                nonlocal captured_docstring
                captured_docstring = call.docstring
                return "done"

        class DocstringAgent(Agent, llm=FakeLLMClient()):
            @strategy(CapturingStrategy())
            async def process(self, name: str, count: int = 10) -> str:
                """Process {name} with {count} iterations."""
                ...

        agent_inst = DocstringAgent()
        await agent_inst.process(name="test", count=42)

        assert captured_docstring is not None
        assert "test" in captured_docstring
        assert "42" in captured_docstring

    @pytest.mark.asyncio
    async def test_docstring_silent_on_missing_vars(self):
        """Test that missing variables don't cause errors (silent mode)."""
        from nooa.strategies.pure_python import PurePythonStrategy

        captured_docstring = None

        class CapturingStrategy(PurePythonStrategy):
            async def execute(self, runtime, call):
                nonlocal captured_docstring
                captured_docstring = call.docstring
                return "done"

        class DocstringAgent(Agent, llm=FakeLLMClient()):
            @strategy(CapturingStrategy())
            async def process(self, data: list) -> str:
                """Process data. Unknown: {unknown_var}."""
                ...

        agent_inst = DocstringAgent()
        await agent_inst.process([1, 2, 3])

        # Should preserve the placeholder on error (silent mode)
        assert captured_docstring is not None
        assert "{unknown_var}" in captured_docstring
