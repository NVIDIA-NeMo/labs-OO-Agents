# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa.prompts — print_prompt() runtime API and rendering helpers."""

import pytest

import nooa
from nooa import Agent, CodeActStrategy, PromptData, strategy
from nooa.prompts import (
    _get_prefill,
    _get_task_prompt,
    build_prompt_data,
    print_prompt,
    render_prompt_data,
)
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient

_LLM = FakeLLMClient()


# ---------------------------------------------------------------------------
# Test agents
# ---------------------------------------------------------------------------


class _SimpleAgent(Agent, llm=_LLM):
    async def analyze(self, data: str) -> str:
        """Analyze {data} and return a summary."""
        ...

    async def no_args(self) -> str:
        """Just do something with no parameters."""
        ...

    @strategy(PurePythonStrategy())
    async def transform(self, items: list) -> list:
        """Transform {items} into processed output."""
        ...

    async def with_pre_ellipsis(self, value: int) -> int:
        """Compute result from {value}."""
        x = value * 2  # noqa: F841
        ...


# ---------------------------------------------------------------------------
# _get_task_prompt
# ---------------------------------------------------------------------------


class TestGetTaskPrompt:
    @pytest.mark.asyncio
    async def test_codeact_contains_method_name_and_docstring(self):
        agent = _SimpleAgent()
        strategy_obj = CodeActStrategy()
        call = CurrentCall(
            id="c3",
            method_name="analyze",
            decorator="plan",
            docstring="Analyze carefully.",
        )
        result = await _get_task_prompt(agent.runtime, strategy_obj, call)
        assert "analyze" in result
        assert "Analyze carefully." in result

    @pytest.mark.asyncio
    async def test_pure_python_contains_docstring(self):
        agent = _SimpleAgent()
        strategy_obj = PurePythonStrategy()
        call = CurrentCall(
            id="c4",
            method_name="transform",
            decorator="plan",
            docstring="Transform the items.",
            signature="(self, items: list) -> list",
            kwargs={"items": ["a", "b"]},
        )
        result = await _get_task_prompt(agent.runtime, strategy_obj, call)
        assert "Transform the items." in result

    @pytest.mark.asyncio
    async def test_strategy_without_build_task_message_falls_back_to_docstring(self):
        """Strategy with no _build_task_message uses raw docstring."""

        class _MinimalStrategy:
            pass

        agent = _SimpleAgent()
        call = CurrentCall(
            id="c5",
            method_name="do_thing",
            decorator="plan",
            docstring="Do something.",
        )
        result = await _get_task_prompt(agent.runtime, _MinimalStrategy(), call)
        assert result == "Do something."

    @pytest.mark.asyncio
    async def test_no_docstring_returns_placeholder(self):
        agent = _SimpleAgent()
        strategy_obj = CodeActStrategy()
        call = CurrentCall(id="c6", method_name="mystery", decorator="plan", docstring=None)
        result = await _get_task_prompt(agent.runtime, strategy_obj, call)
        assert "mystery" in result


# ---------------------------------------------------------------------------
# _get_prefill
# ---------------------------------------------------------------------------


class TestGetPrefill:
    def test_codeact_with_args_has_inspect_code(self):
        strategy_obj = CodeActStrategy()
        call = CurrentCall(
            id="c7",
            method_name="analyze",
            decorator="plan",
            kwargs={"data": "hello"},
        )
        inspect_code, _ = _get_prefill(strategy_obj, call, None)
        assert inspect_code is not None
        assert "data" in inspect_code
        assert "pprint" in inspect_code

    def test_codeact_no_args_inspect_code_is_none(self):
        strategy_obj = CodeActStrategy()
        call = CurrentCall(
            id="c8",
            method_name="no_args",
            decorator="plan",
            kwargs={},
        )
        inspect_code, _ = _get_prefill(strategy_obj, call, None)
        assert inspect_code is None

    def test_pure_python_has_inspect_prefill_by_default(self):
        strategy_obj = PurePythonStrategy()  # InspectInputsPrefill by default
        call = CurrentCall(id="c9", method_name="transform", decorator="plan", kwargs={"items": []})
        inspect_code, _ = _get_prefill(strategy_obj, call, None)
        assert inspect_code is not None
        assert "items" in inspect_code
        assert "pprint" in inspect_code

    def test_other_strategy_no_inspect_prefill(self):
        """Unknown strategy produces no inspect_prefill but preserves pre_ellipsis."""
        from nooa import PredictStrategy

        strategy_obj = PredictStrategy()
        call = CurrentCall(id="c10", method_name="classify", decorator="plan", kwargs={"x": 1})
        inspect_code, pre_ellipsis = _get_prefill(strategy_obj, call, None)
        assert inspect_code is None
        assert pre_ellipsis is None  # no pre-ellipsis code in this call

    def test_codeact_config_prefill_none_disables_inspect(self):
        """#331: print_prompt must honor CodeActConfig.prefill=None (inspect disabled)."""
        from nooa.config.strategy_config import CodeActConfig

        strategy_obj = CodeActStrategy(config=CodeActConfig(prefill=None))
        call = CurrentCall(
            id="c11",
            method_name="analyze",
            decorator="plan",
            kwargs={"data": "hello"},
        )
        inspect_code, _ = _get_prefill(strategy_obj, call, None)
        assert inspect_code is None

    def test_codeact_config_custom_prefill_used(self):
        """#331: a custom configured Prefill drives inspect code, not the hardcoded default."""

        class _CustomPrefill:
            def get_code(self, call, config=None):
                return "# custom-prefill-marker"

        from nooa.config.strategy_config import CodeActConfig

        strategy_obj = CodeActStrategy(config=CodeActConfig(prefill=_CustomPrefill()))
        call = CurrentCall(
            id="c12",
            method_name="analyze",
            decorator="plan",
            kwargs={"data": "hello"},
        )
        inspect_code, _ = _get_prefill(strategy_obj, call, None)
        assert inspect_code == "# custom-prefill-marker"


# ---------------------------------------------------------------------------
# render_prompt_data
# ---------------------------------------------------------------------------


class TestRenderPromptData:
    def _make_data(self, **overrides):
        defaults = {
            "system_prompt": "<system>You are TestAgent.</system>",
            "task_prompt": "## Task: analyze\n\nDo the thing.",
            "inspect_prefill": 'print("Task: analyze()")',
            "pre_ellipsis": None,
            "strategy_name": "CodeActStrategy",
            "method_path": "TestAgent.analyze",
        }
        defaults |= overrides
        return PromptData(**defaults)

    def test_has_section_headers(self):
        data = self._make_data()
        result = render_prompt_data(data)
        assert "=== SYSTEM PROMPT" in result
        assert "=== TASK PROMPT" in result
        assert "=== PREFILL" in result

    def test_contains_content(self):
        data = self._make_data()
        result = render_prompt_data(data)
        assert "You are TestAgent." in result
        assert "Do the thing." in result
        assert 'print("Task: analyze()")' in result

    def test_no_prefill_omits_prefill_section(self):
        data = self._make_data(inspect_prefill=None, pre_ellipsis=None)
        result = render_prompt_data(data)
        assert "PREFILL" not in result

    def test_no_system_prompt_omits_system_section(self):
        data = self._make_data(system_prompt=None)
        result = render_prompt_data(data)
        assert "SYSTEM PROMPT" not in result

    def test_pre_ellipsis_shown_in_prefill(self):
        data = self._make_data(inspect_prefill=None, pre_ellipsis="x = value * 2")
        result = render_prompt_data(data)
        assert "x = value * 2" in result
        assert "PREFILL" in result

    def test_both_prefill_parts_shown(self):
        data = self._make_data(
            inspect_prefill='print("Task: analyze()")',
            pre_ellipsis="x = value * 2",
        )
        result = render_prompt_data(data)
        assert 'print("Task: analyze()")' in result
        assert "x = value * 2" in result

    def test_section_order(self):
        data = self._make_data()
        result = render_prompt_data(data)
        assert result.index("SYSTEM PROMPT") < result.index("TASK PROMPT") < result.index("PREFILL")


# ---------------------------------------------------------------------------
# build_prompt_data and print_prompt (runtime API)
# ---------------------------------------------------------------------------


class TestBuildPromptDataRuntime:
    @pytest.mark.asyncio
    async def test_non_bound_method_raises_type_error(self):
        with pytest.raises(TypeError, match="bound agent method"):
            await build_prompt_data(lambda: None)

    @pytest.mark.asyncio
    async def test_non_agent_bound_method_raises_type_error(self):
        class _NotAnAgent:
            def method(self) -> None: ...

        with pytest.raises(TypeError, match="Agent instance"):
            await build_prompt_data(_NotAnAgent().method)

    @pytest.mark.asyncio
    async def test_method_path(self):
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.analyze, "hello world")
        assert data.method_path == "_SimpleAgent.analyze"

    @pytest.mark.asyncio
    async def test_strategy_name_codeact(self):
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.analyze, "hello")
        assert data.strategy_name == "CodeActStrategy"

    @pytest.mark.asyncio
    async def test_strategy_name_pure_python(self):
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.transform, ["a", "b"])
        assert data.strategy_name == "PurePythonStrategy"

    @pytest.mark.asyncio
    async def test_task_prompt_contains_docstring(self):
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.analyze, "hello")
        # The method docstring is embedded in the strategy's task template.
        # {data} in the docstring is a literal placeholder, not expanded here.
        assert "analyze" in data.task_prompt
        assert "Analyze" in data.task_prompt

    @pytest.mark.asyncio
    async def test_prefill_with_real_args(self):
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.analyze, "test data")
        assert data.inspect_prefill is not None
        assert "data" in data.inspect_prefill

    @pytest.mark.asyncio
    async def test_prefill_no_args_is_none(self):
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.no_args)
        assert data.inspect_prefill is None

    @pytest.mark.asyncio
    async def test_system_prompt_present(self):
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.analyze, "x")
        assert data.system_prompt is not None
        assert "_SimpleAgent" in data.system_prompt

    @pytest.mark.asyncio
    async def test_pre_ellipsis_extracted(self):
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.with_pre_ellipsis, 5)
        assert data.pre_ellipsis is not None
        assert "value * 2" in data.pre_ellipsis


class TestPrintPromptRuntime:
    @pytest.mark.asyncio
    async def test_writes_to_stdout(self, capsys):
        agent = _SimpleAgent()
        await print_prompt(agent.analyze, "hello")
        captured = capsys.readouterr()
        assert "=== TASK PROMPT" in captured.out
        assert "analyze" in captured.out

    @pytest.mark.asyncio
    async def test_unknown_kwarg_forwarded_not_silently_absorbed(self):
        """Kwargs not consumed by print_prompt are forwarded to the method.

        This test documents the behaviour: print_prompt has no special
        parameters of its own, so any kwarg is passed straight through to
        the agent method's call context.  Callers should not pass kwargs
        intended for print_prompt itself (there are none).
        """
        agent = _SimpleAgent()
        data = await build_prompt_data(agent.analyze, data="hello")
        # 'data' kwarg reaches the call and appears in the prefill
        assert data.inspect_prefill is not None
        assert "data" in data.inspect_prefill

    @pytest.mark.asyncio
    async def test_returns_none(self):
        agent = _SimpleAgent()
        result = await print_prompt(agent.analyze, "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_exported_from_nooa(self):
        """print_prompt is available as nooa.print_prompt."""
        assert hasattr(nooa, "print_prompt")
        assert nooa.print_prompt is print_prompt

    def test_prompt_data_exported_from_nooa(self):
        """PromptData is importable from the top-level nooa namespace."""
        assert nooa.PromptData is PromptData
