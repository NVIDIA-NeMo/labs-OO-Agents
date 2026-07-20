# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for doc(instance) on real Agent subclasses from tests/capability/agents/.

These tests guard against regressions in instance rendering — specifically:
- Class-valued fields (sub-agent class attrs) render as just the class name
- ## Referenced Types appears exactly once, not once per sub-agent field
- Smoke tests: doc(instance) doesn't crash for any agent
"""

import sys
from pathlib import Path

import pytest

# Ensure tests/capability/agents is importable
sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))

from nooa.agentdoc import doc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doc_instance(cls) -> str:
    """Create a bare instance (no __init__) and call doc() on it."""
    instance = cls.__new__(cls)
    return doc(instance)


def _referenced_types_count(output: str) -> int:
    return output.count("## Referenced Types")


# ---------------------------------------------------------------------------
# Router — sub-agent class attributes
# ---------------------------------------------------------------------------


class TestRouterTestWrapper:
    @pytest.fixture(scope="class")
    def output(self):
        from capability.agents.router import RouterTestWrapper  # type: ignore[import]

        return _doc_instance(RouterTestWrapper)

    def test_referenced_types_appears_exactly_once(self, output):
        assert _referenced_types_count(output) == 1

    def test_sub_agent_fields_show_as_class_name(self, output):
        # Fields should be: AnalyzerSubAgent = AnalyzerSubAgent (not full expansion)
        assert "AnalyzerSubAgent: type[AnalyzerSubAgent] = AnalyzerSubAgent" in output
        assert "TransformerSubAgent: type[TransformerSubAgent] = TransformerSubAgent" in output
        assert "ValidatorSubAgent: type[ValidatorSubAgent] = ValidatorSubAgent" in output

    def test_sub_agents_in_referenced_types(self, output):
        refs_section = output.split("## Referenced Types", 1)[-1]
        assert "AnalyzerSubAgent" in refs_section

    def test_process_method_shown(self, output):
        assert "async def process" in output

    def test_no_nested_referenced_types_in_field_value(self, output):
        # The old bug: ## Referenced Types appeared inline within a field value line
        # Verify there's no ## before the single expected ## Referenced Types
        first_ref_pos = output.index("## Referenced Types")
        assert "## Referenced Types" not in output[:first_ref_pos]


# ---------------------------------------------------------------------------
# EmployeeSalaryAgent — nested sub-agent class attributes
# ---------------------------------------------------------------------------


class TestEmployeeSalaryAgent:
    @pytest.fixture(scope="class")
    def output(self):
        from capability.agents.employee_lookup import EmployeeSalaryAgent  # type: ignore[import]

        return _doc_instance(EmployeeSalaryAgent)

    def test_referenced_types_appears_exactly_once(self, output):
        assert _referenced_types_count(output) == 1

    def test_sub_agent_fields_show_as_class_name(self, output):
        assert "EmployeeDirectory: type[EmployeeDirectory] = EmployeeDirectory" in output
        assert "PayrollSystem: type[PayrollSystem] = PayrollSystem" in output

    def test_method_shown(self, output):
        assert "async def get_employee_salary" in output


# ---------------------------------------------------------------------------
# OrderTestWrapper — Pydantic fields, Enum, many helpers
# ---------------------------------------------------------------------------


class TestOrderTestWrapper:
    @pytest.fixture(scope="class")
    def output(self):
        from capability.agents.order import OrderTestWrapper  # type: ignore[import]

        return _doc_instance(OrderTestWrapper)

    def test_no_crash(self, output):
        assert output.startswith("class OrderTestWrapper")

    def test_referenced_types_at_most_once(self, output):
        # May have 0 (if no non-hidden referenced types) or 1
        assert _referenced_types_count(output) <= 1

    def test_process_method_shown(self, output):
        assert "async def process" in output


# ---------------------------------------------------------------------------
# Smoke tests — all agents render without crashing
# ---------------------------------------------------------------------------


AGENT_IMPORTS = [
    ("capability.agents.router", "RouterTestWrapper"),
    ("capability.agents.router", "AnalyzerSubAgent"),
    ("capability.agents.router", "TransformerSubAgent"),
    ("capability.agents.router", "ValidatorSubAgent"),
    ("capability.agents.employee_lookup", "EmployeeSalaryAgent"),
    ("capability.agents.order", "OrderTestWrapper"),
    ("capability.agents.needle", "NeedleTestWrapper"),
    ("capability.agents.sentiment", "SentimentAgent"),
    ("capability.agents.calculate_single", "CalculateSingleAgent"),
    ("capability.agents.calculate_batch", "CalculateBatchAgent"),
    ("capability.agents.summarize", "SummarizeAgent"),
    ("capability.agents.summarize", "SummarizeBatchAgent"),
    ("capability.agents.json_extract", "JsonExtractAgent"),
    ("capability.agents.json_qa", "JsonQAAgent"),
    ("capability.agents.error_recovery", "WeatherLookupAgent"),
    ("capability.agents.large_data", "LargeDataAgent"),
    ("capability.agents.refinement", "RefinementTestAgent"),
    ("capability.agents.repl_exploration", "ReplExplorationTestAgent"),
    ("capability.agents.sentiment_batch", "SentimentBatchAgent"),
    ("capability.agents.sentiment_single", "SentimentSingleAgent"),
    ("capability.agents.task_decomposition", "TaskDecompositionTestAgent"),
]


@pytest.mark.parametrize("module_name,class_name", AGENT_IMPORTS)
def test_doc_instance_smoke(module_name, class_name):
    """doc(instance) must not crash and must produce a non-empty string."""
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    output = _doc_instance(cls)
    assert isinstance(output, str)
    assert len(output) > 0
    assert f"class {class_name}" in output


@pytest.mark.parametrize("module_name,class_name", AGENT_IMPORTS)
def test_doc_instance_no_duplicate_referenced_types(module_name, class_name):
    """## Referenced Types must appear at most once."""
    import importlib

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    output = _doc_instance(cls)
    assert _referenced_types_count(output) <= 1, (
        f"{class_name}: '## Referenced Types' appeared {_referenced_types_count(output)} times"
    )
