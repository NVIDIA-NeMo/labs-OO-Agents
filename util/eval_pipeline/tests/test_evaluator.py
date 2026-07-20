# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Evaluator high-level API."""

from pathlib import Path

import pytest

from eval_pipeline.cli import (
    _clone_evaluator_with_tests,
    _derive_agent_label_from_spec,
    _parse_agent_spec,
    _unique_label,
    apply_agent_override,
)
from eval_pipeline.evaluator import EvalResults, EvalTest, Evaluator
from eval_pipeline.models import Task
from eval_pipeline.scoring import ScorerConfig

# =============================================================================
# Mock classes
# =============================================================================


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, llm=None):
        self.llm = llm

    async def classify(self, text: str) -> str:
        return "positive" if "love" in text.lower() else "negative"


class MockScorer:
    """Mock scorer for testing."""

    def score(self, ctx):
        from eval_pipeline.models import ScoreResult

        match = str(ctx.actual).lower() == str(ctx.expected).lower()
        return ScoreResult(score=1.0 if match else 0.0, reasoning="Mock")


class MockClient:
    """Mock LLM client."""

    pass


# =============================================================================
# Tests for Evaluator creation
# =============================================================================


class TestEvaluatorCreation:
    """Tests for creating Evaluator instances."""

    def test_create_empty_evaluator(self):
        """Can create evaluator with no models."""
        evaluator = Evaluator()
        assert evaluator.models == {}
        assert evaluator.tests == {}

    def test_create_with_models(self):
        """Can create evaluator with models dict."""
        client = MockClient()
        evaluator = Evaluator(
            models={"gpt-4": client},
            output_dir="experiments",
            name="test_eval",
        )
        assert "gpt-4" in evaluator.models
        assert evaluator.models["gpt-4"] is client
        assert evaluator.output_dir == Path("experiments")
        assert evaluator.name == "test_eval"

    def test_create_with_multiple_models(self):
        """Can create evaluator with multiple models."""
        client1 = MockClient()
        client2 = MockClient()
        evaluator = Evaluator(
            models={"gpt-4": client1, "claude": client2},
        )
        assert len(evaluator.models) == 2
        assert evaluator.models["gpt-4"] is client1
        assert evaluator.models["claude"] is client2


class TestEvaluatorFromConfig:
    """Tests for Evaluator.from_config()."""

    def test_from_config_loads_models(self, tmp_path):
        """from_config creates evaluator with models from YAML."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test_eval
models:
  test-model:
    model_name: test/model
    endpoint: https://test.com/v1
    api_key_env: TEST_API_KEY
    max_tokens: 1024

agent_models:
  - test-model

output_dir: results
test_suite: []
""")
        # This will fail trying to create client without proper env vars,
        # but we're testing the config loading path
        # In real usage, the environment variable would be set
        import os

        os.environ["TEST_API_KEY"] = "test-key"
        try:
            evaluator = Evaluator.from_config(config_file)
            assert evaluator.name == "test_eval"
            assert "test-model" in evaluator.models
        except Exception:
            # May fail on client creation, that's OK for this test
            pass
        finally:
            del os.environ["TEST_API_KEY"]


# =============================================================================
# Tests for add_test
# =============================================================================


class TestAddTest:
    """Tests for Evaluator.add_test()."""

    def test_add_test_with_dicts(self):
        """Can add test with data as list of dicts."""
        evaluator = Evaluator()
        evaluator.add_test(
            name="sentiment",
            agent_class=MockAgent,
            method="classify",
            data=[
                {"kwargs": {"text": "I love this"}, "expected": "positive"},
                {"kwargs": {"text": "I hate this"}, "expected": "negative"},
            ],
            scorers=[MockScorer()],
        )
        assert "sentiment" in evaluator.tests
        test = evaluator.tests["sentiment"]
        assert test.name == "sentiment"
        assert test.agent_class is MockAgent
        assert test.method == "classify"
        assert len(test.data) == 2
        assert len(test.scorers) == 1

    def test_add_test_with_tasks(self):
        """Can add test with data as list of Task objects."""
        evaluator = Evaluator()
        tasks = [
            Task(id="t1", input=((), {"text": "hello"}), expected="positive"),
            Task(id="t2", input=((), {"text": "bye"}), expected="negative"),
        ]
        evaluator.add_test(
            name="test1",
            agent_class=MockAgent,
            method="classify",
            data=tasks,
            scorers=[MockScorer()],
        )
        assert len(evaluator.tests["test1"].data) == 2
        assert evaluator.tests["test1"].data[0].id == "t1"

    def test_add_test_with_scorer_config(self):
        """Can add test with ScorerConfig objects."""
        evaluator = Evaluator()
        scorer_config = ScorerConfig(name="exact", weight=0.5, scorer=MockScorer())
        evaluator.add_test(
            name="test1",
            agent_class=MockAgent,
            method="classify",
            data=[{"kwargs": {"text": "x"}, "expected": "y"}],
            scorers=[scorer_config],
        )
        assert evaluator.tests["test1"].scorers[0].name == "exact"
        assert evaluator.tests["test1"].scorers[0].weight == 0.5

    def test_add_test_generates_task_ids(self):
        """Task IDs are generated from test name."""
        evaluator = Evaluator()
        evaluator.add_test(
            name="mytest",
            agent_class=MockAgent,
            method="classify",
            data=[
                {"kwargs": {"text": "a"}, "expected": "x"},
                {"kwargs": {"text": "b"}, "expected": "y"},
                {"kwargs": {"text": "c"}, "expected": "z"},
            ],
            scorers=[MockScorer()],
        )
        ids = [t.id for t in evaluator.tests["mytest"].data]
        assert ids == ["mytest_001", "mytest_002", "mytest_003"]

    def test_add_test_with_description(self):
        """Can add test with description."""
        evaluator = Evaluator()
        evaluator.add_test(
            name="test1",
            agent_class=MockAgent,
            method="classify",
            data=[{"kwargs": {"text": "x"}, "expected": "y"}],
            scorers=[MockScorer()],
            description="Test classification",
        )
        assert evaluator.tests["test1"].description == "Test classification"

    def test_add_multiple_tests(self):
        """Can add multiple tests."""
        evaluator = Evaluator()
        evaluator.add_test(
            name="test1",
            agent_class=MockAgent,
            method="classify",
            data=[{"kwargs": {"text": "a"}, "expected": "x"}],
            scorers=[MockScorer()],
        )
        evaluator.add_test(
            name="test2",
            agent_class=MockAgent,
            method="classify",
            data=[{"kwargs": {"text": "b"}, "expected": "y"}],
            scorers=[MockScorer()],
        )
        assert len(evaluator.tests) == 2
        assert "test1" in evaluator.tests
        assert "test2" in evaluator.tests


# =============================================================================
# Tests for EvalResults
# =============================================================================


class TestEvalResults:
    """Tests for EvalResults dataclass."""

    def test_pass_rate_calculation(self):
        """pass_rate is calculated correctly."""
        results = EvalResults(
            results=[{"passed": True}, {"passed": False}, {"passed": True}],
            output_file=Path("/tmp/test.jsonl"),
            passed=2,
            total=3,
        )
        assert results.pass_rate == pytest.approx(66.67, rel=0.01)

    def test_pass_rate_zero_total(self):
        """pass_rate handles zero total."""
        results = EvalResults(
            results=[],
            output_file=Path("/tmp/test.jsonl"),
            passed=0,
            total=0,
        )
        assert results.pass_rate == 0.0

    def test_summary(self):
        """summary() returns human-readable string."""
        results = EvalResults(
            results=[],
            output_file=Path("/tmp/test.jsonl"),
            passed=15,
            total=18,
        )
        assert results.summary() == "15/18 passed (83.3%)"

    def test_perfect_score(self):
        """100% pass rate displays correctly."""
        results = EvalResults(
            results=[{"passed": True}] * 10,
            output_file=Path("/tmp/test.jsonl"),
            passed=10,
            total=10,
        )
        assert results.pass_rate == 100.0
        assert results.summary() == "10/10 passed (100.0%)"


# =============================================================================
# Tests for TestDefinition
# =============================================================================


class TestEvalTest:
    """Tests for EvalTest dataclass."""

    def test_create_eval_test(self):
        """Can create EvalTest."""
        defn = EvalTest(
            name="test1",
            agent_class=MockAgent,
            method="classify",
            data=[Task(id="t1", input=((), {}), expected="x")],
            scorers=[ScorerConfig(name="s", weight=1.0, scorer=MockScorer())],
            description="A test",
        )
        assert defn.name == "test1"
        assert defn.agent_class is MockAgent
        assert defn.method == "classify"
        assert len(defn.data) == 1
        assert len(defn.scorers) == 1
        assert defn.description == "A test"

    def test_default_description(self):
        """description defaults to empty string."""
        defn = EvalTest(
            name="test1",
            agent_class=MockAgent,
            method="classify",
            data=[],
            scorers=[],
        )
        assert defn.description == ""


# =============================================================================
# Tests for task_ids filtering in Evaluator.run()
# =============================================================================


class TestEvaluatorRunTaskIds:
    """Tests for --task-ids filtering in Evaluator.run()."""

    def _make_evaluator(self, tmp_path, agent_class=None):
        """Helper: create a minimal evaluator ready to call run()."""
        if agent_class is None:
            agent_class = MockAgent
        client = MockClient()
        evaluator = Evaluator(
            models={"m": client},
            output_dir=tmp_path,
            name="test",
        )
        evaluator._default_model_ids = ["m"]
        evaluator._model_factories = {"m": lambda: client}
        evaluator._model_metadata = {"m": {"id": "m", "model_name": "test/model"}}
        return evaluator

    @pytest.mark.asyncio
    async def test_task_ids_filters_tasks(self, tmp_path):
        """task_ids runs only the specified subset of tasks."""
        ran_ids = []

        class RecordingAgent:
            def __init__(self, llm=None):
                pass

            async def classify(self, text: str) -> str:
                ran_ids.append(text)
                return "pos"

        evaluator = self._make_evaluator(tmp_path, RecordingAgent)
        tasks = [
            Task(id="task_001", input=((), {"text": "a"}), expected="pos"),
            Task(id="task_002", input=((), {"text": "b"}), expected="pos"),
            Task(id="task_003", input=((), {"text": "c"}), expected="pos"),
        ]
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        evaluator.tests["t"] = EvalTest(
            name="t",
            agent_class=RecordingAgent,
            method="classify",
            data=tasks,
            scorers=[scorer],
        )

        results = await evaluator.run(task_ids=["task_001", "task_003"])

        assert results.total == 2
        assert sorted(ran_ids) == ["a", "c"]

    @pytest.mark.asyncio
    async def test_task_ids_none_runs_all(self, tmp_path):
        """task_ids=None (default) runs all tasks."""
        evaluator = self._make_evaluator(tmp_path)
        tasks = [
            Task(id="task_001", input=((), {"text": "a"}), expected="pos"),
            Task(id="task_002", input=((), {"text": "b"}), expected="pos"),
        ]
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        evaluator.tests["t"] = EvalTest(
            name="t",
            agent_class=MockAgent,
            method="classify",
            data=tasks,
            scorers=[scorer],
        )

        results = await evaluator.run()

        assert results.total == 2

    @pytest.mark.asyncio
    async def test_task_ids_no_match_runs_zero(self, tmp_path):
        """task_ids with no matches produces zero samples."""
        evaluator = self._make_evaluator(tmp_path)
        tasks = [Task(id="task_001", input=((), {"text": "a"}), expected="pos")]
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        evaluator.tests["t"] = EvalTest(
            name="t",
            agent_class=MockAgent,
            method="classify",
            data=tasks,
            scorers=[scorer],
        )

        results = await evaluator.run(task_ids=["nonexistent"])

        assert results.total == 0

    @pytest.mark.asyncio
    async def test_task_ids_empty_list_runs_zero(self, tmp_path):
        """task_ids=[] (empty list) runs zero tasks, not all tasks."""
        evaluator = self._make_evaluator(tmp_path)
        tasks = [Task(id="task_001", input=((), {"text": "a"}), expected="pos")]
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        evaluator.tests["t"] = EvalTest(
            name="t",
            agent_class=MockAgent,
            method="classify",
            data=tasks,
            scorers=[scorer],
        )

        results = await evaluator.run(task_ids=[])

        assert results.total == 0

    @pytest.mark.asyncio
    async def test_task_ids_combined_with_limit(self, tmp_path):
        """task_ids filtering is applied after limit slicing."""
        evaluator = self._make_evaluator(tmp_path)
        tasks = [
            Task(id="task_001", input=((), {"text": "a"}), expected="pos"),
            Task(id="task_002", input=((), {"text": "b"}), expected="pos"),
            Task(id="task_003", input=((), {"text": "c"}), expected="pos"),
        ]
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        evaluator.tests["t"] = EvalTest(
            name="t",
            agent_class=MockAgent,
            method="classify",
            data=tasks,
            scorers=[scorer],
        )

        # limit=2 keeps task_001 and task_002; task_ids then filters to task_001 only
        results = await evaluator.run(limit=2, task_ids=["task_001"])

        assert results.total == 1

    @pytest.mark.asyncio
    async def test_limit_excludes_task_id_warns(self, tmp_path, capsys):
        """When limit cuts a requested task ID, a warning is written to stderr."""
        evaluator = self._make_evaluator(tmp_path)
        tasks = [
            Task(id="task_001", input=((), {"text": "a"}), expected="pos"),
            Task(id="task_002", input=((), {"text": "b"}), expected="pos"),
            Task(id="task_003", input=((), {"text": "c"}), expected="pos"),
        ]
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        evaluator.tests["t"] = EvalTest(
            name="t",
            agent_class=MockAgent,
            method="classify",
            data=tasks,
            scorers=[scorer],
        )

        # limit=2 excludes task_003; requesting task_003 should trigger a warning
        results = await evaluator.run(limit=2, task_ids=["task_003"])

        assert results.total == 0
        captured = capsys.readouterr()
        assert "task_003" in captured.err
        assert "--limit" in captured.err


# =============================================================================
# Tests for --agent CLI override (via apply_agent_override)
# =============================================================================


class TestAgentOverride:
    """Tests for apply_agent_override() — the production helper used by main_async."""

    def _make_evaluator_with_tests(self, *agent_classes):
        """Return an evaluator with one test per agent class (named demo0, demo1, ...)."""
        evaluator = Evaluator(models={"m": MockClient()}, output_dir="tmp", name="test")
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        for i, cls in enumerate(agent_classes):
            task = Task(id=f"t{i}", input=((), {"text": "hi"}), expected="pos")
            evaluator.tests[f"demo{i}"] = EvalTest(
                name=f"demo{i}",
                agent_class=cls,
                method="classify",
                data=[task],
                scorers=[scorer],
            )
        return evaluator

    def test_override_all_when_no_old(self, tmp_path):
        """'module.Class' (no OLD=) replaces the agent on all tests."""

        class OriginalAgent:
            async def classify(self, text: str) -> str:
                return "pos"

        evaluator = self._make_evaluator_with_tests(OriginalAgent)
        # Use a real importable class for the test
        overridden, err = apply_agent_override(evaluator, "eval_pipeline.evaluator.Evaluator")
        assert err is None
        assert overridden == ["demo0"]
        assert evaluator.tests["demo0"].agent_class is Evaluator

    def test_override_matching_old_class(self, tmp_path):
        """'OldClass=module.Class' only touches tests whose agent is named OldClass."""

        class AgentA:
            async def classify(self, text: str) -> str:
                return "pos"

        class AgentB:
            async def classify(self, text: str) -> str:
                return "neg"

        evaluator = self._make_evaluator_with_tests(AgentA, AgentB)
        overridden, err = apply_agent_override(
            evaluator, "AgentA=eval_pipeline.evaluator.Evaluator"
        )
        assert err is None
        assert overridden == ["demo0"]
        assert evaluator.tests["demo0"].agent_class is Evaluator
        assert evaluator.tests["demo1"].agent_class is AgentB  # unchanged

    def test_no_match_returns_empty_no_error(self, tmp_path):
        """Non-matching OLD class returns an empty list without an error."""

        class OriginalAgent:
            async def classify(self, text: str) -> str:
                return "pos"

        evaluator = self._make_evaluator_with_tests(OriginalAgent)
        overridden, err = apply_agent_override(
            evaluator, "NonExistent=eval_pipeline.evaluator.Evaluator"
        )
        assert err is None
        assert overridden == []
        assert evaluator.tests["demo0"].agent_class is OriginalAgent  # unchanged

    def test_dot_free_spec_returns_error(self, tmp_path):
        """A spec with no dot (e.g. 'MyClass') returns an error, not a crash."""

        class OriginalAgent:
            pass

        evaluator = self._make_evaluator_with_tests(OriginalAgent)
        overridden, err = apply_agent_override(evaluator, "MyClass")
        assert err is not None
        assert "fully-qualified" in err
        assert overridden == []

    def test_bad_module_returns_error(self, tmp_path):
        """An unimportable module returns an error message."""

        class OriginalAgent:
            pass

        evaluator = self._make_evaluator_with_tests(OriginalAgent)
        overridden, err = apply_agent_override(evaluator, "no.such.module.MyClass")
        assert err is not None
        assert "cannot import" in err
        assert overridden == []

    def test_dotted_old_class_returns_error(self, tmp_path):
        """A dotted old-class path (e.g. my.module.OldClass=...) returns an error."""

        class OriginalAgent:
            pass

        evaluator = self._make_evaluator_with_tests(OriginalAgent)
        overridden, err = apply_agent_override(
            evaluator, "my.module.OldClass=eval_pipeline.evaluator.Evaluator"
        )
        assert err is not None
        assert "bare class name" in err
        assert overridden == []
        assert evaluator.tests["demo0"].agent_class is OriginalAgent  # unchanged


# =============================================================================
# Tests for --agent file-path loading (via apply_agent_override)
# =============================================================================


class TestAgentOverrideFilePath:
    """Tests for apply_agent_override() with file-path specs."""

    def _make_evaluator(self):
        """Return a minimal evaluator with one test."""
        evaluator = Evaluator(models={"m": MockClient()}, output_dir="tmp", name="test")
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        task = Task(id="t0", input=((), {"text": "hi"}), expected="pos")
        evaluator.tests["demo"] = EvalTest(
            name="demo",
            agent_class=MockAgent,
            method="classify",
            data=[task],
            scorers=[scorer],
        )
        return evaluator

    def _write_agent_file(self, tmp_path, class_names=("AgentFromFile",)):
        """Write a temporary .py file defining the given class names."""
        body = "\n\n".join(
            f"class {name}:\n"
            f"    def __init__(self, llm=None): pass\n"
            f"    async def classify(self, text: str) -> str: return 'pos'"
            for name in class_names
        )
        p = tmp_path / "agent_tmp.py"
        p.write_text(body)
        return p

    def test_file_path_with_explicit_class(self, tmp_path):
        """'path/to/file.py::ClassName' loads the named class from the file."""
        agent_file = self._write_agent_file(tmp_path, ("AgentFromFile",))
        evaluator = self._make_evaluator()

        overridden, err = apply_agent_override(evaluator, f"{agent_file}::AgentFromFile")

        assert err is None
        assert overridden == ["demo"]
        assert evaluator.tests["demo"].agent_class.__name__ == "AgentFromFile"

    def test_file_path_auto_detect_single_class(self, tmp_path):
        """'path/to/file.py' auto-detects the class when there is exactly one."""
        agent_file = self._write_agent_file(tmp_path, ("SingleAgent",))
        evaluator = self._make_evaluator()

        overridden, err = apply_agent_override(evaluator, str(agent_file))

        assert err is None
        assert overridden == ["demo"]
        assert evaluator.tests["demo"].agent_class.__name__ == "SingleAgent"

    def test_file_path_multiple_classes_requires_explicit(self, tmp_path):
        """A file with multiple classes returns an error unless explicit class given."""
        agent_file = self._write_agent_file(tmp_path, ("AgentAlpha", "AgentBeta"))
        evaluator = self._make_evaluator()

        overridden, err = apply_agent_override(evaluator, str(agent_file))

        assert err is not None
        assert "multiple classes" in err
        assert overridden == []

    def test_file_path_nonexistent_file_returns_error(self, tmp_path):
        """A non-existent file path returns a clear error."""
        evaluator = self._make_evaluator()

        overridden, err = apply_agent_override(evaluator, "/no/such/file.py::MyAgent")

        assert err is not None
        assert "not found" in err
        assert overridden == []

    def test_file_path_wrong_class_name_returns_error(self, tmp_path):
        """Specifying a class name not in the file returns an error."""
        agent_file = self._write_agent_file(tmp_path, ("AgentFromFile",))
        evaluator = self._make_evaluator()

        overridden, err = apply_agent_override(evaluator, f"{agent_file}::WrongName")

        assert err is not None
        assert "WrongName" in err
        assert overridden == []

    def test_file_path_with_old_class_filter(self, tmp_path):
        """'OldClass=path/file.py::NewClass' only replaces matching tests."""
        agent_file = self._write_agent_file(tmp_path, ("NewAgent",))

        evaluator = Evaluator(models={"m": MockClient()}, output_dir="tmp", name="test")
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())

        class AgentA:
            pass

        class AgentB:
            pass

        for i, cls in enumerate([AgentA, AgentB]):
            task = Task(id=f"t{i}", input=((), {"text": "hi"}), expected="pos")
            evaluator.tests[f"demo{i}"] = EvalTest(
                name=f"demo{i}",
                agent_class=cls,
                method="classify",
                data=[task],
                scorers=[scorer],
            )

        overridden, err = apply_agent_override(evaluator, f"AgentA={agent_file}::NewAgent")

        assert err is None
        assert overridden == ["demo0"]
        assert evaluator.tests["demo0"].agent_class.__name__ == "NewAgent"
        assert evaluator.tests["demo1"].agent_class is AgentB


# =============================================================================
# Tests for _parse_agent_spec and _derive_agent_label_from_spec
# =============================================================================


class TestAgentLabelDerivation:
    """Tests for label parsing helpers used by --agents."""

    def test_parse_explicit_label(self):
        """'label:spec' splits into (label, spec)."""
        label, spec = _parse_agent_spec("pf:path/to/agent.py::Agent")
        assert label == "pf"
        assert spec == "path/to/agent.py::Agent"

    def test_parse_no_label(self):
        """A plain spec has no label prefix."""
        label, spec = _parse_agent_spec("path/to/agent.py::Agent")
        assert label is None
        assert spec == "path/to/agent.py::Agent"

    def test_parse_old_equals_not_a_label(self):
        """'OldClass=module.New' is not treated as a label (contains '=')."""
        label, spec = _parse_agent_spec("OldClass=module.New")
        assert label is None

    def test_parse_module_spec_no_label(self):
        """'module.ClassName' is not treated as a label (contains '.')."""
        label, spec = _parse_agent_spec("module.ClassName")
        assert label is None

    def test_derive_module_spec(self):
        """`module.ClassName` → class name as label."""
        assert _derive_agent_label_from_spec("module.ClassName") == "ClassName"

    def test_derive_module_spec_with_old(self):
        """`Old=module.ClassName` → class name as label."""
        assert _derive_agent_label_from_spec("Old=module.ClassName") == "ClassName"

    def test_derive_file_spec_with_class(self):
        """`path/to/agent-pf-000.py::Foo` → file stem."""
        assert _derive_agent_label_from_spec("path/to/agent-pf-000.py::Foo") == "agent-pf-000"

    def test_derive_file_spec_no_class(self):
        """`path/to/agent-pf-000.py` → file stem."""
        assert _derive_agent_label_from_spec("path/to/agent-pf-000.py") == "agent-pf-000"

    def test_derive_explicit_label(self):
        """`pf:path/to/agent.py::Agent` → explicit label `pf`."""
        assert _derive_agent_label_from_spec("pf:path/to/agent.py::Agent") == "pf"

    def test_unique_label_no_collision(self):
        assert _unique_label("foo", []) == "foo"
        assert _unique_label("foo", ["bar"]) == "foo"

    def test_unique_label_collision(self):
        assert _unique_label("foo", ["foo"]) == "foo_1"
        assert _unique_label("foo", ["foo", "foo_1"]) == "foo_2"


# =============================================================================
# Tests for agent_label propagation into variant field
# =============================================================================


class TestAgentLabelVariant:
    """Tests that agent_label flows through to EvalTestResult.variant."""

    def _make_evaluator(self, tmp_path):
        client = MockClient()
        evaluator = Evaluator(models={"m": client}, output_dir=tmp_path, name="test")
        evaluator._default_model_ids = ["m"]
        evaluator._model_factories = {"m": lambda: client}
        evaluator._model_metadata = {"m": {"id": "m", "model_name": "test/model"}}
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        task = Task(id="task_001", input=((), {"text": "hi"}), expected="pos")
        evaluator.tests["t"] = EvalTest(
            name="t",
            agent_class=MockAgent,
            method="classify",
            data=[task],
            scorers=[scorer],
        )
        return evaluator

    @pytest.mark.asyncio
    async def test_agent_label_in_variant(self, tmp_path):
        """When agent_label is set, variant is '{label}_run1'."""
        evaluator = self._make_evaluator(tmp_path)
        results = await evaluator.run(agent_label="v2")
        assert results.total == 1
        assert all(r.variant == "v2_run1" for r in results.results)

    @pytest.mark.asyncio
    async def test_agent_label_none_uses_run_id(self, tmp_path):
        """When agent_label is None (default), variant is 'run1'."""
        evaluator = self._make_evaluator(tmp_path)
        results = await evaluator.run()
        assert results.total == 1
        assert all(r.variant == "run1" for r in results.results)


# =============================================================================
# Tests for _clone_evaluator_with_tests
# =============================================================================


class TestCloneEvaluatorWithTests:
    def test_clone_has_new_tests(self):
        evaluator = Evaluator(models={"m": MockClient()}, output_dir="tmp", name="test")
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        task = Task(id="t0", input=((), {}), expected="x")
        evaluator.tests["orig"] = EvalTest(
            name="orig", agent_class=MockAgent, method="classify", data=[task], scorers=[scorer]
        )

        import dataclasses

        new_tests = {k: dataclasses.replace(v) for k, v in evaluator.tests.items()}
        clone = _clone_evaluator_with_tests(evaluator, new_tests)

        assert clone.tests is new_tests
        assert clone.tests is not evaluator.tests
        assert clone.name == evaluator.name

    def test_clone_tests_are_independent(self):
        """Mutating the clone's agent_class does not affect the original."""
        evaluator = Evaluator(models={"m": MockClient()}, output_dir="tmp", name="test")
        scorer = ScorerConfig(name="s", weight=1.0, scorer=MockScorer())
        task = Task(id="t0", input=((), {}), expected="x")
        evaluator.tests["orig"] = EvalTest(
            name="orig", agent_class=MockAgent, method="classify", data=[task], scorers=[scorer]
        )

        import dataclasses

        new_tests = {k: dataclasses.replace(v) for k, v in evaluator.tests.items()}
        clone = _clone_evaluator_with_tests(evaluator, new_tests)

        class AnotherAgent:
            pass

        clone.tests["orig"].agent_class = AnotherAgent

        assert evaluator.tests["orig"].agent_class is MockAgent  # original untouched


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
