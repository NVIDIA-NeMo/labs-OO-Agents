# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval metadata flow: config → test → task → spans → results.

Covers:
- Three-level metadata merge (config < test < task)
- Metadata in OTLP span attributes (both file and OTLP paths)
- Metadata in EvalTestResult serialization
- Reserved key filtering
- Non-scalar task metadata filtering
- load_tasks metadata loading
"""

import json
from pathlib import Path

import pytest

from eval_pipeline.eval_types import EvalTestResult, ScoreDetail, SubprocessTaskInput
from eval_pipeline.models import Task
from eval_pipeline.trace_eval_span import (
    _RESERVED_EVAL_KEYS,
    write_eval_span_to_trace,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_eval_span_attrs(trace_file: Path) -> dict:
    """Read the eval span from a trace file and return flat attributes dict."""
    from tests.otlp_helpers import _otlp_attrs_to_dict

    for line in trace_file.read_text().strip().splitlines():
        payload = json.loads(line)
        for rs in payload.get("resourceSpans", []):
            for ss in rs.get("scopeSpans", []):
                for span in ss.get("spans", []):
                    if span.get("name") == "eval":
                        return _otlp_attrs_to_dict(span.get("attributes", []))
    raise AssertionError("No eval span found in trace file")


_BASIC_SCORES = {"exact": ScoreDetail(score=1.0, passed=True, reasoning="ok")}


# ---------------------------------------------------------------------------
# Three-level merge
# ---------------------------------------------------------------------------


class TestMetadataMerge:
    """Test the three-level metadata merge logic."""

    def test_task_overrides_config(self):
        """Task-level metadata overrides config+test level."""
        from eval_pipeline.pipeline import Sample, _build_eval_metadata

        sample = Sample(
            task=Task(
                id="t1",
                input=((), {}),
                expected=None,
                metadata={"difficulty": "hard", "source": "manual"},
            ),
            method="run",
            agent_class="A",
            scorers=[],
            agent_factory=lambda: None,
            eval_metadata={"difficulty": "easy", "dataset": "kdd"},
        )
        merged = _build_eval_metadata(sample)
        assert merged["difficulty"] == "hard"  # task overrides
        assert merged["dataset"] == "kdd"  # config preserved
        assert merged["source"] == "manual"  # task-only

    def test_config_only(self):
        """Config metadata flows through when task has no metadata."""
        from eval_pipeline.pipeline import Sample, _build_eval_metadata

        sample = Sample(
            task=Task(id="t1", input=((), {}), expected=None),
            method="run",
            agent_class="A",
            scorers=[],
            agent_factory=lambda: None,
            eval_metadata={"dataset": "kdd"},
        )
        merged = _build_eval_metadata(sample)
        assert merged == {"dataset": "kdd"}

    def test_task_only(self):
        """Task metadata flows through when config has no metadata."""
        from eval_pipeline.pipeline import Sample, _build_eval_metadata

        sample = Sample(
            task=Task(
                id="t1",
                input=((), {}),
                expected=None,
                metadata={"difficulty": "easy"},
            ),
            method="run",
            agent_class="A",
            scorers=[],
            agent_factory=lambda: None,
        )
        merged = _build_eval_metadata(sample)
        assert merged == {"difficulty": "easy"}

    def test_empty_metadata(self):
        """No metadata at any level produces empty dict."""
        from eval_pipeline.pipeline import Sample, _build_eval_metadata

        sample = Sample(
            task=Task(id="t1", input=((), {}), expected=None),
            method="run",
            agent_class="A",
            scorers=[],
            agent_factory=lambda: None,
        )
        merged = _build_eval_metadata(sample)
        assert merged == {}

    def test_non_scalar_task_metadata_filtered(self):
        """Non-scalar values in task metadata are silently dropped."""
        from eval_pipeline.pipeline import Sample, _build_eval_metadata

        sample = Sample(
            task=Task(
                id="t1",
                input=((), {}),
                expected=None,
                metadata={
                    "difficulty": "hard",
                    "tags": ["a", "b"],  # list — should be dropped
                    "nested": {"x": 1},  # dict — should be dropped
                    "count": 42,  # int — kept
                    "ratio": 0.5,  # float — kept
                    "flag": True,  # bool — kept
                },
            ),
            method="run",
            agent_class="A",
            scorers=[],
            agent_factory=lambda: None,
        )
        merged = _build_eval_metadata(sample)
        assert merged == {"difficulty": "hard", "count": 42, "ratio": 0.5, "flag": True}


# ---------------------------------------------------------------------------
# Span attributes
# ---------------------------------------------------------------------------


class TestSpanAttributes:
    """Test metadata in OTLP span attributes."""

    def test_write_eval_span_with_metadata(self, tmp_path: Path):
        """Extra metadata appears as eval.* attributes in the trace file."""
        trace_file = tmp_path / "test.jsonl"
        trace_file.write_text("")

        write_eval_span_to_trace(
            trace_file=trace_file,
            test_id="t1",
            passed=True,
            weighted_score=1.0,
            model="m1",
            agent_class="A",
            method="run",
            scores=_BASIC_SCORES,
            run_id=3,
            extra_metadata={"difficulty": "hard", "dataset": "kdd"},
        )

        attrs = _read_eval_span_attrs(trace_file)
        assert attrs["eval.run_id"] == 3
        assert attrs["eval.difficulty"] == "hard"
        assert attrs["eval.dataset"] == "kdd"

    def test_write_eval_span_without_metadata(self, tmp_path: Path):
        """No extra eval.* keys when metadata is None."""
        trace_file = tmp_path / "test.jsonl"
        trace_file.write_text("")

        write_eval_span_to_trace(
            trace_file=trace_file,
            test_id="t1",
            passed=True,
            weighted_score=1.0,
            model="m1",
            agent_class="A",
            method="run",
            scores=_BASIC_SCORES,
        )

        attrs = _read_eval_span_attrs(trace_file)
        # No eval.run_id or user metadata keys
        user_keys = {
            k
            for k in attrs
            if k.startswith("eval.")
            and k.split(".")[1]
            not in (
                "test_id",
                "passed",
                "weighted_score",
                "model",
                "agent_class",
                "method",
                "scorer",
            )
        }
        assert user_keys == set()

    def test_reserved_keys_raise_valueerror(self, tmp_path: Path):
        """Metadata keys that collide with built-in eval.* keys raise ValueError."""
        trace_file = tmp_path / "test.jsonl"
        trace_file.write_text("")

        with pytest.raises(ValueError, match="collide with built-in"):
            write_eval_span_to_trace(
                trace_file=trace_file,
                test_id="t1",
                passed=True,
                weighted_score=1.0,
                model="m1",
                agent_class="A",
                method="run",
                scores=_BASIC_SCORES,
                extra_metadata={"model": "COLLISION", "difficulty": "hard"},
            )

    def test_reserved_keys_set(self):
        """Verify all expected built-in keys are reserved."""
        assert "model" in _RESERVED_EVAL_KEYS
        assert "passed" in _RESERVED_EVAL_KEYS
        assert "run_id" in _RESERVED_EVAL_KEYS
        assert "agent_class" in _RESERVED_EVAL_KEYS
        # User-defined keys are NOT reserved
        assert "difficulty" not in _RESERVED_EVAL_KEYS
        assert "dataset" not in _RESERVED_EVAL_KEYS


# ---------------------------------------------------------------------------
# EvalTestResult serialization
# ---------------------------------------------------------------------------


class TestEvalTestResultMetadata:
    """Test eval_metadata field on EvalTestResult."""

    def test_round_trip(self):
        """eval_metadata survives JSON round-trip."""
        meta = {"dataset": "kdd", "difficulty": "hard", "count": 42}
        result = EvalTestResult(
            test_id="t1",
            agent_class="A",
            method="run",
            model="m1",
            variant="run1",
            passed=True,
            scores={},
            input=None,
            output=None,
            expected=None,
            eval_metadata=meta,
        )
        json_str = result.model_dump_json()
        restored = EvalTestResult.model_validate_json(json_str)
        assert restored.eval_metadata == meta

    def test_default_empty(self):
        """eval_metadata defaults to empty dict."""
        result = EvalTestResult(
            test_id="t1",
            agent_class="A",
            method="run",
            model="m1",
            variant="run1",
            passed=True,
            scores={},
            input=None,
            output=None,
            expected=None,
        )
        assert result.eval_metadata == {}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    """Test eval_metadata in config.yaml and task JSONL."""

    def test_load_config_with_eval_metadata(self, tmp_path: Path):
        """Config-level and test-level eval_metadata are parsed."""
        config_yaml = tmp_path / "config.yaml"
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"args":[], "kwargs":{"x":1}, "expected":1}\n')

        config_yaml.write_text(f"""\
name: test
output_dir: {tmp_path / "out"}
eval_metadata:
  dataset: kdd
  env: ci
models:
  m1:
    model_name: openai/test
    endpoint: http://localhost
    api_key_env: TEST_KEY
test_suite:
  - name: t1
    description: test
    agent:
      module: eval_pipeline.scoring
      class: ExactMatchScorer
    method: score
    data_file: {data_file}
    eval_metadata:
      subset: easy
    scorers:
      - name: exact
        class: ExactMatchScorer
""")

        from eval_pipeline.config import load_config

        config = load_config(config_yaml)
        assert config.eval_metadata == {"dataset": "kdd", "env": "ci"}
        assert config.tests[0].eval_metadata == {"subset": "easy"}

    def test_load_config_without_eval_metadata(self, tmp_path: Path):
        """Config without eval_metadata defaults to None."""
        config_yaml = tmp_path / "config.yaml"
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"args":[], "kwargs":{"x":1}, "expected":1}\n')

        config_yaml.write_text(f"""\
name: test
output_dir: {tmp_path / "out"}
models:
  m1:
    model_name: openai/test
    endpoint: http://localhost
    api_key_env: TEST_KEY
test_suite:
  - name: t1
    description: test
    agent:
      module: eval_pipeline.scoring
      class: ExactMatchScorer
    method: score
    data_file: {data_file}
    scorers:
      - name: exact
        class: ExactMatchScorer
""")

        from eval_pipeline.config import load_config

        config = load_config(config_yaml)
        assert config.eval_metadata is None
        assert config.tests[0].eval_metadata is None

    def test_load_tasks_reads_metadata(self, tmp_path: Path):
        """load_tasks reads metadata from JSONL."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text(
            '{"args":[], "kwargs":{"x":1}, "expected":1, "metadata":{"difficulty":"hard"}}\n'
            '{"args":[], "kwargs":{"x":2}, "expected":2}\n'
        )

        from eval_pipeline.config import load_tasks

        tasks = load_tasks(data_file)
        assert tasks[0].metadata == {"difficulty": "hard"}
        assert tasks[1].metadata == {}  # default when no metadata field


# ---------------------------------------------------------------------------
# SubprocessTaskInput serialization
# ---------------------------------------------------------------------------


class TestSubprocessTaskInputMetadata:
    """Test eval_metadata on SubprocessTaskInput round-trip."""

    def test_round_trip(self):
        from eval_pipeline.eval_types import AgentSpec, TaskSpec

        inp = SubprocessTaskInput(
            agent_spec=AgentSpec(agent_module="m", agent_class="A", method="run", client_config={}),
            task=TaskSpec(id="t1", input=[[], {}], expected=None),
            scorers=[],
            sample_id="s1",
            model="m1",
            trace_dir="/tmp",
            eval_metadata={"difficulty": "hard", "dataset": "kdd"},
        )
        json_str = inp.model_dump_json()
        restored = SubprocessTaskInput.model_validate_json(json_str)
        assert restored.eval_metadata == {"difficulty": "hard", "dataset": "kdd"}
