# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config_adapter - bridging YAML config to Evaluator."""

import pytest


class TestEvaluatorFromConfig:
    """Tests for evaluator_from_config function."""

    def test_loads_models_as_clients(self, tmp_path):
        """Models from config are converted to LLM clients."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("""
name: test_eval
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 4096

agent_models:
  - gpt-4

output_dir: results
test_suite: []
""")
        from eval_pipeline.config import evaluator_from_config

        evaluator = evaluator_from_config(config_file)

        assert "gpt-4" in evaluator.models
        assert evaluator._default_model_ids == ["gpt-4"]

    def test_loads_tests_with_exact_match_scorer(self, tmp_path):
        """Tests with ExactMatchScorer are loaded correctly."""
        # Create minimal agent module
        agent_dir = tmp_path / "agents"
        agent_dir.mkdir()
        (agent_dir / "__init__.py").write_text("")
        (agent_dir / "test_agent.py").write_text("""
from nooa import Agent


class TestAgent(Agent):
    async def classify(self, text: str) -> str:
        ...
""")
        # Create data file
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"kwargs": {"text": "hello"}, "expected": "positive"}\n')

        # Create config
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
name: test_eval
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY

agent_models:
  - gpt-4

output_dir: results

test_suite:
  - name: test1
    agent:
      module: agents.test_agent
      class: TestAgent
    method: classify
    data_file: {data_file}
    scorers:
      - name: exact
        class: ExactMatchScorer
        weight: 1.0
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            from eval_pipeline.config import evaluator_from_config

            evaluator = evaluator_from_config(config_file)

            assert "test1" in evaluator.tests
            assert len(evaluator.tests["test1"].scorers) == 1
            assert evaluator.tests["test1"].scorers[0].name == "exact"
        finally:
            sys.path.remove(str(tmp_path))

    def test_loads_tests_with_llm_judge_scorer(self, tmp_path):
        """Tests with LLMJudgeScorer are loaded correctly with model reference."""
        # Create minimal agent module
        agent_dir = tmp_path / "agents"
        agent_dir.mkdir()
        (agent_dir / "__init__.py").write_text("")
        (agent_dir / "test_agent.py").write_text("""
from nooa import Agent


class TestAgent(Agent):
    async def classify(self, text: str) -> str:
        ...
""")
        # Create data file
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"kwargs": {"text": "hello"}, "expected": "positive"}\n')

        # Create config with LLMJudgeScorer referencing a model
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
name: test_eval
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
  claude-judge:
    model_name: anthropic/claude-3-haiku
    endpoint: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY

agent_models:
  - gpt-4

output_dir: results

test_suite:
  - name: test1
    agent:
      module: agents.test_agent
      class: TestAgent
    method: classify
    data_file: {data_file}
    scorers:
      - name: judge
        class: LLMJudgeScorer
        weight: 1.0
        rubric: "Is this good?"
        model: claude-judge
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            from eval_pipeline.config import evaluator_from_config

            evaluator = evaluator_from_config(config_file)

            assert "test1" in evaluator.tests
            assert len(evaluator.tests["test1"].scorers) == 1
            scorer_cfg = evaluator.tests["test1"].scorers[0]
            assert scorer_cfg.name == "judge"
            # Scorer should have been created with the model_spec (stores _llm for creating fresh agents)
            assert hasattr(scorer_cfg.scorer, "_llm")
            assert hasattr(scorer_cfg.scorer, "_rubric")
        finally:
            sys.path.remove(str(tmp_path))

    def test_llm_judge_scorer_missing_model_raises(self, tmp_path):
        """LLMJudgeScorer with missing model reference raises error."""
        # Create minimal agent module
        agent_dir = tmp_path / "agents"
        agent_dir.mkdir()
        (agent_dir / "__init__.py").write_text("")
        (agent_dir / "test_agent.py").write_text("""
from nooa import Agent


class TestAgent(Agent):
    async def classify(self, text: str) -> str:
        ...
""")
        # Create data file
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"kwargs": {"text": "hello"}, "expected": "positive"}\n')

        # Create config with LLMJudgeScorer referencing non-existent model
        config_file = tmp_path / "config.yaml"
        config_file.write_text(f"""
name: test_eval
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY

agent_models:
  - gpt-4

output_dir: results

test_suite:
  - name: test1
    agent:
      module: agents.test_agent
      class: TestAgent
    method: classify
    data_file: {data_file}
    scorers:
      - name: judge
        class: LLMJudgeScorer
        weight: 1.0
        rubric: "Is this good?"
        model: nonexistent-model
""")

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            from eval_pipeline.config import evaluator_from_config

            with pytest.raises(ValueError, match="model 'nonexistent-model' not found"):
                evaluator_from_config(config_file)
        finally:
            sys.path.remove(str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
