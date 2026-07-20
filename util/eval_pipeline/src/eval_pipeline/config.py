# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration loading for eval_pipeline.

This module provides:
- YAML config parsing (load_config, load_tasks)
- Config-to-Evaluator bridge (evaluator_from_config)

The Evaluator class itself has no knowledge of YAML - this module
handles all configuration loading and conversion.
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from nooa.unifiedllm import CompletionClient, RetryConfig

from .eval_types import ModelSpec, Tier
from .models import Task

if TYPE_CHECKING:
    from .evaluator import Evaluator

# Track if error capture has been enabled (only enable once globally)
_error_capture_enabled = False


def _enable_error_capture_if_requested():
    """Enable HTTP error capture if CAPTURE_LLM_ERRORS env var is set.

    This function is idempotent - will only enable once even if called multiple times.
    """
    global _error_capture_enabled
    if not _error_capture_enabled and os.getenv("CAPTURE_LLM_ERRORS"):
        from nooa.unifiedllm.http_logging import enable_http_request_logging

        # Determine output directory - use CAPTURE_LLM_ERRORS value if it's a path
        capture_setting = os.getenv("CAPTURE_LLM_ERRORS")
        if capture_setting and capture_setting.lower() not in ["1", "true", "yes"]:
            output_dir = Path(capture_setting)
        else:
            output_dir = Path("eval_errors")

        enable_http_request_logging(
            output_dir=output_dir,
            errors_only=True,
            save_responses=True,
            verbose=os.getenv("CAPTURE_LLM_ERRORS_VERBOSE", "false").lower()
            in ["1", "true", "yes"],
        )
        _error_capture_enabled = True
        print(f"✓ LLM error capture enabled - writing to: {output_dir / 'llm_errors.jsonl'}")


# =============================================================================
# Config Dataclasses
# =============================================================================


@dataclass
class TestConfig:
    """Configuration for a single test."""

    name: str
    description: str
    tier: Tier
    agent_module: str
    agent_class: str
    agent_method: str
    data_file: Path
    limit: int | None
    scorers: list[dict]
    eval_metadata: dict[str, str | int | float | bool] | None = None


@dataclass
class StrategyConfig:
    """Configuration for a custom strategy class.

    Attributes:
        module: Module path (e.g., "agents.strategy")
        class_name: Class name (e.g., "MyCustomStrategy")
    """

    module: str
    class_name: str


@dataclass
class EvalConfig:
    """Full evaluation configuration.

    Attributes:
        name: Experiment name
        description: Description of the evaluation
        output_dir: Directory for output files
        models: Dict of model_id -> ModelSpec for all models used
        agent_models: List of model IDs to use for agent evaluation
        tests: List of test configurations
        pass_threshold: Minimum weighted score to pass (default 0.5)
        timeout_seconds: Optional timeout per sample in seconds (None = no timeout)
        default_strategy: Strategy config - either:
            - string: Built-in strategy name ("codeact", "pure_python", "reflexion")
            - StrategyConfig: Custom strategy class to import
    """

    name: str
    description: str
    output_dir: Path
    models: dict[str, ModelSpec]  # All models defined in config, keyed by ID
    agent_models: list[str]  # Which model IDs to use for agent evaluation
    tests: list[TestConfig]
    pass_threshold: float = 0.5
    timeout_seconds: float | None = None
    default_strategy: str | StrategyConfig | None = None
    no_cache: bool = False
    # If set, also send traces to Langfuse (in addition to local JSONL files used for scoring)
    langfuse_host: str | None = None
    # Experiment-wide eval metadata — each key becomes an eval.{key} span attribute
    # and a dynamic column in the trace viewer.  Test-level and task-level metadata
    # override these values.
    eval_metadata: dict[str, str | int | float | bool] | None = None


# =============================================================================
# YAML Loading
# =============================================================================


def _resolve_registry_model(registry_name: str, overrides: dict | None = None) -> dict:
    """Resolve a model from the unifiedllm registry into ModelSpec fields.

    Args:
        registry_name: Registry key (e.g., "claude-sonnet-4-5")
        overrides: Optional dict of field overrides (max_tokens, reasoning_effort, etc.)

    Returns:
        Dict of ModelSpec fields ready for construction.

    Raises:
        ValueError: If the registry name is not found.
    """
    # eval_pipeline is launched as a standalone CLI without the TUI
    # bootstrap, so the registry would otherwise be empty. Trigger the
    # lazy auto-load before reading MODELS.
    from nooa.unifiedllm.registry import MODELS, _registry_lock, ensure_loaded

    ensure_loaded()

    # Snapshot under the lock so a concurrent reload_registry() can't
    # make us observe a half-cleared MODELS dict.
    with _registry_lock:
        config = dict(MODELS.get(registry_name)) if registry_name in MODELS else None
        available_keys = sorted(MODELS.keys())[:10]

    if config is None:
        available = ", ".join(available_keys)
        raise ValueError(
            f"Model '{registry_name}' not found in unifiedllm registry. "
            f"Available (first 10): {available}"
        )

    # Build ModelSpec fields from registry config
    # model_name from config is the litellm-ready string; fall back to registry key
    fields: dict = {
        "model_name": config.get("model_name", registry_name),
    }
    if "api_base" in config:
        fields["endpoint"] = config["api_base"]
    if "api_key_env" in config:
        fields["api_key_env"] = config["api_key_env"]

    # Copy optional fields from registry
    for key in (
        "max_tokens",
        "temperature",
        "top_p",
        "reasoning_effort",
        "max_thinking_tokens",
        "max_retries",
        "retry_on_empty_content",
        "client_type",
    ):
        if key in config:
            fields[key] = config[key]

    # Apply overrides (from config YAML)
    if overrides:
        fields.update(overrides)

    return fields


def load_config(config_path: Path) -> EvalConfig:
    """Load configuration from YAML file.

    Model format - three styles supported:

    1. Full spec (existing):
        models:
          gpt-oss-20b:
            model_name: openai/nvidia/openai/gpt-oss-20b
            endpoint: https://inference-api.nvidia.com/v1
            api_key_env: NVIDIA_INFERENCE_API_KEY
            max_tokens: 16384

    2. Registry reference (new - string value):
        models:
          claude-haiku: azure/anthropic/claude-haiku-4-5

    3. Registry reference with overrides (new - dict with 'registry' key):
        models:
          claude-haiku:
            registry: azure/anthropic/claude-haiku-4-5
            max_tokens: 64000
            reasoning_effort: medium

    4. Auto-resolve from agent_models (new):
        If agent_models lists an ID not in models, it's treated as a
        registry name and auto-resolved.

    Scorers can reference models by ID:
        scorers:
          - name: method_judge
            class: LLMJudgeScorer
            model: nemotron3-nano-30b  # References key from models
    """
    with open(config_path) as f:
        data = yaml.safe_load(f)

    # Parse models as dict keyed by ID
    models: dict[str, ModelSpec] = {}
    models_data = data.get("models", {})

    if models_data and not isinstance(models_data, dict):
        raise TypeError("models must be a dict keyed by model ID")

    for model_id, model_cfg in (models_data or {}).items():
        if isinstance(model_cfg, str):
            # Style 2: "claude-haiku: azure/anthropic/claude-haiku-4-5"
            fields = _resolve_registry_model(model_cfg)
            models[model_id] = ModelSpec(id=model_id, **fields)

        elif isinstance(model_cfg, dict) and "registry" in model_cfg:
            # Style 3: dict with "registry" key + optional overrides
            registry_name = model_cfg["registry"]
            overrides = {k: v for k, v in model_cfg.items() if k not in ("name", "registry")}
            fields = _resolve_registry_model(registry_name, overrides)
            models[model_id] = ModelSpec(id=model_id, **fields)

        elif isinstance(model_cfg, dict):
            # Style 1: Full spec (existing behavior)
            models[model_id] = ModelSpec(
                id=model_id,
                model_name=model_cfg["model_name"],
                endpoint=model_cfg.get("endpoint"),
                api_key_env=model_cfg.get("api_key_env"),
                max_tokens=model_cfg.get("max_tokens"),
                temperature=model_cfg.get("temperature"),
                top_p=model_cfg.get("top_p"),
                reasoning_effort=model_cfg.get("reasoning_effort"),
                max_thinking_tokens=model_cfg.get("max_thinking_tokens"),
                max_retries=model_cfg.get("max_retries"),
                retry_on_empty_content=model_cfg.get("retry_on_empty_content", False),
                no_cache=model_cfg.get("no_cache", False),
            )
        else:
            raise TypeError(
                f"Model '{model_id}': expected string (registry name) or dict, got {type(model_cfg)}"
            )

    # Which models to use for agent evaluation
    agent_models_raw = data.get("agent_models", list(models.keys()))

    # Auto-resolve agent_models entries not found in models dict (Style 4)
    agent_models = []
    for model_ref in agent_models_raw:
        if model_ref not in models:
            # Try to resolve from registry — use the registry name as the ID
            try:
                fields = _resolve_registry_model(model_ref)
                models[model_ref] = ModelSpec(id=model_ref, **fields)
            except ValueError:
                # Not in registry either — will fail later at validation
                pass
        agent_models.append(model_ref)

    tests = []
    for test in data.get("test_suite", []):
        agent = test.get("agent", {})
        tests.append(
            TestConfig(
                name=test["name"],
                description=test.get("description", ""),
                tier=Tier(test.get("tier", "stable")),
                agent_module=agent.get("module", ""),
                agent_class=agent.get("class", ""),
                agent_method=test.get("method", "run"),
                data_file=Path(test.get("data_file", "")),
                limit=test.get("limit"),
                scorers=test.get("scorers", []),
                eval_metadata=test.get("eval_metadata"),
            )
        )

    # Parse default_strategy - can be string or dict with module/class
    default_strategy_raw = data.get("default_strategy")
    default_strategy: str | StrategyConfig | None = None
    if default_strategy_raw is not None:
        if isinstance(default_strategy_raw, str):
            # Built-in strategy name (e.g., "codeact", "pure_python")
            default_strategy = default_strategy_raw
        elif isinstance(default_strategy_raw, dict):
            # Custom strategy class
            default_strategy = StrategyConfig(
                module=default_strategy_raw.get("module", ""),
                class_name=default_strategy_raw.get("class", ""),
            )

    tracing = data.get("tracing") or {}
    langfuse_host = tracing.get("langfuse_host") if isinstance(tracing, dict) else None
    if langfuse_host is None:
        langfuse_host = data.get("langfuse_host")

    return EvalConfig(
        name=data.get("name", "eval"),
        description=data.get("description", ""),
        output_dir=Path(data.get("output_dir", "experiments")),
        models=models,
        agent_models=agent_models,
        tests=tests,
        pass_threshold=data.get("pass_threshold", 0.5),
        timeout_seconds=data.get("timeout_seconds"),
        default_strategy=default_strategy,
        no_cache=data.get("no_cache", False),
        langfuse_host=langfuse_host,
        eval_metadata=data.get("eval_metadata"),
    )


def load_tasks(data_file: Path, limit: int | None = None) -> list[Task]:
    """Load tasks from JSONL data file.

    Data file format (each line):
        {"args": [...], "kwargs": {...}, "expected": ...}
        Optionally: {"id": "custom_id", ...} to preserve specific task IDs

    Args:
        data_file: Path to JSONL file
        limit: Max tasks to load

    Returns:
        List of Tasks with input as (args, kwargs) tuple
    """
    tasks = []
    with open(data_file) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            data = json.loads(line)
            args = tuple(data.get("args", []))
            kwargs = data.get("kwargs", {})
            # Use explicit id if provided, otherwise generate from line number
            task_id = data.get("id") or f"{data_file.stem}_{i + 1:03d}"
            tasks.append(
                Task(
                    id=task_id,
                    input=(args, kwargs),
                    expected=data["expected"],
                    metadata=data.get("metadata", {}),
                )
            )
    return tasks


# =============================================================================
# Config-to-Evaluator Bridge
# =============================================================================


def evaluator_from_config(
    config_path: str | Path, *, no_cache_override: bool | None = None
) -> Evaluator:
    """Create an Evaluator from a YAML config file.

    This is the bridge between YAML config and the pure Python Evaluator API.

    Args:
        config_path: Path to config.yaml
        no_cache_override: If set, overrides both config-level and per-model no_cache settings.
            Useful for CLI --no-cache flag.

    Returns:
        Configured Evaluator instance with all tests loaded
    """
    from .evaluator import EvalTest, Evaluator

    config_path = Path(config_path)
    config = load_config(config_path)
    config_dir = config_path.parent

    # Enable error capture if CAPTURE_LLM_ERRORS is set (global, once)
    _enable_error_capture_if_requested()

    # Drop unsupported params (e.g., tool_choice for some Azure models via NVIDIA)
    import litellm

    litellm.drop_params = True

    # Create LLM client factories from ModelSpecs (fresh client per sample for parallelism)
    models = {}
    model_factories = {}
    model_metadata = {}

    for model_id, spec in config.models.items():
        # Create a factory that produces fresh clients
        def make_client_factory(s=spec):
            def factory():
                # Build retry config if any retry options specified
                retry_config = None
                if s.max_retries is not None or s.retry_on_empty_content:
                    retry_config = RetryConfig(
                        max_retries=s.max_retries if s.max_retries is not None else 3,
                        retry_on_empty_content=s.retry_on_empty_content,
                    )

                # Build config dict
                # Guard against api_key_env=None (public registry models that
                # delegate API key resolution to litellm)
                api_key = os.getenv(s.api_key_env, "") if s.api_key_env else ""
                config_dict = {
                    "model": s.model_name,
                    "api_base": s.endpoint,
                    "api_key": api_key,
                    "max_tokens": s.max_tokens,
                }

                # Forward sampling params from the registry / config so
                # models.yaml temperature/top_p actually reach the client.
                if s.temperature is not None:
                    config_dict["temperature"] = s.temperature
                if s.top_p is not None:
                    config_dict["top_p"] = s.top_p

                # Add reasoning support for Claude and other models
                if s.reasoning_effort:
                    config_dict["reasoning_effort"] = s.reasoning_effort
                    # Tell litellm to allow reasoning_effort through for OpenAI-compatible endpoints
                    config_dict["allowed_openai_params"] = ["reasoning_effort"]

                # Build extra_body for litellm passthrough
                extra_body: dict = {}
                if s.max_thinking_tokens:
                    extra_body["nvext"] = {"max_thinking_tokens": s.max_thinking_tokens}

                # Disable NVIDIA inference API caching (important for pass@k diversity)
                # Priority: CLI override > per-model > config-level
                use_no_cache = (
                    no_cache_override
                    if no_cache_override is not None
                    else (s.no_cache or config.no_cache)
                )
                if use_no_cache:
                    extra_body["cache"] = {"no-cache": True}

                if extra_body:
                    config_dict["extra_body"] = extra_body

                # Dispatch based on client_type from registry config
                if getattr(s, "client_type", None) == "responses":
                    from nooa.unifiedllm import ResponsesClient

                    return ResponsesClient(retry_config=retry_config, **config_dict)
                return CompletionClient(retry_config=retry_config, **config_dict)

            return factory

        # Store both: shared client (for backwards compat) and factory (for parallelism)
        model_factories[model_id] = make_client_factory()
        models[model_id] = model_factories[model_id]()  # Create one for backwards compat

        model_metadata[model_id] = {
            "id": model_id,
            "model_name": spec.model_name,
            "endpoint": spec.endpoint,
            "api_key_env": spec.api_key_env,
            "max_tokens": spec.max_tokens,
            "max_retries": spec.max_retries,
            "retry_on_empty_content": spec.retry_on_empty_content,
            "reasoning_effort": spec.reasoning_effort,
            "max_thinking_tokens": spec.max_thinking_tokens,
            "no_cache": spec.no_cache,
        }

    # Create evaluator
    evaluator = Evaluator(
        models=models,
        output_dir=config.output_dir,
        name=config.name,
        pass_threshold=config.pass_threshold,
        timeout_seconds=config.timeout_seconds,
    )
    evaluator._model_metadata = model_metadata
    evaluator._model_factories = model_factories  # Enable per-sample client creation
    evaluator._default_model_ids = config.agent_models
    evaluator._langfuse_host = getattr(config, "langfuse_host", None)

    # Load all tests from config
    for test_cfg in config.tests:
        # Load agent class
        module = importlib.import_module(test_cfg.agent_module)
        agent_class = getattr(module, test_cfg.agent_class)

        # Load tasks
        data_path = config_dir / test_cfg.data_file
        if not data_path.exists():
            data_path = Path(test_cfg.data_file)
        tasks = load_tasks(data_path, test_cfg.limit)

        # Create scorers
        scorers = _create_scorers(test_cfg.scorers, config.models)
        # Merge config-level and test-level eval_metadata (test overrides config)
        merged_meta: dict[str, str | int | float | bool] = {}
        if config.eval_metadata:
            merged_meta |= config.eval_metadata
        if test_cfg.eval_metadata:
            merged_meta |= test_cfg.eval_metadata

        evaluator.tests[test_cfg.name] = EvalTest(
            name=test_cfg.name,
            agent_class=agent_class,
            method=test_cfg.agent_method,
            data=tasks,
            scorers=scorers,
            description=test_cfg.description,
            tier=test_cfg.tier.value if test_cfg.tier else None,  # Convert Tier enum to string
            eval_metadata=merged_meta or None,
        )

    return evaluator


def _resolve_model_spec(
    model_id: str,
    model_specs: dict[str, ModelSpec],
    scorer_name: str,
) -> ModelSpec:
    """Resolve a model reference to a ModelSpec.

    Resolution order:
    1. Key in config file's models dict
    2. Registry key in unifiedllm registry
    3. Raise ValueError

    Args:
        model_id: Model reference (config key or registry name)
        model_specs: Models already loaded from config
        scorer_name: Scorer name for error messages

    Returns:
        Resolved ModelSpec
    """
    # 1. Config key
    if model_id in model_specs:
        return model_specs[model_id]

    # 2. Registry fallback
    try:
        fields = _resolve_registry_model(model_id)
        spec = ModelSpec(id=model_id, **fields)
        # Cache it so subsequent lookups (and evaluator_from_config) can find it
        model_specs[model_id] = spec
        return spec
    except ValueError:
        pass

    # 3. Not found anywhere
    available = list(model_specs.keys())
    raise ValueError(
        f"Scorer '{scorer_name}': model '{model_id}' not found in config or registry. "
        f"Config models: {available}"
    )


def _create_scorers(
    scorer_configs: list[dict],
    model_specs: dict,  # model_id -> ModelSpec (may be mutated to add registry lookups)
) -> list:
    """Create scorers from config dicts.

    Built-in scorers (ExactMatchScorer, LLMJudgeScorer, etc.) are resolved by
    short name.  Custom scorers use a fully-qualified dotted Python path as the
    ``class`` value; any extra YAML keys beyond ``name``, ``weight``, and
    ``class`` are forwarded as **kwargs to the scorer constructor.

    Custom scorer YAML example::

        scorers:
          - name: kdd_accuracy
            class: my_project.scoring.KDDScorer   # dotted path → dynamic import
            weight: 1.0
            threshold: 0.8                         # forwarded as kwarg
            categories: ["billing", "technical"]   # forwarded as kwarg

    The scorer class must implement ``score(self, ctx) -> ScoreResult``.
    ``ctx.metadata`` carries per-task data from the task JSON — use it for
    per-task rubrics, difficulty tags, or other task-level scorer config.

    Custom scorer Python example::

        class KDDScorer:
            def __init__(self, threshold=0.5, categories=None):
                self.threshold = threshold
                self.categories = categories or []

            def score(self, ctx: ScoringContext) -> ScoreResult:
                rubric = ctx.metadata.get("rubric", "default")
                match = ctx.expected == ctx.actual
                return ScoreResult(score=1.0 if match else 0.0, reasoning=rubric)
    """
    from .scoring import (
        ExactMatchScorer,
        LLMJudgeScorer,
        LLMMethodologyScorer,
        ModeSelectionScorer,
        ScorerConfig,
        TypeMatchScorer,
    )

    scorers = []
    for cfg in scorer_configs:
        name = cfg["name"]
        weight = cfg.get("weight", 1.0)
        class_name = cfg.get("class", "")

        # Track the class path and kwargs for subprocess reconstruction
        scorer_class_path = ""
        scorer_kwargs: dict = {}

        if class_name == "ExactMatchScorer":
            scorer = ExactMatchScorer()
            scorer_class_path = "eval_pipeline.scoring.ExactMatchScorer"
        elif class_name == "LLMJudgeScorer":
            rubric = cfg.get("rubric", "")
            model_id = cfg.get("model")
            skip_prefill = cfg.get("skip_prefill", False)
            if not model_id:
                raise ValueError(f"Scorer '{name}': LLMJudgeScorer requires 'model' parameter")
            spec = _resolve_model_spec(model_id, model_specs, name)
            scorer = LLMJudgeScorer(
                rubric=rubric,
                model_spec=spec,
                skip_prefill=skip_prefill,
            )
            scorer_class_path = "eval_pipeline.scoring.LLMJudgeScorer"
            scorer_kwargs = {
                "rubric": rubric,
                "model_spec": spec.model_dump(),
                "skip_prefill": skip_prefill,
            }
        elif class_name == "ModeSelectionScorer":
            expected = cfg.get("expected")
            if not expected:
                raise ValueError(
                    f"Scorer '{name}': ModeSelectionScorer requires 'expected' parameter"
                )
            scorer = ModeSelectionScorer(expected=expected)
            scorer_class_path = "eval_pipeline.scoring.ModeSelectionScorer"
            scorer_kwargs = {"expected": expected}
        elif class_name == "TypeMatchScorer":
            scorer = TypeMatchScorer()
            scorer_class_path = "eval_pipeline.scoring.TypeMatchScorer"
        elif class_name == "LLMMethodologyScorer":
            rubric = cfg.get("rubric", "")
            model_id = cfg.get("model")
            if not model_id:
                raise ValueError(
                    f"Scorer '{name}': LLMMethodologyScorer requires 'model' parameter"
                )
            spec = _resolve_model_spec(model_id, model_specs, name)
            skip_prefill = cfg.get("skip_prefill", True)
            scorer = LLMMethodologyScorer(
                rubric=rubric,
                model_spec=spec,
                skip_prefill=skip_prefill,
            )
            scorer_class_path = "eval_pipeline.scoring.LLMMethodologyScorer"
            scorer_kwargs = {
                "rubric": rubric,
                "model_spec": spec.model_dump(),
                "skip_prefill": skip_prefill,
            }
        else:
            # Dynamic import for custom scorers: "my_package.module.MyScorer"
            if "." not in class_name:
                raise ValueError(f"Unknown scorer class: {class_name}")
            module_path, cls_name = class_name.rsplit(".", 1)
            try:
                module = importlib.import_module(module_path)
            except ModuleNotFoundError as e:
                raise ValueError(
                    f"Scorer '{name}': cannot import module '{module_path}': {e}"
                ) from e
            try:
                scorer_cls = getattr(module, cls_name)
            except AttributeError:
                raise ValueError(
                    f"Scorer '{name}': module '{module_path}' has no class '{cls_name}'"
                ) from None
            extra_kwargs = {k: v for k, v in cfg.items() if k not in ("name", "weight", "class")}
            scorer = scorer_cls(**extra_kwargs)
            scorer_class_path = class_name
            scorer_kwargs = extra_kwargs

        scorers.append(
            ScorerConfig(
                name=name,
                weight=weight,
                scorer=scorer,
                scorer_class=scorer_class_path,
                scorer_kwargs=scorer_kwargs,
            )
        )

    return scorers
