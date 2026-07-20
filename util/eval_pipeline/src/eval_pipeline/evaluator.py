# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""High-level evaluation API.

This module provides the main `Evaluator` class for running agent evaluations.
It is pure Python with no YAML knowledge - use config_adapter for YAML loading.

Example:
    from eval_pipeline import Evaluator, ExactMatchScorer
    from nooa.unifiedllm import CompletionClient

    evaluator = Evaluator(
        models={"gpt-4": CompletionClient(model="openai/gpt-4", ...)},
        output_dir="experiments/results",
    )

    evaluator.add_test(
        name="sentiment",
        agent_class=SentimentAgent,
        method="classify_single",
        data=[{"kwargs": {"text": "I love this"}, "expected": "positive"}],
        scorers=[ExactMatchScorer()],
    )

    results = await evaluator.run(models=["gpt-4"], runs=3, parallel=10)
"""

import os
import sys
import tempfile
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from .eval_types import Tier
from .experiment_writer import ExperimentWriter, NullExperimentWriter
from .models import Task
from .pipeline import PipelineConfig, Sample, run_evaluation
from .scoring import ScorerConfig

_DEFAULT_OTLP_ENDPOINT = "http://localhost:5001/v1/traces"


def _probe_otlp(endpoint: str, timeout: float = 0.5) -> bool:
    """Return True if the OTLP endpoint is reachable (any HTTP response).

    Uses GET /api/eval/health instead of POSTing to /v1/traces so we don't
    accidentally create ``unknown_*`` sessions in the viewer's store.
    """
    base = endpoint.rstrip("/").removesuffix("/v1/traces").removesuffix("/v1")
    health_url = f"{base}/api/eval/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        urllib.request.urlopen(req, timeout=timeout)
        return True
    except urllib.error.HTTPError:
        return True  # server is up, just rejected the request
    except Exception:
        return False


class LLMClient(Protocol):
    """Protocol for LLM clients."""

    pass  # Duck typing - any client with appropriate methods works


@dataclass
class EvalTest:
    """Definition of a test to run."""

    name: str
    agent_class: type
    method: str
    data: list[Task]
    scorers: list[ScorerConfig]
    tier: Tier = Tier.STABLE
    description: str = ""
    eval_metadata: dict[str, str | int | float | bool] | None = None


@dataclass
class EvalResults:
    """Results from an evaluation run."""

    results: list[Any]
    output_file: Path | None
    passed: int
    total: int

    @property
    def pass_rate(self) -> float:
        """Pass rate as a percentage."""
        return (self.passed / self.total * 100) if self.total > 0 else 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        return f"{self.passed}/{self.total} passed ({self.pass_rate:.1f}%)"


class Evaluator:
    """High-level API for running agent evaluations.

    Pure Python - no YAML knowledge. For YAML config, use:
        from eval_pipeline.config_adapter import evaluator_from_config
        evaluator = evaluator_from_config("config.yaml")

    Attributes:
        models: Dict of model_id -> LLMClient
        output_dir: Directory for output files
        name: Experiment name
        tests: Dict of test_name -> EvalTest
    """

    def __init__(
        self,
        models: dict[str, Any] | None = None,
        output_dir: str | Path = "experiments",
        name: str = "eval",
        pass_threshold: float = 0.5,
        timeout_seconds: float | None = None,
    ):
        """Create an Evaluator.

        Args:
            models: Dict of model_id -> LLMClient
            output_dir: Directory for output files
            name: Experiment name for output file naming
            pass_threshold: Minimum weighted score to pass (default 0.5)
            timeout_seconds: Default timeout per sample in seconds (None = no timeout)
        """
        self.models: dict[str, Any] = models or {}
        self.output_dir = Path(output_dir).resolve()
        self.name = name
        self.pass_threshold = pass_threshold
        self.timeout_seconds = timeout_seconds
        self.tests: dict[str, EvalTest] = {}

        # Optional metadata (set by config_adapter)
        self._model_metadata: dict[str, dict] = {}
        self._default_model_ids: list[str] = []
        # Client factories for per-sample client creation (set by config_adapter)
        # If set, creates a fresh client per sample for better parallelism
        self._model_factories: dict[str, callable] = {}
        self._langfuse_host: str | None = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Evaluator":
        """Create an Evaluator from a YAML config file.

        This is a convenience method that delegates to config.
        """
        from .config import evaluator_from_config

        return evaluator_from_config(config_path)

    def add_test(
        self,
        name: str,
        agent_class: type,
        method: str,
        data: list[dict] | list[Task],
        scorers: list,
        tier: Tier = Tier.STABLE,
        description: str = "",
    ) -> None:
        """Add a test to the evaluator.

        Args:
            name: Test name
            agent_class: Agent class to instantiate
            method: Method name to call
            data: List of test cases (dicts with kwargs/expected or Task objects)
            scorers: List of scorer instances
            tier: Test tier (default: Tier.STABLE)
            description: Optional description
        """
        # Convert dicts to Tasks if needed
        tasks = []
        for i, item in enumerate(data):
            if isinstance(item, Task):
                tasks.append(item)
            else:
                tasks.append(
                    Task(
                        id=f"{name}_{i + 1:03d}",
                        input=((), item.get("kwargs", {})),
                        expected=item.get("expected"),
                    )
                )

        # Convert scorers to ScorerConfig if needed
        scorer_configs = []
        for s in scorers:
            if isinstance(s, ScorerConfig):
                scorer_configs.append(s)
            else:
                scorer_configs.append(ScorerConfig(name=s.__class__.__name__, weight=1.0, scorer=s))

        self.tests[name] = EvalTest(
            name=name,
            agent_class=agent_class,
            method=method,
            data=tasks,
            scorers=scorer_configs,
            description=description,
            tier=tier,
        )

    def _start_tracing(
        self,
        traces_dir: Path,
        no_files: bool,
        run_experiment_name: str,
        trace_files: bool = False,
    ):
        """Start headless OTLP backend and configure tracing exporters.

        Args:
            traces_dir: Directory for trace file output.
            no_files: Suppress all file output (overrides trace_files).
            run_experiment_name: Experiment name for OTLP session grouping.
            trace_files: Write JSONL trace files even when viewer is running (default: False).
                        Has no effect when no_files=True.

        Returns (backend, headless_base_url, use_viewer, external_otlp_endpoint).
        Caller is responsible for calling backend.stop() in a finally block.
        """
        from nooa.tracing import enable_tracing, exporters

        from .headless_backend import HeadlessOtlpBackend

        _backend = HeadlessOtlpBackend()
        _headless_base = _backend.start()

        _external_otlp = os.getenv("OTLP_ENDPOINT", _DEFAULT_OTLP_ENDPOINT)
        _use_viewer = _probe_otlp(_external_otlp)

        _headless_journal = exporters.journal(endpoint=f"{_headless_base}/v1/traces")
        _headless_journal.synchronous = True  # Use SimpleSpanProcessor (no queue drops)
        exporter_list: list = [_headless_journal]
        if _use_viewer:
            exporter_list.append(exporters.journal(endpoint=_external_otlp))
        # File traces: always suppressed when no_files. Otherwise written unless a
        # viewer is active (traces go to OTLP only), which --trace-files overrides.
        if not no_files and (not _use_viewer or trace_files):
            exporter_list.append(exporters.jsonl(traces_dir))

        if self._langfuse_host:
            try:
                exporter_list.append(exporters.langfuse(host=self._langfuse_host))
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(f"Langfuse tracing not enabled: {e}")
        enable_tracing(exporters=exporter_list, experiment=run_experiment_name)
        if _use_viewer:
            viewer_base = _external_otlp.split("/v1/")[0]
            print(f"Viewer: {viewer_base}/eval/experiment/{run_experiment_name}")

        return _backend, _headless_base, _use_viewer, _external_otlp

    async def run(
        self,
        tests: list[str] | None = None,
        models: list[str] | None = None,
        runs: int = 1,
        limit: int | None = None,
        task_ids: list[str] | None = None,
        parallel: int = 1,
        timeout: float | None = None,
        quiet: bool = False,
        engine_type: str = "asyncio",
        batch_size: int = 1,
        on_progress_hook: Callable[[], None] | None = None,
        agent_label: str | None = None,
        agent_variants: list[tuple[str, list]] | None = None,
        no_files: bool = False,
        trace_files: bool = False,
        memory_limit_mb: int | None = None,
    ) -> EvalResults:
        """Run the evaluation.

        Creates all samples upfront, then runs them in one parallel batch.

        Args:
            tests: Which tests to run (None = all)
            models: Which models to use (None = all)
            runs: Number of runs per test for self-consistency
            limit: Limit samples per test
            task_ids: Run only tasks with these IDs (None = all tasks)
            parallel: Max concurrent samples (default: 1)
            timeout: Timeout per sample in seconds (None = no timeout)
            quiet: Minimal output
            engine_type: Execution engine - "asyncio" (default, I/O-bound) or "subprocess" (CPU-bound)
            batch_size: Tasks per subprocess when using subprocess engine (default: 1)
            on_progress_hook: Optional callback called on each sample completion (e.g., for watchdog ping)
            agent_label: Optional label for a single-agent run. Stored in the
                         ``variant`` field as ``"{agent_label}_run{run_id}"``.
            agent_variants: For multi-agent bundled runs (``--agents``). Each entry
                            is ``(label, tests_list)`` where ``tests_list`` is a list
                            of ``EvalTest`` objects with the desired ``agent_class``
                            already set. All variants' samples are batched into a
                            single run and output file, tagged by label in
                            ``variant``. Takes precedence over ``agent_label``.
            no_files: Suppress all file output (default: False).
            trace_files: Write JSONL trace files even when OTLP viewer is running (default: False).
                        By default, trace files are only written when the viewer is unavailable.
                        Has no effect when no_files=True.
            memory_limit_mb: Per-worker memory cap in MB (None = no limit).
                            Only effective with ``engine_type="subprocess"``.
                            Workers are killed when RSS exceeds this value;
                            diagnostics are captured at 85%.

        Returns:
            EvalResults with all results
        """
        # Determine which tests to run
        if tests:
            tests_to_run = [self.tests[t] for t in tests if t in self.tests]
        else:
            tests_to_run = list(self.tests.values())

        # Determine which models to use
        if models:
            model_ids = models
        elif self._default_model_ids:
            model_ids = self._default_model_ids
        else:
            model_ids = list(self.models.keys())

        # Validate models exist
        for model_id in model_ids:
            if model_id not in self.models:
                raise ValueError(
                    f"Model '{model_id}' not found. Available: {list(self.models.keys())}"
                )

        # Setup output directory (unique per run: microseconds + optional runs/parallel)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        suffix = ""
        if agent_label:
            suffix += f"_{agent_label}"
        if runs != 1:
            suffix += f"_r{runs}"
        if parallel != 1:
            suffix += f"_p{parallel}"
        run_experiment_name = f"{self.name}_{ts}{suffix}"
        _tmpdir = tempfile.TemporaryDirectory() if no_files else None
        experiment_dir = Path(_tmpdir.name) if no_files else self.output_dir / run_experiment_name
        if not no_files:
            experiment_dir.mkdir(parents=True, exist_ok=True)
        traces_dir = experiment_dir / "traces"
        traces_dir.mkdir(exist_ok=True)

        _backend, _headless_base, _use_viewer, _external_otlp = self._start_tracing(
            traces_dir, no_files, run_experiment_name, trace_files
        )

        # Create writer
        if no_files:
            writer: ExperimentWriter | NullExperimentWriter = NullExperimentWriter()
        else:
            writer = ExperimentWriter(output_dir=experiment_dir, experiment_name=self.name)
        writer.start(
            suite_name=self.name,
            models=[self._model_metadata.get(m, {"id": m}) for m in model_ids],
            tests=[t.name for t in tests_to_run],
            runs=runs,
        )

        # Phase 1: Create all samples upfront
        # Loop order: run -> test -> task -> model
        # This interleaves models naturally, so task1 runs on all models before task2 starts
        all_samples: list[Sample] = []

        _task_id_set = set(task_ids) if task_ids is not None else None
        # Warn once if limit may have excluded a requested task ID
        if _task_id_set is not None and limit is not None:
            for test in tests_to_run:
                limited_ids = {t.id for t in test.data[:limit]}
                all_ids = {t.id for t in test.data}
                cut_ids = _task_id_set & (all_ids - limited_ids)
                if cut_ids:
                    sys.stderr.write(
                        f"WARNING: --limit {limit} excluded requested task IDs from test"
                        f" '{test.name}': {sorted(cut_ids)}\n"
                    )
        # Determine the set of (label, tests) pairs to iterate over.
        # agent_variants = bundled multi-agent mode (--agents)
        # Otherwise treat as a single anonymous variant.
        if agent_variants:
            labeled_tests = agent_variants  # list of (label, tests_list)
        else:
            labeled_tests = [(agent_label, tests_to_run)]

        for label, variant_tests in labeled_tests:
            for run_id in range(1, runs + 1):
                for test in variant_tests:
                    tasks = test.data[:limit] if limit else test.data
                    if _task_id_set is not None:
                        tasks = [t for t in tasks if t.id in _task_id_set]
                    for i, task in enumerate(tasks):
                        for model_id in model_ids:
                            client = self.models[model_id]
                            sample = self._create_sample(test, task, i, client, model_id, run_id)
                            sample.agent_label = label
                            if label and sample.sample_id:
                                sample.sample_id = f"{label}_{sample.sample_id}"
                            all_samples.append(sample)

        total_samples = len(all_samples)
        if not quiet:
            print(f"Running {total_samples} samples (parallel={parallel})...")

        # Progress callback for real-time output
        passed_count = [0]  # Use list for mutable closure
        import time

        start_time = time.perf_counter()

        def on_progress(completed: int, total: int, result) -> None:
            # Call external hook first (e.g., watchdog ping) - always runs
            if on_progress_hook:
                try:
                    on_progress_hook()
                except Exception:
                    pass  # Don't let hook errors affect evaluation

            if quiet:
                return
            try:
                if result.passed:
                    passed_count[0] += 1
                mem_warn = getattr(result, "memory_diag_file", None) is not None
                icon = "✓" if result.passed else "✗"
                if mem_warn:
                    icon += "M"
                name = getattr(result, "display_name", None) or getattr(
                    result, "test_id", "unknown"
                )
                pct = (completed / total * 100) if total else 0
                elapsed = time.perf_counter() - start_time
                # Per-sample duration from result (if available)
                sample_dur = getattr(result, "duration_seconds", None)
                sample_str = f"({sample_dur:5.1f}s)" if sample_dur else "(--.-s)"
                # Worker ID for tracking parallel execution slots
                wid = getattr(result, "worker_id", None)
                wid_str = f"W{wid:02d}" if wid else "W--"
                # Write to stderr which is unbuffered
                line = f"  {icon} [{completed}/{total}] {elapsed:6.1f}s {sample_str} {wid_str} {name} — {passed_count[0]} passed ({pct:.0f}%)"
                # Show truncated error for failed results (helps debug subprocess crashes)
                err = getattr(result, "error", None)
                if err and not result.passed:
                    line += f"\n    error: {err[:200]}"
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
            except Exception as e:
                sys.stderr.write(f"  [PROGRESS ERROR] {completed}/{total}: {e}\n")
                sys.stderr.flush()

        # Use CLI timeout if provided, otherwise fall back to config default
        effective_timeout = timeout if timeout is not None else self.timeout_seconds

        pipeline_config = PipelineConfig(
            trace_dir=traces_dir,
            on_progress=on_progress,
            pass_threshold=self.pass_threshold,
            timeout_seconds=effective_timeout,
            engine_type=engine_type,
            batch_size=batch_size,
            use_otlp=True,
            no_files=no_files,
            experiment_name=run_experiment_name,
            agent_label=agent_label,
            memory_limit_mb=memory_limit_mb,
            otlp_base_url=_headless_base,
            viewer_endpoint=_external_otlp if _use_viewer else None,
        )

        # Phase 2: Run all samples in one parallel batch
        try:
            all_results = await run_evaluation(
                all_samples, pipeline_config, writer, max_concurrent=parallel
            )
        finally:
            try:
                writer.finalize()
            finally:
                _backend.stop()
                if _tmpdir is not None:
                    _tmpdir.cleanup()

        return EvalResults(
            results=all_results,
            output_file=writer.file_path,
            passed=writer.passed_count,
            total=len(all_results),
        )

    def _create_sample(
        self,
        test: EvalTest,
        task: Task,
        task_index: int,
        client: Any,
        model_id: str,
        run_id: int,
    ) -> Sample:
        """Create a single Sample object for a task/model/run combination."""
        from .agents import AgentWrapper

        # Check if we have a factory for this model (creates fresh client per sample)
        client_factory = self._model_factories.get(model_id)

        # Create a copy to avoid mutating shared tasks
        task_copy = Task(
            id=task.id,
            input=task.input,
            expected=task.expected,
            metadata=task.metadata,
            run_id=run_id,
        )

        # Factory returns AgentWrapper that adapts agent to run() interface
        if client_factory:
            # Create fresh client per sample for better parallelism
            def make_factory(cf=client_factory, cls=test.agent_class, method=test.method):
                def factory():
                    fresh_client = cf()
                    agent_instance = cls(llm=fresh_client)
                    return AgentWrapper(agent_instance, method)

                return factory
        else:
            # Use shared client (original behavior)
            def make_factory(c=client, cls=test.agent_class, method=test.method):
                def factory():
                    agent_instance = cls(llm=c)
                    return AgentWrapper(agent_instance, method)

                return factory

        # Generate unique sample_id using run_id, test name, task index, and model
        # Include model to ensure uniqueness when model loop runs within same second
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = model_id.split("/")[-1]
        sample_id = f"{test.name}_{timestamp}_{run_id:02d}_{task_index:06d}_{model_short}"

        # Build serializable config for subprocess execution
        agent_module = test.agent_class.__module__
        # Detect file-based agents (loaded via --agent flag)
        agent_file = None
        if agent_module.startswith("_agent_override_"):
            import sys as _sys

            agent_mod = _sys.modules.get(agent_module)
            if agent_mod and hasattr(agent_mod, "__file__"):
                agent_file = str(agent_mod.__file__)
            else:
                import logging

                logging.getLogger(__name__).warning(
                    f"File-based agent '{test.agent_class.__name__}' (module={agent_module}) "
                    f"has no resolvable __file__. Subprocess engine may fail to import it."
                )

        client_config = self._build_client_config(model_id)
        scorer_specs = []
        for sc in test.scorers:
            if not sc.scorer_class:
                # Scorers added via add_test() without scorer_class can't be
                # reconstructed in subprocess. Infer from the class name.
                sc.scorer_class = f"{type(sc.scorer).__module__}.{type(sc.scorer).__name__}"
            scorer_specs.append(
                {
                    "name": sc.name,
                    "weight": sc.weight,
                    "scorer_class": sc.scorer_class,
                    "scorer_kwargs": sc.scorer_kwargs,
                }
            )

        return Sample(
            task=task_copy,
            method=test.method,
            agent_class=test.agent_class.__name__,
            scorers=test.scorers,
            agent_factory=make_factory(),
            model=model_id,
            test_name=test.name,
            display_name=f"{test.name}/{task_copy.id}/{model_id}/run{run_id}",
            sample_id=sample_id,
            tier=test.tier,
            eval_metadata=dict(test.eval_metadata) if test.eval_metadata else None,
            # Subprocess config data
            agent_module=agent_module,
            client_config=client_config,
            scorer_specs=scorer_specs,
            agent_file=agent_file,
        )

    def _build_client_config(self, model_id: str) -> dict[str, Any]:
        """Build a serializable client config dict for subprocess execution.

        Extracts the model configuration from _model_metadata in a form that
        can be JSON-serialized and used to reconstruct a CompletionClient
        in a subprocess via agent_from_spec().
        """
        meta = self._model_metadata.get(model_id, {})
        config: dict[str, Any] = {
            "model": meta.get("model_name", model_id),
        }
        if meta.get("endpoint"):
            config["api_base"] = meta["endpoint"]
        if meta.get("api_key_env"):
            config["api_key_env"] = meta["api_key_env"]
        if meta.get("max_tokens"):
            config["max_tokens"] = meta["max_tokens"]
        if meta.get("reasoning_effort"):
            config["reasoning_effort"] = meta["reasoning_effort"]
            config["allowed_openai_params"] = ["reasoning_effort"]
        extra_body: dict[str, Any] = {}
        if meta.get("max_thinking_tokens"):
            extra_body["nvext"] = {"max_thinking_tokens": meta["max_thinking_tokens"]}
        if meta.get("no_cache"):
            extra_body["cache"] = {"no-cache": True}
        if extra_body:
            config["extra_body"] = extra_body
        # Retry config
        if meta.get("max_retries") is not None or meta.get("retry_on_empty_content"):
            config["retry_config"] = {
                "max_retries": meta.get("max_retries", 3),
                "retry_on_empty_content": meta.get("retry_on_empty_content", False),
            }
        return config
