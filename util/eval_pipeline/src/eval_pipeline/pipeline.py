# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Main evaluation pipeline that composes execute, score, and write stages.

This module provides the core pipeline functions:
- process_sample: Run a single sample through all stages
- run_evaluation: Run multiple samples with concurrency control

Pipeline stages:
1. Execute: Run agent method and capture trace
2. Build context: Extract data from trace for scoring
3. Score: Run all configured scorers
4. Write: Append result to experiment file

Output Format:
Results are written in .noo-eval.jsonl format using typed EvalTestResult models.
Each result includes:
- test_id: Unique ID including model and run suffix
- base_test_id: For grouping runs of the same test
- run_id: Which run (1-based) for self-consistency analysis
- scores: Per-scorer results with passed/score/reasoning
"""

import asyncio
import logging
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from eval_pipeline.concurrency import (
    ConcurrencyConfig,
    ConcurrencyEngine,
    SubprocessEngine,
)

from ._utils import classify_error_type as _classify_error_type
from ._utils import merge_eval_metadata as _merge_eval_metadata
from ._utils import sanitize_for_json as _sanitize_for_json
from .eval_types import EvalTestResult, ScoreDetail, Tier
from .execute import Agent, execute_task
from .models import Task
from .scoring import ScorerConfig, build_scoring_context, compute_weighted_score, score_task
from .trace_eval_span import post_eval_span_to_otlp, write_eval_span_to_trace


class ExperimentWriterProtocol(Protocol):
    """Protocol for experiment writers."""

    def start(
        self,
        suite_name: str | None = None,
        models: list | None = None,
        config_file: str | None = None,
        tests: list[str] | None = None,
        runs: int = 1,
        variants: list[str] | None = None,
    ) -> Path | None: ...

    def append_result(self, result: EvalTestResult | dict[str, Any]) -> None: ...

    def finalize(
        self,
        status: str = "completed",
        extra_metrics: dict[str, Any] | None = None,
    ) -> None: ...


@dataclass
class Sample:
    """A sample to process: task + its test context.

    Attributes:
        task: The evaluation task to run
        method: Agent method name (e.g., "classify_single")
        agent_class: Agent class name (e.g., "SentimentAgent")
        scorers: List of scorer configurations to apply
        agent_factory: Callable that creates a fresh agent instance
        model: Model identifier for logging/grouping (e.g., "gpt-4")
        display_name: Human-readable test description
        sample_id: Unique identifier for this sample (generated at creation time).
                   Used for trace filenames and result correlation.
    """

    task: Task
    method: str
    agent_class: str
    scorers: list[ScorerConfig]
    agent_factory: Callable[[], Agent]
    tier: Tier = Tier.STABLE
    model: str | None = None
    test_name: str | None = None  # Config test name for unique grouping in matrix
    display_name: str | None = None
    sample_id: str | None = None  # Set at creation for deterministic trace filenames
    agent_label: str | None = None  # Per-sample agent label for --agents bundled runs
    # Subprocess config data (serializable, for JSON-based subprocess execution)
    agent_module: str | None = None  # e.g. "tests.capability.agents.sentiment"
    client_config: dict[str, Any] | None = None  # CompletionClient kwargs
    scorer_specs: list[dict[str, Any]] | None = None  # [{class, weight, kwargs}]
    agent_file: str | None = None  # File path for --agent loaded agents
    # Merged eval metadata from config + test + task levels (task overrides test overrides config)
    eval_metadata: dict[str, Any] | None = None


def _build_eval_metadata(sample: "Sample") -> dict[str, str | int | float | bool]:
    """Merge eval metadata from sample (config+test) and task levels."""
    return _merge_eval_metadata(sample.eval_metadata, sample.task.metadata)


# Progress callback type: (current, total, result) -> None
ProgressCallback = Callable[[int, int, EvalTestResult], None]


@dataclass
class PipelineConfig:
    """Configuration for the evaluation pipeline.

    Attributes:
        trace_dir: Directory where trace files will be written
        on_progress: Optional callback invoked after each sample completes.
                     Called with (current_index, total_count, result).
        pass_threshold: Minimum weighted score to pass (default 0.5).
                        Set higher (e.g., 0.8) for stricter evaluation.
        timeout_seconds: Optional timeout per sample in seconds.
                        If None (default), no timeout is applied.
        engine_type: Execution engine type - "asyncio" (default, I/O-bound LLM APIs)
                    or "subprocess" (CPU-bound local model inference).
        batch_size: Tasks per subprocess when using subprocess engine (default: 1).
                   Higher values reduce process spawn overhead.
        use_otlp: If True, send traces to viewer API (OTLP) and post eval spans there.
        experiment_name: Experiment name for OTLP (required when use_otlp is True).
        agent_label: Optional label identifying the agent variant (e.g. "agent-pf-000").
                     When set, the ``variant`` field of each result becomes
                     ``"{agent_label}_run{run_id}"`` instead of ``"run{run_id}"``.
    """

    trace_dir: Path
    on_progress: ProgressCallback | None = None
    pass_threshold: float = 0.5
    timeout_seconds: float | None = None
    engine_type: str = "asyncio"
    batch_size: int = 1
    use_otlp: bool = False
    no_files: bool = False
    experiment_name: str | None = None
    agent_label: str | None = None
    memory_limit_mb: int | None = None
    otlp_base_url: str | None = None  # headless backend base URL (for scoring fetches)
    viewer_endpoint: str | None = None  # external viewer /v1/traces URL (if running)


async def process_sample(
    sample: Sample,
    config: PipelineConfig,
    writer: ExperimentWriterProtocol,
) -> EvalTestResult:
    """Process a single sample through the full pipeline.

    Stages:
    1. Execute: Run agent, write trace
    2. Build context: Extract data from trace
    3. Score: Run all scorers
    4. Write: Append result to experiment file

    Args:
        sample: Sample to process
        config: Pipeline configuration
        writer: Experiment writer for output

    Returns:
        The EvalTestResult that was written
    """
    # Generate trace file path using sample_id (deterministic, set at creation time)
    # Fall back to timestamp if sample_id not set (backwards compatibility)
    model_short = sample.model.split("/")[-1] if sample.model else "default"
    if sample.sample_id:
        trace_filename = (
            f"{sample.agent_class}_{model_short}_{sample.method}_{sample.sample_id}.jsonl"
        )
    else:
        # Fallback for samples created without sample_id
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        trace_filename = f"{sample.agent_class}_{model_short}_{sample.method}_{timestamp}.jsonl"
    trace_file = config.trace_dir / trace_filename
    session_id = trace_file.stem

    # Set session for this async context (supports concurrent samples)
    try:
        from nooa.tracing import set_session

        config.trace_dir.mkdir(parents=True, exist_ok=True)
        set_session(session_id)
    except ImportError:
        pass
    except Exception as e:
        import logging

        logging.getLogger(__name__).warning(f"Failed to set session: {e}")

    # Stage 1: Execute
    _trace = None
    try:
        agent = sample.agent_factory()
        result = await execute_task(
            agent=agent,
            task=sample.task,
            trace_file=None if config.use_otlp else trace_file,
            timeout_seconds=config.timeout_seconds,
        )
        result.session_id = session_id
        del agent  # Free conversation history before scoring phases

        _trace = None
        try:
            from nooa.tracing import flush_traces

            flush_traces()
        except ImportError:
            pass

        if config.otlp_base_url:
            # With SimpleSpanProcessor on the headless journal exporter,
            # all spans are exported synchronously on span end.  After
            # flush_traces() the data is in the backend's ingest queue;
            # /v1/sync waits for the queue to drain before we read.
            try:
                urllib.request.urlopen(
                    urllib.request.Request(f"{config.otlp_base_url}/v1/sync", method="POST"),
                    timeout=30,
                )
            except Exception:
                pass
            try:
                from nooa.trace_explorer import TraceExplorer

                _trace = await TraceExplorer.from_viewer(config.otlp_base_url, session_id)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to load trace for session {session_id}: {e}. "
                    "Scoring without trace data."
                )
        elif result.trace_file is not None:
            try:
                from nooa.trace_explorer import TraceExplorer

                _trace = await TraceExplorer.from_file(str(result.trace_file))
            except Exception:
                pass
    finally:
        if config.use_otlp:
            try:
                from nooa.tracing import set_session

                set_session(None)
            except ImportError:
                pass

    run_id = getattr(sample.task, "run_id", 1)
    base_test_id = f"{sample.task.id}_{model_short}"
    unique_test_id = f"{base_test_id}_run{run_id}"
    # trace_file_str is the session reference stored in the result (session_id for OTLP,
    # file path for JSONL-only mode).
    if config.otlp_base_url:
        trace_file_str = session_id
    else:
        # JSONL-only mode: flush then confirm the file landed on disk.
        trace_file_str = str(trace_file)
        try:
            from nooa.tracing import flush_traces

            flush_traces()
            await asyncio.sleep(0.1)

            if trace_file.exists() and trace_file.stat().st_size > 0:
                trace_file_str = str(trace_file)
            elif not config.no_files:
                import logging

                logging.getLogger(__name__).warning(
                    f"Trace file {trace_file} does not exist or is empty for sample {unique_test_id}. "
                    f"This may indicate spans were not exported correctly."
                )
        except ImportError:
            pass
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning(f"Failed to flush traces for {trace_file}: {e}")

    # Stage 2: Build scoring context
    ctx = build_scoring_context(
        result,
        metadata=sample.task.metadata,
        trace=_trace,
        use_otlp=config.use_otlp,
    )

    # Stage 3: Score
    scores = await score_task(ctx, sample.scorers)

    # Compute overall pass/fail using configurable threshold
    weighted_score = compute_weighted_score(scores)
    passed = weighted_score >= config.pass_threshold and result.error is None

    # Convert scores to typed ScoreDetail objects
    typed_scores: dict[str, ScoreDetail] = {}
    for name, score_data in scores.items():
        typed_scores[name] = ScoreDetail(
            score=score_data["score"],
            passed=score_data["score"] >= 0.5,
            reasoning=score_data.get("reasoning"),
            metrics=score_data.get("metadata"),
        )

    merged_eval_meta = _build_eval_metadata(sample)

    # Sanitize output to prevent serialization errors (e.g., unawaited coroutines)
    sanitized_output = _sanitize_for_json(ctx.actual)

    # If output was sanitized to a coroutine placeholder, mark as failed with error
    output_error = None
    if isinstance(sanitized_output, str) and sanitized_output.startswith("<unawaited coroutine"):
        output_error = f"Agent returned unawaited coroutine: {sanitized_output}"
        passed = False

    # Create identity fields (model_short already computed above)
    variant = (
        f"{sample.agent_label or config.agent_label}_run{run_id}"
        if (sample.agent_label or config.agent_label)
        else f"run{run_id}"
    )

    # Stage 4: Write eval span to trace (for trace viewer rendering)
    # Post to external viewer if running; also write to JSONL file when present.
    if config.viewer_endpoint and config.experiment_name:
        post_eval_span_to_otlp(
            session_id=session_id,
            experiment=config.experiment_name,
            test_id=f"{sample.task.id}_{model_short}_run{run_id}",
            passed=passed,
            weighted_score=weighted_score,
            model=sample.model or "unknown",
            agent_class=sample.agent_class,
            method=sample.method,
            scores=typed_scores,
            test_name=sample.test_name,
            display_name=sample.display_name,
            tier=sample.tier.value if hasattr(sample.tier, "value") else str(sample.tier),
            variant=variant,
            run_id=run_id,
            extra_metadata=merged_eval_meta or None,
            endpoint=config.viewer_endpoint,
        )
    if not config.no_files and not config.otlp_base_url:
        # JSONL-only mode: write eval span directly into the trace file
        write_eval_span_to_trace(
            trace_file=trace_file,
            test_id=f"{sample.task.id}_{model_short}_run{run_id}",
            passed=passed,
            weighted_score=weighted_score,
            model=sample.model or "unknown",
            agent_class=sample.agent_class,
            method=sample.method,
            scores=typed_scores,
            run_id=run_id,
            extra_metadata=merged_eval_meta or None,
        )

    # Build typed result
    eval_result = EvalTestResult(
        test_id=unique_test_id,
        base_test_id=base_test_id,
        run_id=run_id,
        test_case=sample.task.id,  # Logical test case ID without model/run
        agent_class=sample.agent_class,
        method=sample.method,
        test_name=sample.test_name,
        display_name=sample.display_name,
        tier=sample.tier,
        model=sample.model or "unknown",
        variant=variant,
        passed=passed,
        scores=typed_scores,
        input=ctx.input,
        output=sanitized_output,
        expected=ctx.expected,
        trace_file=trace_file_str,
        input_tokens=ctx.input_tokens,
        output_tokens=ctx.output_tokens,
        total_tokens=ctx.total_tokens,
        error=output_error or ctx.error,
        # Note: result.error is a string, so we can't get the original exception type.
        # We classify based on error message patterns instead.
        error_type=_classify_error_type(output_error or ctx.error)
        if (output_error or ctx.error)
        else None,
        eval_metadata=merged_eval_meta or {},
    )

    writer.append_result(eval_result)

    return eval_result


async def run_evaluation(
    samples: Iterable[Sample],
    config: PipelineConfig,
    writer: ExperimentWriterProtocol,
    max_concurrent: int = 1,
) -> list[EvalTestResult]:
    """Run evaluation on all samples with concurrency control.

    Uses the unified backend's ConcurrencyEngine for execution orchestration.

    Args:
        samples: Samples to evaluate
        config: Pipeline configuration
        writer: Experiment writer
        max_concurrent: Max samples to run in parallel (default 1 = sequential)

    Returns:
        List of all EvalTestResult objects
    """
    samples_list = list(samples)
    total = len(samples_list)

    # Build mapping from sample ID to sample
    sample_map: dict[str, Sample] = {}
    task_ids: list[str] = []

    for i, sample in enumerate(samples_list):
        # Generate unique task ID for this sample
        sample_id = sample.sample_id or f"sample_{i:06d}"
        sample_map[sample_id] = sample
        task_ids.append(sample_id)

    # Track progress
    completed = [0]  # Use list for mutable closure
    passed = [0]

    def on_task_complete(task_id: str, result: EvalTestResult | Exception) -> None:
        """Progress callback for ConcurrencyEngine."""
        completed[0] += 1

        if not isinstance(result, Exception):
            if result.passed:
                passed[0] += 1

            # Call user's progress callback if provided
            if config.on_progress:
                try:
                    config.on_progress(completed[0], total, result)
                except Exception as e:
                    import sys

                    sys.stderr.write(f"[progress error] {e}\n")

    # Worker ID pool for tracking which slot is running each task
    worker_pool: asyncio.Queue[int] = asyncio.Queue()
    for i in range(1, max_concurrent + 1):
        worker_pool.put_nowait(i)

    async def task_fn(task_id: str) -> EvalTestResult:
        """Execute a single sample and return its result."""
        import time

        sample = sample_map[task_id]
        started_at = datetime.now().isoformat()
        sample_start = time.perf_counter()

        # Get worker ID from pool (will be returned on completion)
        worker_id = await worker_pool.get()

        try:
            result = await process_sample(sample, config, writer)
            # Add timing and worker info to result
            result.started_at = started_at
            result.duration_seconds = time.perf_counter() - sample_start
            result.worker_id = worker_id
            return result

        except BaseException as e:
            # Catch BaseException (not just Exception) so that SystemExit/CancelledError
            # raised by agent code (e.g. sys.exit() in a pytools REPL) doesn't kill
            # the entire eval run.  KeyboardInterrupt is re-raised so Ctrl-C still works.
            if isinstance(e, KeyboardInterrupt):
                raise
            import logging

            logging.getLogger(__name__).exception(f"Sample {sample.display_name} failed")
            run_id = getattr(sample.task, "run_id", 1)
            model_short = sample.model.split("/")[-1] if sample.model else "unknown"
            base_test_id = f"{sample.task.id}_{model_short}"
            unique_test_id = f"{base_test_id}_run{run_id}"
            error_msg = str(e)

            # Try to find any trace file that was created for this sample, or use session_id when OTLP
            trace_file_str = ""
            if config.use_otlp:
                model_short = sample.model.split("/")[-1] if sample.model else "unknown"
                trace_file_str = f"{sample.agent_class}_{model_short}_{sample.method}_{sample.sample_id or 'unknown'}"
            else:
                try:
                    pattern = f"{sample.agent_class}_{model_short}_{sample.method}_*.jsonl"
                    matching_files = sorted(config.trace_dir.glob(pattern), reverse=True)
                    if matching_files:
                        trace_file_str = str(matching_files[0])
                except Exception:
                    pass

            result = EvalTestResult(
                test_id=unique_test_id,
                base_test_id=base_test_id,
                run_id=run_id,
                test_case=sample.task.id,
                agent_class=sample.agent_class,
                method=sample.method,
                test_name=sample.test_name,
                tier=sample.tier,
                model=sample.model or "unknown",
                variant=f"{sample.agent_label or config.agent_label}_run{run_id}"
                if (sample.agent_label or config.agent_label)
                else f"run{run_id}",
                passed=False,
                scores={},
                input=sample.task.input,
                output=None,
                expected=sample.task.expected,
                error=error_msg,
                error_type=_classify_error_type(error_msg),
                display_name=sample.display_name,
                trace_file=trace_file_str,
                eval_metadata=_build_eval_metadata(sample),
            )
            # Add timing and worker info to error result too
            result.started_at = started_at
            result.duration_seconds = time.perf_counter() - sample_start
            result.worker_id = worker_id
            writer.append_result(result)
            return result

        finally:
            # Always return worker ID to pool
            worker_pool.put_nowait(worker_id)

    # Use unified backend's execution engine
    if config.engine_type == "subprocess":
        engine = SubprocessEngine()

        # Build JSON task data for subprocess workers
        from .eval_types import AgentSpec, ScorerSpec, SubprocessTaskInput, TaskSpec

        _otlp_endpoint = (
            f"{config.otlp_base_url}/v1/traces"
            if config.otlp_base_url
            else "http://localhost:5001/v1/traces"
        )
        task_data: dict[str, str] = {}
        for sample_id, sample in sample_map.items():
            task_input = SubprocessTaskInput(
                agent_spec=AgentSpec(
                    agent_module=sample.agent_module or "",
                    agent_class=sample.agent_class,
                    method=sample.method,
                    client_config=sample.client_config or {},
                    agent_file=sample.agent_file,
                ),
                task=TaskSpec(
                    id=sample.task.id,
                    input=sample.task.input,
                    expected=sample.task.expected,
                    metadata=sample.task.metadata,
                    run_id=getattr(sample.task, "run_id", 1),
                ),
                scorers=[ScorerSpec(**s) for s in (sample.scorer_specs or [])],
                sample_id=sample.sample_id or sample_id,
                model=sample.model or "unknown",
                test_name=sample.test_name,
                display_name=sample.display_name,
                tier=sample.tier.value if hasattr(sample.tier, "value") else str(sample.tier),
                agent_label=sample.agent_label or config.agent_label,
                trace_dir=str(config.trace_dir),
                use_otlp=config.use_otlp,
                write_trace_file=not config.no_files,
                otlp_endpoint=_otlp_endpoint,
                viewer_endpoint=config.viewer_endpoint,
                experiment_name=config.experiment_name,
                pass_threshold=config.pass_threshold,
                timeout_seconds=config.timeout_seconds,
                memory_limit_mb=config.memory_limit_mb,
                eval_metadata=sample.eval_metadata or {},
            )
            task_data[sample_id] = task_input.model_dump_json()

        # Subprocess on_task_complete: parent writes results
        def subprocess_on_complete(task_id: str, result: EvalTestResult | Exception) -> None:
            completed[0] += 1
            if isinstance(result, Exception):
                # Worker crashed or communication failed — create an error result
                # so it appears in the output file and progress reporting.
                sample = sample_map.get(task_id)
                run_id = getattr(sample.task, "run_id", 1) if sample else 1
                result = EvalTestResult(
                    test_id=task_id,
                    base_test_id=task_id,
                    run_id=run_id,
                    test_case=task_id,
                    agent_class=sample.agent_class if sample else "unknown",
                    method=sample.method if sample else "unknown",
                    model=sample.model or "unknown" if sample else "unknown",
                    variant=f"run{run_id}",
                    passed=False,
                    scores={},
                    input=None,
                    output=None,
                    expected=None,
                    error=str(result),
                    error_type="SubprocessError",
                    eval_metadata=_build_eval_metadata(sample) if sample else {},
                )
            if result.passed:
                passed[0] += 1
            try:
                writer.append_result(result)
            except Exception as e:
                logging.getLogger(__name__).warning("Failed to write result for %s: %s", task_id, e)
            if config.on_progress:
                try:
                    config.on_progress(completed[0], total, result)
                except Exception:
                    pass

        engine_config = ConcurrencyConfig(
            max_concurrent=max_concurrent,
            timeout_seconds=None,  # timeout handled by subprocess worker
        )
        results = await engine.run_tasks(
            task_ids=task_ids,
            task_data=task_data,
            config=engine_config,
            on_task_complete=subprocess_on_complete,
        )
    else:
        engine = ConcurrencyEngine()
        engine_config = ConcurrencyConfig(
            max_concurrent=max_concurrent,
            timeout_seconds=config.timeout_seconds,
        )
        results = await engine.run_tasks(
            task_ids=task_ids,
            task_fn=task_fn,
            config=engine_config,
            on_task_complete=on_task_complete,
        )

    return results
