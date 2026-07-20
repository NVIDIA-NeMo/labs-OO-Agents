# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Subprocess worker: reads SubprocessTaskInput JSON from stdin, writes EvalTestResult JSON to stdout.

This script runs in a completely clean, spawned subprocess. No inherited state from
the parent — all tracing, agents, and scorers are constructed from the JSON config.

Usage (by SubprocessEngine, not directly):
    echo '{"agent_spec": ..., "task": ..., ...}' | python -m eval_pipeline.subprocess_worker

Protocol:
    stdin  → SubprocessTaskInput JSON (one object)
    stdout → EvalTestResult JSON (one object)
    stderr → logs, warnings, tracing output (not parsed by parent)
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from ._utils import classify_error_type, merge_eval_metadata, sanitize_for_json
from .agents import agent_from_spec
from .eval_types import EvalTestResult, ScoreDetail, SubprocessTaskInput, Tier
from .execute import execute_task
from .models import Task
from .scoring import (
    build_scoring_context,
    compute_weighted_score,
    score_task,
    scorer_from_spec,
)

log = logging.getLogger(__name__)


class _TaskOutputPrefixer:
    """Prefix task-owned stdout/stderr lines without touching protocol stdout."""

    def __init__(self, stream, task_id: str):
        self._stream = stream
        self._prefix = f"[{task_id}] "
        self._at_line_start = True

    def write(self, text: str) -> int:
        for chunk in text.splitlines(keepends=True):
            if self._at_line_start and chunk:
                self._stream.write(self._prefix)
            self._stream.write(chunk)
            self._at_line_start = chunk[-1] in ("\n", "\r") if chunk else self._at_line_start
        return len(text)

    def writelines(self, lines) -> None:
        for line in lines:
            self.write(line)

    def flush(self) -> None:
        self._stream.flush()

    def isatty(self) -> bool:
        return self._stream.isatty()

    @property
    def encoding(self) -> str | None:
        return getattr(self._stream, "encoding", None)


async def run_task(task_input: SubprocessTaskInput) -> EvalTestResult:
    """Execute a single evaluation task in this subprocess.

    Sets up tracing fresh, reconstructs agent and scorers from specs,
    runs execution + scoring, and returns the result.
    """
    started_at = datetime.now().isoformat()
    sample_start = time.perf_counter()

    # Reconstruct Task from TaskSpec
    task = Task(
        id=task_input.task.id,
        input=tuple(task_input.task.input),  # JSON arrays → tuple
        expected=task_input.task.expected,
        metadata=task_input.task.metadata,
        run_id=task_input.task.run_id,
    )

    # Fix input format: should be (args_tuple, kwargs_dict)
    if isinstance(task.input, (list, tuple)) and len(task.input) == 2:
        args_part, kwargs_part = task.input
        task.input = (tuple(args_part) if isinstance(args_part, list) else args_part, kwargs_part)

    # Build merged eval metadata early — needed by both happy path and error paths.
    merged_eval_meta = merge_eval_metadata(task_input.eval_metadata, task.metadata)

    # Build identifiers
    model_short = task_input.model.split("/")[-1] if task_input.model else "default"
    run_id = task_input.task.run_id
    base_test_id = f"{task.id}_{model_short}"
    unique_test_id = f"{base_test_id}_run{run_id}"
    variant = f"{task_input.agent_label}_run{run_id}" if task_input.agent_label else f"run{run_id}"
    trace_dir = Path(task_input.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Build trace file / session ID — include a short experiment hash so that two
    # experiments running the same agent class concurrently never share a session_id.
    _exp_hash = hashlib.sha256((task_input.experiment_name or "").encode()).hexdigest()[:8]
    trace_filename = f"{task_input.agent_spec.agent_class}_{model_short}_{task_input.agent_spec.method}_{task_input.sample_id}_{_exp_hash}.jsonl"
    trace_file = trace_dir / trace_filename
    session_id = trace_file.stem

    # Set up tracing (fresh process — no inherited state)
    if task_input.use_otlp:
        try:
            from nooa.tracing import (
                enable_tracing,
                exporters,
                set_session,
            )

            # Always send to headless backend (otlp_endpoint); also to external viewer if running.
            _exporter_list = [exporters.journal(endpoint=task_input.otlp_endpoint)]
            if task_input.viewer_endpoint:
                _exporter_list.append(exporters.journal(endpoint=task_input.viewer_endpoint))
            if task_input.write_trace_file:
                _exporter_list.append(exporters.jsonl(trace_dir))
            enable_tracing(exporters=_exporter_list, experiment=task_input.experiment_name)
            set_session(session_id)
        except Exception as e:
            log.warning(f"Failed to set up tracing: {e}")

    # Reconstruct agent and scorers
    try:
        agent = agent_from_spec(task_input.agent_spec)
        scorers = [scorer_from_spec(s) for s in task_input.scorers]
    except Exception as e:
        return _error_result(
            task_input,
            unique_test_id,
            base_test_id,
            run_id,
            model_short,
            str(e),
            started_at,
            time.perf_counter() - sample_start,
            session_id,
            eval_metadata=merged_eval_meta,
        )

    # Stage 1: Execute
    _execute_error: Exception | None = None
    try:
        result = await execute_task(
            agent=agent,
            task=task,
            trace_file=None if task_input.use_otlp else trace_file,
            timeout_seconds=task_input.timeout_seconds,
        )
    except Exception as e:
        _execute_error = e

    # End any active spans that were never closed (e.g. due to timeout cancellation).
    # OTel SDK only exports ended spans; without this, timed-out executions lose
    # all tracing data because shutdown_traces() silently drops un-ended spans.
    if _execute_error is not None or (result and result.error):
        try:
            from nooa.tracing import end_active_spans

            _reason = str(_execute_error) if _execute_error else result.error
            end_active_spans(_reason)
        except Exception as e:
            log.debug(f"end_active_spans() failed (non-fatal): {e}")

    # Shut down tracing regardless of success/failure.
    #
    # flush_traces() alone (force_flush) is not sufficient:
    # 1. On error: execute_task raises before outer spans end.
    # 2. Under parallel load: force_flush() returns before the BSP worker thread
    #    finishes its HTTP POST.  shutdown_traces() joins the worker, guaranteeing
    #    all exports complete before we read the trace back for scoring.
    #
    # Run in a thread executor so the event loop stays responsive during the
    # BSP flush.  Blocking the event loop thread directly causes LiteLLM's
    # LoggingWorker asyncio.wait_for() to time out (CancelledError → TimeoutError).
    try:
        from nooa.tracing import shutdown_traces

        await asyncio.get_event_loop().run_in_executor(None, shutdown_traces)
    except Exception as e:
        log.debug(f"shutdown_traces() failed (non-fatal): {e}")

    if _execute_error is not None:
        return _error_result(
            task_input,
            unique_test_id,
            base_test_id,
            run_id,
            model_short,
            str(_execute_error),
            started_at,
            time.perf_counter() - sample_start,
            session_id,
            eval_metadata=merged_eval_meta,
        )

    # Fetch trace from headless backend for scoring
    _viewer_base = task_input.otlp_endpoint.rstrip("/").removesuffix("/v1/traces")
    _trace = None
    if task_input.use_otlp:
        # Wait for the backend's write queue to drain, then load the trace.
        try:
            import urllib.request

            urllib.request.urlopen(
                urllib.request.Request(f"{_viewer_base}/v1/sync", method="POST"),
                timeout=30,
            )
        except Exception:
            pass
        try:
            from nooa.trace_explorer import TraceExplorer

            _trace = await TraceExplorer.from_viewer(_viewer_base, session_id)
        except Exception as e:
            log.warning(f"Failed to load trace for session {session_id}: {e}")
    elif result.trace_file is not None:
        try:
            from nooa.trace_explorer import TraceExplorer

            _trace = await TraceExplorer.from_file(str(result.trace_file))
        except Exception:
            pass

    # Clean up session
    if task_input.use_otlp:
        try:
            from nooa.tracing import set_session

            set_session(None)
        except Exception:
            pass

    # Stage 2: Build scoring context
    ctx = build_scoring_context(
        result,
        metadata=task.metadata,
        trace=_trace,
        use_otlp=task_input.use_otlp,
    )

    # Stage 3: Score
    scores = await score_task(ctx, scorers)

    weighted_score = compute_weighted_score(scores)
    passed = weighted_score >= task_input.pass_threshold and result.error is None

    typed_scores: dict[str, ScoreDetail] = {}
    for name, score_data in scores.items():
        typed_scores[name] = ScoreDetail(
            score=score_data["score"],
            passed=score_data["score"] >= 0.5,
            reasoning=score_data.get("reasoning"),
            metrics=score_data.get("metadata"),
        )

    # Stage 4: Post eval span to external viewer (if running)
    if task_input.viewer_endpoint and task_input.experiment_name:
        try:
            from .trace_eval_span import post_eval_span_to_otlp

            post_eval_span_to_otlp(
                session_id=session_id,
                experiment=task_input.experiment_name,
                test_id=unique_test_id,
                passed=passed,
                weighted_score=weighted_score,
                model=task_input.model or "unknown",
                agent_class=task_input.agent_spec.agent_class,
                method=task_input.agent_spec.method,
                scores=typed_scores,
                test_name=task_input.test_name,
                display_name=task_input.display_name,
                tier=task_input.tier,
                variant=variant,
                run_id=run_id,
                extra_metadata=merged_eval_meta or None,
                endpoint=task_input.viewer_endpoint,
            )
        except Exception as e:
            log.warning(f"Failed to post eval span: {e}")
    if not task_input.use_otlp:
        try:
            from .trace_eval_span import write_eval_span_to_trace

            write_eval_span_to_trace(
                trace_file=trace_file,
                test_id=unique_test_id,
                passed=passed,
                weighted_score=weighted_score,
                model=task_input.model or "unknown",
                agent_class=task_input.agent_spec.agent_class,
                method=task_input.agent_spec.method,
                scores=typed_scores,
                run_id=run_id,
                extra_metadata=merged_eval_meta or None,
            )
        except Exception:
            pass

    # Build result
    sanitized_output = sanitize_for_json(ctx.actual)
    output_error = None
    if isinstance(sanitized_output, str) and sanitized_output.startswith("<unawaited coroutine"):
        output_error = f"Agent returned unawaited coroutine: {sanitized_output}"
        passed = False

    trace_file_str = session_id if task_input.use_otlp else str(trace_file)

    eval_result = EvalTestResult(
        test_id=unique_test_id,
        base_test_id=base_test_id,
        run_id=run_id,
        test_case=task.id,
        agent_class=task_input.agent_spec.agent_class,
        method=task_input.agent_spec.method,
        test_name=task_input.test_name,
        display_name=task_input.display_name,
        tier=Tier(task_input.tier) if task_input.tier else Tier.STABLE,
        model=task_input.model or "unknown",
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
        error_type=classify_error_type(output_error or ctx.error)
        if (output_error or ctx.error)
        else None,
        eval_metadata=merged_eval_meta or {},
    )
    eval_result.started_at = started_at
    eval_result.duration_seconds = time.perf_counter() - sample_start

    return eval_result


def _annotate_memory(result: EvalTestResult, monitor: object | None) -> EvalTestResult:
    """Stamp memory monitoring info onto an eval result.

    Args:
        result: The eval result to annotate.
        monitor: A ``MemoryMonitor`` instance (or None if memory limiting is off).
                 Typed as ``object`` to avoid importing ``_memory_monitor`` at
                 module level.
    """
    if monitor is None:
        return result
    result.peak_rss_mb = monitor.peak_rss_mb
    if monitor.soft_limit_hit:
        result.memory_diag_file = monitor.diag_file
        # If the task otherwise succeeded, downgrade to a warning
        if not result.error:
            from pathlib import Path

            diag_name = Path(monitor.diag_file).name if monitor.diag_file else "N/A"
            result.error = (
                f"Memory soft limit hit: {monitor.peak_rss_mb:.1f} MB "
                f"(limit: {monitor.limit_mb} MB). "
                f"Diagnostics: {diag_name}"
            )
            result.error_type = "MemoryWarning"
            result.passed = False
    return result


def _error_result(
    task_input: SubprocessTaskInput,
    unique_test_id: str,
    base_test_id: str,
    run_id: int,
    model_short: str,
    error_msg: str,
    started_at: str,
    duration: float,
    session_id: str,
    eval_metadata: dict | None = None,
) -> EvalTestResult:
    """Build an EvalTestResult for an error case."""
    trace_file_str = session_id if task_input.use_otlp else ""
    variant = f"{task_input.agent_label}_run{run_id}" if task_input.agent_label else f"run{run_id}"
    result = EvalTestResult(
        test_id=unique_test_id,
        base_test_id=base_test_id,
        run_id=run_id,
        test_case=task_input.task.id,
        agent_class=task_input.agent_spec.agent_class,
        method=task_input.agent_spec.method,
        test_name=task_input.test_name,
        tier=Tier(task_input.tier) if task_input.tier else Tier.STABLE,
        model=task_input.model or "unknown",
        variant=variant,
        passed=False,
        scores={},
        input=task_input.task.input,
        output=None,
        expected=task_input.task.expected,
        error=error_msg,
        error_type=classify_error_type(error_msg),
        display_name=task_input.display_name,
        trace_file=trace_file_str,
        eval_metadata=eval_metadata or {},
    )
    result.started_at = started_at
    result.duration_seconds = duration
    return result


def _make_error_result(
    task_input: SubprocessTaskInput | None,
    error_msg: str,
    error_type: str,
    **extra_fields,
) -> EvalTestResult:
    """Build an EvalTestResult for error cases, preserving task identity when available."""
    if task_input:
        ms = task_input.model.split("/")[-1] if task_input.model else "default"
        rid = task_input.task.run_id
        btid = f"{task_input.task.id}_{ms}"
        var = f"{task_input.agent_label}_run{rid}" if task_input.agent_label else f"run{rid}"
        return EvalTestResult(
            test_id=f"{btid}_run{rid}",
            base_test_id=btid,
            run_id=rid,
            test_case=task_input.task.id,
            agent_class=task_input.agent_spec.agent_class,
            method=task_input.agent_spec.method,
            test_name=task_input.test_name,
            display_name=task_input.display_name,
            model=task_input.model or "unknown",
            variant=var,
            passed=False,
            scores={},
            input=None,
            output=None,
            expected=task_input.task.expected,
            error=error_msg,
            error_type=error_type,
            **extra_fields,
        )
    return EvalTestResult(
        test_id="unknown",
        base_test_id="unknown",
        run_id=1,
        test_case="unknown",
        agent_class="unknown",
        method="unknown",
        model="unknown",
        variant="run1",
        passed=False,
        scores={},
        input=None,
        output=None,
        expected=None,
        error=error_msg,
        error_type=error_type,
        **extra_fields,
    )


def main() -> None:
    """Entry point: read JSON lines from stdin, write JSON lines to stdout.

    Each line is one ``SubprocessTaskInput`` JSON → one ``EvalTestResult`` JSON.
    When stdin closes (EOF), the worker exits. This amortizes Python startup
    cost across all tasks routed to this worker.
    """
    # Suppress litellm's "coroutine was never awaited" warning on loop close.
    # This is a known litellm issue: its async HTTP client cleanup coroutine
    # is still pending when the event loop closes.
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="coroutine 'close_litellm_async_clients' was never awaited",
        category=RuntimeWarning,
    )

    # Drop unsupported LLM params (e.g. tool_choice for some Azure models).
    # This is a litellm global that the parent sets in config loading;
    # the subprocess must set it independently.
    try:
        import litellm

        litellm.drop_params = True
    except ImportError:
        pass

    # Capture the real stdout for protocol output, then redirect sys.stdout
    # to stderr so that print() in agent code / libraries doesn't corrupt
    # the JSON line protocol.
    _proto_out = sys.stdout.buffer
    _real_stderr = sys.stderr
    sys.stdout = sys.stderr

    # Enable tracemalloc early so allocations from the first task are tracked.
    # Lazy import: only pull in _memory_monitor when memory limiting is active.
    _memory_mod = None

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        for line in sys.stdin.buffer:
            line = line.strip()
            if not line:
                continue

            monitor = None  # _memory_monitor.MemoryMonitor | None
            task_input: SubprocessTaskInput | None = None
            _eval_meta: dict = {}
            try:
                task_input = SubprocessTaskInput.model_validate_json(line)
                _eval_meta = merge_eval_metadata(task_input.eval_metadata, task_input.task.metadata)
                _prefixed_output = _TaskOutputPrefixer(_real_stderr, task_input.task.id)
                sys.stdout = _prefixed_output
                sys.stderr = _prefixed_output

                # -- memory cap setup ------------------------------------ #
                if task_input.memory_limit_mb:
                    if _memory_mod is None:
                        from . import _memory_monitor as _memory_mod

                        _memory_mod.enable_tracking()
                    _memory_mod.set_hard_limit(task_input.memory_limit_mb)
                    # Pre-compute identity fields so the monitor can write
                    # a proper EvalTestResult if it hard-kills the process.
                    _ms = task_input.model.split("/")[-1] if task_input.model else "default"
                    _rid = task_input.task.run_id
                    _btid = f"{task_input.task.id}_{_ms}"
                    _var = (
                        f"{task_input.agent_label}_run{_rid}"
                        if task_input.agent_label
                        else f"run{_rid}"
                    )
                    _sess = (
                        f"{task_input.agent_spec.agent_class}_{_ms}"
                        f"_{task_input.agent_spec.method}_{task_input.sample_id}"
                    )
                    monitor = _memory_mod.MemoryMonitor(
                        limit_mb=task_input.memory_limit_mb,
                        trace_dir=task_input.trace_dir,
                        sample_id=task_input.sample_id,
                        proto_out=_proto_out,
                        task_meta={
                            "test_id": f"{_btid}_run{_rid}",
                            "base_test_id": _btid,
                            "run_id": _rid,
                            "test_case": task_input.task.id,
                            "agent_class": task_input.agent_spec.agent_class,
                            "method": task_input.agent_spec.method,
                            "test_name": task_input.test_name,
                            "display_name": task_input.display_name,
                            "model": task_input.model or "unknown",
                            "variant": _var,
                            "trace_file": _sess if task_input.use_otlp else "",
                        },
                    )
                    monitor.start()

                result = loop.run_until_complete(run_task(task_input))
                result = _annotate_memory(result, monitor)
                _result_bytes = result.model_dump_json().encode()

            except MemoryError:
                # Hard limit (RLIMIT_AS) was hit.
                diag_file = monitor.diag_file if monitor else None
                peak = monitor.peak_rss_mb if monitor else None
                limit = task_input.memory_limit_mb if task_input else None
                error_msg = f"MemoryError: process exceeded {limit or '?'} MB memory limit."
                if peak:
                    error_msg += f" Peak RSS: {peak:.1f} MB."
                if diag_file:
                    from pathlib import Path

                    error_msg += f" Diagnostics: {Path(diag_file).name}"
                _result_bytes = (
                    _make_error_result(
                        task_input,
                        error_msg,
                        "MemoryError",
                        memory_diag_file=diag_file,
                        peak_rss_mb=peak,
                        eval_metadata=_eval_meta,
                    )
                    .model_dump_json()
                    .encode()
                )

            except Exception as e:
                # Must write a valid EvalTestResult so the parent can parse it
                _result_bytes = (
                    _make_error_result(
                        task_input,
                        str(e),
                        "WorkerError",
                        eval_metadata=_eval_meta,
                    )
                    .model_dump_json()
                    .encode()
                )

            finally:
                # -- memory cap teardown --------------------------------- #
                sys.stdout = _real_stderr
                sys.stderr = _real_stderr
                if monitor:
                    monitor.stop()
                if task_input and task_input.memory_limit_mb and _memory_mod:
                    _memory_mod.clear_hard_limit()
                    try:
                        import tracemalloc

                        tracemalloc.clear_traces()
                    except Exception:
                        pass

            # When memory limiting is active, write result atomically via
            # os.write() (single syscall, no Python buffering) then exit
            # immediately so the parent spawns a fresh worker with clean
            # memory.  Mirrors the hard-kill path in _memory_monitor.py.
            if task_input and task_input.memory_limit_mb:
                import os as _os

                _os.write(1, _result_bytes + b"\n")
                _os._exit(0)
            else:
                _proto_out.write(_result_bytes)
                _proto_out.write(b"\n")
                _proto_out.flush()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
