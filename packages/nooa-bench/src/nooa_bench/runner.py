# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Harbor agent runner for NOOA — executed inside the Harbor container.

Harbor invokes this as a CLI process::

    nemo-harbor \\
        --instruction '...' \\
        --model 'anthropic/claude-opus-4-6' \\
        --agent-type bench

The runner is benchmark-agnostic. Its only jobs are:

1. Instantiate the right agent class from ``--agent-type``.
2. Call ``agent._run_evaluation({"user_message": instruction})``.
3. Write ``result.json`` to ``/logs/agent/`` and answer text to ``/app/answer.txt``.

Benchmark-specific logic (system prompts, instruction parsing, data paths) lives
inside each agent class, not here.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger("nooa_bench.runner")

# Harbor container path conventions.
#
# Harbor bind-mounts ONLY /logs/agent and /logs/verifier from the host
# (harbor.models.trial.paths). Everything else under /logs lives in the
# container's own writable layer and is destroyed when the trial container is
# removed -- which is the default. Traces therefore have to be written inside
# /logs/agent to survive the run; /logs/artifacts silently discarded them.
LOGS_DIR = Path("/logs/agent")
TRACES_DIR = LOGS_DIR / "traces"
ANSWER_FILE = Path("/app/answer.txt")


def _setup_logging() -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(LOGS_DIR / "nooa_bench.log")))
    except OSError:
        # Outside a Harbor container /logs may not exist or be writable — stderr only.
        pass
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )


def _setup_tracing(model: str, agent_type: str) -> None:
    """Enable OTel tracing, always on disk and additionally live when reachable.

    JSONL files under ``/logs/agent/traces/`` (a host-mounted directory)
    are written unconditionally — they are the record failure analysis runs on,
    and they are importable later via ``nemo-oo import-harbor``.  When
    ``OTLP_ENDPOINT`` (default ``http://localhost:5001``) is reachable the
    journal exporter is added *alongside* the file exporter so the viewer sees
    spans live without that becoming the only copy.

    Streaming used to replace the file exporter rather than supplement it, so a
    run against a reachable viewer left no trajectory on disk at all.

    Note: Apptainer containers share the host network namespace, so
    ``localhost:5001`` inside the container resolves to the developer's host.
    For Docker containers set ``OTLP_ENDPOINT=http://host.docker.internal:5001``.
    """
    try:
        from nooa.tracing import (
            enable_tracing,
            probe_otlp_endpoint,
        )
        from nooa.tracing import exporters as nemo_exporters
    except ImportError:
        logger.warning("nooa.tracing not available, no tracing")
        return

    endpoint = os.environ.get("OTLP_ENDPOINT", "http://localhost:5001/v1/traces")

    TRACES_DIR.mkdir(parents=True, exist_ok=True)
    exporters = [nemo_exporters.jsonl(TRACES_DIR)]

    if probe_otlp_endpoint(endpoint):
        exporters.append(nemo_exporters.journal(endpoint=endpoint))
        logger.info("OTLP endpoint reachable (%s) — also streaming live", endpoint)
    else:
        logger.info("OTLP endpoint unreachable (%s) — writing files only", endpoint)

    enable_tracing(
        exporters=exporters,
        extra_resource_attrs={"eval.model": model, "eval.agent_type": agent_type},
    )
    logger.info("OTel tracing enabled → %s (%d exporter(s))", TRACES_DIR, len(exporters))


def _import_agent_class(agent_type: str) -> type:
    from nooa_bench import AGENT_CLASSES

    entry = AGENT_CLASSES.get(agent_type)
    if entry is None:
        raise ValueError(
            f"Unknown agent_type: {agent_type!r}.  Must be one of: {sorted(AGENT_CLASSES)}"
        )
    module_path, class_name = entry.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _write_result(result: dict[str, Any], model: str, agent_type: str) -> None:
    """Write result metadata to LOGS_DIR/result.json."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "agent_type": agent_type,
        "success": result.get("success", False),
        "response": result.get("response", ""),
        "error": result.get("error"),
        "n_input_tokens": result.get("n_input_tokens"),
        "n_output_tokens": result.get("n_output_tokens"),
    }
    out = LOGS_DIR / "result.json"
    out.write_text(json.dumps(payload, indent=2))
    logger.info("Result written → %s", out)


def _write_trajectory(agent: Any) -> None:
    """Dump the agent's full event history to LOGS_DIR/trajectory.json.

    The OTLP spans under ``agent/traces/`` remain the canonical record, but
    failure analysis starts in the per-task ``agent/`` directory — which
    otherwise holds only ``nooa_bench.log`` and a ``result.json`` carrying just
    the final response.  Anyone looking there for the turn-by-turn trajectory
    previously found nothing.
    """
    manager = getattr(agent, "event_manager", None)
    if manager is None:
        logger.warning("Agent exposes no event_manager — no trajectory written")
        return

    try:
        events = [
            {
                "event_id": event_id,
                "event_type": type(event).__name__,
                **event.model_dump(mode="json"),
            }
            for event_id, event in manager.items()
        ]
    except Exception as e:  # never fail the task over a debug artifact
        logger.warning("Could not serialise trajectory: %s", e)
        return

    out = LOGS_DIR / "trajectory.json"
    try:
        out.write_text(json.dumps(events, indent=2, default=str))
    except OSError as e:
        logger.warning("Could not write %s: %s", out, e)
        return
    logger.info("Trajectory written → %s (%d events)", out, len(events))


def _write_behavior_report(model: str, agent_type: str) -> None:
    """Write deterministic interface-behavior metrics beside the trajectory.

    Behavior analysis is observability only: malformed or missing artifacts must
    never turn a completed benchmark task into a failure.
    """
    trajectory = LOGS_DIR / "trajectory.json"
    try:
        from nooa_bench.behavior_analyzer import analyze_trajectory

        change_id = os.environ.get("NOOA_INTERFACE_CHANGE_ID", "baseline")
        report = analyze_trajectory(
            trajectory, model=model, agent_type=agent_type, change_id=change_id
        )
        out = LOGS_DIR / "behavior.json"
        out.write_text(json.dumps(report.to_dict(), indent=2))
    except Exception as e:  # noqa: BLE001 - analysis must not fail the benchmark
        logger.warning("Could not write interface behavior report: %s", e)
        return
    logger.info("Behavior report written → %s", out)


def _write_answer(result: dict[str, Any]) -> None:
    """Write the agent's answer to /app/answer.txt for Harbor's verifier."""
    answer = result.get("answer") or result.get("response", "")
    if not answer:
        logger.warning("No answer to write to %s", ANSWER_FILE)
        return
    try:
        ANSWER_FILE.parent.mkdir(parents=True, exist_ok=True)
        ANSWER_FILE.write_text(str(answer))
        logger.info("Answer written → %s", ANSWER_FILE)
    except OSError as e:
        logger.warning("Could not write answer file %s: %s", ANSWER_FILE, e)


async def _run(
    instruction: str,
    model: str,
    agent_type: str,
    api_base: str | None,
    working_dir: str | None = None,
) -> int:
    """Async main: instantiate, wire, run.  Returns exit code (0 = success)."""
    from nooa.unifiedllm import get_llm_client

    # Build LLM client — honour env-var overrides for local vLLM deployments.
    llm_overrides: dict[str, str] = {}
    if api_base:
        llm_overrides["api_base"] = api_base
    elif base_url := os.environ.get("OPENAI_BASE_URL"):
        llm_overrides["api_base"] = base_url
    if api_key := os.environ.get("OPENAI_API_KEY"):
        llm_overrides["api_key"] = api_key

    llm_client = get_llm_client(model, **llm_overrides)

    agent: Any = None
    try:
        # Instantiate inside the lifecycle guard so a constructor failure still
        # closes the already-created model client.
        AgentClass = _import_agent_class(agent_type)
        agent = AgentClass(llm=llm_client)

        # All agents share the same interface: {"user_message": instruction}.
        # Benchmark-specific parsing (system prompts, data paths, etc.) happens
        # inside the agent's _run_evaluation method.
        from nooa.runtime.token_usage import get_task_tokens, start_task_tokens

        logger.info("Running agent %s (model=%s)...", agent_type, model)
        start_task_tokens()
        task_input: dict[str, Any] = {"user_message": instruction}
        if working_dir:
            task_input["working_dir"] = working_dir
        result = await agent._run_evaluation(task_input)
        result.update(get_task_tokens())
        _write_result(result, model, agent_type)
        _write_trajectory(agent)
        _write_behavior_report(model, agent_type)
        _write_answer(result)

        if result.get("success"):
            logger.info("Agent completed successfully.")
            return 0
        logger.error("Agent reported failure.")
        return 1
    finally:
        try:
            close = getattr(agent, "close", None) if agent is not None else None
            if callable(close):
                close_result = close()
                if inspect.isawaitable(close_result):
                    await close_result
        finally:
            aclose = getattr(llm_client, "aclose", None)
            if callable(aclose):
                close_result = aclose()
                if inspect.isawaitable(close_result):
                    await close_result


@click.command()
@click.option("--instruction", required=True, help="Task instruction / problem statement")
@click.option("--model", required=True, help="Model name in litellm format")
@click.option("--agent-type", default="bench", show_default=True, help="Agent variant to run")
@click.option("--working-dir", default=None, help="Working directory for the agent shell session")
@click.option("--api-base", default=None, help="Override API base URL")
def main(
    instruction: str,
    model: str,
    agent_type: str,
    working_dir: str | None,
    api_base: str | None,
) -> None:
    """Run a NOOA agent on a task inside a Harbor container."""
    _setup_logging()
    logger.info("nooa-bench runner starting")
    logger.info("  model:      %s", model)
    logger.info("  agent_type: %s", agent_type)
    if api_base:
        logger.info("  api_base:   %s", api_base)

    # Validate agent_type early so we get a clean error before any heavy imports.
    from nooa_bench import AGENT_CLASSES

    if agent_type not in AGENT_CLASSES:
        logger.error(
            "Unknown agent_type: %r.  Must be one of: %s", agent_type, sorted(AGENT_CLASSES)
        )
        sys.exit(1)

    _setup_tracing(model=model, agent_type=agent_type)

    try:
        exit_code = asyncio.run(_run(instruction, model, agent_type, api_base, working_dir))
    except Exception as e:
        logger.exception("Runner failed with unhandled exception: %s", e)
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
