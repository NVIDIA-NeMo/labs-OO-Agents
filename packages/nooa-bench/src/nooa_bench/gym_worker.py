# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one Gym episode inside the task sandbox: ``python -m nooa_bench.gym_worker in.json out.json``.

This is the process boundary between Gym and NOOA. Gym writes the payload, starts this
module with the staged NOOA runtime, and reads the result; the worker needs NOOA only.
Gym's model server already holds the authoritative copy of every model call, so the
result file is the episode summary, not the sole evidence.

Payload keys (all but ``task``, ``agent`` and ``model_url`` optional):

* ``agent``: ``module:Class`` of any NOOA agent, constructed with ``llm=`` and ``agent_kwargs``.
  A constructor that names ``run_command`` (a shell into the task workspace), ``workdir`` or
  ``working_dir`` also receives that; nothing else is guessed, so a class that forwards
  ``**kwargs`` to ``Agent`` never sees arguments it did not ask for.
* ``turn_method``: the method awaited with the task text (default ``solve``).
* ``tools`` and ``tool_access``: Gym function-tool schemas and the HTTP grant for them.
  ``tool_binding`` names a ``module:callable(agent, tools)`` that replaces the default
  translation (:func:`nooa_bench.gym.bind_gym_tools`).
* ``artifacts_dir`` with ``atif`` / ``trace_jsonl`` flags: NOOA's own trajectory export and
  OTLP span files are written there for Gym to collect after the episode. ``otlp_endpoint``
  additionally ships the same spans live to a collector (NOOA's native OTLP tracing).
* ``model_url``, ``model_name``, ``max_policy_calls``, ``headers``, ``cookies``,
  ``context_window``, ``replay_reasoning``, ``media_inputs``: the model seam.

When the sandbox supervisor stops the worker at the agent's time budget it sends
SIGTERM and waits a short grace period before SIGKILL. The worker uses that window
to stop the agent and write the episode it has so far with stop reason
``timeout_exceeded``, so Gym grades the sandbox state instead of failing the rollout.
The checkpoint is written the moment the run is interrupted, before any cleanup, and
only SIGTERM earns it: SIGINT, an internal cancellation or a crash leave no result and
stay failures. Gym cross-checks the stop reason against the supervisor's receipt.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import shlex
import signal
import socket
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from nooa_bench.gym import bind_gym_tools, resource_tools, run_gym_episode


async def _local_shell(command: str, *, cwd: str, timeout: float) -> str:
    """The sandbox is the local machine here: run the command where the task lives."""
    process = await asyncio.create_subprocess_shell(
        command, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    try:
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise RuntimeError(
            f"command timed out after {timeout:g}s: {shlex.quote(command)}"
        ) from None
    text = stdout.decode(errors="replace").rstrip()
    if process.returncode != 0:
        raise RuntimeError(f"{text}\n[exit {process.returncode}]")
    return f"{text}\n[exit 0]"


def _import(reference: str) -> Any:
    module_name, _, attribute = reference.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"expected 'module:attribute', got {reference!r}")
    return getattr(importlib.import_module(module_name), attribute)


def _constructor_kwargs(
    agent_cls: type, run_command: Callable[[str], Any], workdir: str, agent_kwargs: dict[str, Any]
) -> dict[str, Any]:
    """``agent_kwargs`` verbatim, plus the worker-supplied values the constructor names explicitly."""
    params = inspect.signature(agent_cls).parameters
    supplied = {"run_command": run_command, "workdir": workdir, "working_dir": workdir}
    return {**{k: v for k, v in supplied.items() if k in params}, **agent_kwargs}


def _checkpoint_then_cancel(checkpoint: Callable[[], None], cancel: Callable[[], object]) -> None:
    """Write the deadline checkpoint, then cancel the run; the cancel happens even if the write fails.

    The checkpoint comes first because the supervisor's TERM grace is short and the cancelled
    agent's own cleanup may take longer than that; the final write refreshes it when the run
    winds down in time.
    """
    try:
        checkpoint()
    except Exception as error:  # a failed checkpoint must not keep the worker running past the stop
        print(
            f"nooa_bench.gym_worker: checkpoint failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
    finally:
        cancel()


def _trace_exporters(
    payload: dict[str, Any], root: Path | None
) -> tuple[list[Any], dict[str, str]]:
    """NOOA's native OTLP span exporters for this episode: a live collector, a JSONL spool, or both.

    ``otlp_endpoint`` ships spans as they happen to the job's collector (Gym runs one beside
    each job and the trace viewer reads it). ``trace_jsonl`` keeps the same spans as OTLP JSONL
    files under the artifacts directory, for collection after the episode when the pod cannot
    reach a collector or when the run must keep its own copy.
    """
    from nooa.tracing import exporters

    selected: list[Any] = []
    enabled: dict[str, str] = {}
    if payload.get("otlp_endpoint"):
        selected.append(exporters.otlp(payload["otlp_endpoint"]))
        enabled["otlp_endpoint"] = str(payload["otlp_endpoint"])
    if payload.get("trace_jsonl") and root is not None:
        selected.append(exporters.jsonl(root / "traces"))
        enabled["trace_jsonl"] = str(root / "traces")
    return selected, enabled


def _enable_artifacts(payload: dict[str, Any]) -> dict[str, str]:
    """Turn on NOOA's own exports (ATIF, OTLP tracing); return what was enabled and where."""
    directory = payload.get("artifacts_dir")
    root = Path(directory) if directory else None
    enabled: dict[str, str] = {}
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
        enabled["artifacts_dir"] = str(root)
        if payload.get("atif"):
            from nooa.atif import enable_atif

            enable_atif(output_dir=root / "atif")
            enabled["atif"] = str(root / "atif")
    selected, tracing = _trace_exporters(payload, root)
    if selected:
        from nooa.tracing import enable_tracing

        enable_tracing(exporters=selected)
        enabled.update(tracing)
    return enabled


async def main(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text())
    trace_env = ",".join(sorted(name for name in os.environ if name.startswith(("OTLP_", "OTEL_"))))
    agent_cls = _import(payload["agent"])
    turn_method = payload.get("turn_method") or "solve"
    workdir = payload.get("workdir") or str(Path.cwd())
    shell_timeout = float(payload.get("shell_timeout_seconds") or 180)
    artifacts = _enable_artifacts(payload)

    async def run_command(command: str) -> str:
        return await _local_shell(command, cwd=workdir, timeout=shell_timeout)

    # Gym Resources tools, granted over direct HTTP for this episode only.
    resource_cookies: dict[str, str] = dict((payload.get("tool_access") or {}).get("cookies") or {})
    tools = []
    if payload.get("tools") and payload.get("tool_access"):
        access = payload["tool_access"]
        tools = resource_tools(
            payload["tools"],
            base_url=access["base_url"],
            headers=access.get("headers") or {},
            cookies=resource_cookies,
        )
    supplied = payload.get("agent_kwargs") or {}
    policy_required = (
        getattr(agent_cls, "benchmark_policy_required", False) is True
        or "benchmark_policy" in supplied
    )
    if policy_required and (payload.get("tools") or payload.get("tool_binding") is not None):
        raise ValueError("Atom benchmark requires no Gym tools and tool_binding=null")
    if "benchmark_policy_path" in supplied:
        raise ValueError("benchmark_policy_path is worker-owned, not an agent_kwargs override")
    policy_path = (
        output_path.with_name(output_path.name + ".atom-policy.json") if policy_required else None
    )
    bind_tools = _import(payload["tool_binding"]) if payload.get("tool_binding") else bind_gym_tools
    kwargs = _constructor_kwargs(
        agent_cls,
        run_command,
        workdir,
        {**supplied, **({"benchmark_policy_path": str(policy_path)} if policy_required else {})},
    )

    # Only the supervisor's SIGTERM (sent at the agent's time budget) makes a cancelled run a
    # `timeout_exceeded` episode. Any other cancellation, and SIGINT, propagate and leave no result,
    # so they stay failures on the Gym side instead of becoming a gradable timeout.
    stop: dict[str, str | None] = {"signal": None}
    render: list = []  # filled by run_gym_episode once the run starts
    run = asyncio.current_task()
    assert run is not None

    def write(episode) -> None:
        policy_failed = (
            policy_required and episode.runtime.get("benchmark_policy_verdict") != "pass"
        )
        if policy_failed:
            episode.stop_reason, episode.return_value = "error", None
            episode.error = "benchmark policy verdict missing, failed, or incomplete"
        if stop["signal"] and episode.stop_reason != "timeout_exceeded" and not policy_failed:
            # The run wound down after the stop as something else (the agent swallowed the
            # cancellation, or its teardown raised). The stop still classifies the episode; what
            # the teardown reported is kept as a diagnostic.
            episode.runtime["teardown"] = f"{episode.stop_reason}: {episode.error}"
            episode.stop_reason, episode.error = (
                "timeout_exceeded",
                "the agent was stopped before it finished",
            )
        episode.runtime.update(
            python=sys.executable,
            version=sys.version.split()[0],
            hostname=socket.gethostname(),
            trace_env=trace_env,
            **artifacts,
            **({"stop_signal": stop["signal"]} if stop["signal"] else {}),
        )
        tmp = output_path.with_suffix(".tmp")
        tmp.write_text(episode.to_json())
        tmp.replace(output_path)

    def on_sigterm() -> None:
        stop["signal"] = "SIGTERM"
        _checkpoint_then_cancel(
            lambda: write(render[0]("timeout_exceeded")) if render else None, run.cancel
        )

    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, on_sigterm)
    try:
        episode = await run_gym_episode(
            lambda llm: agent_cls(llm=llm, **kwargs),
            lambda agent: getattr(agent, turn_method)(payload["task"]),
            model_url=payload["model_url"],
            model_name=payload.get("model_name", "gym-policy"),
            max_policy_calls=payload.get("max_policy_calls"),
            extra_headers=payload.get("headers") or {},
            cookies=payload.get("cookies") or {},
            resource_cookies=resource_cookies,
            context_window=payload.get("context_window"),
            replay_reasoning=bool(payload.get("replay_reasoning", False)),
            media_inputs=tuple(payload.get("media_inputs") or ()),
            tools=tools,
            bind_tools=bind_tools,
            interrupt=lambda: "timeout_exceeded" if stop["signal"] else None,
            snapshot=render.append,
            policy_required=policy_required,
            policy_path=policy_path,
        )
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
    write(episode)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
