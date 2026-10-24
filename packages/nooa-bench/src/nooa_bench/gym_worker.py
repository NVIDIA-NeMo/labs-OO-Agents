# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one Gym episode inside the task sandbox: ``python -m nooa_bench.gym_worker in.json out.json``.

The worker needs NOOA only. Gym's model server already holds the authoritative
copy of every model call, so the result file is the episode summary, not the
sole evidence.

When the sandbox supervisor stops the worker at the agent's time budget it sends
SIGTERM and waits a short grace period before SIGKILL. The worker uses that window
to stop the agent and write the episode it has so far with stop reason
``timeout_exceeded``, so Gym grades the sandbox state instead of failing the rollout.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import shlex
import signal
import socket
import sys
from pathlib import Path

from nooa_bench.gym import resource_tools, run_gym_episode


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


async def main(input_path: Path, output_path: Path) -> None:
    payload = json.loads(input_path.read_text())
    module_name, _, class_name = payload["agent"].partition(":")
    agent_cls = getattr(importlib.import_module(module_name), class_name)
    workdir = payload.get("workdir") or str(Path.cwd())
    shell_timeout = float(payload.get("shell_timeout_seconds") or 180)

    async def run_command(command: str) -> str:
        return await _local_shell(command, cwd=workdir, timeout=shell_timeout)

    # Gym Resources tools, granted over direct HTTP for this episode only.
    resource_cookies: dict[str, str] = dict((payload.get("tool_access") or {}).get("cookies") or {})
    extra_tools = []
    if payload.get("tools") and payload.get("tool_access"):
        access = payload["tool_access"]
        extra_tools = resource_tools(
            payload["tools"],
            base_url=access["base_url"],
            headers=access.get("headers") or {},
            cookies=resource_cookies,
        )

    # Harness classes take what they need: ShellAgent uses run_command, Atom uses workdir.
    extra = {
        "run_command": run_command,
        "workdir": workdir,
        "extra_tools": extra_tools,
        **(payload.get("agent_kwargs") or {}),
    }
    params = inspect.signature(agent_cls).parameters
    accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = extra if accepts_any else {k: v for k, v in extra.items() if k in params}

    # The supervisor's SIGTERM at the time budget cancels the run; run_gym_episode turns that
    # into a `timeout_exceeded` episode and the result below is still written.
    run = asyncio.current_task()
    assert run is not None
    loop = asyncio.get_running_loop()
    loop.add_signal_handler(signal.SIGTERM, run.cancel)
    try:
        episode = await run_gym_episode(
            lambda llm: agent_cls(llm=llm, **kwargs),
            lambda agent: agent.solve(payload["task"]),
            model_url=payload["model_url"],
            model_name=payload.get("model_name", "gym-policy"),
            max_policy_calls=payload.get("max_policy_calls"),
            extra_headers=payload.get("headers") or {},
            cookies=payload.get("cookies") or {},
            resource_cookies=resource_cookies,
            context_window=payload.get("context_window"),
            replay_reasoning=bool(payload.get("replay_reasoning", False)),
            media_inputs=tuple(payload.get("media_inputs") or ()),
            interrupt_stop_reason="timeout_exceeded",
        )
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
    episode.runtime = {
        "python": sys.executable,
        "version": sys.version.split()[0],
        "hostname": socket.gethostname(),
    }
    tmp = output_path.with_suffix(".tmp")
    tmp.write_text(episode.to_json())
    tmp.replace(output_path)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), Path(sys.argv[2])))
