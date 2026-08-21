# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-agent turn policy owned by the TUI composition root.

The renderer consumes immutable agent state.  Concrete-agent reflection and
keep-going behavior stays here, beside the local runtime adapter that invokes it.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable
from typing import Any

from nooa_cli.interactive.runtime import AgentRuntime

logger = logging.getLogger(__name__)


def _short_exception_message(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return text[:160]


class LocalTurnPolicy:
    """Concrete local-agent policy kept outside ``TUIApplication``."""

    def __init__(
        self,
        agent: Any,
        runtime: AgentRuntime,
        config: Any,
        *,
        emit_output: Callable[[Any], Awaitable[None]],
        invalidate: Callable[[], None],
    ) -> None:
        self._agent = agent
        self._runtime = runtime
        self._config = config
        self._emit_output = emit_output
        self._invalidate = invalidate
        self._tasks: set[asyncio.Task[Any]] = set()
        self._generation = 0
        self._closed = False
        self._state_lock = threading.Lock()

    async def before_handle(self, agent: Any) -> None:
        if not self._is_active():
            return
        reflection = getattr(agent, "_tui_reflection_runner", None)
        if reflection is not None:
            await reflection.interrupt()

    async def after_handle(self, agent: Any, result: Any) -> None:
        """Apply keep-going and reflection policy after one completed turn."""
        if not self._is_active():
            return
        explanation = getattr(result, "explanation", "")
        logger.info(
            "[DISPATCHER] handle() returned kind=%r explanation=%r",
            result.kind,
            explanation,
        )
        reflection_deferred = False
        enabled, model = self._keep_going_state(agent)
        if enabled and str(result.kind) == "DONE":
            if model:
                reflection_deferred = True
                with self._state_lock:
                    if self._closed:
                        return
                    generation = self._generation
                    task = asyncio.create_task(
                        self._run_keep_going_audit(agent, result, model, generation),
                        name="keep-going-audit",
                    )
                    self._tasks.add(task)
                task.add_done_callback(self._discard_task)
            else:
                from .output import StopReasonOutput

                await self._emit_output(
                    StopReasonOutput(
                        "KEEP_GOING",
                        "disabled: configure a model with /keep-going model <model-id>",
                    )
                )
                if not self._is_active():
                    return
        if not reflection_deferred:
            self._schedule_reflection(agent)
        if explanation and self._is_active():
            from .output import StopReasonOutput

            await self._emit_output(StopReasonOutput(result.kind, explanation))

    def on_notification(self, notification: dict[str, list[Any]]) -> None:
        if "user_messages" in notification or "slash_commands" in notification:
            self.invalidate_keep_going()

    def invalidate_keep_going(self) -> None:
        """Make in-flight audits stale and cancel them on their owning loop."""
        with self._state_lock:
            if self._closed:
                return
            self._generation += 1
            tasks = tuple(self._tasks)
        self._runtime.cancel_tasks(list(tasks))

    async def shutdown(self) -> None:
        """Atomically close and quiesce policy producers before host teardown."""
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._generation += 1
            tasks = tuple(self._tasks)
        self._runtime.cancel_tasks(list(tasks))
        if tasks:

            async def _await_tasks() -> None:
                await asyncio.gather(*tasks, return_exceptions=True)

            await self._runtime.run_async(_await_tasks)
        await self.interrupt_reflection(teardown=True)

    async def interrupt_reflection(self, *, teardown: bool = False) -> None:
        runner = self._reflection_runner()
        if runner is None:
            return

        async def _stop() -> None:
            await runner.interrupt()
            if teardown:
                runner.teardown()

        await self._runtime.run_async(_stop)

    def _keep_going_state(self, agent: Any) -> tuple[bool, str | None]:
        vars_obj = getattr(agent, "vars", None)
        if vars_obj is not None and "tui_keep_going" in vars_obj:
            enabled = bool(vars_obj.get("tui_keep_going"))
        else:
            enabled = bool(
                self._config is not None
                and getattr(getattr(self._config, "tui", None), "keep_going", False)
            )
        if vars_obj is not None and "tui_keep_going_model" in vars_obj:
            value = vars_obj.get("tui_keep_going_model")
        else:
            value = getattr(getattr(self._config, "tui", None), "keep_going_model", None)
        model = str(value).strip() if value is not None else ""
        if not model or model.lower() == "none":
            model = ""
        return enabled, model or None

    async def _run_keep_going_audit(
        self, agent: Any, result: Any, model: str, generation: int
    ) -> None:
        from .output import StopReasonOutput

        try:
            from .keep_going import build_keep_going_prompt

            prompt = await build_keep_going_prompt(agent, result, model=model)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if not self._is_active(generation):
                return
            message = _short_exception_message(exc)
            logger.warning("keep-going audit failed: %s", message)
            await self._emit_output(
                StopReasonOutput(
                    "KEEP_GOING",
                    f"judge failed: {message}; use /keep-going off or /keep-going model <model-id>",
                )
            )
            if self._is_active(generation):
                self._schedule_reflection(agent)
            return
        if not self._is_active(generation):
            return
        if not prompt:
            self._schedule_reflection(agent)
            return
        enabled, current_model = self._keep_going_state(agent)
        if not enabled or current_model != model:
            self._schedule_reflection(agent)
            return

        prompt_text = getattr(prompt, "prompt", prompt)
        display_reason = getattr(prompt, "display_reason", "continuing unfinished work")
        if display_reason:
            await self._emit_output(StopReasonOutput("KEEP_GOING", str(display_reason)))
            if not self._is_active(generation):
                return
        route = self._runtime.queue_continuation(str(prompt_text))
        if route is not None:
            logger.info("[DISPATCHER] keep-going queued %s continuation prompt", route)
            return
        self._schedule_reflection(agent)

    def _is_active(self, generation: int | None = None) -> bool:
        with self._state_lock:
            return not self._closed and (generation is None or generation == self._generation)

    def _discard_task(self, task: asyncio.Task[Any]) -> None:
        with self._state_lock:
            self._tasks.discard(task)

    def _reflection_runner(self) -> Any | None:
        runner = getattr(self._agent, "_tui_reflection_runner", None)
        if runner is not None:
            runner.invalidate = self._invalidate
        return runner

    def _schedule_reflection(self, agent: Any) -> None:
        if not self._is_active():
            return
        runner = getattr(agent, "_tui_reflection_runner", None)
        if runner is not None:
            runner.invalidate = self._invalidate
            runner.on_response_done()
