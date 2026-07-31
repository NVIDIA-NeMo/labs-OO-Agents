# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Copilot SDK-backed benchmark agent.

This agent preserves the Harbor runner's ``_run_evaluation(dict) -> dict``
contract while delegating execution to the official GitHub Copilot SDK runtime.
It intentionally does not subclass ``nooa.Agent``: Copilot SDK already provides
the agent loop, built-in coding tools, permissions, and session lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from copilot import CopilotClient, ToolInvocation, ToolResult, define_tool
from copilot.session import PermissionHandler
from copilot.session_events import (
    AssistantMessageData,
    AssistantUsageData,
    SessionErrorData,
    SessionEvent,
    SessionIdleData,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator

_logger = logging.getLogger(__name__)

ReasoningEffort = Literal["low", "medium", "high", "xhigh"]
ContextTier = Literal["default", "long_context"]

DEFAULT_COPILOT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT: ReasoningEffort = "xhigh"
DEFAULT_CONTEXT_TIER: ContextTier = "long_context"
DEFAULT_TIMEOUT_SECONDS = 3600.0
_CLEANUP_TIMEOUT_SECONDS = 10.0

_SDK_REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
_SDK_CONTEXT_TIERS = {"default", "long_context"}

_SYSTEM_MESSAGE_APPEND = """\
You are running as the NOOA benchmark agent inside an isolated benchmark
container or worktree. Use the SDK-provided coding tools to inspect, edit, and
verify the task autonomously. Non-interactive tool permission approval is enabled
only for this isolated benchmark environment.

Do not ask the user for clarification; make reasonable assumptions and proceed.
Before finishing, run the most targeted verification command that demonstrates
the task is solved.

When the task is complete, call the return_task_result tool exactly once with:
- solution_description: what changed and why it solves the task
- evidence: commands run and concrete observed output
- command_to_verify: one shell command a verifier can run

Do not finish by printing JSON text; the benchmark runner only accepts the
return_task_result tool call as completion.
"""


class TaskResult(BaseModel):
    """Strict structured result for the Copilot-backed benchmark agent."""

    model_config = ConfigDict(extra="forbid", strict=True)

    solution_description: str = Field(
        description="What changed and why it solves the benchmark task."
    )
    evidence: str = Field(description="Concrete commands/output observed during verification.")
    command_to_verify: str = Field(description="Verifier command expected to exit 0 on success.")

    @field_validator("solution_description", "evidence", "command_to_verify")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must be a non-blank string")
        return stripped


def _problem_statement(task_input: dict) -> str:
    """Extract the task text from supported Harbor/benchmark field names."""
    for key in ("user_message", "problem_statement", "task_description"):
        value = task_input.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(
        "task_input must include a non-empty user_message, problem_statement, or task_description"
    )


class _CopilotSessionProtocol(Protocol):
    def on(self, handler: Any) -> Any: ...

    async def send(self, prompt: str, **kwargs: Any) -> str: ...

    async def disconnect(self) -> None: ...


class _CopilotClientProtocol(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def force_stop(self) -> None: ...

    async def list_models(self) -> list[Any]: ...

    async def create_session(self, **kwargs: Any) -> _CopilotSessionProtocol: ...


@dataclass
class _UsageAccumulator:
    n_input_tokens: int = 0
    n_output_tokens: int = 0
    _seen_api_call_ids: set[str] = field(default_factory=set)

    def add(self, data: AssistantUsageData) -> None:
        api_call_id = data.api_call_id
        if api_call_id:
            if api_call_id in self._seen_api_call_ids:
                return
            self._seen_api_call_ids.add(api_call_id)

        self.n_input_tokens += data.input_tokens or 0
        self.n_output_tokens += data.output_tokens or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "n_input_tokens": self.n_input_tokens,
            "n_output_tokens": self.n_output_tokens,
        }


@dataclass
class _CopilotRunOutcome:
    final_response: str
    usage: _UsageAccumulator
    task_result: TaskResult | None = None
    error: str | None = None


def _prompt_for_task(description: str, instructions: str, initial_observation: str) -> str:
    parts = [
        "Solve this benchmark task end-to-end.",
        "",
        "Task:",
        description,
    ]
    if instructions.strip():
        parts.extend(["", "Additional instructions:", instructions.strip()])
    if initial_observation.strip():
        parts.extend(["", "Initial observation:", initial_observation.strip()])
    parts.extend(
        [
            "",
            "Finish by calling the return_task_result tool.",
        ]
    )
    return "\n".join(parts)


class CopilotBenchAgent:
    """Benchmark agent implemented with the official GitHub Copilot SDK."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_COPILOT_MODEL,
        reasoning_effort: ReasoningEffort | None = DEFAULT_REASONING_EFFORT,
        context_tier: ContextTier | None = DEFAULT_CONTEXT_TIER,
        timeout_seconds: float | None = DEFAULT_TIMEOUT_SECONDS,
        github_token: str | None = None,
        client_factory: Any = CopilotClient,
    ) -> None:
        self.model = model
        self.reasoning_effort = (
            DEFAULT_REASONING_EFFORT if reasoning_effort is None else reasoning_effort
        )
        self.context_tier = DEFAULT_CONTEXT_TIER if context_tier is None else context_tier
        self.timeout_seconds = (
            DEFAULT_TIMEOUT_SECONDS if timeout_seconds is None else timeout_seconds
        )
        self.github_token = github_token or os.environ.get("COPILOT_GITHUB_TOKEN")
        self._client_factory = client_factory

    async def _run_evaluation(self, task_input: dict) -> dict:
        """Entry point called by the Harbor runner."""
        description = _problem_statement(task_input)
        working_dir = self._working_dir(task_input.get("working_dir"))
        instructions = task_input.get("system_prompt") or task_input.get("instructions") or ""
        initial_observation = task_input.get("initial_observation") or ""

        prompt = _prompt_for_task(description, instructions, initial_observation)
        usage = _UsageAccumulator()
        try:
            if self.timeout_seconds is None:
                outcome = await self._run_copilot_session(
                    prompt=prompt, working_dir=working_dir, usage=usage
                )
            else:
                outcome = await asyncio.wait_for(
                    self._run_copilot_session(prompt=prompt, working_dir=working_dir, usage=usage),
                    timeout=self.timeout_seconds,
                )
        except TimeoutError:
            error = f"Copilot session timed out after {self.timeout_seconds:g} seconds"
            _logger.error(error)
            return {
                "response": "",
                "success": False,
                "error": error,
                **usage.as_dict(),
            }
        except asyncio.CancelledError:
            raise
        except ValueError as exc:
            _logger.exception("CopilotBenchAgent configuration failed: %s", exc)
            return {
                "response": "",
                "success": False,
                "error": str(exc),
                **usage.as_dict(),
            }
        except Exception as exc:
            _logger.exception("CopilotBenchAgent failed: %s", exc)
            return {
                "response": "",
                "success": False,
                "error": str(exc),
                **usage.as_dict(),
            }
        usage = outcome.usage.as_dict()

        if outcome.error:
            return {
                "response": outcome.final_response,
                "success": False,
                "error": outcome.error,
                **usage,
            }

        if outcome.task_result is None:
            return {
                "response": outcome.final_response,
                "success": False,
                "error": "Copilot session went idle before return_task_result was called",
                **usage,
            }

        result = outcome.task_result
        success = all(getattr(result, field_name) for field_name in TaskResult.model_fields)
        return {
            "response": result.command_to_verify,
            "success": success,
            "result": result.model_dump(),
            "copilot_response": outcome.final_response,
            **usage,
        }

    def _working_dir(self, configured: Any) -> str:
        if isinstance(configured, str) and configured.strip():
            path = Path(configured)
        else:
            path = next((Path(d) for d in ("/testbed", "/app") if Path(d).is_dir()), Path.cwd())

        if not path.is_dir():
            raise ValueError(f"working_dir does not exist: {str(path)!r}")
        return str(path)

    async def _run_copilot_session(
        self, *, prompt: str, working_dir: str, usage: _UsageAccumulator
    ) -> _CopilotRunOutcome:
        _prepare_copilot_home()
        done = asyncio.Event()
        loop = asyncio.get_running_loop()
        final_messages: list[str] = []
        errors: list[str] = []

        def mark_done() -> None:
            if not done.is_set():
                done.set()

        def on_event(event: SessionEvent) -> None:
            data = event.data
            match data:
                case AssistantMessageData() as message:
                    final_messages.append(message.content)
                case AssistantUsageData() as usage_data:
                    usage.add(usage_data)
                case SessionErrorData() as error_data:
                    errors.append(error_data.message)
                    loop.call_soon_threadsafe(mark_done)
                case SessionIdleData() as idle_data:
                    if idle_data.aborted:
                        errors.append("Copilot session aborted")
                    loop.call_soon_threadsafe(mark_done)

        client_kwargs: dict[str, Any] = {"working_directory": working_dir}
        if self.github_token:
            client_kwargs["github_token"] = self.github_token
            client_kwargs["use_logged_in_user"] = False
        else:
            client_kwargs["use_logged_in_user"] = True

        client: _CopilotClientProtocol = self._client_factory(**client_kwargs)
        session: _CopilotSessionProtocol | None = None
        unsubscribe: Any = None
        try:
            await client.start()
            model_info = await self._find_model(client)
            completed_result: TaskResult | None = None

            async def return_task_result(
                params: TaskResult, invocation: ToolInvocation
            ) -> ToolResult:
                nonlocal completed_result
                completed_result = params
                return ToolResult(
                    text_result_for_llm="TaskResult recorded. Stop now.",
                    result_type="success",
                    session_log="TaskResult recorded",
                )

            return_tool = define_tool(
                "return_task_result",
                description="Record the final benchmark TaskResult. Call exactly once when done.",
                params_type=TaskResult,
                handler=return_task_result,
                skip_permission=True,
                defer="never",
            )
            session_kwargs: dict[str, Any] = {
                "model": self.model,
                "on_permission_request": PermissionHandler.approve_all,
                "system_message": {"mode": "append", "content": _SYSTEM_MESSAGE_APPEND},
                "tools": [return_tool],
                "working_directory": working_dir,
            }
            session_kwargs.update(self._supported_model_options(model_info))

            session = await client.create_session(**session_kwargs)
            unsubscribe = session.on(on_event)
            await session.send(prompt, agent_mode="autopilot")
            await done.wait()
        finally:
            try:
                if callable(unsubscribe):
                    try:
                        unsubscribe()
                    except Exception as exc:
                        _logger.exception("Copilot session unsubscribe failed: %s", exc)
                if session is not None:
                    await _bounded_cleanup("Copilot session", session.disconnect())
            finally:
                await _cleanup_client(client)

        return _CopilotRunOutcome(
            final_response=final_messages[-1] if final_messages else "",
            usage=usage,
            task_result=completed_result,
            error=errors[-1] if errors else None,
        )

    async def _find_model(self, client: _CopilotClientProtocol) -> Any:
        models = await client.list_models()
        model = next((candidate for candidate in models if candidate.id == self.model), None)
        if model is None:
            available = ", ".join(sorted(candidate.id for candidate in models))
            raise ValueError(
                f"Copilot model {self.model!r} is not available. Available models: {available}"
            )
        return model

    def _supported_model_options(self, model_info: Any) -> dict[str, str]:
        options: dict[str, str] = {}

        if self.reasoning_effort is not None:
            if self.reasoning_effort not in _SDK_REASONING_EFFORTS:
                raise ValueError(
                    f"Unsupported Copilot SDK reasoning effort: {self.reasoning_effort!r}"
                )

            supported = model_info.supported_reasoning_efforts
            supports_reasoning = model_info.capabilities.supports.reasoning_effort
            if supported is not None:
                if self.reasoning_effort not in supported:
                    raise ValueError(
                        f"Model {self.model!r} does not support reasoning effort "
                        f"{self.reasoning_effort!r}. Supported: {supported}"
                    )
            elif not supports_reasoning:
                raise ValueError(f"Model {self.model!r} does not support reasoning_effort")
            options["reasoning_effort"] = self.reasoning_effort

        if self.context_tier is not None:
            if self.context_tier not in _SDK_CONTEXT_TIERS:
                raise ValueError(f"Unsupported Copilot SDK context tier: {self.context_tier!r}")
            options["context_tier"] = self.context_tier

        return options


def _prepare_copilot_home() -> None:
    configured = os.environ.get("COPILOT_HOME")
    if not configured:
        return
    path = Path(configured).expanduser()
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.chmod(0o700)


async def _bounded_cleanup(label: str, awaitable: Any) -> None:
    try:
        await asyncio.wait_for(awaitable, timeout=_CLEANUP_TIMEOUT_SECONDS)
    except TimeoutError:
        _logger.error("%s cleanup timed out after %s seconds", label, _CLEANUP_TIMEOUT_SECONDS)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _logger.exception("%s cleanup failed: %s", label, exc)


async def _cleanup_client(client: _CopilotClientProtocol) -> None:
    try:
        await asyncio.wait_for(client.stop(), timeout=_CLEANUP_TIMEOUT_SECONDS)
        return
    except TimeoutError:
        _logger.error(
            "Copilot client stop timed out after %s seconds; forcing runtime termination",
            _CLEANUP_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        await _force_stop_client(client)
        raise
    except Exception as exc:
        _logger.exception("Copilot client stop failed; forcing runtime termination: %s", exc)

    await _force_stop_client(client)


async def _force_stop_client(client: _CopilotClientProtocol) -> None:
    try:
        await asyncio.wait_for(client.force_stop(), timeout=_CLEANUP_TIMEOUT_SECONDS)
    except TimeoutError:
        _logger.error(
            "Copilot client force_stop timed out after %s seconds", _CLEANUP_TIMEOUT_SECONDS
        )
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        _logger.exception("Copilot client force_stop failed: %s", exc)
