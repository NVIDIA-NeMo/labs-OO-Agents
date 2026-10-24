# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a NOOA agent as a NeMo Gym policy harness, on the Gym side of the wire.

Spike for the "thin" Gym integration. Gym supplies a rollout-prefixed
Responses URL, headers and cookies; this module runs the agent against it with
NOOA's own ``ResponsesClient`` and returns the exact wire evidence Gym needs:
every request/response pair as sent, every tool execution, and a stop reason.
Nothing here imports Gym.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel

from nooa import Agent, strategy
from nooa.context_blocks.events import ResultStatus, ToolCallEvent, ToolResult
from nooa.errors import (
    GenerationError,
    IterationLimitExceeded,
    LoopDetectedError,
    RetryLimitExceeded,
)
from nooa.runtime.hooks import call_after_hook, call_before_hook, hooks_scope
from nooa.strategies.base import GenerationStrategy, RuntimeServices
from nooa.strategies.current_call import CurrentCall
from nooa.unifiedllm import HttpConfig, ResponsesClient, Tool

__all__ = [
    "GymEpisode",
    "resource_tools",
    "ModelCallRecord",
    "PolicyCallBudgetExceeded",
    "OutputTokenLimitExceeded",
    "ToolLoopStrategy",
    "run_gym_episode",
]


class PolicyCallBudgetExceeded(RuntimeError):
    """The rollout used its configured number of policy calls."""


class OutputTokenLimitExceeded(RuntimeError):
    """A policy response stopped at max_output_tokens; its tool calls were not executed."""


@dataclass(slots=True)
class ModelCallRecord:
    """One wire request/response pair, exactly as sent and received."""

    invocation_id: str
    request: dict[str, Any]
    response: dict[str, Any] | None = None
    status_code: int | None = None
    started_at: float = 0.0
    completed_at: float | None = None


@dataclass(slots=True)
class ToolCallRecord:
    invocation_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    output: Any
    started_at: float
    completed_at: float
    status: str  # completed | failed


@dataclass(slots=True)
class GymEpisode:
    """What a Gym agent server needs to answer `/v1/responses` and `close`."""

    output_items: list[dict[str, Any]]
    conversation: list[dict[str, Any]]
    model_calls: list[ModelCallRecord]
    tool_calls: list[ToolCallRecord]
    return_value: Any
    stop_reason: str  # completed | policy_budget_exceeded | max_output_tokens_exceeded | max_iterations_exceeded | loop_detected | retry_limit_exceeded | timeout_exceeded | error
    error: str | None = None
    resource_cookies: dict[str, str] = field(default_factory=dict)
    # Where the agent ran (interpreter path, version, host), filled in by the worker.
    runtime: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        def default(value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump(mode="json")
            return str(value)

        return json.dumps(asdict(self), default=default)

    @classmethod
    def from_json(cls, text: str) -> GymEpisode:
        data = json.loads(text)
        data["model_calls"] = [ModelCallRecord(**m) for m in data["model_calls"]]
        data["tool_calls"] = [ToolCallRecord(**t) for t in data["tool_calls"]]
        return cls(**data)


_current_invocation: ContextVar[str] = ContextVar("nooa_gym_invocation", default="root")


class _RecordingClient(ResponsesClient):
    """NOOA's Responses client plus Gym's per-rollout budget and header, with wire capture."""

    def __init__(
        self,
        *args: Any,
        max_policy_calls: int | None,
        records: list[ModelCallRecord],
        context_window: int | None = None,
        replay_reasoning: bool = False,
        media_inputs: tuple[str, ...] = (),
        **kwargs: Any,
    ):
        # NOOA's media policy: image/audio/video/pdf blocks reach the policy only when listed
        # here; everything else is replaced by a short note before the request is built.
        kwargs.setdefault("media_inputs", tuple(media_inputs) or None)
        # Gym's model server validates requests strictly (no extra body keys) and fronts vLLM,
        # so NOOA's OpenAI explicit prompt-cache option (`extra_body.prompt_cache_options`)
        # would be a 400 there. Off unless the harness asks for it explicitly.
        kwargs.setdefault("cache_breakpoint", None)
        # ResponsesClient sends `truncation: disabled` by default; Gym's chat-backed model servers
        # cannot downconvert that field and refuse the request. None drops it from the wire.
        kwargs.setdefault("truncation", None)
        super().__init__(*args, **kwargs)
        self._max_policy_calls = max_policy_calls
        self._records = records
        # Client metadata only: passed as config it would leak into the request body.
        self._context_window_override = context_window
        self._replay_reasoning = replay_reasoning
        self._pending: dict[int, ModelCallRecord] = {}
        self._http.httpx_async.event_hooks = {
            "request": [self._on_request],
            "response": [self._on_response],
        }

    @property
    def context_window(self) -> int | None:  # NOOA's runtime reads this for context budgeting
        return self._context_window_override or super().context_window

    def _prepare_input(
        self, messages, state_scope, model, native_encrypted_reasoning=False, *, media_inputs=None
    ):
        # The Gym route looks like OpenAI, so ResponsesClient would replay every prior reasoning
        # item (summary text, no encrypted state). Gym's chat-backed servers render those back
        # into the prompt as <think> text, which the thick nooa_agent never sends: about 1.5x the
        # input per call and a different model view of its own history. Off unless asked for.
        input_messages, instructions = self._transform_messages(
            messages, state_scope, native_encrypted_reasoning, media_inputs=media_inputs
        )
        if not self._replay_reasoning:
            input_messages = [
                m
                for m in input_messages
                if not (isinstance(m, dict) and m.get("type") == "reasoning")
            ]
        # Gym's request schema requires `detail` on input_image; NOOA only carries one when the
        # image came with it. "auto" is the Responses API default.
        for item in input_messages:
            if not isinstance(item, dict):
                continue
            blocks = (
                item.get("output")
                if item.get("type") == "function_call_output"
                else item.get("content")
            )
            if isinstance(blocks, list):
                for block in blocks:
                    if isinstance(block, dict) and block.get("type") == "input_image":
                        block.setdefault("detail", "auto")
        return self._prepare_cache_boundary(
            input_messages, responses=True, model=model, instructions=instructions
        )

    async def _on_request(self, request: httpx.Request) -> None:
        # Gym attributes a model call to an invocation through this header.
        request.headers["x-session-id"] = _current_invocation.get()
        try:
            body = json.loads(request.content or b"{}")
        except ValueError:
            body = {"_raw": request.content.decode(errors="replace")}
        record = ModelCallRecord(
            invocation_id=_current_invocation.get(), request=body, started_at=time.time()
        )
        self._pending[id(request)] = record
        self._records.append(record)

    async def _on_response(self, response: httpx.Response) -> None:
        record = self._pending.pop(id(response.request), None)
        if record is None:
            return
        await response.aread()
        record.status_code = response.status_code
        record.completed_at = time.time()
        try:
            record.response = response.json()
        except ValueError:
            record.response = {"_raw": response.text}

    async def acall(
        self, messages: Any, tools: Any = None, output_model: Any = None, **kwargs: Any
    ) -> Any:
        used = sum(1 for r in self._records if r.status_code is not None and r.status_code < 400)
        if self._max_policy_calls is not None and used >= self._max_policy_calls:
            raise PolicyCallBudgetExceeded(
                f"policy call budget of {self._max_policy_calls} exhausted"
            )
        response = await super().acall(messages, tools, output_model, **kwargs)
        if response.finish_reason == "length" and response.tool_calls:
            # Never execute a tool call that was cut off mid-generation.
            raise OutputTokenLimitExceeded(
                "policy response exhausted max_output_tokens before completion"
            )
        return response


_SCHEMA_TYPES = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


def resource_tools(
    schemas: list[dict[str, Any]],
    *,
    base_url: str,
    headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    timeout: float = 300.0,
) -> list[Tool]:
    """Gym Resources tools (OpenAI function schemas, POST ``<base_url>/<name>``) as NOOA tools.

    Server failures (5xx) raise and end the rollout; other HTTP errors and invalid
    arguments are returned to the model as tool output. ``cookies`` is updated in
    place with the Resources server's session cookies.
    """
    from jsonschema import Draft202012Validator, ValidationError
    from pydantic import Field, create_model

    cookie_jar = cookies if cookies is not None else {}
    tools: list[Tool] = []
    for schema_item in schemas:
        if schema_item.get("type") != "function":
            raise ValueError(
                f"resource tools support function tools only, got {schema_item.get('type')!r}"
            )
        name = schema_item["name"]
        schema = schema_item.get("parameters") or {"type": "object", "properties": {}}
        validator = Draft202012Validator(schema)
        required = set(schema.get("required", []))
        fields = {
            key: (
                _SCHEMA_TYPES.get(prop.get("type"), Any),
                Field(
                    ... if key in required else prop.get("default"),
                    description=prop.get("description"),
                ),
            )
            for key, prop in (schema.get("properties") or {}).items()
        }
        model = create_model(f"{name}_arguments", **fields)

        async def call(
            __name: str = name, __validator: Draft202012Validator = validator, **arguments: Any
        ) -> Any:
            try:
                __validator.validate(arguments)
            except ValidationError as error:
                return {"error": f"Invalid arguments for {__name}: {error.message}"}
            async with httpx.AsyncClient(timeout=timeout, cookies=cookie_jar) as http:
                response = await http.post(
                    f"{base_url.rstrip('/')}/{__name}", json=arguments, headers=headers or {}
                )
            cookie_jar.update({c.name: c.value for c in http.cookies.jar})
            if response.status_code >= 500:
                response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return response.text

        tools.append(
            Tool(
                name=name,
                description=schema_item.get("description") or name,
                callable=call,
                parameters_model=model,
            )
        )
    return tools


class ToolLoopStrategy(GenerationStrategy):
    """Expose named agent methods as function tools and loop until the model answers in text.

    Minimal on purpose: each tool call is executed and recorded as a
    ``ToolCallEvent`` so the next turn's history carries the result. Tool
    execution fires the standard ``before/after_tool_execution`` hooks.
    """

    def __init__(self, tool_methods: tuple[str, ...], *, max_iterations: int = 50) -> None:
        self.tool_methods = tool_methods
        self.max_iterations = max_iterations

    def _tools(self, agent: Any) -> dict[str, Tool]:
        tools: dict[str, Tool] = {}
        for name in self.tool_methods:
            method = getattr(agent, name)
            tools[name] = Tool(
                name=name, description=(method.__doc__ or name).strip(), callable=method
            )
        for tool in getattr(agent, "extra_tools", None) or ():
            tools[tool.name] = tool
        return tools

    async def execute(self, runtime: RuntimeServices, call: CurrentCall) -> Any:
        by_name = self._tools(runtime.agent)
        tools = list(by_name.values())
        for _ in range(self.max_iterations):
            response, _event_id = await runtime.generate(tools=tools, tool_choice="auto")
            if not response.tool_calls:
                return response.content
            for tool_call in response.tool_calls:
                try:
                    arguments = json.loads(tool_call.arguments or "{}")
                except ValueError:
                    arguments = {}
                execution_id = str(uuid4())
                context = call_before_hook(
                    "before_tool_execution",
                    agent=runtime.agent,
                    tool_name=tool_call.name,
                    arguments=arguments,
                    execution_id=execution_id,
                    tool_call_id=tool_call.id,
                    generation_id=runtime.get_generation_id(),
                )
                result: Any = None
                exception: BaseException | None = None
                try:
                    tool = by_name.get(tool_call.name)
                    if tool is None:
                        raise ValueError(f"unknown tool {tool_call.name!r}")
                    result = tool.callable(**arguments)
                    if asyncio.iscoroutine(result):
                        result = await result
                except Exception as error:  # the model sees tool errors as results
                    exception = error
                    result = str(error) or type(error).__name__
                finally:
                    call_after_hook(
                        "after_tool_execution",
                        context,
                        agent=runtime.agent,
                        tool_name=tool_call.name,
                        arguments=arguments,
                        result=result,
                        exception=exception,
                        execution_id=execution_id,
                        tool_call_id=tool_call.id,
                    )
                content = result if isinstance(result, str) else json.dumps(result, default=str)
                runtime.event_manager.add(
                    ToolCallEvent(
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        arguments=arguments,
                        llm_response_id=str(response.id),
                        result=ToolResult(
                            tool_call_id=tool_call.id,
                            content=content,
                            result_status=ResultStatus.ERROR
                            if exception
                            else ResultStatus.COMPLETE,
                        ),
                    )
                )
        raise IterationLimitExceeded(f"tool loop exceeded {self.max_iterations} iterations")


class _ToolHooks:
    """Record tool executions, including delegated children, through NOOA's hooks."""

    def __init__(self, records: list[ToolCallRecord]) -> None:
        self.records = records
        self._open: dict[str, tuple[ToolCallRecord, str]] = {}

    def before_agent_call(self, *, call_id: str, **_: Any) -> Any:
        return _current_invocation.set(call_id)

    def after_agent_call(self, *, context: Any, **_: Any) -> None:
        if context is not None:
            _current_invocation.reset(context)

    def before_tool_execution(
        self, *, tool_name: str, arguments: dict[str, Any], execution_id: str, **kw: Any
    ) -> Any:
        record = ToolCallRecord(
            invocation_id=_current_invocation.get(),
            tool_call_id=str(kw.get("tool_call_id") or execution_id),
            name=tool_name,
            arguments=arguments,
            output=None,
            started_at=time.time(),
            completed_at=0.0,
            status="incomplete",
        )
        self.records.append(record)
        return record

    def after_tool_execution(
        self, *, context: Any, result: Any, exception: BaseException | None, **_: Any
    ) -> None:
        if isinstance(context, ToolCallRecord):
            context.output = result
            context.completed_at = time.time()
            context.status = "failed" if exception is not None else "completed"

    # CodeAct runs model-authored code through these; record it like any other tool.
    def before_code_execution(self, *, code: str, execution_id: str, **kw: Any) -> Any:
        return self.before_tool_execution(
            tool_name=str(kw.get("tool_name") or "python_cell"),
            arguments={"code": code},
            execution_id=execution_id,
            **kw,
        )

    def after_code_execution(
        self, *, context: Any, result: Any, exception: BaseException | None, **_: Any
    ) -> None:
        output = result
        if result is not None and not isinstance(result, (str, int, float, bool, list, dict)):
            output = str(
                getattr(result, "stdout", None) or getattr(result, "output", None) or result
            )
        self.after_tool_execution(context=context, result=output, exception=exception)

    # The remaining hooks are not needed here.
    def before_generation(self, **_: Any) -> None: ...
    def after_generation(self, **_: Any) -> None: ...
    def before_method_invocation(self, **_: Any) -> None: ...
    def after_method_invocation(self, **_: Any) -> None: ...
    def on_messages_built(self, **_: Any) -> None: ...


class ShellAgent(Agent):
    """The smallest useful Gym harness: one shell tool, a tool-call loop, a text answer."""

    def __init__(
        self,
        llm: Any,
        run_command: Callable[[str], Awaitable[str]],
        extra_tools: list[Tool] | None = None,
        **_: Any,
    ) -> None:
        super().__init__(llm=llm)
        self._run_command = run_command
        self.extra_tools = list(extra_tools or [])

    async def shell(self, command: str) -> str:
        """Run a shell command in the task workspace and return its combined output."""
        return await self._run_command(command)

    @strategy(ToolLoopStrategy(("shell",)))
    async def solve(self, task: str) -> str:
        """Complete the task with the shell tool, then reply with the final answer as plain text."""
        ...


# Policy-side stops are gradable ("incomplete"); everything else is an error.
_STOP_REASONS: tuple[tuple[type[BaseException], str], ...] = (
    (PolicyCallBudgetExceeded, "policy_budget_exceeded"),
    (OutputTokenLimitExceeded, "max_output_tokens_exceeded"),
    (IterationLimitExceeded, "max_iterations_exceeded"),
    (LoopDetectedError, "loop_detected"),
    (RetryLimitExceeded, "retry_limit_exceeded"),
)


def _classify(exc: BaseException) -> tuple[str, str]:
    """Walk the cause chain: NOOA wraps limit errors, and Atom wraps turn failures."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        for error_type, reason in _STOP_REASONS:
            if isinstance(current, error_type):
                return reason, str(current)
        current = current.__cause__ or current.__context__
    return "error", f"{type(exc).__name__}: {exc}"


def _episode_items(records: list[ModelCallRecord]) -> tuple[list[dict], list[dict]]:
    """The conversation the policy last saw plus its final answer, verbatim."""
    done = [r for r in records if r.response is not None and (r.status_code or 500) < 400]
    if not done:
        return [], []
    last = done[-1]
    history = last.request.get("input") or []
    if isinstance(history, str):
        history = [{"role": "user", "content": history}]
    output = list(last.response.get("output") or [])
    # Response output holds typed items only: policy-authored items and tool
    # outputs. Role messages (the task, rendered context) stay in the conversation.
    typed = [item for item in history if isinstance(item, dict) and item.get("type")]
    return [*typed, *output], [*history, *output]


async def run_gym_episode(
    agent_factory: Callable[[ResponsesClient], Any],
    invoke: Callable[[Any], Awaitable[Any]],
    *,
    model_url: str,
    model_name: str = "gym-policy",
    max_policy_calls: int | None = None,
    http_config: HttpConfig | None = None,
    extra_headers: dict[str, str] | None = None,
    cookies: dict[str, str] | None = None,
    resource_cookies: dict[str, str] | None = None,
    context_window: int | None = None,
    replay_reasoning: bool = False,
    media_inputs: tuple[str, ...] = (),
    interrupt_stop_reason: str | None = None,
) -> GymEpisode:
    """Run one NOOA agent activation against a Gym model server and return the evidence.

    ``resource_cookies`` is the Resources server's cookie jar; tools built with
    :func:`resource_tools` update it in place and the episode reports it back.
    ``media_inputs`` lists what the policy accepts (``image``, ``audio``, ``video``, ``pdf``);
    empty is text-only.
    ``interrupt_stop_reason`` turns a cancellation of this coroutine into a finished episode
    with that stop reason, keeping the model and tool calls made so far (the worker uses it
    when the sandbox supervisor stops it at the agent's time budget). By default a
    cancellation propagates.
    """
    model_calls: list[ModelCallRecord] = []
    tool_calls: list[ToolCallRecord] = []
    client = _RecordingClient(
        f"openai/{model_name}",
        api_base=model_url,
        api_key="nemo-gym",
        extra_headers=dict(extra_headers or {}),
        http_config=http_config or HttpConfig(read_timeout=600, request_timeout=None),
        max_policy_calls=max_policy_calls,
        records=model_calls,
        context_window=context_window,
        replay_reasoning=replay_reasoning,
        media_inputs=media_inputs,
    )
    if cookies:
        client._http.httpx_async.cookies.update(cookies)
    stop_reason, error, return_value = "completed", None, None
    try:
        agent = agent_factory(client)
        with hooks_scope(_ToolHooks(tool_calls)):
            return_value = await invoke(agent)
    except PolicyCallBudgetExceeded as exc:
        stop_reason, error = "policy_budget_exceeded", str(exc)
    except OutputTokenLimitExceeded as exc:
        stop_reason, error = "max_output_tokens_exceeded", str(exc)
    except GenerationError as exc:
        stop_reason, error = _classify(exc)
    except Exception as exc:
        stop_reason, error = _classify(exc)
    except asyncio.CancelledError:
        if interrupt_stop_reason is None:
            raise
        stop_reason, error = interrupt_stop_reason, "the agent was stopped before it finished"
    finally:
        cookies_out = {c.name: c.value for c in client._http.httpx_async.cookies.jar}
        if cookies is not None:
            cookies.update(cookies_out)
        await client.aclose()
    output_items, conversation = _episode_items(model_calls)
    return GymEpisode(
        output_items=output_items,
        conversation=conversation,
        model_calls=model_calls,
        tool_calls=tool_calls,
        return_value=return_value,
        stop_reason=stop_reason,
        error=error,
        resource_cookies=dict(resource_cookies or {}),
    )
