# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert ATIF trajectories into OTLP JSON that the trace viewer renders.

This is the reverse of :class:`nooa.atif.AtifExporter`: it turns an ATIF
trajectory (written by NOOA or by any other ATIF producer) back into spans, so
trajectories stored as ATIF can be browsed in the viewer.

Span layout per trajectory::

    method span (the trajectory)
    └── generation span (one per agent step)
        ├── LLM span          llm.input_messages.* / llm.output_messages.*
        └── tool span         one per tool call; code tools render as code
            └── method span   a nested method call or embedded subagent trajectory

ATIF has no field for agent methods. NOOA's exporter records ``method_name``,
``generation_id`` and ``parent_generation_id`` in each agent step's ``extra``;
with them, steps of a nested same-agent method call (flattened into the
trajectory) are grouped under their own method span, placed in the parent's
code execution that made the call. Other producers' trajectories are named
from ``extra.method`` or NOOA's ``## Task:`` header, else the trajectory ID.

ATIF stores one linear history. The input of each top-level LLM call is
rebuilt from the steps before it, so it is exact only when the agent did not
rewrite its context; nested calls show only their own turns, because their
prompts are not recorded. Agent steps carry their start timestamp; other steps
carry none, and untimed spans are laid out on a 1 ms grid after the last known
time.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

__all__ = ["DEFAULT_CODE_TOOLS", "trajectory_to_otlp"]

#: Tool names whose calls carry ``{"code": ...}`` and render as code execution.
DEFAULT_CODE_TOOLS = frozenset({"execute_python", "python_cell"})

_SCOPE = {"name": "nooa.atif.importer", "version": "1"}
_OK, _ERROR = 1, 2
_INTERNAL, _CLIENT = 1, 3


def _value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"stringValue": json.dumps(value, ensure_ascii=False, default=str)}


def _attrs(pairs: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": _value(v)} for k, v in pairs.items() if v is not None]


def _span_id(seed: str) -> str:
    return hashlib.sha256(seed.encode()).hexdigest()[:16]


def _ns(seconds: float) -> str:
    return str(int(seconds * 1e9))


def _timestamp(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _text(message: Any) -> str:
    """ATIF messages and observation contents are strings or content parts."""
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        pieces = []
        for part in message:
            if isinstance(part, dict):
                if part.get("type") == "image":
                    source = part.get("source") or {}
                    pieces.append(f"[image: {source.get('path', '')}]")
                else:
                    pieces.append(part.get("text") or "")
            else:
                pieces.append(str(part))
        return "\n".join(pieces)
    return "" if message is None else str(message)


# NOOA's CodeAct task message starts with "## Task: <method name>".
_TASK_HEADER = re.compile(r"^##\s*Task:\s*([A-Za-z_][\w.]*)", re.MULTILINE)


def _method_name(trajectory: dict[str, Any], first_user: str | None) -> str | None:
    """The agent method a trajectory ran; ATIF has no field for it.

    Uses ``extra.method`` when a producer records it, else the NOOA task header
    in the first user message.
    """
    method = (trajectory.get("extra") or {}).get("method")
    if isinstance(method, str) and method:
        return method
    match = _TASK_HEADER.search(first_user or "")
    return match.group(1) if match else None


def _arguments(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _message_attrs(prefix: str, index: int, message: dict[str, Any]) -> dict[str, Any]:
    base = f"{prefix}.{index}.message"
    out: dict[str, Any] = {
        f"{base}.role": message["role"],
        f"{base}.content": message.get("content") or "",
    }
    if message.get("tool_call_id"):
        out[f"{base}.tool_call_id"] = message["tool_call_id"]
    for j, call in enumerate(message.get("tool_calls") or []):
        tool_call = f"{base}.tool_calls.{j}.tool_call"
        out[f"{tool_call}.id"] = call["id"]
        out[f"{tool_call}.function.name"] = call["name"]
        out[f"{tool_call}.function.arguments"] = call["arguments"]
    return out


def _cover_children(spans: list[dict[str, Any]]) -> None:
    """Widen parents so each span's interval contains its children's."""
    by_id = {span["spanId"]: span for span in spans}
    # Children are emitted after their parents, so one reverse pass suffices
    # except for re-parented nested calls; repeat until stable (depth-bounded).
    for _ in range(len(spans)):
        changed = False
        for span in reversed(spans):
            parent = by_id.get(span.get("parentSpanId", ""))
            if parent is None:
                continue
            if int(span["endTimeUnixNano"]) > int(parent["endTimeUnixNano"]):
                parent["endTimeUnixNano"] = span["endTimeUnixNano"]
                changed = True
            if int(span["startTimeUnixNano"]) < int(parent["startTimeUnixNano"]):
                parent["startTimeUnixNano"] = span["startTimeUnixNano"]
                changed = True
        if not changed:
            return


class _Converter:
    def __init__(self, session_id: str, trace_id: str, code_tools: frozenset[str]):
        self.session_id = session_id
        self.trace_id = trace_id
        self.code_tools = code_tools
        self.spans: list[dict[str, Any]] = []

    def _span(self, seed, parent, name, start, end, attributes, *, kind=_INTERNAL, ok=True):
        span = {
            "traceId": self.trace_id,
            "spanId": _span_id(f"{self.session_id}/{seed}"),
            "name": name,
            "kind": kind,
            "startTimeUnixNano": _ns(start),
            "endTimeUnixNano": _ns(max(end, start)),
            "attributes": _attrs({"session.id": self.session_id, **attributes}),
            "status": {"code": _OK if ok else _ERROR},
        }
        if parent:
            span["parentSpanId"] = parent
        self.spans.append(span)
        return span

    def trajectory(
        self,
        trajectory: dict[str, Any],
        seed: str,
        parent: str | None,
        not_before: float,
        *,
        output: str | None = None,
        failed: bool = False,
    ) -> float:
        """Emit the spans for one trajectory; return its end time."""
        agent = trajectory.get("agent") or {}
        agent_name = agent.get("name") or "agent"
        name = trajectory.get("trajectory_id") or agent_name
        steps = trajectory.get("steps") or []
        subagents: dict[str, dict[str, Any]] = {
            str(t.get("trajectory_id") or f"#{i}"): t
            for i, t in enumerate(trajectory.get("subagent_trajectories") or [])
            if isinstance(t, dict)
        }
        used_subagents: set[str] = set()
        times = [t for s in steps if (t := _timestamp(s.get("timestamp"))) is not None]
        start = min(times) if times else not_before
        clock = start
        root = self._span(seed, parent, f"{agent_name}.{name}", start, start, {})

        context: list[dict[str, Any]] = []  # system/user steps + top-level turns
        histories: dict[str, list[dict[str, Any]]] = {}  # nested calls: own turns
        first_user = None
        final = ""
        top_gid: str | None = None
        top_method: str | None = None
        counts: dict[str | None, int] = {}
        methods: dict[str, dict[str, Any]] = {}  # generation_id -> method span
        nested: list[tuple[dict[str, Any], str, float]] = []  # (span, parent gid, start)
        calls: dict[str, list[tuple[float, dict[str, Any], list[dict[str, Any]]]]] = {}
        for index, step in enumerate(steps):
            source = step.get("source")
            if source in ("system", "user"):
                content = _text(step.get("message"))
                context.append({"role": source, "content": content})
                if source == "user" and first_user is None:
                    first_user = content
                continue
            if source != "agent":
                continue

            extra = step.get("extra") or {}
            gid = (
                extra.get("generation_id") if isinstance(extra.get("generation_id"), str) else None
            )
            pgid = extra.get("parent_generation_id")
            pgid = pgid if isinstance(pgid, str) else None
            step_method = extra.get("method_name")
            step_method = step_method if isinstance(step_method, str) else None
            if top_gid is None and pgid is None:
                top_gid, top_method = gid, step_method
            is_top = gid is None or gid == top_gid or pgid is None
            history = context if is_top else histories.setdefault(gid or "", [])
            counts[gid] = generation = counts.get(gid, 0) + 1
            metrics = step.get("metrics") or {}
            step_start = _timestamp(step.get("timestamp"))
            step_start = max(clock, not_before) if step_start is None else step_start
            llm_end = step_start + float(extra.get("duration_ms") or 1.0) / 1000.0
            message = {
                "role": "assistant",
                "content": _text(step.get("message")),
                "tool_calls": [
                    {
                        "id": call.get("tool_call_id") or f"call_{k}",
                        "name": call.get("function_name") or "tool",
                        "arguments": _arguments(call.get("arguments") or {}),
                    }
                    for k, call in enumerate(step.get("tool_calls") or [])
                ],
            }
            generation_id = _span_id(f"{self.session_id}/{seed}/gen/{index}")
            owner = root
            if gid is not None and gid != top_gid:
                if gid not in methods:
                    # Parented under root for now; placed under the calling code
                    # execution once all steps are known (inner steps can be
                    # sealed before the outer step that made the call).
                    methods[gid] = self._span(
                        f"{seed}/method/{gid}",
                        root["spanId"],
                        f"{agent_name}.{step_method or 'method'}",
                        step_start,
                        step_start,
                        {
                            "openinference.span.kind": "AGENT",
                            "nooa.viewer.plugin": "method",
                            "agent.name": agent_name,
                            "agent.method": step_method,
                            "agent.call_id": _span_id(f"{self.session_id}/{seed}/method/{gid}"),
                            "generation.id": gid,
                            "generation.parent_id": pgid,
                        },
                    )
                    if pgid is not None:
                        nested.append((methods[gid], pgid, step_start))
                owner = methods[gid]
            gen = self._span(
                f"{seed}/gen/{index}",
                owner["spanId"],
                f"generation {generation}",
                step_start,
                llm_end,
                {
                    "openinference.span.kind": "CHAIN",
                    "nooa.viewer.plugin": "generation",
                    "agent.name": agent_name,
                    "generation.id": generation_id,
                },
            )

            llm = {
                "openinference.span.kind": "LLM",
                "llm.model_name": step.get("model_name") or agent.get("model_name"),
                "llm.token_count.prompt": metrics.get("prompt_tokens"),
                "llm.token_count.completion": metrics.get("completion_tokens"),
                "llm.token_count.prompt_details.cache_read": metrics.get("cached_tokens"),
                "llm.cost.total": metrics.get("cost_usd"),
                "llm.reasoning_content": step.get("reasoning_content"),
                "llm.finish_reason": extra.get("finish_reason"),
                "atif.step_id": step.get("step_id"),
                "atif.context_rewritten": extra.get("context_rewritten"),
                "output.value": message["content"],
            }
            prompt, completion = metrics.get("prompt_tokens"), metrics.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                llm["llm.token_count.total"] = prompt + completion
            for i, previous in enumerate(history):
                llm.update(_message_attrs("llm.input_messages", i, previous))
            llm.update(_message_attrs("llm.output_messages", 0, message))
            self._span(
                f"{seed}/llm/{index}",
                gen["spanId"],
                "llm_call",
                step_start,
                llm_end,
                llm,
                kind=_CLIENT,
            )

            results = {
                r.get("source_call_id"): r
                for r in (step.get("observation") or {}).get("results") or []
                if isinstance(r, dict)
            }
            tool_clock = llm_end
            tool_messages = []
            tool_spans: list[dict[str, Any]] = []
            for k, call in enumerate(message["tool_calls"]):
                result = results.get(call["id"]) or {}
                result_extra = result.get("extra") or {}
                output_text = _text(result.get("content"))
                tool_start = tool_clock
                tool_end = tool_start + float(result_extra.get("duration_ms") or 1.0) / 1000.0
                is_code = call["name"] in self.code_tools
                tool_ok = not result_extra.get("error_type") and result_extra.get("status") in (
                    None,
                    "completed",
                    "complete",
                    "ok",
                )
                tool = self._span(
                    f"{seed}/tool/{index}/{k}",
                    gen["spanId"],
                    "code_execution" if is_code else f"tool:{call['name']}",
                    tool_start,
                    tool_end,
                    {
                        "openinference.span.kind": "TOOL",
                        "nooa.viewer.plugin": "code_execution" if is_code else "tool_execution",
                        "tool.name": call["name"],
                        "agent.name": agent_name,
                        "execution.id": call["id"],
                        "generation.id": generation_id,
                        "input.value": call["arguments"],
                        "input.mime_type": "application/json",
                        "output.value": output_text,
                        "error.type": result_extra.get("error_type"),
                    },
                    ok=tool_ok,
                )
                if is_code:
                    tool_spans.append(tool)
                for ref in result.get("subagent_trajectory_ref") or []:
                    sub_id = (ref or {}).get("trajectory_id")
                    if (
                        isinstance(sub_id, str)
                        and sub_id in subagents
                        and sub_id not in used_subagents
                    ):
                        used_subagents.add(sub_id)
                        sub_end = self.trajectory(
                            subagents[sub_id], f"{seed}/sub/{sub_id}", tool["spanId"], tool_start
                        )
                        tool_end = max(tool_end, sub_end)
                        tool["endTimeUnixNano"] = _ns(tool_end)
                tool_clock = tool_end
                tool_messages.append(
                    {"role": "tool", "content": output_text, "tool_call_id": call["id"]}
                )
            gen["endTimeUnixNano"] = _ns(tool_clock)
            history.append(message)
            history.extend(tool_messages)
            if gid is not None:
                calls.setdefault(gid, []).append((step_start, gen, tool_spans))
            if is_top:
                final = message["content"] or final
            clock = tool_clock + 0.001

        # Place each nested method call in the parent's latest code execution
        # (else generation, else method) that started before it.
        for span, pgid, call_start in nested:
            candidates = [c for c in calls.get(pgid, []) if c[0] <= call_start]
            if candidates:
                _, parent_gen, parent_tools = max(candidates, key=lambda c: c[0])
                target = parent_tools[-1] if parent_tools else parent_gen
                span["parentSpanId"] = target["spanId"]
            elif pgid in methods:
                span["parentSpanId"] = methods[pgid]["spanId"]

        # Embedded subagents no observation referenced still belong to the trace.
        for sub_id, sub in subagents.items():
            if sub_id not in used_subagents:
                clock = self.trajectory(sub, f"{seed}/sub/{sub_id}", root["spanId"], clock) + 0.001

        end = max(clock, start + 0.001)
        method = top_method or _method_name(trajectory, first_user) or name
        root["name"] = f"{agent_name}.{method}"
        root["endTimeUnixNano"] = _ns(end)
        root["attributes"] = _attrs(
            {
                "session.id": self.session_id,
                "openinference.span.kind": "AGENT",
                "nooa.viewer.plugin": "method",
                "agent.name": agent_name,
                "agent.method": method,
                "agent.call_id": root["spanId"],
                "input.value": first_user or "",
                "output.value": final if output is None else output,
                "atif.schema_version": trajectory.get("schema_version"),
                "atif.trajectory_id": trajectory.get("trajectory_id"),
                "atif.final_metrics": trajectory.get("final_metrics"),
                "atif.notes": trajectory.get("notes"),
                "atif.extra": trajectory.get("extra"),
            }
        )
        root["status"] = {"code": _ERROR if failed else _OK}
        return end


def trajectory_to_otlp(
    trajectory: Any,
    session_id: str,
    *,
    experiment: str | None = None,
    resource: dict[str, Any] | None = None,
    code_tools: Iterable[str] = DEFAULT_CODE_TOOLS,
    output: str | None = None,
    failed: bool = False,
) -> dict[str, Any]:
    """Return one OTLP ``{"resourceSpans": [...]}`` body for an ATIF trajectory.

    Args:
        trajectory: A :class:`nooa.atif.Trajectory` or its JSON dict.
        session_id: Viewer session ID, written to ``session.id``.
        experiment: Optional experiment name for the viewer's evaluation view.
        resource: Extra resource attributes, e.g. ``{"batch_id": ...}``.
        code_tools: Tool names whose calls render as code execution.
        output: Overrides the root span output (default: final agent message).
        failed: Marks the root span as an error, e.g. for a failed task.
    """
    if hasattr(trajectory, "model_dump"):
        trajectory = trajectory.model_dump(mode="json", exclude_none=True)
    if not isinstance(trajectory, dict):
        raise TypeError("trajectory must be an ATIF Trajectory or dict")
    trace_id = hashlib.sha256(f"{session_id}/trace".encode()).hexdigest()[:32]
    converter = _Converter(session_id, trace_id, frozenset(code_tools))
    converter.trajectory(trajectory, "root", None, 0.0, output=output, failed=failed)
    _cover_children(converter.spans)

    agent = trajectory.get("agent") or {}
    attributes = {
        "session.id": session_id,
        "service.name": agent.get("name") or "agent",
        "agent.version": agent.get("version"),
        "llm.model_name": agent.get("model_name"),
        "atif.session_id": trajectory.get("session_id"),
        "atif.trajectory_id": trajectory.get("trajectory_id"),
        "experiment": experiment,
        **(resource or {}),
    }
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs(attributes)},
                "scopeSpans": [{"scope": _SCOPE, "spans": converter.spans}],
            }
        ]
    }
