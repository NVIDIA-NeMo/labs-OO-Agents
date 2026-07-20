# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-time span->event exporter that populates the reference TUI's per-game panels.

The self-contained multi-game viewer (``viewer.py``)
builds every per-game panel from ``arc_agi_3.viewer.update_state(event, state)``, which
tails ``agent_logs/<team>/team_leader/events.jsonl`` and dispatches on ``event["event"]``.
Our harness recorder (``recorder.py``) only writes ``solver_start`` / ``env_step`` /
``agent_turn``, so the REPL (``$``), rounds, reasoning/prompt/content and token panels
stay empty.

This exporter is a *sibling* of ``viewer_trace_exporter.ViewerMessageExporter`` (added to
the same OpenTelemetry tracing pipeline). It receives live ``ReadableSpan`` objects as
they complete and appends the reference-schema events the panels need, derived from the
agent's trace spans, to the SAME ``events.jsonl`` the recorder uses (same line format:
``{"timestamp": <iso>, "unix_time_s": <float>, "event": <name>, "agent_id": ..., **kw}``).
It never re-emits ``env_step`` / ``solver_start`` (those come from the harness recorder).

Span -> event mapping (viewer.py handler line numbers are as of this writing):

  LLM spans  (``aresponses`` / ``responses`` / ``acompletion`` / ``completion``, with
              ``llm.input_messages.*`` present)
    -> ``llm_call``        consumed at viewer.py ~2238  (drives total_llm_calls,
                           input_tokens, output_tokens, current_depth_round and, via
                           ``_read_assistant_file(state, depth_round, step)`` @~1908, the
                           Reasoning panel content -- which reads the ``messages/`` .md
                           files the sibling ViewerMessageExporter writes; so our
                           ``step`` + ``depth_round`` numbering is kept in lockstep with
                           the sibling's per-step round counter).
    -> ``system_prompt``   consumed at viewer.py ~2363  (once, on the first LLM span that
                           carries a system message).
    -> ``user_message``    consumed at viewer.py ~2367  (once, first user content).
      Token attrs: ``llm.token_count.prompt`` / ``.completion`` /
      ``.completion_details.reasoning``.  Messages via ``_indexed(at, "input"/"output")``
      over ``llm.<kind>_messages.*`` (identical helper to the sibling exporter).

  ``code_execution`` spans
    -> ``repl_execute``    consumed at viewer.py ~2282  (drives the ``$``/REPL Stats
                           panel: repl_total, repl_success, repl_function_counts,
                           repl_last_error). code from ``input.value`` {"code": ...} and
                           ``code.length``; stdout/stderr from ``output.value``
                           {"stdout","stderr","returned_value"} -> success = empty stderr.

  ``method.handle`` spans  (one agent turn)
    -> ``round_complete``  consumed at viewer.py ~2450  (one per agent reasoning round in
                           the turn -> total_rounds / total_round_time / round_timing).
    -> ``step_complete``   consumed at viewer.py ~2345  (once per turn -> completed_steps /
                           total_step_time / status "★ answer" line).
      A "round" is an agent LLM span (``aresponses`` with ``llm.tools.*`` and a system
      message) whose start falls inside the handle's [start, end] window; memory/embedding
      LLM calls (no tools) are excluded from the round count. Timings are derived from the
      buffered child span durations (children always end -- and export -- before the
      parent handle span, so they are already buffered by the time a handle arrives).

Every emitted event carries a ``step`` mapped to the env-step in progress at the span's
start time, using the harness-written ``env_step`` timestamps exactly like
``viewer_trace_exporter._step_for``.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

# LLM span names -- identical set to viewer_trace_exporter.ViewerMessageExporter so the
# per-step round counter stays in lockstep with the messages/*.md files it writes.
LLM_SPANS = {"aresponses", "responses", "acompletion", "completion"}
CODE_SPAN = "code_execution"
HANDLE_SPAN = "method.handle"

# Bare function names worth surfacing in the REPL "Top functions" line, in addition to
# every ``self.<method>(`` call found in the executed code.
_INTERESTING_CALLS = {
    "reasoning",
    "message",
    "print",
    "pprint",
    "doc",
    "submit_actions",
    "return_result",
}
_CALL_RE = re.compile(r"(?:self\.)?([A-Za-z_]\w*)\s*\(")


def _pricing_key(model: str) -> str:
    """Map a litellm model id (e.g. ``openai/openai/gpt-5.5``) to a key that
    ``arc_agi_3.llm_configs.get_pricing`` recognises, so the Status panel's ``$``
    cost line resolves a real price. model_tokens is keyed by the llm_call
    ``agent_id``, and get_pricing knows the short names ``opus`` / ``gpt5.5`` /
    ``flash`` (it strips an ``arc3-`` prefix and trailing ``-<suffix>`` rungs, but
    not provider ``/`` prefixes or the ``gpt-5.5``↔``gpt5.5`` hyphen)."""
    m = (model or "").lower()
    if "gpt-5.5" in m or "gpt5.5" in m:
        return "gpt5.5"
    if "opus" in m:
        return "opus"
    if "flash" in m:
        return "flash"
    # Fall back to the last path segment (best effort; may price at 0).
    return m.rsplit("/", 1)[-1] if m else "unknown"


def _indexed(at: dict, kind: str) -> list[tuple[str, str]]:
    """(role, content[+tool_call]) for llm.<kind>_messages.*, in index order.

    Identical to viewer_trace_exporter._indexed so message extraction matches the
    sibling exporter exactly.
    """
    prefix = f"llm.{kind}_messages."
    grp: dict[int, dict] = {}
    for k, v in at.items():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        i = int(rest.split(".", 1)[0])
        grp.setdefault(i, {})[rest.split(".", 1)[1]] = v
    out = []
    for i in sorted(grp):
        m = grp[i]
        role = m.get("message.role", "?")
        body = str(m.get("message.content", "") or "")
        # Some providers put assistant text under contents.<j>.message_content.text
        if not body:
            texts = [str(v) for k, v in m.items() if k.endswith("message_content.text")]
            body = "".join(texts)
        tcs = [
            (("name" if k.endswith("function.name") else "args"), str(v))
            for k, v in m.items()
            if "tool_calls" in k
            and (k.endswith("function.name") or k.endswith("function.arguments"))
        ]
        if tcs:
            body = (body + "\n[tool_call] " + " ".join(f"{a}={b}" for a, b in tcs)).strip()
        out.append((role, body))
    return out


def _count_output(at: dict) -> tuple[int, int, bool]:
    """(num_tool_calls, num_python_blocks, has_text_block) for an LLM span's output."""
    num_tool_calls = 0
    num_python_blocks = 0
    has_text = False
    for k, v in at.items():
        if not k.startswith("llm.output_messages."):
            continue
        if k.endswith("tool_call.function.name"):
            num_tool_calls += 1
            if str(v) == "execute_python":
                num_python_blocks += 1
        elif k.endswith("message.content") or k.endswith("message_content.text"):
            if str(v or "").strip():
                has_text = True
    return num_tool_calls, num_python_blocks, has_text


def _functions_used(code: str) -> list[str]:
    """Best-effort list of function/method names invoked in a REPL code block."""
    if not code:
        return []
    seen: list[str] = []
    for m in _CALL_RE.finditer(code):
        name = m.group(1)
        # Keep ``self.<name>(`` method calls and a small set of interesting bare calls.
        is_self = code[: m.start()].endswith("self.")
        if (is_self or name in _INTERESTING_CALLS) and name not in seen:
            seen.append(name)
    return seen


class ViewerEventExporter(SpanExporter):
    """Appends reference-schema events derived from live trace spans to events.jsonl."""

    def __init__(self, run_dir: str | Path, team: str = "nemo", role: str = "team_leader"):
        self.events_path = Path(run_dir) / "agent_logs" / team / role / "events.jsonl"
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._agent_id = role
        # Per-step counters (mirror the sibling's ViewerMessageExporter._round_by_step
        # so llm_call.step/depth_round line up with the messages/*.md file names).
        self._round_by_step: dict[int, int] = {}
        self._block_by_step: dict[int, int] = {}
        # Guard against processing the same span twice across overlapping batches.
        self._seen: set[tuple[int, int]] = set()
        # Buffered child-span timings, needed to derive round_complete/step_complete from
        # a method.handle span (children export before their parent handle).
        # Each: {"start": ns, "end": ns, "agent_round": bool}
        self._llm_records: list[dict] = []
        # Each: {"start": ns, "end": ns}
        self._code_records: list[dict] = []
        # Emit system_prompt/user_message only once, on the first LLM span with a sys msg.
        self._prompt_emitted = False

    # ------------------------------------------------------------- helpers

    def _span_key(self, sp) -> tuple[int, int] | None:
        ctx = getattr(sp, "context", None)
        if ctx is None:
            return None
        try:
            return (ctx.trace_id, ctx.span_id)
        except AttributeError:
            return None

    def _step_for(self, call_start_s: float) -> int:
        """env-step in progress at ``call_start_s`` (mirrors viewer_trace_exporter)."""
        cur = 0
        if not self.events_path.exists():
            return cur
        try:
            for line in self.events_path.read_text().splitlines():
                e = json.loads(line)
                if e.get("event") == "env_step":
                    t = float(e.get("unix_time_s", 0))
                    if t <= call_start_s:
                        cur = int(e.get("step", 0))
                    else:
                        break
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return cur

    def _emit(self, out, sort_ns: int, event: str, **kwargs) -> None:
        """Queue an event line in the recorder's exact format, keyed for time ordering."""
        unix = sort_ns / 1e9
        ts = datetime.fromtimestamp(unix, tz=UTC).isoformat()
        line = {
            "timestamp": ts,
            "unix_time_s": unix,
            "event": event,
            "agent_id": self._agent_id,
            **kwargs,
        }
        out.append((sort_ns, line))

    # --------------------------------------------------------------- export

    def export(self, spans) -> SpanExportResult:
        try:
            pending: list[tuple[int, dict]] = []
            handles: list = []

            # Phase 1: LLM + code spans (buffer child timings, emit their events).
            for sp in spans:
                key = self._span_key(sp)
                if key is not None:
                    if key in self._seen:
                        continue
                    self._seen.add(key)
                name = getattr(sp, "name", "")
                if name == HANDLE_SPAN:
                    handles.append(sp)  # deferred to phase 2 (needs child timings)
                    continue
                at = dict(getattr(sp, "attributes", None) or {})
                start_ns = int(getattr(sp, "start_time", 0) or 0)
                end_ns = int(getattr(sp, "end_time", 0) or start_ns)
                if name in LLM_SPANS:
                    self._handle_llm_span(pending, at, start_ns, end_ns)
                elif name == CODE_SPAN:
                    self._handle_code_span(pending, at, start_ns, end_ns)

            # Phase 2: handle spans (rounds derived from buffered child timings).
            for sp in handles:
                at = dict(getattr(sp, "attributes", None) or {})
                start_ns = int(getattr(sp, "start_time", 0) or 0)
                end_ns = int(getattr(sp, "end_time", 0) or start_ns)
                self._handle_turn_span(pending, at, start_ns, end_ns)

            # Append in monotonic time order (stable: preserves insertion for ties).
            pending.sort(key=lambda x: x[0])
            if pending:
                with self.events_path.open("a") as f:
                    for _, line in pending:
                        f.write(json.dumps(line) + "\n")
        except Exception:  # never raise out of export()
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    # ------------------------------------------------------- span handlers

    def _handle_llm_span(self, pending, at: dict, start_ns: int, end_ns: int) -> None:
        # Match the sibling's filter so the per-step round counter stays in lockstep.
        if not any(k.startswith("llm.input_messages") for k in at):
            return
        step = self._step_for(start_ns / 1e9)
        dr = self._round_by_step.get(step, 0)
        self._round_by_step[step] = dr + 1

        # An agent reasoning round = an LLM call with tool schemas + a system message
        # (memory/consolidation LLM calls have neither).
        has_tools = any(k.startswith("llm.tools.") for k in at)
        inputs = _indexed(at, "input")
        has_system = any(r == "system" for r, _ in inputs)
        self._llm_records.append(
            {"start": start_ns, "end": end_ns, "agent_round": has_tools and has_system}
        )

        inp = int(at.get("llm.token_count.prompt", 0) or 0)
        out = int(at.get("llm.token_count.completion", 0) or 0)
        n_tool, n_py, has_text = _count_output(at)
        latency_s = max((end_ns - start_ns) / 1e9, 0.0)

        # system_prompt + user_message once, on the first LLM span with a system message.
        if not self._prompt_emitted and has_system:
            self._prompt_emitted = True
            sys_content = next((c for r, c in inputs if r == "system"), "")
            usr_content = next((c for r, c in inputs if r == "user"), "")
            self._emit(
                pending, start_ns, "system_prompt", content_length=len(sys_content), step=step
            )
            self._emit(
                pending,
                start_ns,
                "user_message",
                content_length=len(usr_content),
                content_tokens=len(usr_content) // 4,
                depth_round=0,
                step=step,
            )

        model = str(at.get("llm.model_name", "") or "")
        cache_read = int(at.get("llm.token_count.prompt_details.cache_read", 0) or 0)
        self._emit(
            pending,
            start_ns,
            "llm_call",
            # agent_id keys state.model_tokens, which the Status ``$`` line prices
            # via get_pricing(); use a pricing-resolvable model key, not the role.
            agent_id=_pricing_key(model),
            depth_round=dr,
            step=step,
            latency_s=round(latency_s, 3),
            input_tokens=inp,
            output_tokens=out,
            cache_read_tokens=cache_read,
            # This gateway has no cache-write concept; the field must still be
            # an int — the reference viewer's update_state hard-indexes it.
            cache_creation_tokens=0,
            reasoning_tokens=int(at.get("llm.token_count.completion_details.reasoning", 0) or 0),
            num_python_blocks=n_py,
            num_tool_calls=n_tool,
            has_text_block=has_text,
            model=model,
            # model_uri keys/prices model_tokens in the reference multi-game TUI
            # (a required llm_call field there); pricing-resolvable key.
            model_uri=_pricing_key(model),
        )

    def _handle_code_span(self, pending, at: dict, start_ns: int, end_ns: int) -> None:
        step = self._step_for(start_ns / 1e9)
        bi = self._block_by_step.get(step, 0)
        self._block_by_step[step] = bi + 1
        self._code_records.append({"start": start_ns, "end": end_ns})

        code = ""
        iv = at.get("input.value")
        if isinstance(iv, str):
            try:
                code = str(json.loads(iv).get("code", "") or "")
            except (json.JSONDecodeError, AttributeError, TypeError):
                code = ""
        code_length = int(at.get("code.length", len(code)) or len(code))

        stdout, stderr = "", ""
        ov = at.get("output.value")
        if isinstance(ov, str):
            try:
                d = json.loads(ov)
                if isinstance(d, dict):
                    stdout = str(d.get("stdout", "") or "")
                    stderr = str(d.get("stderr", "") or "")
            except json.JSONDecodeError:
                pass
        success = not stderr.strip()
        latency_s = max((end_ns - start_ns) / 1e9, 0.0)

        self._emit(
            pending,
            start_ns,
            "repl_execute",
            block_index=bi,
            step=step,
            code_length=code_length,
            stdout_length=len(stdout),
            success=success,
            error=(stderr[:500] if not success else None),
            latency_s=round(latency_s, 3),
            functions_used=_functions_used(code),
        )

    def _handle_turn_span(self, pending, at: dict, start_ns: int, end_ns: int) -> None:
        step = self._step_for(start_ns / 1e9)
        # Agent reasoning rounds within this turn's window, in start order.
        rounds = sorted(
            (r for r in self._llm_records if r["agent_round"] and start_ns <= r["start"] < end_ns),
            key=lambda r: r["start"],
        )
        codes = sorted(
            (c for c in self._code_records if start_ns <= c["start"] < end_ns),
            key=lambda c: c["start"],
        )
        for i, r in enumerate(rounds):
            llm_s = max((r["end"] - r["start"]) / 1e9, 0.0)
            nxt = rounds[i + 1]["start"] if i + 1 < len(rounds) else end_ns
            # REPL time attributed to this round = code executed after the LLM call,
            # before the next round starts.
            repl_s = sum(
                max((c["end"] - c["start"]) / 1e9, 0.0)
                for c in codes
                if r["end"] <= c["start"] < nxt
            )
            total_s = llm_s + repl_s
            self._emit(
                pending,
                r["start"],
                "round_complete",
                depth_round=i,
                step=step,
                compaction_s=0.0,
                llm_s=round(llm_s, 3),
                tools_s=0.0,
                repl_s=round(repl_s, 3),
                round_total_s=round(total_s, 3),
            )

        answer = str(at.get("output.value", "") or "")
        wall_time_s = max((end_ns - start_ns) / 1e9, 0.0)
        self._emit(
            pending,
            end_ns,
            "step_complete",
            step=step,
            wall_time_s=round(wall_time_s, 3),
            depth_rounds_used=len(rounds),
            answer=answer,
            timed_out=False,
            exhausted_depth=False,
        )

        # Turns are sequential and non-overlapping: buffered records at or before this
        # turn's end are now consumed -- drop them to bound memory.
        self._llm_records = [r for r in self._llm_records if r["start"] >= end_ns]
        self._code_records = [c for c in self._code_records if c["start"] >= end_ns]

    # ------------------------------------------------------------- lifecycle

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
