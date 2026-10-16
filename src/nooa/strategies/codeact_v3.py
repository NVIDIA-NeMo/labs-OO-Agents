# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Experimental CodeAct V3 strategy: named markdown snippets plus ``run_snippet``.

The model writes Python in its normal reply as fenced blocks whose header names
the snippet, then calls the single provider tool ``run_snippet(name)``:

    ```python scan-files
    print(sorted(p.name for p in Path(".").iterdir()))
    ```

    run_snippet(name="scan-files")

Code therefore stays in the assistant text instead of a JSON-escaped tool
argument, and only snippets the model explicitly runs are executed. Everything
else (session state, ``return_result`` inside code, context blocks) is
inherited from :class:`CodeActV2`.
"""

import json
import re
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nooa.context_blocks import ResultStatus, ToolResult
from nooa.decorators import strategy
from nooa.events import Error, PythonOutput
from nooa.runtime.harness_metrics import get_harness_metrics
from nooa.strategies.base import RuntimeServices
from nooa.strategies.codeact import TextOnlyResponseAction, TextOnlyResponseContext
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.strategies.loop_guard import action_fingerprint
from nooa.strategies.template import TemplateStrategy
from nooa.unifiedllm import Tool

if TYPE_CHECKING:
    from nooa.strategies.codeact import CodeActSession, _ToolCallsResult
    from nooa.strategies.current_call import CurrentCall

RUN_SNIPPET = "run_snippet"
SNIPPET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_PYTHON_LANGUAGES = frozenset({"python", "py", "python3"})
_OPEN_FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
_TITLE_ATTR = re.compile(r"""^(?:title|name|file|filename)\s*=\s*(?:"([^"]*)"|'([^']*)'|(\S+))""")


@dataclass(frozen=True)
class Snippet:
    """One fenced Python block from an assistant reply."""

    name: str | None
    code: str
    header: str
    variant: str | None = None
    closed: bool = True
    invalid_name: str | None = None


def _parse_header(info: str) -> tuple[str, str | None, str | None]:
    """Return ``(language, name, variant)`` for a fence info string.

    The taught form is ``python <name>``; ``variant`` is ``None`` for it.
    Common documentation conventions are accepted and labelled so harness
    metrics can show whether models follow the taught form.
    """
    info = info.strip()
    if info.startswith("{") and info.endswith("}"):
        # Pandoc ``{.python #name}`` and knitr ``{python name, opt=1}``.
        tokens = [t for t in re.split(r"[\s,]+", info[1:-1]) if t]
        if not tokens:
            return "", None, None
        language = tokens[0].lstrip(".")
        for token in tokens[1:]:
            if token.startswith("#"):
                return language, token[1:], "pandoc"
        rest = [t for t in tokens[1:] if "=" not in t and not t.startswith(".")]
        return language, (rest[0] if rest else None), "knitr"

    first, _, rest = info.partition(" ")
    rest = rest.strip()
    if ":" in first:
        language, _, name = first.partition(":")
        return language, name or None, "colon"
    language = first
    if not rest:
        return language, None, None
    if rest.startswith("[") and "]" in rest:
        return language, rest[1 : rest.index("]")].strip() or None, "bracket"
    match = _TITLE_ATTR.match(rest)
    if match:
        name = next((g for g in match.groups() if g is not None), "")
        return language, name or None, "title"
    return language, rest.split()[0], None


def parse_snippets(text: str) -> list[Snippet]:
    """Extract fenced Python blocks from markdown text, named or not.

    Fences follow CommonMark: three or more backticks or tildes, indented by at
    most three spaces, closed by a fence of the same character that is at least
    as long. Non-Python blocks are ignored.
    """
    snippets: list[Snippet] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _OPEN_FENCE.match(lines[i])
        if not match or (match.group(2)[0] == "`" and "`" in match.group(3)):
            i += 1
            continue
        indent, fence, info = len(match.group(1)), match.group(2), match.group(3)
        close = re.compile(rf"^ {{0,3}}{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
        body: list[str] = []
        i += 1
        closed = False
        while i < len(lines):
            if close.match(lines[i]):
                closed = True
                i += 1
                break
            line = lines[i]
            strip = len(line) - len(line.lstrip(" "))
            body.append(line[min(indent, strip) :])
            i += 1
        language, name, variant = _parse_header(info)
        if language.lower() not in _PYTHON_LANGUAGES:
            continue
        invalid_name = None
        if name is not None and not SNIPPET_NAME_PATTERN.match(name):
            invalid_name, name = name, None
        snippets.append(
            Snippet(
                name=name,
                code="\n".join(body),
                header=info.strip(),
                variant=variant if name is not None else None,
                closed=closed,
                invalid_name=invalid_name,
            )
        )
    return snippets


def resolve_snippet(snippets: list[Snippet], name: Any) -> tuple[Snippet | None, str | None]:
    """Return ``(snippet, None)`` or ``(None, model-facing error)`` for *name*."""
    example = f'{RUN_SNIPPET}(name="scan-files")'
    if not isinstance(name, str) or not name:
        return None, (
            f"`{RUN_SNIPPET}` takes one string argument `name`, e.g. {example}, naming a "
            "```python <name>``` block written in the same reply."
        )
    matches = [s for s in snippets if s.name == name]
    if len(matches) > 1:
        return None, (
            f"Snippet name `{name}` appears {len(matches)} times in this reply. "
            "Give each snippet in a reply a unique name."
        )
    if matches:
        if not matches[0].closed:
            return None, f"Snippet `{name}` is missing its closing ``` fence."
        return matches[0], None

    named = sorted({s.name for s in snippets if s.name})
    parts = [f"Snippet `{name}` was not found in this reply."]
    if named:
        parts.append(f"Named snippets in this reply: {', '.join(f'`{n}`' for n in named)}.")
    else:
        parts.append("This reply has no named Python snippets.")
    invalid = sorted({s.invalid_name for s in snippets if s.invalid_name})
    if invalid:
        parts.append(
            f"Invalid snippet names: {', '.join(f'`{n}`' for n in invalid)}; names use "
            "letters, digits, `-`, `_` or `.` (at most 64 characters)."
        )
    unnamed = sum(1 for s in snippets if s.name is None and not s.invalid_name)
    if unnamed:
        parts.append(f"{unnamed} unnamed ```python block(s) were treated as plain text.")
    parts.append(
        f"Write the code in the same reply as ```python {name}``` ... ``` and call "
        f'{RUN_SNIPPET}(name="{name}"). Snippets from earlier replies cannot be run; '
        "write the code again to rerun it."
    )
    return None, " ".join(parts)


# Snippets parsed from the assistant reply whose tool calls are being processed.
# A context variable keeps concurrent and nested generations independent.
_TURN_SNIPPETS: ContextVar[list[Snippet] | None] = ContextVar("codeact_v3_snippets", default=None)


class CodeActV3(CodeActV2):
    """Experimental CodeAct variant where code lives in named markdown snippets.

    The model's only provider tool is ``run_snippet(name)``. It runs the
    ```` ```python <name> ```` block written in the same reply, in the same
    persistent Python session CodeActV2 uses. Unnamed blocks and other
    languages are plain text. ``return_result`` inside a snippet finishes.
    """

    _CONTEXT_SCOPE_LINE = "# Already in scope inside every snippet; state persists across snippets."

    @property
    def name(self) -> str:
        return "CODEACT_V3"

    def _python_tool_name(self) -> str:
        return RUN_SNIPPET

    def _available_tool_names(self) -> str:
        return RUN_SNIPPET

    def _python_cell_reference(self) -> str:
        return "a named ```python snippet"

    def _tool_arguments_hint(self) -> str:
        return f'call {RUN_SNIPPET} with arguments like {{"name": "scan-files"}}.'

    def _prefill_tool_arguments(self, code: str, prefill_type: str) -> dict[str, Any]:
        del code
        return {"name": prefill_type.replace("_", "-")}

    def _build_execute_python_tool(self) -> Any:
        """Build the sole provider tool, including the snippet contract."""

        def run_snippet(name: str) -> str:
            """Run a named Python snippet written in this same reply.

            Args:
                name: Name from the snippet's opening fence, e.g. ``scan-files``
                    for a block opened with ```python scan-files.
            """
            return ""

        description = f"""Run one named Python snippet from this same reply in the current \
method call's Python session.

Write code in your reply as a fenced block whose opening fence names it, then call
this tool with that name. For example, write

```python scan-files
files = sorted(p.name for p in Path(".").iterdir())
print(len(files), files[:20])
```

and call {RUN_SNIPPET}(name="scan-files").

Snippet rules:
- Only ```python blocks with a name run, and only when you call {RUN_SNIPPET}. Unnamed
  blocks and other languages are plain text, so you can still show example code.
- Names use letters, digits, `-`, `_` or `.` (at most 64 characters) and must be
  unique within a reply. A later reply may reuse a name for new code.
- {RUN_SNIPPET} only sees snippets in the same reply. To rerun earlier code, write it
  again.
- To run several snippets, call {RUN_SNIPPET} once per name. They run in order and
  stop at the first error.

Parameters are pre-loaded as locals. Names defined in one snippet remain available in
later snippets of this call; reuse them instead of recreating unchanged values. The
caller controls whether locals survive after the method returns, so follow the agent's
application-specific state guidance. {self._always_available_text()} Use
`await` directly. This is your only provider tool: call it on every turn because
plain-text replies do not execute work or finish the task.

To finish, run a snippet that calls `return_result(value)`. It immediately submits a
value matching the method's annotated return type. A bare final expression does not
finish the task. In particular, a trailing string is not shown; use `print(text)`
when you want to inspect prose before submitting it.

Use Python for arithmetic, iteration, transforms, and batches rather than manually
constructing large outputs. Define reusable helpers at the top of a snippet. Existing
methods on `self` may be called with `await` when async.

If `self` exposes delegation, inspect its documentation with `doc(self.delegate)`.
Use bounded objectives when an independent context helps; run independent work
with `asyncio.gather` and inspect each result. For single-shot extraction or
classification, use a documented `@strategy(PredictStrategy())` helper.

Restrictions (will throw):
{self._restrictions_text()}
"""
        return Tool(
            name=RUN_SNIPPET,
            description=description,
            callable=run_snippet,
            parameters_model=None,
        )

    async def _process_tool_calls(
        self,
        tool_calls: list[Any],
        runtime: RuntimeServices,
        builtins: dict[str, Any],
        session: "CodeActSession",
        call: "CurrentCall",
        return_type: Any,
        event_id: str,
        preserve_llm_response: bool = False,
    ) -> "_ToolCallsResult":
        """Parse this reply's snippets once, then process its tool calls."""
        response = runtime.event_manager.get(event_id) if event_id else None
        content = getattr(response, "content", "") or ""
        token = _TURN_SNIPPETS.set(parse_snippets(content))
        try:
            return await super()._process_tool_calls(
                tool_calls,
                runtime,
                builtins,
                session,
                call,
                return_type,
                event_id,
                preserve_llm_response=preserve_llm_response,
            )
        finally:
            _TURN_SNIPPETS.reset(token)

    def _loop_guard_action(self, tool_call: Any) -> tuple[str, Any]:
        """Fingerprint a run by its snippet code, so reused names are not repeats."""
        action, parsed = super()._loop_guard_action(tool_call)
        if tool_call.name == RUN_SNIPPET and isinstance(parsed, dict):
            snippet, _ = resolve_snippet(_TURN_SNIPPETS.get() or [], parsed.get("name"))
            if snippet is not None:
                action = action_fingerprint(RUN_SNIPPET, {"code": snippet.code}, RUN_SNIPPET)
        return action, parsed

    async def _handle_execute_python(
        self,
        runtime: RuntimeServices,
        tool_call: Any,
        args: dict[str, Any],
        builtins: dict[str, Any],
        session: "CodeActSession",
        call: Any,
        return_type: Any,
        tool_call_event_id: str,
    ) -> Any | None:
        """Resolve ``run_snippet(name)`` to code, then run it as a V2 cell.

        Translated tool calls (``translate_tool_calls``) already carry code and
        are run unchanged.
        """
        if tool_call.name == RUN_SNIPPET:
            unexpected = sorted(str(k) for k in args if k != "name")
            snippet, error = resolve_snippet(_TURN_SNIPPETS.get() or [], args.get("name"))
            if unexpected and error is None:
                error = (
                    f"`{RUN_SNIPPET}` takes only `name`; received "
                    f"{', '.join(f'`{k}`' for k in unexpected)}. Put code in a named "
                    "```python block in your reply instead of a tool argument."
                )
            if error is not None or snippet is None:
                return self._snippet_error(runtime, tool_call, session, tool_call_event_id, error)
            if snippet.variant is not None:
                get_harness_metrics().snippet_header_variant(f"{snippet.variant}: {snippet.header}")
            runtime.event_manager.update(
                tool_call_event_id,
                metadata={"snippet_name": snippet.name, "snippet_code": snippet.code},
            )
            args = {"code": snippet.code, "name": snippet.name}
        return await super()._handle_execute_python(
            runtime,
            tool_call,
            args,
            builtins,
            session,
            call,
            return_type,
            tool_call_event_id,
        )

    @staticmethod
    def _snippet_error(
        runtime: RuntimeServices,
        tool_call: Any,
        session: "CodeActSession",
        tool_call_event_id: str,
        error: str | None,
    ) -> None:
        """Record an unresolved snippet as a failed cell the model can correct."""
        message = error or "Snippet could not be resolved."
        get_harness_metrics().snippet_error(message)
        session.record_error()
        execution_count = session.record_execution()
        runtime.event_manager.update(
            tool_call_event_id,
            result=ToolResult(
                tool_call_id=tool_call.id,
                content="status: error",
                result_status=ResultStatus.ERROR,
            ),
        )
        runtime.event_manager.add(
            PythonOutput(
                tool_call_id=tool_call.id,
                execution_count=execution_count,
                stdout="",
                stderr="",
                error=f"Snippet error: {message}",
                value=None,
                explicit_return=False,
                execution_status=ResultStatus.ERROR,
            )
        )

    def _empty_code_error(self, args: dict[str, Any]) -> str:
        return f"Execution error: snippet `{args.get('name', '')}` is empty."

    def _unknown_tool_feedback(
        self, tool_name: str, args: dict[str, Any], runtime: RuntimeServices
    ) -> str:
        del args, runtime
        shown = tool_name if len(tool_name) <= 80 else tool_name[:77] + "..."
        header = f"Unknown tool `{shown}`. Available tools: {RUN_SNIPPET}."
        if tool_name == "return_result":
            return (
                f"{header} return_result is a Python builtin, not a provider tool. To finish, "
                f"write a snippet such as ```python finish``` containing `return_result(value)` "
                f'and call {RUN_SNIPPET}(name="finish").'
            )
        return (
            f"{header} Call agent APIs from Python: write them in a named ```python block "
            f"in your reply and call {RUN_SNIPPET} with its name. Use `print(doc(self))` in a "
            f"snippet to inspect the available APIs. Do not call `{shown}` as a tool again."
        )

    @strategy(TemplateStrategy())
    async def _tool_use_reminder(self, runtime: RuntimeServices, reason: str) -> str:
        """{reason} Write a named ```python block and call `run_snippet(name)`. To finish, run a snippet that calls `return_result(value)`."""
        ...

    @staticmethod
    def _retry_text_only_response(context: TextOnlyResponseContext) -> TextOnlyResponseAction:
        names = [s.name for s in parse_snippets(context.content) if s.name]
        method = context.call.method_name
        if names:
            get_harness_metrics().snippets_not_run()
            shown = ", ".join(f"`{n}`" for n in dict.fromkeys(names))
            message = (
                f"Your last reply wrote snippet(s) {shown} but made no tool call, so nothing "
                f"ran. Snippets run only when you call `{RUN_SNIPPET}` in the same reply: "
                f"write the snippet again and call {RUN_SNIPPET}(name={json.dumps(names[-1])})."
            )
        else:
            message = (
                "Your last reply was plain text with no tool call. It was preserved, but a "
                "bare message cannot end the turn or run code. "
                f"To finish `{method}`, write a named ```python snippet that calls "
                f"`return_result(value)` and call `{RUN_SNIPPET}` with its name. To continue "
                "working, do the same with the next computation."
            )
        return TextOnlyResponseAction.retry(Error(content=message))
