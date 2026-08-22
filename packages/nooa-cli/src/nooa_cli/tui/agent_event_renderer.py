# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-turn rendering for agent events into the TUI's block queue.

``AgentEventRenderer`` subscribes to the agent's ``Reasoning``,
``ToolCallEvent``, ``PythonOutput``, and semantic ``FileEdit`` events and forwards them to
``emit_text`` as Rich renderables. It also owns the monkey-patched
``agent._render_message`` hook so ``self.message()`` output lands in
the transcript.

State (per-turn):

- ``_agent_has_messaged`` — True after the first ``self.message()`` in
  the current turn. Drives the one-time ``"OO ─"`` rule that separates
  agent output from user input.

``self.message()`` emits inline. There was a previous attempt to buffer
messages and only flush them after the next ``PythonOutput`` / tool
call, so they'd visually land *below* the preceding code cell; that
created a silent-loss bug where any message emitted after the last
``PythonOutput`` of a turn sat in the buffer until the user typed
something else (``reset_turn`` was the only reliable flush path). The
ordering constraint isn't worth the correctness cost: ``self.message()``
is called mid-cell, the ``∴`` preview fires on the enclosing
``ToolCallEvent`` *before* the cell body runs, and cell output arrives
*after* the cell body — so inline emit naturally lands between preview
and output.

The class has no reference to ``Session``. Callers pass ``emit_text``
(the ANSI-enqueue function), display-preference getters, and a shared
``pending_code`` dict (keyed by ``tool_call_id``) that pairs a code
preview with its eventual output.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from rich.markdown import Markdown
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

from .code_preview import _code_preview


class _ReasoningEvent(Protocol):
    """Fields the renderer reads from a ``Reasoning`` event."""

    content: str


class _ToolCallEvent(Protocol):
    """Fields the renderer reads from a ``ToolCallEvent``."""

    name: str
    tool_call_id: str
    arguments: dict[str, Any]


class _PythonOutputEvent(Protocol):
    """Fields the renderer reads from a ``PythonOutput`` event."""

    tool_call_id: str
    stdout: str
    stderr: str


class _SummaryEvent(Protocol):
    """Fields the renderer reads from a ``Summary`` event."""

    summary_tag: str
    replaced_range: tuple[int, int]
    children_tags: list[str]
    summary_text: str | None


class _FileEditEvent(Protocol):
    """Fields the renderer reads from a semantic coding-host file edit."""

    path: str
    operation: str
    diff: str
    content_complete: bool
    diff_complete: bool


class AgentEventRenderer:
    """Render agent events into the TUI block queue, in turn order."""

    def __init__(
        self,
        *,
        agent: Any,
        emit_text: Callable[..., None],
        show_python: Callable[[], bool],
        show_diffs: Callable[[], bool],
        pending_code: dict[str, str],
        colors: dict[str, str],
    ) -> None:
        self._agent = agent
        self._emit_text = emit_text
        self._show_python = show_python
        self._show_diffs = show_diffs
        self._pending_code = pending_code
        self._colors = colors
        self._unsubscribes: list[Callable[[], None]] = []
        self._agent_has_messaged = False
        # Assigned by attach() with whatever was on agent._render_message
        # before; detach() restores to this value. Declaring it here
        # means detach() can read it without a defensive getattr and
        # pyright sees the field.
        self._prior_render_message: Callable[[str], None] | None = None

    # ── lifecycle ──────────────────────────────────────────────────────

    def attach(self) -> None:
        """Subscribe to the agent's event manager and wire _render_message.

        Idempotent: a second call is a no-op (handlers already subscribed).
        Chains onto any existing ``agent._render_message`` so an observer
        hook installed by another component (e.g. a web mirror) still
        fires when ``self.message()`` is called.
        """
        if self._unsubscribes:
            return
        em = self._agent.event_manager
        self._unsubscribes.extend(
            [
                em.on("Reasoning", self._on_reasoning),
                em.on("ToolCallEvent", self._on_tool_call),
                em.on("PythonOutput", self._on_python_output),
                em.on("FileEdit", self._on_file_edit),
                em.on("Summary", self._on_summary),
                em.on("TextOnlyReply", self._on_text_only_reply),
            ]
        )
        # Chain onto any prior hook so we don't silently stomp observers.
        prior = getattr(self._agent, "_render_message", None)
        self._prior_render_message = prior
        if hasattr(self._agent, "_render_message"):

            def _hook(text: str, **emit_kwargs: Any) -> None:
                if prior is not None:
                    try:
                        prior(text)
                    except TypeError:
                        try:
                            prior(text, **emit_kwargs)
                        except Exception:
                            pass
                    except Exception:
                        pass
                self._render_message(text, **emit_kwargs)

            self._agent._render_message = _hook

    def detach(self) -> None:
        """Unsubscribe all handlers and clear the agent hook.

        Safe to call multiple times. Restores ``agent._render_message``
        to whatever was there before ``attach()`` (usually ``None``) so
        a post-shutdown ``agent.message()`` doesn't invoke a dead renderer.
        """
        for fn in self._unsubscribes:
            try:
                fn()
            except Exception:
                pass
        self._unsubscribes.clear()
        # Restore the prior hook if the agent still has our chain installed.
        if hasattr(self._agent, "_render_message"):
            self._agent._render_message = self._prior_render_message

    def reset_turn(self) -> None:
        """Called on new user submission — reset the per-turn ``OO ─`` guard.

        Nothing else to clean up now that ``self.message()`` emits
        inline; the OO rule wants to re-fire once per turn so each
        turn gets its own separator between agent output and user
        input.
        """
        self._agent_has_messaged = False

    # ── message rendering (agent self.message()) ───────────────────────

    def _render_message(self, text: str, **emit_kwargs: Any) -> None:
        """Hook plugged into ``agent._render_message``.

        Emits the message inline as a Markdown block. Ordering vs
        surrounding events is naturally correct because ``self.message()``
        is called mid-cell, after the ``∴`` preview fires on the
        enclosing ``ToolCallEvent`` and before the cell's
        ``PythonOutput`` arrives.
        """
        self._emit_markdown(str(text), **emit_kwargs)

    def _emit_markdown(self, text: str, **emit_kwargs: Any) -> None:
        if not self._agent_has_messaged:
            self._agent_has_messaged = True
            self._emit_text(
                Rule(Text("OO ", style=self._colors["mauve"]), style="dim", align="left"),
                **emit_kwargs,
            )
        self._emit_text(Markdown(str(text)), **emit_kwargs)

    # ── agent event handlers ───────────────────────────────────────────

    def _event_emit_kwargs(self, event: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        event_id = getattr(event, "id", None)
        if event_id is not None:
            kwargs["event_id"] = str(event_id)
        tag = getattr(event, "tag", None)
        if tag is not None:
            kwargs["tags"] = {str(tag)}
        return kwargs

    def _on_reasoning(self, event: _ReasoningEvent) -> None:
        content = getattr(event, "content", "") or ""
        if not content.strip():
            return
        first_line = content.strip().split("\n", 1)[0][:80]
        if "inspecting inputs" in first_line.lower():
            return
        self._emit_text(
            Text(f"∴ Reasoning: {first_line}", style="dim italic"), **self._event_emit_kwargs(event)
        )

    def _on_tool_call(self, event: _ToolCallEvent) -> None:
        name = getattr(event, "name", "")
        if name not in {"execute_python", "python_cell"}:
            # ``return_result`` and other tool calls have no visible
            # rendering of their own — self.message() output (if any)
            # was already emitted inline when the agent called it.
            return
        tool_call_id = getattr(event, "tool_call_id", "")
        arguments = getattr(event, "arguments", {})
        code = arguments.get("code", "") if isinstance(arguments, dict) else ""
        if not code:
            return
        self._pending_code[tool_call_id] = code

        # show_python=True mode renders the full cell from _on_python_output;
        # no teaser preview line needed.
        if self._show_python():
            return

        if tool_call_id.startswith("prefill_"):
            return
        preview = _code_preview(code)
        if not preview:
            return
        first_line = preview.split("\n", 1)[0]
        styled = Text(f"∴ {first_line}", style="dim")
        if first_line.lstrip().startswith("#"):
            styled.stylize("not dim", 0, len(first_line) + 2)
        if "\n" in preview:
            styled.append("\n  " + preview.split("\n", 1)[1], style="dim")
        self._emit_text(styled, **self._event_emit_kwargs(event))

    def _on_python_output(self, event: _PythonOutputEvent) -> None:
        tool_call_id = getattr(event, "tool_call_id", "")
        code = self._pending_code.pop(tool_call_id, None)
        if tool_call_id.startswith("prefill_"):
            return

        stdout = str(getattr(event, "stdout", "") or "")
        stderr = str(getattr(event, "stderr", "") or "")
        error = str(getattr(event, "error", "") or "")
        if error and not stderr:
            stderr = error

        # show_python=True: render a notebook-style cell — 'oo python'
        # rule, syntax-highlighted code, 'oo stdout' rule + stdout,
        # 'oo stderr' rule + stderr.
        if self._show_python() and code:
            mauve = self._colors["mauve"]
            red = self._colors["red"]
            text_color = self._colors["text"]
            self._emit_text(Rule(Text("oo python", style=mauve), style="dim", align="left"))
            self._emit_text(
                Syntax(code.strip(), "python", theme="monokai", background_color="default")
            )
            if stdout.strip():
                self._emit_text(Rule(Text("oo stdout", style=mauve), style="dim", align="left"))
                self._emit_text(Text(stdout.rstrip("\n"), style=text_color))
            if stderr.strip():
                self._emit_text(Rule(Text("oo stderr", style=red), style="dim", align="left"))
                self._emit_text(Text(stderr.rstrip("\n"), style=red))
            return

        # Preview mode: show a one-line error summary (muted red).
        if stderr.strip():
            first_err = stderr.strip().split("\n")[-1][:120]
            self._emit_text(
                Text(f"  ⚠ {first_err}", style="dim red"), **self._event_emit_kwargs(event)
            )

    def _on_file_edit(self, event: _FileEditEvent) -> None:
        """Render the bounded diff supplied by ``ActivityShellTools``."""
        if not self._show_diffs():
            return
        diff = str(getattr(event, "diff", "") or "")
        if not diff.strip():
            return

        path = str(getattr(event, "path", "") or "")
        operation = str(getattr(event, "operation", "edit") or "edit")
        kwargs = self._event_emit_kwargs(event)
        self._emit_text(
            Rule(
                Text(f"oo {operation} · {path}", style=self._colors["mauve"]),
                style="dim",
                align="left",
            ),
            **kwargs,
        )
        self._emit_text(
            Syntax(diff, "diff", theme="monokai", background_color="default", word_wrap=True),
            **kwargs,
        )
        if not bool(getattr(event, "diff_complete", True)):
            self._emit_text(Text("diff is truncated", style="dim italic"), **kwargs)

    def _on_summary(self, event: _SummaryEvent) -> None:
        """Render a dim one-line notice when a summary/truncation is applied.

        Makes the summarizer visible: seeing back-to-back lines here is the
        signal that lets the user catch cascade bugs (same diagnosis we just
        did via the DB, but live). Format:

            ∴ summarized 1..600 · 600 events → 2.5KB summary
            ∴ truncated 1..600 · 600 events (no summary)
        """
        tag = getattr(event, "summary_tag", "") or ""
        rng = getattr(event, "replaced_range", None) or (0, 0)
        children = getattr(event, "children_tags", None) or []
        text = getattr(event, "summary_text", None)

        # Event count: prefer children_tags len (exact, survives nested
        # summaries) over replaced_range width (always an upper bound).
        n_events = len(children) if children else max(0, rng[1] - rng[0] + 1)

        if text is None:
            detail = f"{n_events} events (no summary)"
            verb = "truncated"
        else:
            chars = len(text)
            if chars >= 1024:
                size = f"{chars / 1024:.1f}KB"
            else:
                size = f"{chars}B"
            detail = f"{n_events} events → {size} summary"
            verb = "summarized"

        self._emit_text(
            Text(f"∴ {verb} {tag} · {detail}", style="dim italic"), **self._event_emit_kwargs(event)
        )

    def _on_text_only_reply(self, event: Any) -> None:
        """Render a notice when a code-act turn replies without a tool call."""
        recovered = getattr(event, "recovered", False)
        n = getattr(event, "consecutive_text_only", 0)
        state = "recovered" if recovered else f"#{n}"
        self._emit_text(
            Text(
                f"⚠ agent replied without a tool call ({state}).",
                style="yellow",
            ),
            **self._event_emit_kwargs(event),
        )
