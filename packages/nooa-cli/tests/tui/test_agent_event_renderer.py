# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``AgentEventRenderer``.

The renderer owns one piece of per-turn state (``_agent_has_messaged``)
and three event handlers. These tests construct the renderer directly
with a ``_FakeAgent`` and a list-append ``emit_text`` so we can assert
on the exact sequence of emitted Rich renderables — no prompt_toolkit,
no block queue, no frontend.

Covers:
- ``self.message()`` emits inline as a Markdown block the moment it's
  called — no buffering. (See the module docstring for why the old
  buffer-until-next-PythonOutput scheme was removed.)
- The first ``self.message()`` of a turn emits the ``"OO ─"`` rule
  once; subsequent messages don't re-emit it until ``reset_turn()``.
- prefill tool-call-ids short-circuit the full-cell render (no
  user-visible code / stdout for internal inspection calls).
- ``show_python=True`` path renders the full cell
  (``oo python`` / ``oo stdout``).
- ``detach()`` restores any prior ``_render_message`` hook.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from nooa_cli.tui.agent_event_renderer import AgentEventRenderer
from nooa_cli.tui.theme import COLORS
from rich.markdown import Markdown
from rich.rule import Rule
from rich.syntax import Syntax
from rich.text import Text

# ── helpers ────────────────────────────────────────────────────────────


class _FakeEventManager:
    """Records ``.on(type, handler)`` calls so we can drive the handlers
    directly — we don't need a real EventManager's dispatch path to
    unit-test the renderer's *reactions* to events."""

    def __init__(self) -> None:
        self.handlers: dict[str, list[Any]] = {}

    def on(self, event_type: str, handler: Any) -> Any:
        self.handlers.setdefault(event_type, []).append(handler)

        def _unsub() -> None:
            self.handlers.get(event_type, []).remove(handler)

        return _unsub

    def fire(self, event_type: str, event: Any) -> None:
        for h in self.handlers.get(event_type, []):
            h(event)


class _FakeAgent:
    """Minimal agent surface the renderer touches: event_manager + render hook."""

    def __init__(self) -> None:
        self.event_manager = _FakeEventManager()
        self._render_message: Any = None


def _mk(
    show_python: bool = False, show_diffs: bool = True
) -> tuple[_FakeAgent, list[Any], AgentEventRenderer]:
    agent = _FakeAgent()
    emitted: list[Any] = []
    pending_code: dict[str, str] = {}
    r = AgentEventRenderer(
        agent=agent,  # type: ignore[arg-type]
        emit_text=emitted.append,
        show_python=lambda: show_python,
        show_diffs=lambda: show_diffs,
        pending_code=pending_code,
        colors=COLORS,
    )
    return agent, emitted, r


def _fire_reasoning(em: _FakeEventManager, content: str) -> None:
    em.fire("Reasoning", SimpleNamespace(content=content))


def _fire_tool_call(em: _FakeEventManager, code: str, tool_call_id: str = "t1") -> None:
    em.fire(
        "ToolCallEvent",
        SimpleNamespace(
            name="execute_python",
            tool_call_id=tool_call_id,
            arguments={"code": code},
        ),
    )


def _fire_python_output(
    em: _FakeEventManager, stdout: str = "", stderr: str = "", tool_call_id: str = "t1"
) -> None:
    em.fire(
        "PythonOutput",
        SimpleNamespace(tool_call_id=tool_call_id, stdout=stdout, stderr=stderr),
    )


def _fire_file_edit(
    em: _FakeEventManager,
    *,
    content_complete: bool = True,
    diff_complete: bool = True,
) -> None:
    em.fire(
        "FileEdit",
        SimpleNamespace(
            path="/workspace/example.py",
            operation="update",
            diff="--- a/example.py\n+++ b/example.py\n@@ -1 +1 @@\n-old\n+new\n",
            content_complete=content_complete,
            diff_complete=diff_complete,
        ),
    )


def _fire_summary(
    em: _FakeEventManager,
    *,
    summary_tag: str = "1..10",
    replaced_range: tuple[int, int] = (1, 10),
    children_tags: list[str] | None = None,
    summary_text: str | None = "Summary of events.",
) -> None:
    em.fire(
        "Summary",
        SimpleNamespace(
            summary_tag=summary_tag,
            replaced_range=replaced_range,
            children_tags=children_tags
            if children_tags is not None
            else [str(i) for i in range(1, 11)],
            summary_text=summary_text,
        ),
    )


# ── tests ──────────────────────────────────────────────────────────────


def test_message_emits_inline_not_buffered() -> None:
    """Regression: ``self.message()`` renders inline the moment it's
    called, rather than being buffered until the next
    ``PythonOutput``. The old buffer-then-flush scheme stranded
    messages for the full lifetime of the last turn whenever the
    flush triggers (``PythonOutput``, ``return_result``) didn't fire
    after the final ``self.message()`` — the user only saw them after
    typing their next input. Inline emit kills that class of bug.
    """
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    _fire_tool_call(agent.event_manager, "print('hi')")
    assert agent._render_message is not None
    agent._render_message("mid-cell note")

    # Message must have rendered immediately — not held until the
    # cell completes.
    markdowns = [e for e in emitted if isinstance(e, Markdown)]
    assert len(markdowns) == 1
    assert "mid-cell note" in str(markdowns[0].markup)

    # Subsequent PythonOutput shouldn't add more Markdown renderables
    # (no buffer to flush any more).
    _fire_python_output(agent.event_manager, stderr="warn")
    markdowns_after = [e for e in emitted if isinstance(e, Markdown)]
    assert len(markdowns_after) == 1


def test_message_between_preview_and_cell_output_preserves_natural_order() -> None:
    """``self.message()`` is called *inside* the cell body, so the
    ``∴`` code preview (fired on the enclosing ``ToolCallEvent``)
    renders first, then the message renders inline, and finally the
    cell's stderr/stdout lands when the ``PythonOutput`` event
    arrives. This is the natural ordering inline emit produces.
    """
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    _fire_tool_call(agent.event_manager, "print('hi')")
    agent._render_message("mid-cell note")
    _fire_python_output(agent.event_manager, stderr="warn")

    preview_idx = next(
        i for i, e in enumerate(emitted) if isinstance(e, Text) and "print" in str(e)
    )
    message_idx = next(i for i, e in enumerate(emitted) if isinstance(e, Markdown))
    stderr_idx = next(i for i, e in enumerate(emitted) if isinstance(e, Text) and "warn" in str(e))
    assert preview_idx < message_idx < stderr_idx, (
        "expected preview → message → stderr; got "
        f"preview={preview_idx} message={message_idx} stderr={stderr_idx}"
    )


def test_return_result_tool_call_is_a_renderer_noop() -> None:
    """``return_result`` as its own tool call emits no user-visible
    rendering from this class — ``self.message()`` output, if any,
    was already emitted inline when the agent called it, so there is
    nothing for ``_on_tool_call(name='return_result')`` to do.
    """
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    agent._render_message("Final word")
    # Inline emit — the message is already on the transcript.
    assert any(isinstance(e, Markdown) and "Final word" in str(e.markup) for e in emitted)

    before = list(emitted)
    agent.event_manager.fire(
        "ToolCallEvent",
        SimpleNamespace(name="return_result", tool_call_id="rr1", arguments={"result": {}}),
    )
    assert emitted == before, (
        "return_result must be a no-op; nothing to flush in the inline-emit world"
    )


def test_prefill_tool_call_does_not_render_code_output() -> None:
    """Prefill executions (internal inspection, e.g. ``prefill_abc``)
    show a friendly preview instead of code — and the matching
    ``PythonOutput`` short-circuits the full cell render so stdout
    from the prefill doesn't leak into the transcript."""
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    _fire_tool_call(agent.event_manager, "inspect(x)", tool_call_id="prefill_a")
    agent._render_message("post-inspect")
    _fire_python_output(agent.event_manager, stdout="noise", tool_call_id="prefill_a")

    # Prefill preview is suppressed — no "Inspecting inputs" text emitted.
    assert not any(isinstance(e, Text) and "Inspecting inputs" in str(e) for e in emitted)
    # The stdout from the prefill should NOT render — it's internal noise.
    assert not any(isinstance(e, Text) and "noise" in str(e) for e in emitted)
    # self.message() between preview and prefill-output rendered inline.
    assert any(isinstance(e, Markdown) for e in emitted)


def test_show_python_true_renders_full_cell() -> None:
    """With show_python=True, PythonOutput emits a notebook-style cell:
    'oo python' rule, Syntax(code), 'oo stdout' rule, stdout Text."""
    agent, emitted, r = _mk(show_python=True)
    r.attach()

    _fire_tool_call(agent.event_manager, "print('ok')")
    _fire_python_output(agent.event_manager, stdout="ok\n")

    # Exactly one Syntax renderable for the code.
    syntaxes = [e for e in emitted if isinstance(e, Syntax)]
    assert len(syntaxes) == 1

    # 'oo python' rule precedes syntax; 'oo stdout' rule precedes stdout.
    rules = [e for e in emitted if isinstance(e, Rule)]
    titles = [str(r.title) if r.title is not None else "" for r in rules]
    assert any("python" in t for t in titles)
    assert any("stdout" in t for t in titles)


def test_detach_restores_prior_render_message_hook() -> None:
    """``attach`` chains onto any prior ``_render_message`` so observers
    (e.g. a web mirror) still fire. ``detach`` restores the prior hook
    verbatim so a post-shutdown ``agent.message()`` doesn't invoke a
    dead renderer."""
    agent, emitted, r = _mk(show_python=False)

    observer_calls: list[str] = []
    agent._render_message = observer_calls.append

    r.attach()
    assert agent._render_message is not observer_calls.append  # chained

    # A message fires the observer AND the renderer emits it inline.
    agent._render_message("hi")
    assert observer_calls == ["hi"]
    assert any(isinstance(e, Markdown) and "hi" in str(e.markup) for e in emitted)

    r.detach()
    # Prior hook restored — observer still works, renderer does not
    # queue (any more messages go only to the observer).
    emitted.clear()
    observer_calls.clear()
    agent._render_message("post-shutdown")
    assert observer_calls == ["post-shutdown"]
    assert emitted == []  # renderer is gone


def test_reasoning_event_emits_dim_italic_text() -> None:
    """``Reasoning`` content is emitted as a dim italic ``Text`` block."""
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    _fire_reasoning(agent.event_manager, "thinking...")

    texts = [e for e in emitted if isinstance(e, Text)]
    assert len(texts) == 1
    assert "thinking" in str(texts[0])


def test_empty_reasoning_content_is_ignored() -> None:
    """Blank reasoning events don't emit anything."""
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    _fire_reasoning(agent.event_manager, "")
    _fire_reasoning(agent.event_manager, "   \n ")

    assert emitted == []


def test_reset_turn_is_quiet() -> None:
    """``reset_turn`` just clears the per-turn ``OO ─`` guard flag —
    it must never emit a stray rule or re-render anything."""
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    # First turn: one message renders inline + fires the OO rule once.
    agent._render_message("turn 1")
    assert any(isinstance(e, Rule) for e in emitted)
    emitted.clear()

    # reset_turn is a no-op on the emission side.
    r.reset_turn()
    assert emitted == []

    # Next turn's first message fires a fresh OO rule (guard was reset).
    agent._render_message("turn 2")
    assert any(isinstance(e, Rule) for e in emitted)


def test_attach_is_idempotent() -> None:
    """A second ``attach`` call while already attached is a no-op —
    handlers aren't double-subscribed."""
    agent, emitted, r = _mk(show_python=False)
    r.attach()
    r.attach()  # second call

    _fire_reasoning(agent.event_manager, "x")

    texts = [e for e in emitted if isinstance(e, Text)]
    assert len(texts) == 1  # not 2


def test_summary_event_emits_dim_preview_line() -> None:
    """Applied summaries surface as ``∴ summarized <tag> · N events → NB
    summary`` so the user can see each collapse live.

    Regression guard: if Summary events stop being emitted by
    ``event_manager.collapse``, the TUI silently loses visibility into the
    summarizer (which is how a pathological cascade bug went undiagnosed
    until a DB audit).
    """
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    _fire_summary(
        agent.event_manager,
        summary_tag="1..600",
        replaced_range=(1, 600),
        children_tags=[str(i) for i in range(1, 601)],
        summary_text="x" * 2501,
    )

    texts = [e for e in emitted if isinstance(e, Text) and "∴" in str(e)]
    assert len(texts) == 1
    line = str(texts[0])
    assert "summarized" in line
    assert "1..600" in line
    assert "600 events" in line
    # 2501 chars → "2.4KB" (2501/1024 ≈ 2.4)
    assert "KB" in line


def test_summary_event_without_text_shows_truncation() -> None:
    """A collapse with ``summary_text=None`` is a truncation — shown with
    ``truncated`` and a ``(no summary)`` tail so it's visually distinct
    from a real summarization."""
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    _fire_summary(
        agent.event_manager,
        summary_tag="1..50",
        replaced_range=(1, 50),
        children_tags=[str(i) for i in range(1, 51)],
        summary_text=None,
    )

    texts = [e for e in emitted if isinstance(e, Text) and "∴" in str(e)]
    assert len(texts) == 1
    line = str(texts[0])
    assert "truncated" in line
    assert "1..50" in line
    assert "no summary" in line


@pytest.mark.parametrize(
    "code,expected_fragment",
    [
        ("x = 1", "x = 1"),
        ("# explain what this does\nrun_it()", "explain"),
    ],
)
def test_code_preview_first_line_shape(code: str, expected_fragment: str) -> None:
    """The ∴ code-preview line includes the first (comment or code) line."""
    agent, emitted, r = _mk(show_python=False)
    r.attach()

    _fire_tool_call(agent.event_manager, code)

    preview_texts = [e for e in emitted if isinstance(e, Text) and "∴" in str(e)]
    assert preview_texts, "expected a ∴ preview line"
    assert expected_fragment in str(preview_texts[0])


def test_file_edit_renders_inline_diff_when_enabled() -> None:
    agent, emitted, renderer = _mk(show_diffs=True)
    renderer.attach()

    _fire_file_edit(agent.event_manager)

    assert any(isinstance(item, Rule) and "example.py" in str(item.title) for item in emitted)
    diffs = [item for item in emitted if isinstance(item, Syntax)]
    assert len(diffs) == 1
    assert "+new" in diffs[0].code


def test_file_edit_is_hidden_when_inline_diffs_are_disabled() -> None:
    agent, emitted, renderer = _mk(show_diffs=False)
    renderer.attach()

    _fire_file_edit(agent.event_manager)

    assert emitted == []


def test_file_edit_marks_bounded_diff_as_truncated() -> None:
    agent, emitted, renderer = _mk(show_diffs=True)
    renderer.attach()

    _fire_file_edit(agent.event_manager, diff_complete=False)

    assert any(isinstance(item, Text) and "truncated" in str(item) for item in emitted)


def test_file_edit_does_not_confuse_bounded_content_with_a_truncated_diff() -> None:
    agent, emitted, renderer = _mk(show_diffs=True)
    renderer.attach()

    _fire_file_edit(agent.event_manager, content_complete=False, diff_complete=True)

    assert not any(isinstance(item, Text) and "truncated" in str(item) for item in emitted)


def test_session_emit_text_adds_code_copy_metadata_only_in_fullscreen() -> None:
    from nooa_cli.tui.config import DisplayMode
    from nooa_cli.tui.copyable_markdown import CopyableMarkdown
    from nooa_cli.tui.session import Session

    class FakeApp:
        display_mode = DisplayMode.FULLSCREEN

        def __init__(self) -> None:
            self.calls = []

        def emit_block(self, text, **kwargs) -> None:
            self.calls.append((text, kwargs))

    session = Session.__new__(Session)
    session._app = FakeApp()
    session._render_to_ansi = lambda renderable: (
        str(renderable.markup) if isinstance(renderable, Markdown) else str(renderable)
    )

    Session._emit_text(session, Markdown("```python\nx = 1\n```"))

    _text, kwargs = session._app.calls[0]
    assert callable(kwargs["replay"])
    assert list(kwargs["code_copy_actions"].values()) == ["x = 1"]
    assert isinstance(kwargs["replay"].__defaults__[0], CopyableMarkdown)


def test_session_emit_text_leaves_native_markdown_without_copy_metadata() -> None:
    from nooa_cli.tui.config import DisplayMode
    from nooa_cli.tui.session import Session

    class FakeApp:
        display_mode = DisplayMode.NATIVE

        def __init__(self) -> None:
            self.calls = []

        def emit_block(self, text, **kwargs) -> None:
            self.calls.append((text, kwargs))

    session = Session.__new__(Session)
    session._app = FakeApp()
    session._render_to_ansi = lambda renderable: str(renderable.markup)

    Session._emit_text(session, Markdown("```python\nx = 1\n```"))

    _text, kwargs = session._app.calls[0]
    assert "replay" not in kwargs
    assert "code_copy_actions" not in kwargs
