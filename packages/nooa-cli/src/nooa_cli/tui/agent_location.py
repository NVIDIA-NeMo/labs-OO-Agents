# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""On-demand "where is the agent stopped?" probe for the TUI.

Answers the question the ``/activity`` command poses — not just *what phase*
the agent is in (python / llm / idle), but *where in the code* it is currently
suspended.

Why this lives here and not in the core framework:

- The agent runs as a coroutine on a dedicated asyncio loop on its own thread
  (owned by ``LocalAgentRunner``). A point-in-time "where is it"
  snapshot is answered by asyncio task introspection (``Task.get_stack()``),
  which must run *on the agent loop* — ``asyncio.Task`` is not safe to walk
  from another thread. ``ActivityCommand`` receives a runtime dispatcher from
  the composition root, so the probe runs there and
  returns plain data back across the thread boundary.
- ``sys.monitoring`` is the wrong tool for an on-demand probe: its callbacks
  fire on the *executing* thread (not the asker's), there is no per-thread
  instrumentation, and always-on ``LINE`` events cost ~3x runtime. It is built
  for continuous collection (debuggers, coverage), not snapshots.

No core (``src/``) changes are required.
"""

from __future__ import annotations

import asyncio
import gc
import linecache
import re
from dataclasses import dataclass, field

# The agent's turn entrypoint task is named ``<AgentClass>.handle`` by
# ActorRuntime._call_plan (see runtime/actor.py). We match on the method name
# so any agent class works.
_TURN_METHOD = "handle"


@dataclass
class SourceLine:
    lineno: int
    text: str
    is_current: bool


@dataclass
class FrameInfo:
    qualname: str
    lineno: int
    filename: str
    # ±N lines of source around ``lineno``, when resolvable (real files and
    # registered REPL/CodeAct cells both work via linecache). Empty otherwise.
    context: list[SourceLine] = field(default_factory=list)

    def short(self) -> str:
        return f"{self.qualname} (line {self.lineno})"


def _source_context(filename: str, lineno: int, ctx: int = 3) -> list[SourceLine]:
    """±``ctx`` source lines around ``lineno``.

    ``linecache.getlines`` resolves both real files and code cells registered
    by the CodeAct/REPL runtime (their ``co_filename`` is a ``"Cell In[N]"``
    key in ``linecache.cache``). Returns ``[]`` when no source is available
    (e.g. C frames, generated code).
    """
    lines = linecache.getlines(filename)
    if not lines:
        return []
    lo = max(1, lineno - ctx)
    hi = min(len(lines), lineno + ctx)
    return [SourceLine(n, lines[n - 1].rstrip("\n"), n == lineno) for n in range(lo, hi + 1)]


# A REPL/CodeAct cell's ``co_filename`` is a Jupyter-style ``Cell In[N]`` key
# (see ActorRuntime._execute_code). The agent's own ``execute_python`` code runs
# under such a frame, so this is how we tell *the agent's code* apart from the
# framework/stdlib plumbing it is currently awaiting.
_CELL_FILENAME = re.compile(r"^Cell In\[\d+\]$")


def _is_cell_frame(frame: FrameInfo) -> bool:
    return bool(_CELL_FILENAME.match(frame.filename))


@dataclass
class AgentLocation:
    """Where the agent coroutine is currently suspended."""

    task_name: str | None
    stack: list[FrameInfo]

    @property
    def innermost(self) -> FrameInfo | None:
        # Outermost-first; the await/suspend point is the last frame.
        return self.stack[-1] if self.stack else None

    @property
    def innermost_cell(self) -> FrameInfo | None:
        """The deepest frame that is the agent's own ``execute_python`` cell.

        The true ``innermost`` is usually framework/stdlib plumbing the cell is
        blocked in (e.g. ``asyncio.to_thread`` → ``run_in_executor``, an HTTP
        client, ``asyncio.sleep``). That is a faithful suspend point but unhelpful
        to a user asking "which line of *my* cell is running?". This returns the
        last ``Cell In[N]`` frame in the await chain — the line of the agent's
        code that kicked off the pending await — or ``None`` if no cell frame is
        on the stack (e.g. suspended in ``handle`` plumbing before any cell ran).
        """
        for frame in reversed(self.stack):
            if _is_cell_frame(frame):
                return frame
        return None

    @property
    def highlight(self) -> FrameInfo | None:
        """Frame to render as the highlighted suspend-point source.

        Prefer the agent's own cell line (``innermost_cell``); fall back to the
        true suspend point when no cell frame is present.
        """
        return self.innermost_cell or self.innermost


def _frame_info(frame) -> FrameInfo:
    code = frame.f_code
    return FrameInfo(code.co_qualname, frame.f_lineno, code.co_filename)


def _coro_qualname(coro) -> str:
    code = getattr(coro, "cr_code", getattr(coro, "gi_code", None))
    return code.co_qualname if code is not None else repr(coro)


def _coro_frames(coro) -> list[FrameInfo]:
    """Frames of one coroutine and everything it directly awaits (cr_await).

    Stops at the first thing that is not a coroutine/generator — typically a
    ``FutureIter`` (the await of a Task/Future), which has no inspectable frame.
    Task boundaries are crossed by the caller via ``Task._fut_waiter``.
    """
    frames: list[FrameInfo] = []
    obj = coro
    seen = 0
    while obj is not None and seen < 200:
        seen += 1
        frame = getattr(obj, "cr_frame", getattr(obj, "gi_frame", None))
        if frame is None or frame.f_code is None:
            break
        frames.append(_frame_info(frame))
        nxt = getattr(obj, "cr_await", None)
        if nxt is None:
            nxt = getattr(obj, "gi_yieldfrom", None)
        if nxt is obj:
            break
        obj = nxt
    return frames


def _await_chain(task) -> list[FrameInfo]:
    """Walk from *task* down to the actual suspend point, across Task boundaries.

    ``Task.get_stack()`` only returns the task's own frame; the interesting part
    — which tool / LLM call / sleep the agent is blocked in — lives in the
    coroutines it is awaiting. Within a single task we follow ``cr_await``; when
    a task is blocked on a child Task/Future (``asyncio.gather``, a Doer
    subagent), we hop to it via ``Task._fut_waiter`` and keep walking.

    Returns outermost-first, so the last entry is where the agent is suspended.
    """
    frames: list[FrameInfo] = []
    current: asyncio.Task | None = task
    seen_tasks: set[int] = set()
    while current is not None and id(current) not in seen_tasks:
        seen_tasks.add(id(current))
        frames.extend(_coro_frames(current.get_coro()))
        current = _next_task(getattr(current, "_fut_waiter", None), seen_tasks)
    return frames


def _next_task(waiter, seen_tasks: set[int]) -> asyncio.Task | None:
    """Resolve the child Task a waiter blocks on, or ``None``.

    A task awaiting a single child Task has that Task as its ``_fut_waiter``.
    A task awaiting ``asyncio.gather`` blocks on a ``_GatheringFuture`` whose
    pending children live in its ``_children`` list — we follow the first
    not-yet-seen pending child so the stack descends into (one of) the
    concurrent sub-tasks (e.g. a Doer subagent) rather than stopping at gather.

    NOTE: depends on the private CPython attr ``Task._fut_waiter`` and the
    ``_GatheringFuture._children`` layout (the latter reached via
    ``gc.get_referents`` since it is closure-held, not a public attribute). If a
    future CPython changes either, the walk degrades silently to a shallower
    stack — ``test_locate_descends_through_gather`` pins the behaviour so an
    interpreter upgrade surfaces the break.
    """
    if isinstance(waiter, asyncio.Task):
        return waiter
    if isinstance(waiter, asyncio.Future):
        for ref in gc.get_referents(waiter):
            children = ref.get("_children") if isinstance(ref, dict) else None
            if not children:
                continue
            for child in children:
                if (
                    isinstance(child, asyncio.Task)
                    and not child.done()
                    and id(child) not in seen_tasks
                ):
                    return child
    return None


def locate_agent_on_loop(turn_method: str = _TURN_METHOD) -> AgentLocation | None:
    """Build an :class:`AgentLocation` for the agent's current turn.

    MUST be called on the agent loop (e.g. via ``Command.agent_run``). Returns
    ``None`` if no turn task is currently running (the agent is idle between
    turns).
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return None

    def _is_turn(task: asyncio.Task) -> bool:
        # Match the method as a whole name segment, not a bare substring, so an
        # unrelated coroutine like ``SomeHandler.handle_event`` isn't picked up.
        segment = f".{turn_method}"
        qual = _coro_qualname(task.get_coro())
        name = task.get_name()
        return (
            qual == turn_method or segment in qual or name == turn_method or name.endswith(segment)
        )

    turn_task = next(
        (t for t in asyncio.all_tasks(loop) if not t.done() and _is_turn(t)),
        None,
    )
    if turn_task is None:
        return None

    stack = _await_chain(turn_task)
    if not stack:
        stack = [_frame_info(f) for f in turn_task.get_stack()]
    location = AgentLocation(task_name=turn_task.get_name(), stack=stack)
    # Attach source context to the frame the UI will highlight. That is the
    # agent's own cell frame when present (``highlight`` prefers ``innermost_cell``),
    # falling back to the true suspend point — so the rendered code block matches
    # what /activity points its arrow at. Without this the cell frame carries no
    # source and the highlighted-code block silently disappears.
    if location.highlight is not None:
        frame = location.highlight
        frame.context = _source_context(frame.filename, frame.lineno)
    return location
