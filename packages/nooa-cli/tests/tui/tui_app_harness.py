# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test harness for driving a ``TUIApplication`` in unit tests.

Usage::

    async with TUIHarness() as h:
        await h.type_keys("hello")
        await h.press("enter")
        assert "hello" in h.capture_output()

The harness owns:

- a ``PipeInput`` — bytes written with ``send_text()`` look like keystrokes
- a prompt_toolkit ``Output`` — ``DummyOutput`` by default, or a mutable
  recording output for production resize-path tests
- a background task running ``app.run_async()``
- a scriptable ``FakeAgent`` that yields controllable responses

Most assertions read the app's *logical* state (input/output buffer text,
queue messages, status line). Resize tests additionally record prompt_toolkit
redraw operations and the transcript bytes written around a replay barrier.

Timing model: writing to the pipe is synchronous, but prompt_toolkit
parses input on the event loop. Every driver method ends with an
``asyncio.sleep(0)`` (or an explicit ``wait_for``) so the event loop has
a chance to drain the pipe before the next assertion runs.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import TYPE_CHECKING, Any

from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput, Output

if TYPE_CHECKING:
    from nooa_cli.tui.tui_application import TUIApplication


# ── key-name → terminal escape-sequence --------------------------------------

# Only what the tests actually press. Extend as new tests demand it.
_KEY_SEQUENCES: dict[str, str] = {
    "enter": "\r",
    "escape": "\x1b",
    "option-[": "\x1b[",
    "option-]": "\x1b]",
    "q": "q",
    "tab": "\t",
    "backspace": "\x7f",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "right": "\x1b[C",
    "left": "\x1b[D",
    "s-up": "\x1b[1;2A",
    "s-down": "\x1b[1;2B",
    "s-right": "\x1b[1;2C",
    "s-left": "\x1b[1;2D",
    "home": "\x1b[H",
    "end": "\x1b[F",
    "pageup": "\x1b[5~",
    "pagedown": "\x1b[6~",
    "c-home": "\x1b[1;5H",
    "c-end": "\x1b[1;5F",
    "delete": "\x1b[3~",
    "c-c": "\x03",
    "c-d": "\x04",
    "c-x": "\x18",
    "c-j": "\n",  # bare LF — used by prompt_toolkit as "Shift+Enter"
    "c-u": "\x15",
    "c-y": "\x19",
    "f2": "\x1bOQ",
    "f6": "\x1b[17~",
    "s-enter": "\x1b\r",  # Alt+Enter / Esc+Enter — prompt_toolkit treats as newline
}


def _key_sequence(key: str) -> str:
    try:
        return _KEY_SEQUENCES[key.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown key {key!r}. Add it to _KEY_SEQUENCES in tui_app_harness."
        ) from None


# ── scriptable agent mock ----------------------------------------------------


class ThreadGate:
    """Small awaitable gate safe to set from a different event loop thread."""

    def __init__(self, *, initially_set: bool = False) -> None:
        self._set = initially_set
        self._lock = threading.Lock()
        self._waiters: list[asyncio.Future[None]] = []

    def set(self) -> None:
        with self._lock:
            self._set = True
            waiters = list(self._waiters)
            self._waiters.clear()
        for waiter in waiters:
            if waiter.done():
                continue
            waiter.get_loop().call_soon_threadsafe(waiter.set_result, None)

    def clear(self) -> None:
        with self._lock:
            self._set = False

    def is_set(self) -> bool:
        with self._lock:
            return self._set

    async def wait(self) -> None:
        with self._lock:
            if self._set:
                return
            waiter: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._waiters.append(waiter)
        try:
            await waiter
        except asyncio.CancelledError:
            with self._lock:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
            raise


class FakeAgent:
    """Scriptable stand-in for ``TUIAgent`` used by harness tests.

    Matches the per-turn contract: ``handle((queue_name, item))``
    is invoked once per turn by the dispatcher, and returns a
    ``RespondResult``-shaped object telling the dispatcher what to do
    next.

    Control knobs:

    - ``self.script`` — callables run one per received message.
    - ``self.block`` — when cleared, ``handle()`` awaits it before
      returning, keeping the dispatcher (and spinner) visibly "working"
      for as long as the test needs. Default: set, so handle returns
      quickly.
    - ``self.next_kind`` — the ``kind`` the FakeAgent returns by
      default. Tests raise ``DispatcherExit`` to end the session, or
      flip to ``"WAIT"`` to exercise multi-queue races.
    """

    def __init__(self) -> None:
        from nooa.runtime.channels import QueueManager

        self.script: list[Callable[[FakeAgent, str], Any]] = []
        self.messages_received: list[str] = []
        self.block = ThreadGate(initially_set=True)  # default: handle returns immediately
        # Tests don't need event-mode channels, so no event_manager.
        self.queue_manager = QueueManager()
        self._user_messages_in = self.queue_manager.queue("user_messages")
        self.user_messages = self._user_messages_in.reader
        self.next_kind: str = "GET_USER_INPUT"

    @staticmethod
    def render_message(text: str) -> str:
        """Render Markdown exactly as the production message frontend does."""
        import io as _io

        from rich.console import Console
        from rich.markdown import Markdown

        buf = _io.StringIO()
        Console(
            file=buf, force_terminal=True, color_system="256", width=80, legacy_windows=False
        ).print(Markdown(text))
        return buf.getvalue()

    async def handle(
        self,
        notification: dict[str, list],
    ) -> Any:
        """Per-turn stub: record inputs, run a scripted step, return a result.

        Returns a simple namespace with ``kind`` (matches the shape
        ``RespondResult`` exposes) so the dispatcher's ``getattr``
        calls find what they need without the harness needing to
        import the TUI agent's Pydantic model.
        """
        for items in notification.values():
            for item in items:
                self.messages_received.append(str(item))
        if self.script:
            step = self.script.pop(0)
            await _maybe_await(step(self, item))
        await self.block.wait()

        class _Result:
            pass

        r = _Result()
        r.kind = self.next_kind
        return r

    def queue(self, step: Callable[[FakeAgent, str], Any]) -> None:
        """Add one scripted step to the end of the response sequence."""
        self.script.append(step)


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


# ── the harness itself -------------------------------------------------------


class MutableRecordingOutput(DummyOutput):
    """A prompt_toolkit output whose terminal geometry tests can change.

    ``DummyOutput`` always reports 80×40, which means calling the real
    prompt_toolkit resize callback cannot exercise Nooa's geometry observer.
    This small test output keeps DummyOutput's no-op terminal behavior while
    recording enough of the redraw path to prove a resize reached the renderer.
    """

    def __init__(self, columns: int = 80, rows: int = 40) -> None:
        super().__init__()
        self._size = Size(rows=int(rows), columns=int(columns))
        self.events: list[tuple[Any, ...]] = []
        self._physical_replay_buffer: list[str] | None = None

    def get_size(self) -> Size:
        self.events.append(("get_size", self._size.columns, self._size.rows))
        return self._size

    def get_rows_below_cursor_position(self) -> int:
        return self._size.rows

    def set_size(self, columns: int, rows: int) -> None:
        self._size = Size(rows=int(rows), columns=int(columns))

    def write(self, data: str) -> None:
        self.events.append(("write", data))
        if self._physical_replay_buffer is not None:
            self._physical_replay_buffer.append(data)

    def write_raw(self, data: str) -> None:
        self.events.append(("write_raw", data))
        if "\x1b[3J" in data:
            self._physical_replay_buffer = []
        if self._physical_replay_buffer is not None:
            self._physical_replay_buffer.append(data)

    def erase_down(self) -> None:
        self.events.append(("erase_down",))

    def reset_attributes(self) -> None:
        self.events.append(("reset_attributes",))

    def enable_mouse_support(self) -> None:
        self.events.append(("enable_mouse_support",))

    def disable_mouse_support(self) -> None:
        self.events.append(("disable_mouse_support",))

    def flush(self) -> None:
        self.events.append(("flush",))
        if self._physical_replay_buffer is None:
            return
        data = "".join(self._physical_replay_buffer)
        self._physical_replay_buffer = None
        import sys

        if sys.__stdout__ is not None:
            sys.__stdout__.write(data)
            sys.__stdout__.flush()


def _wire_local_turn_policy(agent: Any, runner: Any, app: Any, config: Any) -> Any:
    """Mirror Session's composition-root policy wiring for focused tests."""
    from nooa_cli.tui.local_turn_policy import LocalTurnPolicy
    from nooa_cli.tui.tui_application import DispatcherExit

    async def _emit_output(output: Any) -> None:
        display_text = getattr(output, "display_text", None)
        if callable(display_text):
            app.emit_block(f"\x1b[2m{display_text()}\x1b[0m\n")

    policy = LocalTurnPolicy(
        agent,
        runner,
        config,
        emit_output=_emit_output,
        invalidate=app.invalidate,
    )
    runner.set_dispatch_hooks(
        on_state_change=app.runtime_state_changed,
        on_before_handle=policy.before_handle,
        on_after_handle=policy.after_handle,
        on_notification=lambda notification: (
            policy.on_notification(notification),
            app.runtime_notification_received(),
        ),
        dispatcher_exit=DispatcherExit,
        on_cancelled=app.runtime_cancelled,
    )
    app._test_turn_policy = policy
    return policy


def make_local_tui_app(agent: Any, **kwargs: Any) -> Any:
    """Compose a TUIApplication with its lifecycle owner for focused tests."""
    from nooa_cli.interactive import LocalAgentRunner
    from nooa_cli.tui.tui_application import TUIApplication

    app_ref: list[TUIApplication] = []
    runner = LocalAgentRunner(
        agent,
        emit_text=lambda text: app_ref[0].emit_block(text),
        agent_id=f"test-{id(agent):x}",
    )
    from nooa_cli.tui.host_services import TUIHostServices

    reflection = getattr(agent, "_tui_reflection_runner", None)
    if "host_services" not in kwargs:
        kwargs["host_services"] = TUIHostServices(
            auxiliary_status=(None if reflection is None else reflection.indicator_frame)
        )
    config = kwargs.get("config")
    kwargs.setdefault("display_mode", "native-replay")
    app = TUIApplication(agent=runner, **kwargs)
    app_ref.append(app)
    app.observe_agent()
    app._test_agent_runner = runner
    policy = _wire_local_turn_policy(agent, runner, app, config)
    app._on_agent_activity = policy.invalidate_keep_going
    return app


class TUIHarness(AbstractAsyncContextManager["TUIHarness"]):
    """Drive a ``TUIApplication`` from tests.

    Enters as an async context manager: starts the app's event loop task
    on ``__aenter__``, tears it down (with a timeout) on ``__aexit__``.
    """

    def __init__(
        self,
        agent: FakeAgent | None = None,
        config: Any = None,
        full_screen: bool | None = None,
        display_mode: Any = None,
        output: Output | None = None,
    ) -> None:
        self.agent = agent or FakeAgent()
        self._config = config
        self._full_screen = full_screen
        self._display_mode = display_mode
        self.output = output or DummyOutput()
        self._pipe_ctx: Any = None
        self._session_ctx: Any = None
        self._run_task: asyncio.Task | None = None
        self.app: TUIApplication | None = None
        self.runner: Any | None = None

    async def __aenter__(self) -> TUIHarness:
        self._pipe_ctx = create_pipe_input()
        pipe = self._pipe_ctx.__enter__()
        self._session_ctx = create_app_session(input=pipe, output=self.output)
        self._session_ctx.__enter__()

        # Import locally so the stub can evolve without breaking harness
        # consumers that don't import it.
        from nooa_cli.interactive import LocalAgentRunner
        from nooa_cli.tui.host_services import TUIHostServices
        from nooa_cli.tui.tui_application import TUIApplication
        from prompt_toolkit.completion import WordCompleter

        # Canonical slash/bang set that covers the completion tests. Production
        # wiring passes a CommandRegistry-backed completer instead.
        completer = WordCompleter(
            ["/help", "/exit", "/clear", "/compact", "!bash", "!git"],
            sentence=True,
        )
        kwargs = {}
        if self._full_screen is not None:
            kwargs["full_screen"] = self._full_screen
        if self._display_mode is not None:
            kwargs["display_mode"] = self._display_mode
        if self._full_screen is None and self._display_mode is None:
            kwargs["display_mode"] = "native-replay"
        app_ref: list[TUIApplication] = []
        agent_runner = LocalAgentRunner(
            self.agent,
            emit_text=lambda text: app_ref[0].emit_block(text),
            agent_id=f"test-{id(self.agent):x}",
        )
        self.runner = agent_runner
        self.app = TUIApplication(
            agent=agent_runner,
            host_services=TUIHostServices(
                auxiliary_status=(
                    self.agent._tui_reflection_runner.indicator_frame
                    if hasattr(self.agent, "_tui_reflection_runner")
                    else None
                )
            ),
            completer=completer,
            config=self._config,
            **kwargs,
        )
        app_ref.append(self.app)
        policy = _wire_local_turn_policy(self.agent, agent_runner, self.app, self._config)
        self.app._on_agent_activity = policy.invalidate_keep_going
        self._pipe = pipe

        agent_runner.activate(asyncio.get_running_loop())
        self._run_task = asyncio.create_task(self.app.run_async())
        # Let the app install its input reader before the test sends keys.
        await self._wait_for_app_ready()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            # Cancel any pending agent task so FakeAgent.handle() awaiting
            # on agent.block doesn't get orphaned at teardown.
            if self.app is not None:
                runner = self.runner
                agent_task = None if runner is None else runner.task
                if agent_task is not None and not agent_task.done():
                    agent_task.cancel()
                    try:
                        await agent_task
                    except (asyncio.CancelledError, BaseException):
                        pass
                self.app.exit()
            if self._run_task is not None:
                try:
                    await asyncio.wait_for(self._run_task, timeout=2.0)
                except (TimeoutError, asyncio.CancelledError):
                    self._run_task.cancel()
        finally:
            policy = None if self.app is None else getattr(self.app, "_test_turn_policy", None)
            if policy is not None:
                await policy.shutdown()
            if self.runner is not None:
                await self.runner.shutdown()
            if self._session_ctx is not None:
                self._session_ctx.__exit__(None, None, None)
            if self._pipe_ctx is not None:
                self._pipe_ctx.__exit__(None, None, None)

    async def _wait_for_app_ready(self) -> None:
        """Spin until the underlying Application reports ready."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            app = self.app
            if app is not None and app.is_running:
                return
            await asyncio.sleep(0.01)
        raise RuntimeError("TUIApplication did not start within 2s")

    # ── input driving --------------------------------------------------

    async def type_keys(self, text: str) -> None:
        """Send literal characters as if the user typed them."""
        self._pipe.send_text(text)
        await asyncio.sleep(0)

    async def press(self, key: str) -> None:
        """Press a named key (``"enter"``, ``"up"``, ``"c-c"``, …)."""
        self._pipe.send_text(_key_sequence(key))
        await asyncio.sleep(0)

    async def resize_from_terminal(self, columns: int, rows: int) -> None:
        """Change terminal geometry and drive the supported redraw path."""
        output = self.output
        if not isinstance(output, MutableRecordingOutput):
            raise TypeError("resize_from_terminal requires MutableRecordingOutput")
        assert self.app is not None
        previous_size_reads = sum(event[0] == "get_size" for event in output.events)
        output.set_size(columns, rows)
        self.app._app._on_resize()
        await self.wait_for(
            lambda: sum(event[0] == "get_size" for event in output.events) > previous_size_reads
        )

    async def submit_async(self, text: str) -> None:
        """Type ``text`` and press Enter. Doesn't wait for any side-effect."""
        await self.type_keys(text)
        await self.press("enter")

    # ── state-polling helpers -----------------------------------------

    async def wait_for(
        self, predicate: Callable[[], bool], timeout: float = 2.0, interval: float = 0.01
    ) -> None:
        """Yield control until ``predicate()`` returns truthy, or raise."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            await asyncio.sleep(interval)
        raise AssertionError(
            f"Predicate never became true within {timeout}s.\n"
            f"  input={self.capture_input()!r}\n"
            f"  output (tail)={self.capture_output()[-200:]!r}\n"
            f"  queued={self.capture_queued()!r}\n"
            f"  status={self.capture_status()!r}"
        )

    async def wait_input_equals(self, expected: str, timeout: float = 1.0) -> None:
        await self.wait_for(lambda: self.capture_input() == expected, timeout=timeout)

    async def wait_output_contains(self, needle: str, timeout: float = 1.0) -> None:
        await self.wait_for(lambda: needle in self.capture_output(), timeout=timeout)

    # ── introspection --------------------------------------------------

    def capture_input(self) -> str:
        assert self.app is not None
        return self.app.input_buffer.text

    def capture_output(self) -> str:
        assert self.app is not None
        return self.app.output_buffer.text

    def capture_output_ansi(self) -> str:
        """Raw retained sources for ANSI round-trip assertions."""
        assert self.app is not None
        return "".join(block.source for block in self.app._transcript_blocks)

    def capture_queued(self) -> list[str]:
        """Return queued items (pending user messages) in submission order.

        Reads ``agent._user_messages_in`` — the hidden InputQueue the
        dispatcher pumps from — so tests see what the UI's queue
        window is showing. Commands are no longer queued in app state;
        they dispatch immediately via ``on_command``/``on_bang``.
        """
        assert self.app is not None
        if self.runner is None:
            return []
        return list(self.runner.pending_user_messages())

    def capture_status(self) -> str:
        assert self.app is not None
        return self.app.status_text()
