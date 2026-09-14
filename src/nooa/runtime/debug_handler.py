# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Debug signal handler for nooa.

Provides SIGUSR2 handler that dumps:
1. Full Python traceback with source lines
2. All registered Cell code from linecache
3. Pending LLM calls (if any)
4. Debug file location

Usage:
    from nooa.runtime.debug_handler import install_debug_handler
    install_debug_handler()  # Call once at startup

Then send SIGUSR2 to dump debug info:
    kill -USR2 <pid>

Debug output is written to:
    debug_dump_<pid>.txt  (in the current working directory, or dump_dir if set)

LLM Call Tracking:
    from nooa.runtime.debug_handler import llm_call_context

    with llm_call_context(model="gpt-4", prompt_tokens=1500):
        response = client.complete(...)
"""

import faulthandler
import linecache
import logging
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from types import FrameType
from typing import IO, Any

logger = logging.getLogger(__name__)

# Registry for tracking pending LLM calls
# Key: unique call ID, Value: dict with metadata (model, start_time, prompt_tokens, etc.)
_pending_llm_calls: dict[str, dict[str, Any]] = {}
_llm_call_lock = threading.Lock()
_llm_call_counter = 0

# Registry for tracking in-flight Python code executions (one per running cell).
# Key: unique exec ID, Value: dict with start_time + code preview.
_pending_code_execs: dict[str, dict[str, Any]] = {}
_code_exec_lock = threading.Lock()
_code_exec_counter = 0


def register_llm_call(
    model: str,
    prompt_tokens: int | None = None,
    endpoint: str | None = None,
    **extra_metadata: Any,
) -> str:
    """Register a pending LLM call for debug tracking.

    Args:
        model: Model name (e.g., "gpt-4", "claude-3-opus")
        prompt_tokens: Approximate number of prompt tokens
        endpoint: API endpoint being called
        **extra_metadata: Any additional metadata to track

    Returns:
        Call ID to use when unregistering
    """
    global _llm_call_counter
    with _llm_call_lock:
        _llm_call_counter += 1
        call_id = f"llm_{_llm_call_counter}"
        _pending_llm_calls[call_id] = {
            "model": model,
            "start_time": time.monotonic(),
            "start_timestamp": datetime.now().isoformat(),
            "prompt_tokens": prompt_tokens,
            "endpoint": endpoint,
            "thread": threading.current_thread().name,
            **extra_metadata,
        }
        return call_id


def unregister_llm_call(call_id: str) -> None:
    """Unregister a completed LLM call."""
    with _llm_call_lock:
        _pending_llm_calls.pop(call_id, None)


def get_pending_llm_calls() -> list[dict[str, Any]]:
    """Snapshot of in-flight LLM calls, oldest first.

    Each entry carries ``call_id`` plus the metadata recorded at
    ``register_llm_call`` time and a live ``elapsed`` (seconds) field.
    Used by the ``/activity`` slash command to report whether the agent
    is blocked waiting on a model response.
    """
    now = time.monotonic()
    with _llm_call_lock:
        calls = list(_pending_llm_calls.items())
    out: list[dict[str, Any]] = []
    for call_id, info in calls:
        entry = dict(info)
        entry["call_id"] = call_id
        entry["elapsed"] = now - info["start_time"]
        out.append(entry)
    out.sort(key=lambda e: e["start_time"])
    return out


def register_code_exec(code: str | None = None) -> str:
    """Register an in-flight Python code execution for activity tracking."""
    global _code_exec_counter
    preview = ""
    if code:
        preview = code.strip().splitlines()[0][:120] if code.strip() else ""
    with _code_exec_lock:
        _code_exec_counter += 1
        exec_id = f"exec_{_code_exec_counter}"
        _pending_code_execs[exec_id] = {
            "start_time": time.monotonic(),
            "start_timestamp": datetime.now().isoformat(),
            "preview": preview,
            "thread": threading.current_thread().name,
        }
        return exec_id


def unregister_code_exec(exec_id: str) -> None:
    """Unregister a completed Python code execution."""
    with _code_exec_lock:
        _pending_code_execs.pop(exec_id, None)


@contextmanager
def code_exec_context(code: str | None = None) -> Generator[str, None, None]:
    """Context manager tracking a Python code execution for ``/activity``."""
    exec_id = register_code_exec(code)
    try:
        yield exec_id
    finally:
        unregister_code_exec(exec_id)


def get_pending_code_execs() -> list[dict[str, Any]]:
    """Snapshot of in-flight Python code executions, oldest first."""
    now = time.monotonic()
    with _code_exec_lock:
        execs = list(_pending_code_execs.items())
    out: list[dict[str, Any]] = []
    for exec_id, info in execs:
        entry = dict(info)
        entry["exec_id"] = exec_id
        entry["elapsed"] = now - info["start_time"]
        out.append(entry)
    out.sort(key=lambda e: e["start_time"])
    return out


def get_activity() -> dict[str, Any]:
    """Best-effort snapshot of what the agent is doing *right now*.

    Returns a dict with:
        - ``phase``: the *primary* activity — one of ``"executing_python"``,
          ``"waiting_llm"``, ``"idle"``. ``executing_python`` wins when both are
          in flight: an LLM call made from inside a running cell means the agent
          is executing code that is itself blocked on the model. The full
          ``code_execs``/``llm_calls`` lists below are always complete, so a
          caller wanting to surface concurrent LLM calls reads those directly.
        - ``code_execs``: list from ``get_pending_code_execs()``.
        - ``llm_calls``: list from ``get_pending_llm_calls()``.

    This reads process-global registries, so it reflects the live state even
    when called from a different thread (e.g. a slash command handler).
    """
    code_execs = get_pending_code_execs()
    llm_calls = get_pending_llm_calls()
    if code_execs:
        phase = "executing_python"
    elif llm_calls:
        phase = "waiting_llm"
    else:
        phase = "idle"
    return {"phase": phase, "code_execs": code_execs, "llm_calls": llm_calls}


@contextmanager
def llm_call_context(
    model: str,
    prompt_tokens: int | None = None,
    endpoint: str | None = None,
    **extra_metadata: Any,
) -> Generator[str, None, None]:
    """Context manager for tracking LLM calls.

    Usage:
        with llm_call_context(model="gpt-4", prompt_tokens=1500):
            response = client.complete(...)
    """
    call_id = register_llm_call(
        model=model,
        prompt_tokens=prompt_tokens,
        endpoint=endpoint,
        **extra_metadata,
    )
    try:
        yield call_id
    finally:
        unregister_llm_call(call_id)


def _dump_pending_llm_calls(file: IO[str] | None = None) -> None:
    """Dump information about pending LLM calls."""
    if file is None:
        file = sys.stderr
    assert file is not None

    file.write("\n" + "=" * 60 + "\n")
    file.write("PENDING LLM CALLS\n")
    file.write("=" * 60 + "\n")

    with _llm_call_lock:
        calls = list(_pending_llm_calls.items())

    if not calls:
        file.write("No pending LLM calls registered.\n")
    else:
        now = time.monotonic()
        for call_id, info in calls:
            elapsed = now - info["start_time"]
            model = info.get("model", "unknown")
            tokens = info.get("prompt_tokens")
            endpoint = info.get("endpoint", "")
            thread = info.get("thread", "")
            started = info.get("start_timestamp", "")

            file.write(f"\n--- {call_id} ---\n")
            file.write(f"  Model: {model}\n")
            file.write(f"  Waiting: {elapsed:.1f}s\n")
            file.write(f"  Started: {started}\n")
            if tokens:
                file.write(f"  Prompt tokens: ~{tokens}\n")
            if endpoint:
                file.write(f"  Endpoint: {endpoint}\n")
            if thread:
                file.write(f"  Thread: {thread}\n")

    file.write("\n")
    file.flush()


# Patterns that indicate we're in an LLM-related call
_LLM_STACK_PATTERNS = [
    # HTTP clients
    ("httpx", "HTTP client (httpx)"),
    ("aiohttp", "HTTP client (aiohttp)"),
    ("urllib3", "HTTP client (urllib3)"),
    ("requests/", "HTTP client (requests)"),
    # LLM libraries
    ("litellm", "LiteLLM"),
    ("openai/", "OpenAI SDK"),
    ("anthropic/", "Anthropic SDK"),
    ("google/generativeai", "Google AI SDK"),
    # Our wrappers
    ("unifiedllm", "UnifiedLLM"),
    ("completion_client", "CompletionClient"),
]


def _detect_llm_in_stack(frame: FrameType | None) -> list[str]:
    """Analyze stack frames to detect if we're in an LLM call.

    Returns list of detected LLM-related contexts.
    """
    detected = []
    seen = set()

    try:
        # Walk up the stack
        current = frame
        while current is not None:
            filename = current.f_code.co_filename
            funcname = current.f_code.co_name

            for pattern, description in _LLM_STACK_PATTERNS:
                if pattern in filename and description not in seen:
                    # Get more context
                    lineno = current.f_lineno
                    detected.append(f"{description} at {filename}:{lineno} in {funcname}()")
                    seen.add(description)

            current = current.f_back
    except Exception:
        pass  # Don't fail if stack inspection fails

    return detected


def _get_debug_dump_path() -> Path:
    """Get debug dump path with PID. Uses _dump_dir, defaulting to cwd."""
    return _dump_dir / f"debug_dump_{os.getpid()}.txt"


def _dump_cell_code(file: IO[str] | None = None) -> None:
    """Dump all Cell code registered in linecache."""
    if file is None:
        file = sys.stderr
    assert file is not None

    file.write("\n" + "=" * 60 + "\n")
    file.write("REGISTERED CELL CODE (from linecache)\n")
    file.write("=" * 60 + "\n")

    # Find all Cell entries in linecache (format: "Cell <exec_id>[N]" or "Cell In[N]")
    cell_entries = sorted(
        [(name, data) for name, data in linecache.cache.items() if name.startswith("Cell ")],
        key=lambda x: x[0],
    )

    if not cell_entries:
        file.write("No Cell code registered in linecache.\n")
        return

    for cell_name, cell_data in cell_entries:
        _size, _mtime, lines, _fullname = cell_data  # type: ignore[misc]
        file.write(f"\n--- {cell_name} ({len(lines)} lines) ---\n")
        for i, line in enumerate(lines, 1):
            file.write(f"{i:4d}: {line.rstrip()}\n")

    file.write("\n" + "=" * 60 + "\n")
    file.flush()


def _debug_signal_handler(signum: int, frame: FrameType | None) -> None:
    """Handle SIGUSR2 by dumping debug information."""
    # Write to both stderr and a debug file for reliability
    debug_file_path = _get_debug_dump_path()

    # Detect LLM calls in stack before we do anything else
    llm_in_stack = _detect_llm_in_stack(frame)

    try:
        with open(debug_file_path, "w") as debug_file:
            for file in [sys.stderr, debug_file]:
                assert file is not None
                file.write("\n")
                file.write("=" * 60 + "\n")
                file.write(f"DEBUG DUMP at {datetime.now().isoformat()}\n")
                file.write(f"Signal: {signum} (SIGUSR2)\n")
                file.write(f"Dump file: {debug_file_path}\n")
                file.write("=" * 60 + "\n\n")

                # 1. Quick summary - are we stuck in an LLM call?
                if llm_in_stack or _pending_llm_calls:
                    file.write("⚠️  LIKELY STUCK IN LLM CALL\n")
                    file.write("-" * 40 + "\n")
                    if llm_in_stack:
                        file.write("Detected in stack:\n")
                        for item in llm_in_stack:
                            file.write(f"  • {item}\n")
                    if _pending_llm_calls:
                        file.write(f"Registered pending calls: {len(_pending_llm_calls)}\n")
                    file.write("\n")

                # 2. Standard traceback (like faulthandler but with source lines)
                file.write("CURRENT TRACEBACK:\n")
                file.write("-" * 40 + "\n")
                try:
                    # Use traceback module which uses linecache for source lines
                    traceback.print_stack(frame, file=file)
                except Exception as e:
                    file.write(f"Error getting traceback: {e}\n")

                # 3. Dump pending LLM calls (registered via llm_call_context)
                _dump_pending_llm_calls(file)

                # 4. Dump all registered cell code
                _dump_cell_code(file)

                file.write("\nDEBUG DUMP COMPLETE\n")
                file.write("=" * 60 + "\n\n")
                file.flush()
    except Exception as e:
        # If file write fails, still try stderr
        sys.stderr.write(f"Debug dump error: {e}\n")
        sys.stderr.flush()


_handler_installed = False
_dump_dir: Path = Path(".")


def install_debug_handler(dump_dir: Path | None = None) -> None:
    """Install SIGUSR2 handler for debug dumps.

    Safe to call multiple times - only installs once.

    Args:
        dump_dir: Directory to write debug_dump_<pid>.txt files. Defaults to cwd.
    """
    global _handler_installed, _dump_dir
    # Update _dump_dir before the early-return so callers can redirect the dump
    # location even after the handler is already installed (e.g. second call
    # with a different dump_dir).  Handler re-installation is intentionally
    # skipped; only the output path changes.
    if dump_dir is not None:
        _dump_dir = dump_dir
    if _handler_installed:
        return

    # Enable basic faulthandler for segfaults etc.
    # Best-effort: skip when stderr lacks a real file descriptor (Textual TUI,
    # Jupyter notebooks, or any host that replaces sys.stderr).
    try:
        faulthandler.enable()
    except (RuntimeError, AttributeError, OSError) as exc:
        logger.debug("faulthandler.enable() skipped: %s", exc)

    # Install SIGUSR2 handler (less commonly used than SIGUSR1)
    try:
        sig = getattr(signal, "SIGUSR2", None)
        if sig is not None:
            signal.signal(sig, _debug_signal_handler)
            _handler_installed = True
    except (ValueError, OSError):
        # Can't set signal handler (not main thread, or platform issue)
        pass


def dump_debug_info(file: IO[str] | None = None) -> None:
    """Manually dump debug info (for programmatic use)."""
    if file is None:
        file = sys.stderr
    assert file is not None

    file.write("\n--- Manual debug dump ---\n")
    _dump_pending_llm_calls(file)
    _dump_cell_code(file)


# ---------------------------------------------------------------------------
# Event-driven activity tracking (LLMCallStart / LLMCallEnd "on" hooks)
# ---------------------------------------------------------------------------


def attach_activity_tracking(event_manager: Any) -> Callable[[], None]:
    """Subscribe to ``LLMCallStart`` / ``LLMCallEnd`` on *event_manager* so
    ``get_activity()`` reflects in-flight LLM calls without relying on the
    unifiedllm context manager.

    These are ``on()`` (observer) hooks — fire-and-forget, after the event is
    emitted. Each ``LLMCallStart`` registers a pending call keyed by
    ``(generation_id, turn_number)``; the matching ``LLMCallEnd`` unregisters
    it (whether the call succeeded or raised).

    Returns an unsubscribe callable that removes both handlers and clears any
    calls this tracker still has registered.
    """
    # generation_id + turn_number → registry call_id from register_llm_call.
    live: dict[tuple[str, int], str] = {}

    def _key(event: Any) -> tuple[str, int]:
        return (getattr(event, "generation_id", ""), getattr(event, "turn_number", 0))

    def _on_start(event: Any) -> None:
        call_id = register_llm_call(
            model=getattr(event, "strategy", "") or "llm",
            endpoint=getattr(event, "method_name", None),
            generation_id=getattr(event, "generation_id", ""),
            turn_number=getattr(event, "turn_number", 0),
        )
        live[_key(event)] = call_id

    def _on_end(event: Any) -> None:
        call_id = live.pop(_key(event), None)
        if call_id is not None:
            unregister_llm_call(call_id)

    unsub_start = event_manager.on("LLMCallStart", _on_start)
    unsub_end = event_manager.on("LLMCallEnd", _on_end)

    def _unsubscribe() -> None:
        unsub_start()
        unsub_end()
        for call_id in live.values():
            unregister_llm_call(call_id)
        live.clear()

    return _unsubscribe
