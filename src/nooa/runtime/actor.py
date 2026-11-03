# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Actor runtime: serialized execution with signal queue."""

import ast
import asyncio
import contextvars
import inspect
import io
import linecache
import logging
import math
import re as _re
import tokenize
import types
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast, get_type_hints
from uuid import uuid4

from pydantic import BaseModel

from nooa.agentdoc import FileBackedTruncatingStringIO, TruncatingStringIO
from nooa.agentdoc.introspect import methods, variables
from nooa.context_blocks import (
    DynamicContext,
    ResolvedBlock,
    render_context,
)
from nooa.context_blocks.scoped import _scoped_blocks_var, _scoped_events_var

if TYPE_CHECKING:
    from nooa.config.truncation_config import TruncationConfig
    from nooa.context_blocks.models import ContextWindowStats
    from nooa.runtime.event_query import EventQuery
    from nooa.runtime.restrictions import RestrictionsConfig

from nooa.events import (
    ExecutionResult,
    ExecutionSignal,
    LLMCallEnd,
    LLMCallStart,
    LLMComplete,
    LLMOutput,
    SystemPrompt,
)
from nooa.runtime.context_vars import (
    _current_event_format_var,
    _in_exec_middleware,
    _in_generation_session,
    _parent_agent_var,
)
from nooa.runtime.harness_metrics import (
    HarnessMetrics,
    get_harness_metrics,
    restore_harness_metrics,
    start_harness_metrics,
)
from nooa.runtime.hooks import call_after_hook, call_before_hook

logger = logging.getLogger(__name__)


@contextmanager
def _harness_metrics_lifecycle(should_trace: bool):
    """Start harness metrics + unifiedllm bridge, flush/restore on exit.

    Yields hm (a HarnessMetrics instance, or None if should_trace is False).
    """
    _hm, _prev_hm = start_harness_metrics() if should_trace else (None, None)
    _llm_cb_token = None
    if _hm is not None:
        try:
            from nooa.unifiedllm.unifiedllm import _llm_metrics_callback

            _llm_cb_token = _llm_metrics_callback.set(_make_llm_metrics_bridge(_hm))
        except ImportError:
            pass
    try:
        yield _hm
    finally:
        if _hm is not None:
            try:
                _hm.flush_to_span()
            finally:
                if _prev_hm is not None:
                    restore_harness_metrics(_prev_hm)
        if _llm_cb_token is not None:
            try:
                from nooa.unifiedllm.unifiedllm import _llm_metrics_callback

                _llm_metrics_callback.reset(_llm_cb_token)
            except ImportError:
                pass


def _make_llm_metrics_bridge(hm: "HarnessMetrics") -> Callable[[str, Any], None]:
    """Create a callback that bridges unifiedllm metric events to HarnessMetrics."""
    from nooa.runtime.token_usage import accumulate_tokens

    def _handle_token_usage(usage: Any) -> None:
        if isinstance(usage, dict):
            accumulate_tokens(
                input_tokens=usage.get("prompt_tokens", 0) or 0,
                output_tokens=usage.get("completion_tokens", 0) or 0,
            )

    _dispatch: dict[str, Callable[[Any], None]] = {
        "think_tag_extracted": lambda _: hm.record_think_tag_extracted(),
        "malformed_think_tag_fixed": lambda _: hm.record_malformed_think_tag(),
        "json_fence_removed": lambda _: hm.record_json_fence_removed(),
        "json_control_chars_removed": lambda _: hm.record_json_control_chars_removed(),
        "json_escape_fixed": lambda _: hm.record_json_escape_fixed(),
        "json_nested_extraction": lambda _: hm.record_json_nested_extraction(),
        "json_double_decoded": lambda _: hm.record_json_double_decoded(),
        "reasoning_as_structured_output": lambda _: hm.record_reasoning_as_structured_output(),
        "token_usage": _handle_token_usage,
    }

    def bridge(event: str, detail: Any = None) -> None:
        handler = _dispatch.get(event)
        if handler:
            handler(detail)
        else:
            logger.debug("[HARNESS_METRICS] Unknown LLM metric event: %s", event)

    return bridge


# Suppress SyntaxWarning for invalid escape sequences (e.g. '\s' instead of r'\s')
# in LLM-generated code.  The string value is identical either way, so the warning
# is pure noise — and showing it to the LLM just wastes a turn trying to "fix" it.
# Other warnings (RuntimeWarning, DeprecationWarning, …) are unaffected.
warnings.filterwarnings(
    "ignore",
    message=r"invalid escape sequence",
    category=SyntaxWarning,
)


_TRAILING_CONTEXT_RE = _re.compile(r"^(.*?)(<context>.*?</context>)\s*\Z", _re.DOTALL)


def _extract_trailing_context_envelope(messages: list[dict[str, Any]]) -> str:
    """Pull the trailing ``<context>…</context>`` envelope from messages.

    ``CachedBlockFormatter`` emits dynamic SYSTEM-role blocks as a
    trailing ``role: "user"`` message that wraps them in a ``<context>``
    envelope (see ``context_blocks/renderers/cached.py``). This helper
    extracts that envelope so observability consumers can record
    per-turn dynamic-block state on the corresponding agent step.

    Returns the envelope (including the ``<context>…</context>`` tags)
    when the last message is a user message whose content ends in such
    an envelope; ``""`` otherwise. Idempotent if the envelope is missing
    (e.g. agents without dynamic blocks).
    """
    if not messages:
        return ""
    last = messages[-1]
    if not isinstance(last, dict) or last.get("role") != "user":
        return ""
    content = last.get("content", "")
    if not isinstance(content, str):
        return ""
    m = _TRAILING_CONTEXT_RE.match(content)
    return m.group(2) if m is not None else ""


def _snapshot_llm_request(
    event_manager: Any, messages: list[dict[str, Any]], generation_id: str
) -> str:
    """Snapshot the rendered request for observability consumers (e.g. ATIF).

    Emits a :class:`SystemPrompt` event carrying ``messages[0].content``
    and returns the trailing ``<context>…</context>`` envelope from the
    same list. Called right after ``_build_messages`` so both reflect the
    exact bytes about to be sent to the LLM; the returned envelope is
    stamped onto the matching :class:`LLMComplete`. ``record=False`` keeps
    the snapshot out of the LLM-visible event timeline.
    """
    if messages and isinstance(messages[0], dict) and messages[0].get("role") == "system":
        content = messages[0].get("content", "")
        if not isinstance(content, str):
            content = str(content)
        event_manager.add(
            SystemPrompt(content=content, generation_id=generation_id),
            record=False,
        )
    return _extract_trailing_context_envelope(messages)


def _resolve_provider_formatter(llm_client: Any, default_formatter: Any) -> Any:
    """Auto-select provider formatter based on LLM client type.

    ResponsesClient needs ResponsesProviderFormatter to emit native Responses
    API wire format.  All other clients (including Anthropic via LiteLLM) use
    the agent's configured formatter.

    This is intentionally runtime-dispatched rather than config-driven:
    Anthropic formatting is handled by LiteLLM (so OpenAIProviderFormatter
    works), but the Responses API has a fundamentally different wire shape
    that LiteLLM does not translate, requiring its own formatter.
    """
    from nooa.unifiedllm import ResponsesClient

    if isinstance(llm_client, ResponsesClient):
        from nooa.context_blocks.formatter import ResponsesProviderFormatter

        return ResponsesProviderFormatter()
    return default_formatter


# ---------------------------------------------------------------------------
# Shared collapse/archival helpers (L4 boundary + error-driven archival)
# ---------------------------------------------------------------------------

# When the LLM API returns a context-window error, archive events until utilization
# drops to this fraction of the token budget cap. Lower = more headroom before re-triggering.
_ARCHIVE_TARGET_UTILIZATION = 0.60

# Cold-start chars→tokens ratio (~4 chars per token) used before any provider
# response has calibrated _tokens_per_char from real usage.
_DEFAULT_TOKENS_PER_CHAR = 0.25


def _collapse_oldest(
    event_manager: Any,
    active_tags: list[str],
    target: int,
    summary_text: str,
    *,
    log_prefix: str = "collapse",
) -> int:
    """Collapse the *target* oldest active events in one shot.

    Collapses a single contiguous range from the start of active_tags,
    producing one summary event instead of many tiny fragments.

    Returns the number of events actually archived.
    """
    if target <= 0:
        return 0
    n = min(target, len(active_tags))
    if n <= 0:
        return 0
    try:
        event_manager.collapse(active_tags[0], active_tags[n - 1], summary_text=summary_text)
        return n
    except Exception as exc:  # noqa: BLE001
        logger.debug(
            "%s failed for %s..%s: %s", log_prefix, active_tags[0], active_tags[n - 1], exc
        )
        return 0


# ---------------------------------------------------------------------------
# ContextWindowExceededError recovery helpers
# ---------------------------------------------------------------------------

_MIN_RECOVERY_OUTPUT_TOKENS = 1024


_PROMPT_TOKENS_RE = _re.compile(
    r"(?:prompt|request)[^0-9]*"
    # "contains/has N tokens" (OpenAI/NVIDIA) OR
    # "is too long: N tokens" (Anthropic / Azure-Anthropic gateway)
    r"(?:contains?|has|is\s+too\s+long\s*:?)\s+(?:at\s+least\s+)?(\d[\d,]*)\s*(?:input\s+)?tokens",
    _re.IGNORECASE,
)

_CONTEXT_WINDOW_TOKENS_RE = _re.compile(
    r"maximum\s+context\s+length\s+is\s+(\d[\d,]*)\s+tokens"
    r"|context\s+window\s+of\s+(?:this\s+model\s+is\s+)?(\d[\d,]*)\s+tokens"
    r"|(\d[\d,]*)\s+tokens\s*>\s*(\d[\d,]*)\s+maximum",
    _re.IGNORECASE,
)


def _is_context_window_error(exc: BaseException) -> bool:
    """Recognize context-window errors using typed and provider-normalized signals."""
    try:
        from litellm.exceptions import ContextWindowExceededError
    except Exception:  # pragma: no cover - litellm is optional at import time
        ContextWindowExceededError = None  # type: ignore[assignment]

    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if ContextWindowExceededError is not None and isinstance(cur, ContextWindowExceededError):
            return True

        name = type(cur).__name__.lower()
        msg = str(cur).lower()
        if (
            "contextwindowexceeded" in name
            or "context length" in msg
            or "context_length_exceeded" in msg
            or "context window" in msg
            # Anthropic / Azure-Anthropic gateway phrasing, surfaced as a
            # BadRequestError rather than a typed ContextWindowExceededError:
            #   "prompt is too long: 1017198 tokens > 1000000 maximum"
            or "prompt is too long" in msg
        ):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _parse_prompt_tokens(exc: BaseException) -> int | None:
    """Extract prompt token count from a ContextWindowExceededError chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        m = _PROMPT_TOKENS_RE.search(str(cur))
        if m:
            return int(m.group(1).replace(",", ""))
        cur = cur.__cause__ or cur.__context__
    return None


def _parse_context_window_tokens(exc: BaseException) -> int | None:
    """Extract model context-window size from a ContextWindowExceededError chain."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        m = _CONTEXT_WINDOW_TOKENS_RE.search(str(cur))
        if m:
            # Alternatives capture either the window directly or "input > maximum".
            for group in reversed(m.groups()):
                if group:
                    return int(group.replace(",", ""))
        cur = cur.__cause__ or cur.__context__
    return None


def _context_window_for_error(llm_client: Any, exc: BaseException) -> int | None:
    """Return the provider-reported context window, falling back to the client."""
    reported = _parse_context_window_tokens(exc)
    if reported:
        return reported
    return getattr(llm_client, "context_window", None)


def _compute_reduced_max_tokens(
    exc: BaseException,
    ctx_window: int | None,
    original_max_tokens: int | None,
) -> int | None:
    """Compute reduced max_tokens for context window recovery.

    Returns the reduced value, or None if recovery is not possible.
    """
    prompt_tok = _parse_prompt_tokens(exc)
    if ctx_window and prompt_tok:
        margin = max(int(ctx_window * 0.02), 256)
        reduced = ctx_window - prompt_tok - margin
    elif ctx_window and original_max_tokens:
        reduced = original_max_tokens // 2
    else:
        return None
    if reduced < _MIN_RECOVERY_OUTPUT_TOKENS:
        return None
    return reduced


def _strip_blocked_modules(
    exec_globals: dict[str, Any],
    blocked_modules: frozenset[str],
) -> dict[str, Any]:
    """Remove blocked modules and their members from exec_globals."""
    from nooa.runtime.restrictions import is_from_blocked_module

    if not blocked_modules:
        return exec_globals
    filtered = {
        name: obj
        for name, obj in exec_globals.items()
        if not is_from_blocked_module(obj, blocked_modules)
    }
    removed_count = len(exec_globals) - len(filtered)
    if removed_count > 0:
        hm = get_harness_metrics()
        removed = set(exec_globals) - set(filtered)
        for name in removed:
            hm.blocked_module_removed(name)
    return filtered


# Context variables for current generation context (per-task, not per-runtime)
# This allows parallel nested calls via asyncio.gather to work correctly
_current_call_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_call", default=None
)
_current_method_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_method", default=None
)
_current_llm_var: contextvars.ContextVar[Any] = contextvars.ContextVar("current_llm", default=None)

# Context variable for the resolved truncation config for the current generation call.
# Set by _execute_with_generation() so that method-level @strategy(truncation=...)
# overrides are visible to runtime.truncation_config throughout strategy execution.
_current_truncation_config_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_truncation_config", default=None
)

# Context variable for current strategy being executed
_current_strategy_var: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "current_strategy", default=None
)

# Context variable for inherited decorator context.
# Set by _execute_with_generation() so that @strategy(context={...})
# blocks propagate to nested method calls on the same agent.
# Read by _prepare_context() and passed explicitly to build_context().
_decorator_context_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "decorator_context", default=None
)

# Context variable for inherited decorator event query.
# Set by _execute_with_generation() so that @strategy(ScopedContext(events=...))
# event filtering propagates to nested method calls on the same agent.
# Read by _prepare_context() and passed explicitly to build_context().
_decorator_events_var: contextvars.ContextVar["EventQuery | None"] = contextvars.ContextVar(
    "decorator_events", default=None
)

# Task-local stdout/stderr buffers for async-safe capture
# Contextvars are per-task, so parallel async executions get isolated buffers
_stdout_buffer_var: contextvars.ContextVar[io.StringIO | TruncatingStringIO | None] = (
    contextvars.ContextVar("stdout_buffer", default=None)
)
_stderr_buffer_var: contextvars.ContextVar[io.StringIO | TruncatingStringIO | None] = (
    contextvars.ContextVar("stderr_buffer", default=None)
)

# Re-export image buffer ContextVar so it's set alongside stdout/stderr
# Agent call stack - defined in context_vars.py to avoid circular imports.
from nooa.runtime.context_vars import _get_agent_call_stack  # noqa: E402
from nooa.runtime.media_capture import (  # noqa: E402
    _media_buffer_var,
    _MediaBuffer,
)
from nooa.runtime.stream_wrappers import (  # noqa: E402
    BlockedStdinWrapper,
    ContextVarStream,
    _block_stdin_var,
)

# Task-local stack for generation ID tracking.
# Uses immutable tuples for parallel task isolation (same pattern as agent call stack).
_generation_id_stack_var: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "generation_id_stack", default=()
)


def _get_generation_id_stack() -> tuple[str, ...]:
    """Get the generation ID stack for the current async context.

    Returns an immutable tuple - use _push_generation_id() and _pop_generation_id()
    for stack operations.
    """
    return _generation_id_stack_var.get()


def _push_generation_id(generation_id: str) -> None:
    """Push a generation_id to the stack.

    Creates a new immutable tuple with the added value. This ensures parallel
    tasks (via asyncio.gather) each have their own isolated stack - pushing
    in one task doesn't affect sibling tasks.
    """
    current = _generation_id_stack_var.get()
    _generation_id_stack_var.set((*current, generation_id))


def _pop_generation_id() -> str | None:
    """Pop from the generation ID stack.

    Creates a new immutable tuple without the last element.
    Returns the popped value or None if stack is empty.
    """
    current = _generation_id_stack_var.get()
    if not current:
        return None
    popped = current[-1]
    _generation_id_stack_var.set(current[:-1])
    return popped


def _make_capture_buffer(
    limit: int, tail_chars: int | None, file_backed: bool
) -> TruncatingStringIO:
    """Create a stdout/stderr capture buffer, optionally file-backed."""
    if file_backed:
        return FileBackedTruncatingStringIO(limit=limit, tail_chars=tail_chars)
    return TruncatingStringIO(limit=limit, tail_chars=tail_chars)


def _extract_captured_locals(exec_globals: dict[str, Any]) -> dict[str, Any]:
    """Extract captured locals from __repl_captured_locals__, filtering out bound methods.

    The wrapper's finally block captures locals into exec_globals["__repl_captured_locals__"].
    This function extracts them, filtering out bound methods. Modules are preserved
    so that `import X` in one cell persists to subsequent cells.

    Args:
        exec_globals: The globals dict from code execution

    Returns:
        Dict of captured local variables (filtered)
    """
    captured: dict[str, Any] = {}
    raw_captured = exec_globals.get("__repl_captured_locals__", {})
    for k, v in raw_captured.items():
        # Skip callables that are helper methods (they're handled separately)
        if callable(v) and hasattr(v, "__self__"):
            continue  # Skip bound methods
        captured[k] = v
    return captured


class _TopLevelReturnFinder(ast.NodeVisitor):
    """Find return statements at the top level, not inside nested functions/classes.

    This avoids false positives where a return inside a nested function definition
    would incorrectly prevent implicit return transformation at the top level.
    """

    def __init__(self) -> None:
        self.found = False

    def visit_Return(self, node: ast.Return) -> None:
        self.found = True

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Do not recurse into nested function definitions
        pass

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        # Do not recurse into nested async function definitions
        pass

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Do not recurse into class definitions
        pass


def _has_top_level_return(tree: ast.Module) -> bool:
    """Check if the AST has any explicit return statements at the top level.

    Does not recurse into nested function/class definitions, only checks
    top-level statements and their non-function/class children.

    Args:
        tree: Parsed AST module

    Returns:
        True if there's an explicit return at the top level
    """
    finder = _TopLevelReturnFinder()
    for node in tree.body:
        finder.visit(node)
        if finder.found:
            return True
    return False


class ActorRuntime:
    """
    Actor runtime for serialized execution.

    Features:
    - One generation session at a time per agent (serialized)
    - Task introspection

    Usage:
        agent = MyAgent(llm=client)
        # Runtime is automatically created as agent.runtime
    """

    def __init__(self, agent: Any):
        """
        Initialize actor runtime for an agent.

        Args:
            agent: Agent instance to manage

        For callbacks on LLM output, use:
            agent.event_manager.on("Message", handler)
            agent.event_manager.on("LLMOutput", handler)
        """
        # Agent instance
        self.agent: Any = agent

        # Lock for generation serialization (only one LLM generation session at a time)
        self._generation_lock = asyncio.Lock()
        self._generation_lock_loop: asyncio.AbstractEventLoop | None = None

        # NOTE: Current generation context uses context variables (not instance attributes)
        # to support parallel nested calls via asyncio.gather.
        # See: _current_call_var, _current_method_var

        # NOTE: Agent call stack and generation ID stack are now tracked via context
        # variables (_get_agent_call_stack(), _get_generation_id_stack()) to prevent
        # context leakage during parallel async execution.

        # render_context for prompt generation (dead simple: takes pre-resolved blocks)

        # Most recent context window utilization stats (updated each _build_messages call).
        # NOTE: This is an instance attribute, not a ContextVar. Under concurrent
        # asyncio.gather on the same agent, the last write wins. For per-task
        # isolation, read stats from the on_messages_built hook's context_stats kwarg.
        self._last_context_stats: ContextWindowStats | None = None
        self._last_prompt_tokens_actual: int | None = None
        self._event_format_cache: dict[tuple[tuple[str, Any], ...], Any] = {}
        self._event_format_cache_max_entries = 32
        # Chars→tokens ratio, calibrated from the last provider response
        # (prompt_tokens / total_chars). Every token estimate in the system is
        # chars × this ratio, anchored to the real provider count. Defaults to
        # the ~4-chars-per-token heuristic before the first response.
        self._tokens_per_char: float = _DEFAULT_TOKENS_PER_CHAR

    def _event_format_for_event(self, event: Any) -> Any:
        """Return the FormatConfig to use when serializing an event.

        Events produced during a generation method carry a persisted
        ``metadata["truncation_event_format"]`` written by ``EventManager.add()``
        from the active ``_current_event_format_var``. Reading that metadata here
        keeps historical events stable across session resume. Events without
        persisted format metadata fall back to the agent-level event format.
        """
        metadata = getattr(event, "metadata", {})
        event_format = metadata.get("truncation_event_format")
        if event_format is None:
            return self.agent._truncation.event_format
        if not isinstance(event_format, dict):
            from nooa.config.truncation_config import FormatConfig

            return FormatConfig.model_validate(event_format)
        cache_key = tuple(sorted(event_format.items()))
        cached = self._event_format_cache.get(cache_key)
        if cached is not None:
            return cached
        from nooa.config.truncation_config import FormatConfig

        resolved = FormatConfig.model_validate(event_format)
        if len(self._event_format_cache) >= self._event_format_cache_max_entries:
            self._event_format_cache.pop(next(iter(self._event_format_cache)))
        self._event_format_cache[cache_key] = resolved
        return resolved

    def _ensure_generation_lock_on_current_loop(self) -> None:
        """Recreate _generation_lock if the event loop changed (gl-212).

        asyncio.Lock binds to the first loop that contends it. In the TUI,
        the agent loop can be recreated after a crash or ESC-cancel, leaving
        the lock bound to the dead loop. Recreating it avoids RuntimeError.
        """
        loop = asyncio.get_running_loop()
        if self._generation_lock_loop is None:
            self._generation_lock_loop = loop
        elif self._generation_lock_loop is not loop:
            self._generation_lock = asyncio.Lock()
            self._generation_lock_loop = loop

    def _archive_on_context_error(
        self,
        ctx_window: int | None,
        exc: BaseException | None = None,
    ) -> None:
        """Archive oldest events after a ContextWindowExceededError from the API.

        Only called when the LLM actually rejects the request as too large.
        Uses the last context stats to estimate how many events to shed.
        """
        stats = self._last_context_stats
        if not stats or not ctx_window:
            return
        active_tags = list(self.event_manager.keys())
        n_active = len(active_tags)
        if n_active == 0:
            return
        # Shed against the REAL current prompt size vs a REAL 70% budget of the
        # USABLE window (model window minus the output-token reserve — the
        # provider rejected prompt + completion budget, so headroom must
        # account for the completion too).
        # After successful calls, stats.total_tokens is provider usage. On a
        # COLD-START context error (the first call IS the overflow, so no prior
        # response has reported usage), stats.total_tokens is None — prefer the
        # authoritative token count the error itself reports; if the error text
        # omits a parseable count, fall back to chars × the calibrated ratio
        # (the same chars→tokens estimate used everywhere else) to size the shed.
        usable_window = max(1, ctx_window - (stats.reserved_output_tokens or 0))
        cap = int(usable_window * 0.70)
        target_tok = int(cap * _ARCHIVE_TARGET_UTILIZATION)
        # Char-based estimates used only when no provider token count exists.
        events_tok = stats.events_tokens
        if events_tok is None:
            events_tok = round(stats.events_chars * self._tokens_per_char)
        total_tok = stats.total_tokens
        if total_tok is None:
            total_tok = round(
                (stats.context_blocks_chars + stats.events_chars) * self._tokens_per_char
            )
        if exc is not None:
            reported = _parse_prompt_tokens(exc)
            if reported and reported > total_tok:
                total_tok = reported
            elif reported is None and total_tok <= target_tok:
                # The provider rejected the request, so the prompt exceeded the
                # true window even if our estimate is still under target and the
                # error text omits a parseable token count. Shed at least one
                # average event instead of no-op'ing and retrying the same prompt.
                avg_event_guess = math.ceil(max(1, events_tok) / max(1, n_active))
                total_tok = target_tok + max(1, avg_event_guess)
        tokens_to_shed = max(0, total_tok - target_tok)
        if tokens_to_shed == 0:
            return
        # Only event tokens are archiveable. total_tok may include fixed
        # tool-schema tokens, and archiving events cannot remove those; using
        # total_tok - context_blocks here would overestimate per-event size and
        # under-archive for tool-heavy agents. If attribution says events are
        # zero-token but active events exist (e.g. structured ToolCallEvents
        # rendered with content=""), fall back to a prompt-wide average so one
        # context error does not collapse the entire active history.
        if events_tok > 0:
            avg_event_tok = events_tok / max(1, n_active)
        else:
            avg_event_tok = total_tok / max(1, n_active)
        n_to_archive = min(
            int(math.ceil(tokens_to_shed / max(1, avg_event_tok))),
            n_active,
        )
        if n_to_archive > 0:
            summary_text = (
                f"context-window API error: total_tokens={total_tok:,}, "
                f"cap={cap:,}, archiving {n_to_archive} events to reach "
                f"{_ARCHIVE_TARGET_UTILIZATION:.0%} utilization"
            )
            _collapse_oldest(
                self.event_manager,
                active_tags,
                n_to_archive,
                summary_text,
                log_prefix="context-error-archival",
            )
            hm = get_harness_metrics()
            if isinstance(hm, HarnessMetrics):
                hm.context_limits_events_collapsed += n_to_archive
                hm.context_limits_tokens_archived += int(n_to_archive * avg_event_tok)

    @property
    def _agent_call_stack(self) -> tuple[str | None, ...]:
        """Agent call stack for method call tracking (per async context).

        Returns an immutable tuple. Use _push_agent_call_id() and
        _pop_agent_call_id() for stack operations.
        """
        return _get_agent_call_stack()

    @property
    def _generation_id_stack(self) -> tuple[str, ...]:
        """Generation ID stack for nested generation tracking (per async context).

        Returns an immutable tuple. Use _push_generation_id() and
        _pop_generation_id() for stack operations.
        """
        return _get_generation_id_stack()

    @property
    def _agent_call_id(self) -> str | None:
        """
        Current agent call ID (method invocation ID).

        Returns top of agent_call_stack if method is active, otherwise None.
        This ID is unique per method invocation.
        """
        stack = _get_agent_call_stack()
        return stack[-1] if stack else None

    @property
    def _current_call(self) -> Any:
        """Current call context (from context variable for concurrent task support)."""
        return _current_call_var.get()

    @property
    def current_call(self) -> Any:
        """Current call context.

        Provides access to the current method invocation's call object (a
        ``CurrentCall``), which exposes ``id``, arguments, and other execution
        metadata. Useful for filtering events, tracking execution, or accessing
        call-specific information.

        Example:
            # In DynamicContext expressions
            @strategy(
                CodeActStrategy(),
                ScopedContext(events={
                    "history": DynamicContext(
                        "self.runtime.event_manager.filter(call_id=self.runtime.current_call.id)"
                    )
                })
            )
            async def my_method(self):
                ...  # Only sees events from this method's call

            # Or directly in code
            current_id = self.runtime.current_call.id
        """
        return _current_call_var.get()

    @property
    def last_prompt_tokens_actual(self) -> int | None:
        """Provider-reported prompt token count from the most recent generation.

        Read-only public accessor for the actual (API-reported) prompt token
        count recorded after the last LLM call, or ``None`` if no provider count
        is available yet (e.g. before the first response). Used by budget-based
        summarizers (e.g. ``TokenBudgetSummarizer``).

        NOTE: This is a plain instance attribute, not a ContextVar. Under
        concurrent ``asyncio.gather`` on the same agent the last write wins; it is
        intended for coarse budget checks, not per-task accounting.
        """
        return self._last_prompt_tokens_actual

    @property
    def _current_method(self) -> Any:
        """Current method being generated (from context variable for concurrent task support)."""
        return _current_method_var.get()

    # --- RuntimeServices Protocol Implementation ---

    @property
    def event_manager(self) -> Any:
        """Event manager for conversation management (RuntimeServices protocol)."""
        return self.agent.event_manager

    @property
    def truncation_config(self) -> "TruncationConfig":
        """Truncation configuration for the current generation call (RuntimeServices protocol).

        Returns the method-level override (set by @strategy(truncation=...)) if active,
        otherwise falls back to the agent-level config.
        """
        override = _current_truncation_config_var.get()
        if override is not None:
            return override
        return self.agent._truncation

    async def generate(
        self,
        *,
        tools: list[dict[str, Any]] | None = None,
        output_model: type | None = None,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """Build messages from context + events, call LLM (RuntimeServices protocol).

        Builds messages using prompt builder from:
        - System message: context blocks + strategy.strategy_prompt
        - Events: conversation events

        Creates an LLMOutput event and adds it to event manager.

        Args:
            tools: Optional list of tool definitions.
            output_model: Optional Pydantic model for structured output.
            **kwargs: Additional LLM options.

        Returns:
            Tuple of (LLMResponse, event_id) where:
            - LLMResponse from unifiedllm with content, reasoning, usage
            - event_id can be used for event_manager.update() or event_manager.get()
        """
        if self._current_method is None:
            raise RuntimeError("generate() called with no current method context")

        # Build messages from context + events (timers inside _build_messages).
        # No proactive clamping — recovery is error-driven. If the API rejects
        # with ContextWindowExceededError, the except handler archives events,
        # rebuilds messages, and retries.
        messages = await self._build_messages(
            self._current_method,
            call_args=self._current_call.args if self._current_call else (),
            call_kwargs=self._current_call.kwargs if self._current_call else {},
            tools=tools,
            max_output_tokens=kwargs.get("max_tokens"),
        )
        _gen_hm = get_harness_metrics()

        # Get LLM client from context var (set by _execute_with_generation)
        llm_client = _current_llm_var.get()
        if llm_client is None:
            raise RuntimeError("generate() called with no LLM client in context")

        current_generation_id = self._generation_id_stack[-1] if self._generation_id_stack else None

        # Snapshot system prompt + dynamic context from the rendered messages
        # (consumed by the ATIF exporter). Captured here, at render time.
        _dynamic_context = _snapshot_llm_request(
            self.event_manager, messages, current_generation_id or ""
        )

        # --- Middleware: llm_call -------------------------------------------
        # record_turn() first so turn_count reflects THIS turn (1-indexed) —
        # both the lifecycle events below and the metric read the same value.
        _gen_hm.record_turn()
        em = self.event_manager

        # --- Observable LLM-call lifecycle events --------------------------
        # RUNTIME_EVENTs (never recorded) emitted around the acall() round-trip
        # so observers can subscribe via on("LLMCallStart"/"LLMCallEnd") to
        # surface "waiting on model" without inferring it from cell boundaries.
        _llm_strategy = _current_strategy_var.get()
        _llm_event_meta = {
            "method_name": self._current_method.__name__,
            "strategy": type(_llm_strategy).__name__ if _llm_strategy is not None else "default",
            "generation_id": current_generation_id or "",
            "turn_number": _gen_hm.turn_count if isinstance(_gen_hm.turn_count, int) else 0,
        }

        def _emit_llm_start() -> None:
            em.add(LLMCallStart(**_llm_event_meta), record=False)

        def _emit_llm_end(*, success: bool, exception_type: str | None = None) -> None:
            em.add(
                LLMCallEnd(**_llm_event_meta, success=success, exception_type=exception_type),
                record=False,
            )

        has_mw = bool(em._middleware.get("llm_call"))
        with _gen_hm.timer("time_llm_call"):
            if has_mw:
                from nooa.runtime.middleware import LLMCallContext

                params: dict[str, Any] = {**kwargs, "tools": tools}
                if output_model is not None:
                    params["output_model"] = output_model
                ctx = LLMCallContext(
                    messages=messages,
                    params=params,
                    agent=self.agent,
                    runtime=self,
                )

                async def _core_llm(ctx: LLMCallContext) -> LLMCallContext:
                    # Tracing hook fires AFTER middleware pre-processing,
                    # so it sees the final (possibly modified) messages.
                    call_before_hook(
                        "on_messages_built",
                        agent=self.agent,
                        method_name=self._current_method.__name__,
                        messages=ctx.messages,
                        generation_id=current_generation_id or "",
                        context_stats=self._last_context_stats,
                    )
                    om = ctx.params.get("output_model", None)
                    call_params = {k: v for k, v in ctx.params.items() if k != "output_model"}
                    _mw_strategy = _current_strategy_var.get()
                    _mw_strategy_tag = (
                        type(_mw_strategy).__name__ if _mw_strategy is not None else "default"
                    )
                    call_params.setdefault(
                        "prompt_cache_key", f"{self.agent._agent_id}-{_mw_strategy_tag}"
                    )
                    ctx.response = await llm_client.acall(
                        ctx.messages,
                        output_model=om,
                        **call_params,
                    )
                    return ctx

                _emit_llm_start()
                try:
                    try:
                        ctx = await em.run_middleware("llm_call", ctx, _core_llm)
                    except Exception as _cw_exc:
                        if not _is_context_window_error(_cw_exc):
                            raise
                        # Always archive first — even if we can't reduce max_tokens,
                        # shedding events lets the retry (or caller's next attempt) succeed.
                        _ctx_window = _context_window_for_error(llm_client, _cw_exc)
                        self._archive_on_context_error(
                            _ctx_window,
                            exc=_cw_exc,
                        )
                        _reduced = _compute_reduced_max_tokens(
                            _cw_exc,
                            _ctx_window,
                            ctx.params.get("max_tokens"),
                        )
                        if _reduced is None:
                            raise
                        logger.warning(
                            "context-window recovery (middleware): reducing max_tokens %s -> %d",
                            ctx.params.get("max_tokens"),
                            _reduced,
                        )
                        # Re-build messages after archival so the retry sees the
                        # reduced event store.
                        ctx.messages = await self._build_messages(
                            self._current_method,
                            call_args=self._current_call.args if self._current_call else (),
                            call_kwargs=self._current_call.kwargs if self._current_call else {},
                            tools=ctx.params.get("tools"),
                            max_output_tokens=_reduced,
                        )
                        _dynamic_context = _snapshot_llm_request(
                            self.event_manager, ctx.messages, current_generation_id or ""
                        )
                        ctx.params["max_tokens"] = _reduced
                        ctx = await em.run_middleware("llm_call", ctx, _core_llm)
                except Exception as _exc:
                    _emit_llm_end(success=False, exception_type=type(_exc).__name__)
                    raise
                _emit_llm_end(success=True)
                response = ctx.response
                if response is None:
                    raise RuntimeError(
                        "llm_call middleware returned without setting ctx.response. "
                        "Short-circuiting middleware must set ctx.response before returning."
                    )
            else:
                # Fast path — no middleware registered
                call_before_hook(
                    "on_messages_built",
                    agent=self.agent,
                    method_name=self._current_method.__name__,
                    messages=messages,
                    generation_id=current_generation_id or "",
                    context_stats=self._last_context_stats,
                )
                # Default prompt_cache_key to per-(agent, strategy) so:
                #   - Parallel agents (same prefix) get distinct shards by
                #     agent_id, avoiding the cybergym multi-trial collision
                #     above the ~15 req/min per-shard ceiling.
                #   - Within one agent, calls that use different strategies
                #     (e.g. CodeAct vs Predict) get separate shards because
                #     their cacheable prefixes diverge anyway — preventing
                #     them from compounding load on a shared shard.
                #   - Same-strategy methods on one agent stay on one shard,
                #     keeping shared-prefix caching (system prompt + strategy
                #     prompt + execution_context + agent doc all match).
                _kwargs = dict(kwargs)
                _current_strategy = _current_strategy_var.get()
                _strategy_tag = (
                    type(_current_strategy).__name__ if _current_strategy is not None else "default"
                )
                _kwargs.setdefault("prompt_cache_key", f"{self.agent._agent_id}-{_strategy_tag}")
                _emit_llm_start()
                try:
                    try:
                        response = await llm_client.acall(
                            messages,
                            tools=tools,
                            output_model=output_model,
                            **_kwargs,
                        )
                    except Exception as _cw_exc:
                        if not _is_context_window_error(_cw_exc):
                            raise
                        # Always archive first — even if we can't reduce max_tokens,
                        # shedding events lets the retry (or caller's next attempt) succeed.
                        _ctx_window = _context_window_for_error(llm_client, _cw_exc)
                        self._archive_on_context_error(
                            _ctx_window,
                            exc=_cw_exc,
                        )
                        _reduced = _compute_reduced_max_tokens(
                            _cw_exc,
                            _ctx_window,
                            kwargs.get("max_tokens"),
                        )
                        if _reduced is None:
                            # Can't compute a reduced max_tokens, but archival already ran.
                            # Re-raise so the caller (e.g. CodeAct) retries with fresh messages.
                            raise
                        logger.warning(
                            "context-window recovery: reducing max_tokens %s -> %d "
                            "(prompt=%s, ctx_window=%s)",
                            kwargs.get("max_tokens"),
                            _reduced,
                            _parse_prompt_tokens(_cw_exc),
                            _ctx_window,
                        )
                        # Re-build messages after archival so the retry sees the
                        # reduced event store. Retrying with the same messages would
                        # fail again when input tokens exceed the context window.
                        messages = await self._build_messages(
                            self._current_method,
                            call_args=self._current_call.args if self._current_call else (),
                            call_kwargs=self._current_call.kwargs if self._current_call else {},
                            tools=tools,
                            max_output_tokens=_reduced,
                        )
                        _dynamic_context = _snapshot_llm_request(
                            self.event_manager, messages, current_generation_id or ""
                        )
                        # Reuse _kwargs (already has the per-(agent, strategy) key set)
                        # so recovery lands on the same shard as the original attempt.
                        _recovery_kw = {**_kwargs, "max_tokens": _reduced}
                        response = await llm_client.acall(
                            messages,
                            tools=tools,
                            output_model=output_model,
                            **_recovery_kw,
                        )
                except Exception as _exc:
                    _emit_llm_end(success=False, exception_type=type(_exc).__name__)
                    raise
                _emit_llm_end(success=True)

        # Emit LLMComplete BEFORE LLMOutput so subscribers that build per-turn
        # records (e.g. the ATIF exporter) populate metrics/tool_calls before
        # the assistant-message content arrives. record=False keeps this off
        # the LLM-visible event timeline (Role.RUNTIME_EVENT) but on()
        # subscribers still receive it.
        _model_name = getattr(llm_client, "model", "") or ""
        usage = getattr(response, "usage", None)
        # Normalize usage to a dict regardless of whether the provider returned
        # a dict, a Pydantic model with attributes, or nothing at all. The
        # token-calibration logic above already grovels through both shapes;
        # mirror that here so LLMComplete metrics don't silently zero out.
        _usage_raw = usage if usage is not None else getattr(response, "usage", None)
        _usage_dict: dict[str, Any] = {}
        if isinstance(_usage_raw, dict):
            _usage_dict = _usage_raw
        elif _usage_raw is not None:
            for _key in (
                "prompt_tokens",
                "input_tokens",
                "completion_tokens",
                "output_tokens",
                "cached_tokens",
                "cache_read_input_tokens",
                "reasoning_tokens",
                "cost",
                "cost_usd",
            ):
                _val = getattr(_usage_raw, _key, None)
                if _val is not None:
                    _usage_dict[_key] = _val
            _prompt_details = getattr(_usage_raw, "prompt_tokens_details", None)
            if _prompt_details is not None:
                _usage_dict["prompt_tokens_details"] = (
                    _prompt_details
                    if isinstance(_prompt_details, dict)
                    else {
                        "cached_tokens": getattr(_prompt_details, "cached_tokens", None),
                    }
                )
            _completion_details = getattr(_usage_raw, "completion_tokens_details", None)
            if _completion_details is not None:
                _usage_dict["completion_tokens_details"] = (
                    _completion_details
                    if isinstance(_completion_details, dict)
                    else {
                        "reasoning_tokens": getattr(_completion_details, "reasoning_tokens", None),
                    }
                )
        _prompt_tokens = int(
            _usage_dict.get("prompt_tokens") or _usage_dict.get("input_tokens") or 0
        )
        if _prompt_tokens > 0:
            self._last_prompt_tokens_actual = _prompt_tokens
            if self._last_context_stats is not None:
                # The provider's exact prompt-token count is the single source of
                # truth for ctx% display, summarization triggers, and archive
                # sizing. render_context leaves prompt_tokens=None (no local
                # estimate); we write the authoritative value back here.
                stats = self._last_context_stats
                self._last_context_stats = stats.model_copy(
                    update={"prompt_tokens": _prompt_tokens}
                )
                # Recalibrate the chars→tokens ratio from this real response:
                # tokens_per_char = prompt_tokens / total_chars. The next
                # render's eviction sizing uses it instead of a fixed heuristic
                # or the litellm tokenizer.
                total_chars = stats.context_blocks_chars + stats.events_chars
                if total_chars > 0:
                    self._tokens_per_char = _prompt_tokens / total_chars
        _completion_tokens = int(
            _usage_dict.get("completion_tokens") or _usage_dict.get("output_tokens") or 0
        )
        _cached_tokens = int(
            _usage_dict.get("cached_tokens")
            or _usage_dict.get("cache_read_input_tokens")
            or (_usage_dict.get("prompt_tokens_details") or {}).get("cached_tokens")
            or 0
        )
        _reasoning_tokens = int(
            (_usage_dict.get("completion_tokens_details") or {}).get("reasoning_tokens")
            or _usage_dict.get("reasoning_tokens")
            or 0
        )
        _cost_usd = float(_usage_dict.get("cost") or _usage_dict.get("cost_usd") or 0.0)
        _tool_calls_payload = [
            {"tool_call_id": tc.id, "function_name": tc.name, "arguments": tc.arguments}
            for tc in (getattr(response, "tool_calls", None) or [])
        ]
        # _dynamic_context was captured at render time alongside the
        # SystemPrompt snapshot (see _snapshot_llm_request), so it reflects
        # the exact messages sent to the LLM even across context-window retry.
        self.event_manager.add(
            LLMComplete(
                model_name=_model_name,
                prompt_tokens=_prompt_tokens,
                completion_tokens=_completion_tokens,
                cached_tokens=_cached_tokens,
                reasoning_tokens=_reasoning_tokens,
                cost_usd=_cost_usd,
                tool_calls=_tool_calls_payload,
                reasoning_content=getattr(response, "reasoning", None) or "",
                generation_id=current_generation_id or "",
                dynamic_context=_dynamic_context,
            ),
            record=False,
        )

        # Create and record LLMOutput
        # Serialize Pydantic models to JSON for proper event storage
        content = response.content or ""
        if isinstance(content, BaseModel):
            # Pydantic model - serialize to JSON string
            content = content.model_dump_json()
        elif not isinstance(content, str):
            # Other non-string types - convert to string representation
            content = str(content)
        event = LLMOutput(content=content)
        event_id = self.event_manager.add(event)

        return response, event_id

    async def execute_code(
        self,
        code: str,
        *,
        builtins: dict[str, Any] | None = None,
        validate: bool = True,
        wrap_in_function: bool = False,
        timeout: float | None = 90.0,
        tool_call_id: str | None = None,
        execution_count: int = 1,
        restrictions: "RestrictionsConfig | None" = None,
        sandbox_executor: Any = None,
    ) -> "ExecutionResult":
        """Execute Python code with namespace + strategy builtins (RuntimeServices protocol).

        Args:
            code: Python code to execute.
            builtins: Strategy-provided functions (return_result, message, method args, etc.)
            validate: Run planning language validation first.
            wrap_in_function: If True, wrap code in async function and capture return value.
            timeout: Maximum execution time in seconds (default 30s). None to disable.
            tool_call_id: LLM's tool call ID for trace correlation (OpenAI format).
            execution_count: Execution number for Jupyter-style "Cell In[N]" filename.
            restrictions: Code execution restrictions (blocked modules/calls).
                None uses defaults from RestrictionsConfig().
            sandbox_executor: When set (a ``SandboxedExecutor``), the cell runs in the
                guarded worker process instead of in-process. The surrounding
                middleware pipeline, before/after_code_execution hooks and events
                still run here on the parent, so the sandbox composes with them.

        Returns:
            ExecutionResult with stdout, error, defined_methods, and optionally returned_value.
        """
        # --- Middleware: execute_python --------------------------------------
        em = self.event_manager
        mid = em._middleware_id
        in_reentry = mid in _in_exec_middleware.get()

        if em._middleware.get("execute_python") and not in_reentry:
            from nooa.runtime.middleware import ExecutePythonContext

            ep_params: dict[str, Any] = {
                "builtins": builtins,
                "validate": validate,
                "wrap_in_function": wrap_in_function,
                "timeout": timeout,
                "tool_call_id": tool_call_id,
                "execution_count": execution_count,
                "restrictions": restrictions,
                "sandbox_executor": sandbox_executor,
            }
            ep_ctx = ExecutePythonContext(
                code=code,
                params=ep_params,
                agent=self.agent,
                runtime=self,
            )

            async def _core_exec(ctx: ExecutePythonContext) -> ExecutePythonContext:
                # Add our id to the re-entry guard so recursive calls skip middleware.
                token = _in_exec_middleware.set(_in_exec_middleware.get() | {mid})
                try:
                    ctx.result = await self.execute_code(
                        ctx.code,
                        **ctx.params,
                    )
                finally:
                    _in_exec_middleware.reset(token)
                return ctx

            ep_ctx = await em.run_middleware("execute_python", ep_ctx, _core_exec)
            if ep_ctx.result is None:
                raise RuntimeError(
                    "execute_python middleware returned without setting ctx.result. "
                    "Short-circuiting middleware must set ctx.result before returning."
                )
            return cast(ExecutionResult, ep_ctx.result)

        # If we are in re-entry (recursive call from _core_exec above), clear
        # our id from the guard so that nested generation methods called during
        # code execution (e.g. code does `await self.other_method()`) can
        # re-enter the middleware pipeline for their own execute_code() calls.
        # The _core_exec finally-block will restore the authoritative state
        # via its own token.
        if in_reentry:
            _in_exec_middleware.set(_in_exec_middleware.get() - {mid})

        from nooa.events import _NO_RETURN

        # Generate execution ID and call hooks
        execution_id = str(uuid4())
        # Get current generation_id for correlation (if in a generation session)
        current_generation_id = self._generation_id_stack[-1] if self._generation_id_stack else None
        hook_context = call_before_hook(
            "before_code_execution",
            agent=self.agent,
            code=code,
            execution_id=execution_id,
            generation_id=current_generation_id,
            tool_call_id=tool_call_id,  # LLM's tool call ID for trace correlation
        )

        result: ExecutionResult | None = None
        stdout_token: contextvars.Token[Any] | None = None  # Track for cleanup in finally
        stderr_token: contextvars.Token[Any] | None = None
        stdin_token: contextvars.Token[Any] | None = None
        media_token: contextvars.Token[Any] | None = None
        # Initialize buffer vars so the finally block can safely
        # isinstance-check them even if an early validation return
        # exited before they were assigned (caught by NameError previously).
        stdout_buffer: Any = None
        stderr_buffer: Any = None
        # Set parent agent context for LLM inheritance by subagents
        parent_token = _parent_agent_var.set(self.agent)
        try:
            # Build execution globals first (needed for validation error messages).
            # Walk the MRO so an agent inheriting from a parent defined in another
            # module also gets that module's module-level symbols (functions,
            # constants, types). Leaf module wins on name collisions.
            from nooa.agentdoc.visibility import filter_mro_module_globals

            exec_globals = filter_mro_module_globals(type(self.agent))

            # Import agentdoc functions for use in generated code
            # Import decorators and strategies for use in generated code
            # NOTE: stdout/stderr capture is handled by ContextVarStream wrappers
            # installed below.
            import typing as _typing

            from nooa.agentdoc import doc
            from nooa.decorators import strategy
            from nooa.media import Audio, File, Image, Media, Video
            from nooa.runtime.media_capture import show
            from nooa.runtime.pprint import pprint
            from nooa.strategies import (
                CodeActStrategy,
                CompositeStrategy,
                PredictStrategy,
                ReflexionStrategy,
                TemplateStrategy,
            )
            from nooa.strategies.pure_python import PurePythonStrategy

            exec_globals.update(
                {
                    "self": self.agent,
                    "asyncio": asyncio,
                    "typing": _typing,
                    # Common typing constructs — LLMs use these constantly
                    "Annotated": _typing.Annotated,
                    "Any": _typing.Any,
                    "Literal": _typing.Literal,
                    "Optional": _typing.Optional,
                    "Union": _typing.Union,
                    # agentdoc introspection (doc respects hidden fields)
                    "doc": doc,
                    "methods": methods,
                    "variables": variables,
                    "help": doc,  # Shadow built-in help() to prevent blocking on stdin
                    # Pretty printing with Rich-compatible API
                    "pprint": pprint,
                    # Media display for multimodal models
                    "show": show,
                    "Media": Media,
                    "Image": Image,
                    "Audio": Audio,
                    "Video": Video,
                    "File": File,
                    # decorators and strategies for LLM-generated methods
                    "strategy": strategy,
                    "PurePythonStrategy": PurePythonStrategy,
                    "CodeActStrategy": CodeActStrategy,
                    "ReflexionStrategy": ReflexionStrategy,
                    "PredictStrategy": PredictStrategy,
                    "TemplateStrategy": TemplateStrategy,
                    "CompositeStrategy": CompositeStrategy,
                }
            )

            # Add strategy builtins (includes reasoning, message, method args)
            if builtins:
                exec_globals.update(builtins)

            # ── Pre-execution cleanup ────────────────────────────────────
            # Intercept point: all code cleanup transforms are bundled here.
            # Consider making this an extensible middleware point in the future.
            from nooa.runtime.response_cleanup import strip_code_fences
            from nooa.runtime.restrictions import RestrictionsConfig

            hm = get_harness_metrics()

            # 1. Strip markdown code fences (safety net — CodeAct already strips
            #    at the tool-call boundary before helper extraction; this covers
            #    PurePython and any direct execute_code callers).
            code, fence_token = strip_code_fences(code)
            if fence_token:
                hm.fence_removal(fence_token)

            # 2. Strip blocked modules and their members from exec_globals
            effective_restrictions = restrictions or RestrictionsConfig()
            exec_globals = _strip_blocked_modules(
                exec_globals, effective_restrictions.blocked_modules
            )

            # 3. Strip redundant imports (from typing import Literal, etc.)
            from nooa.runtime.code_validator import strip_redundant_imports

            code, stripped_imports = strip_redundant_imports(code, set(exec_globals.keys()))
            for stmt in stripped_imports:
                hm.import_stripped(stmt)

            # Validate code if requested (unified validator handles all checks)
            with get_harness_metrics().timer("time_code_validation"):
                if validate:
                    try:
                        from nooa.runtime.code_validator import (
                            UnifiedCodeValidator,
                            ValidationContext,
                        )

                        # Prevent recursive calls to the method currently being generated
                        forbidden_self_calls: set[str] = set()
                        current_call = self._current_call
                        if current_call and hasattr(current_call, "method_name"):
                            forbidden_self_calls = {current_call.method_name}
                        return_type = getattr(current_call, "return_type", None)

                        # Build importable_modules from actual module names (not aliases)
                        # e.g., 'import pandas as pd' means 'pandas' is importable, not 'os'
                        importable_modules: set[str] = set()
                        for obj in exec_globals.values():
                            if isinstance(obj, types.ModuleType):
                                actual_name = getattr(obj, "__name__", None)
                                if actual_name:
                                    importable_modules.add(actual_name)

                        # Create validation context
                        context = ValidationContext(
                            code=code,
                            agent_class=type(self.agent),
                            available_names=set(exec_globals.keys()),
                            importable_modules=importable_modules,
                            restricted_imports=effective_restrictions.restricted_imports,
                            blocked_modules=effective_restrictions.blocked_modules,
                            forbidden_self_calls=forbidden_self_calls,
                            execution_count=execution_count,
                            agent=self.agent,
                            exec_globals=exec_globals,
                            return_type=return_type,
                        )

                        # Run unified validation (security, blocking calls, REPL policy)
                        validator = UnifiedCodeValidator(
                            restrictions=effective_restrictions,
                        )
                        validator.validate(code, context)
                    except Exception as e:
                        result = ExecutionResult(stdout="", error=e, defined_methods={})
                        return result

            # Sandbox backend: delegate the exec to the guarded worker AFTER the
            # shared cleanup + validation above (so restrictions/cell-guard parity
            # holds), while the middleware pipeline, before/after hooks and events
            # still wrap the cell on the parent. The worker owns the namespace,
            # stdout capture and wrapper, so we skip the in-process exec below.
            if sandbox_executor is not None:
                result = await sandbox_executor.run_cell(code, execution_count=execution_count)
                return result

            # Set up stdout/stderr capture BEFORE ast.parse/compile so that
            # SyntaxWarnings (e.g. invalid escape sequences in LLM-generated code)
            # and RuntimeWarnings are captured into the execution result instead of
            # leaking to the terminal as "<unknown>:N: SyntaxWarning: ..."
            #
            # Each async task gets its own buffers via contextvars, so parallel
            # executions are isolated. TruncatingStringIO prevents LLMs from
            # filling the context window.
            truncation_config = self.truncation_config
            stdout_buffer = _make_capture_buffer(
                limit=truncation_config.capture.max_stdout,
                tail_chars=truncation_config.capture.tail,
                file_backed=truncation_config.capture.file_backed,
            )
            stderr_buffer = _make_capture_buffer(
                limit=truncation_config.capture.max_stderr,
                tail_chars=truncation_config.capture.tail,
                file_backed=truncation_config.capture.file_backed,
            )
            stdout_token = _stdout_buffer_var.set(stdout_buffer)
            stderr_token = _stderr_buffer_var.set(stderr_buffer)
            # Block stdin reads for this async task (prevents hangs from input(), etc.)
            stdin_token = _block_stdin_var.set(True)
            # Media capture buffer for show() calls; scoped per cell by design.
            media_buffer = _MediaBuffer(
                max_attachments=truncation_config.media_capture.max_attachments_per_execution
            )
            media_token = _media_buffer_var.set(media_buffer)

            # Install stream wrappers around the CURRENT sys.stdout/sys.stderr/sys.stdin
            # (which may have been replaced by pytest or other tools)
            # We only wrap if not already wrapped, and we NEVER restore - the wrappers
            # are transparent (fall through to original when no buffer/block is set) and
            # must persist for parallel async executions to work correctly.
            import sys

            if not isinstance(sys.stdout, ContextVarStream):
                sys.stdout = ContextVarStream(sys.stdout, _stdout_buffer_var, "stdout")
            if not isinstance(sys.stderr, ContextVarStream):
                sys.stderr = ContextVarStream(sys.stderr, _stderr_buffer_var, "stderr")
            if not isinstance(sys.stdin, BlockedStdinWrapper):
                sys.stdin = BlockedStdinWrapper(sys.stdin)

            # Parse AST to find method definitions
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                result = ExecutionResult(stdout="", error=e, defined_methods={})
                return result

            # Extract method sources (functions with self as first param)
            method_sources: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    if node.args.args and node.args.args[0].arg == "self":
                        method_code = ast.get_source_segment(code, node)
                        if method_code:
                            method_sources[node.name] = method_code

            # Check for explicit return BEFORE any transformation
            # Use a NodeVisitor that doesn't recurse into nested functions/classes
            # to avoid false positives from returns in nested definitions
            has_explicit_return = _has_top_level_return(tree)

            # Transform last expression to implicit return (REPL/Jupyter style)
            # This allows `doc(self)` or `result` at the end to return their values
            # IMPORTANT: We modify the code TEXT directly instead of using ast.unparse()
            # because ast.unparse() removes comments and changes line numbers, which
            # breaks error line number reporting for users.
            implicit_return_added = False
            if wrap_in_function and tree.body and not has_explicit_return:
                last_stmt = tree.body[-1]
                if isinstance(last_stmt, ast.Expr):
                    # Get the line number of the last expression
                    last_line_no = last_stmt.lineno
                    code_lines = code.split("\n")
                    if 1 <= last_line_no <= len(code_lines):
                        # Insert 'return ' at the statement's COLUMN (not the line
                        # start): a ';'-compound final line must keep its earlier
                        # statements live — line-prefixing made them dead code.
                        # This preserves comments and line structure.
                        original_line = code_lines[last_line_no - 1]
                        col = last_stmt.col_offset
                        code_lines[last_line_no - 1] = (
                            f"{original_line[:col]}return {original_line[col:]}"
                        )
                        code = "\n".join(code_lines)
                        implicit_return_added = True
                        get_harness_metrics().implicit_return()

            returned_value = _NO_RETURN
            captured_locals: dict[str, Any] = {}
            wrapper_line_offset = 0  # Lines of wrapper before user code (for error adjustment)

            # Use Jupyter-style "Cell In[N]" as filename for better error messages
            cell_filename = f"Cell In[{execution_count}]"
            try:
                if wrap_in_function:
                    # REPL mode: wrap entire code in async function to capture return
                    # Use try/finally to capture locals regardless of how function exits
                    indented = self._indent_code(code, "        ")
                    # Initialize the captured locals dict in exec_globals
                    exec_globals["__repl_captured_locals__"] = {}

                    # Build global declarations for session variables (REPL persistence)
                    # This allows code like `x = x + 5` to work when x was defined in a prior call
                    # Without this, Python sees `x = ...` and marks x as local, failing on the RHS
                    # We need to declare as global any variable that was passed in builtins
                    global_vars = [
                        name
                        for name, val in (builtins or {}).items()
                        if (
                            not name.startswith("_")
                            and name not in ("self", "asyncio", "__builtins__")
                            and not callable(val)
                            and not isinstance(val, types.ModuleType)
                        )
                    ]
                    global_decl = f"    global {', '.join(global_vars)}\n" if global_vars else ""

                    # Build the set of global var names for the finally block
                    global_var_set = set(global_vars)

                    # Calculate wrapper header line count for error line adjustment
                    # User code starts after: "async def __repl_wrapper__():\n" + global_decl + "    try:\n"
                    wrapper_header = f"async def __repl_wrapper__():\n{global_decl}    try:\n"
                    wrapper_line_offset = wrapper_header.count("\n")

                    wrapper = f"""async def __repl_wrapper__():
{global_decl}    try:
{indented}
    finally:
        # Capture locals for REPL persistence (exclude internals and builtins)
        # Also capture updated global variables that were declared global
        __repl_captured_locals__.update({{
            k: v for k, v in locals().items()
            if not k.startswith('_') and k not in ('self', 'asyncio')
        }})
        # Capture updated values of global variables we declared
        for _gvar in {repr(global_var_set)}:
            if _gvar in globals():
                __repl_captured_locals__[_gvar] = globals()[_gvar]
"""
                    # Register wrapper with linecache so traceback can show source lines
                    linecache.cache[cell_filename] = (
                        len(wrapper),
                        None,
                        wrapper.splitlines(keepends=True),
                        cell_filename,
                    )

                    exec(compile(wrapper, cell_filename, "exec"), exec_globals)
                    # Execute with async safety (blocks Future.result() etc from event loop)
                    from nooa.runtime.async_safety import agent_async_safety_context

                    coro = exec_globals["__repl_wrapper__"]()
                    with agent_async_safety_context():
                        if timeout is not None:
                            try:
                                result_value = await asyncio.wait_for(coro, timeout=timeout)
                            except TimeoutError:
                                raise TimeoutError(
                                    f"Code execution timed out after {timeout} seconds. "
                                    "Check for infinite loops or blocking operations."
                                ) from None
                        else:
                            result_value = await coro

                    # Extract captured locals (filter out non-serializable/internal items)
                    captured_locals = _extract_captured_locals(exec_globals)

                    # Capture return value based on return type:
                    # - Explicit return: always capture (even None)
                    # - Implicit return: only capture non-None (like IPython)
                    if has_explicit_return:
                        returned_value = result_value
                    elif implicit_return_added and result_value is not None:
                        # Suppress None from implicit returns (matches IPython behavior)
                        returned_value = result_value

                    # Still extract method definitions for helper functions
                    # Re-execute just the function defs to bind them
                    func_defs: list[ast.stmt] = [
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                    ]
                    if func_defs:
                        func_tree = ast.Module(body=func_defs, type_ignores=[])
                        exec(compile(func_tree, cell_filename, "exec"), exec_globals)

                    # Attach source code to defined functions for has_ellipsis_body() detection
                    for method_name, method_code in method_sources.items():
                        if method_name in exec_globals and callable(exec_globals[method_name]):
                            try:
                                exec_globals[method_name]._generated_source = method_code
                            except (AttributeError, TypeError):
                                # Some built-in functions don't allow attribute assignment
                                pass
                else:
                    # Original mode: separate function defs from other statements
                    func_defs = [
                        n
                        for n in tree.body
                        if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                    ]
                    other_stmts: list[ast.stmt] = [
                        n
                        for n in tree.body
                        if not isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                    ]

                    # Execute function definitions
                    if func_defs:
                        func_tree = ast.Module(body=func_defs, type_ignores=[])
                        exec(compile(func_tree, cell_filename, "exec"), exec_globals)

                    # Execute other statements
                    if other_stmts:
                        other_tree = ast.Module(body=other_stmts, type_ignores=[])
                        has_await = any(isinstance(n, ast.Await) for n in ast.walk(other_tree))
                        if has_await:
                            other_code = ast.unparse(other_tree)
                            indented = "\n".join("    " + line for line in other_code.split("\n"))
                            wrapper = f"async def __wrapper__():\n{indented}"
                            # Register wrapper with linecache so traceback can show source lines
                            linecache.cache[cell_filename] = (
                                len(wrapper),
                                None,
                                wrapper.splitlines(keepends=True),
                                cell_filename,
                            )
                            exec(compile(wrapper, cell_filename, "exec"), exec_globals)
                            # Execute with async safety (blocks Future.result() etc)
                            from nooa.runtime.async_safety import (
                                agent_async_safety_context,
                            )

                            coro = exec_globals["__wrapper__"]()
                            with agent_async_safety_context():
                                if timeout is not None:
                                    try:
                                        await asyncio.wait_for(coro, timeout=timeout)
                                    except TimeoutError:
                                        raise TimeoutError(
                                            f"Code execution timed out after {timeout} seconds. "
                                            "Check for infinite loops or blocking operations."
                                        ) from None
                                else:
                                    await coro
                        else:
                            # Sync code execution - enable async safety for Future.result() etc
                            from nooa.runtime.async_safety import (
                                agent_async_safety_context,
                            )

                            with agent_async_safety_context():
                                exec(compile(other_tree, cell_filename, "exec"), exec_globals)

                # Attach source code to defined functions for has_ellipsis_body() detection
                for method_name, method_code in method_sources.items():
                    if method_name in exec_globals and callable(exec_globals[method_name]):
                        try:
                            exec_globals[method_name]._generated_source = method_code
                        except (AttributeError, TypeError):
                            # Some built-in functions don't allow attribute assignment
                            pass

                # Helpers stay as REPL locals (captured via __repl_captured_locals__).
                # The names list is kept in ExecutionResult.defined_methods for feedback only.
                defined_methods: dict[str, Any] = {
                    method_name: exec_globals[method_name]
                    for method_name in method_sources
                    if method_name in exec_globals and callable(exec_globals[method_name])
                }

                result = ExecutionResult(
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                    error=None,
                    defined_methods=defined_methods,
                    returned_value=returned_value,
                    explicit_return=has_explicit_return,
                    captured_locals=captured_locals,
                    wrapper_line_offset=wrapper_line_offset,
                    images=media_buffer.blocks,
                )
                return result

            except ExecutionSignal as sig:
                # Extract captured locals even on signal (IPython behavior).
                # The wrapper function's finally block still runs before this handler,
                # populating exec_globals["__repl_captured_locals__"] with all variables
                # that were defined before the signal was raised.
                captured_locals = _extract_captured_locals(exec_globals)

                result = ExecutionResult(
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                    error=None,  # NOT an error
                    signal=sig,  # Store signal separately
                    defined_methods={},
                    returned_value=_NO_RETURN,
                    explicit_return=False,
                    captured_locals=captured_locals,
                    wrapper_line_offset=wrapper_line_offset,
                    images=media_buffer.blocks,
                )
                return result

            except (SystemExit, KeyboardInterrupt) as e:
                # A generated cell used raise SystemExit / sys.exit() / exit() / quit()
                # (or Ctrl-C surfaced as KeyboardInterrupt). These are BaseException,
                # not Exception, so without this handler they would escape execute_code,
                # propagate through the strategy into the parent async flow, and cancel
                # sibling tasks (e.g. a surrounding asyncio.TaskGroup). Convert to an
                # ordinary execution error so one cell cannot terminate the parent run.
                # NOTE: asyncio.CancelledError (also BaseException) is deliberately NOT
                # caught here — real cancellation must still propagate.
                captured_locals = _extract_captured_locals(exec_globals)

                # ExecutionResult.error is typed Exception | None, so wrap the
                # BaseException in a RuntimeError (original kept as __cause__).
                wrapped = RuntimeError(
                    f"{type(e).__name__} raised inside generated code. "
                    "Do not use raise SystemExit / sys.exit() / exit() / quit() to stop "
                    "a cell — use break, a flag, a helper return, or return_result()."
                )
                wrapped.__cause__ = e

                result = ExecutionResult(
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                    error=wrapped,
                    defined_methods={},
                    returned_value=_NO_RETURN,
                    explicit_return=False,
                    captured_locals=captured_locals,
                    wrapper_line_offset=wrapper_line_offset,
                    images=media_buffer.blocks,
                )
                return result

            except Exception as e:
                # Extract captured locals even on error (IPython behavior).
                # The wrapper function's finally block still runs before this handler,
                # populating exec_globals["__repl_captured_locals__"] with all variables
                # that were defined before the exception was raised.
                captured_locals = _extract_captured_locals(exec_globals)

                result = ExecutionResult(
                    stdout=stdout_buffer.getvalue(),
                    stderr=stderr_buffer.getvalue(),
                    error=e,
                    defined_methods={},
                    returned_value=_NO_RETURN,
                    explicit_return=False,
                    captured_locals=captured_locals,
                    wrapper_line_offset=wrapper_line_offset,
                    images=media_buffer.blocks,
                )
                return result

        finally:
            # NOTE: We do NOT restore sys.stdout/sys.stderr here.
            # The ContextVarStream wrappers are transparent (fall through to original
            # when no buffer is set) and must persist for parallel async executions.

            # Reset task-local stdout/stderr buffers and stdin blocking (only if they were set)
            if stdout_token is not None:
                _stdout_buffer_var.reset(stdout_token)
            if stderr_token is not None:
                _stderr_buffer_var.reset(stderr_token)
            if stdin_token is not None:
                _block_stdin_var.reset(stdin_token)
            if media_token is not None:
                _media_buffer_var.reset(media_token)
            # Close file handles for file-backed buffers (but keep files on disk
            # so the LLM can reference the full output via the path in the notice).
            # stdout_buffer/stderr_buffer may not be defined if an early return
            # (validation, SyntaxError) exited before buffer creation.
            try:
                if isinstance(stdout_buffer, FileBackedTruncatingStringIO):
                    stdout_buffer.close()
                if isinstance(stderr_buffer, FileBackedTruncatingStringIO):
                    stderr_buffer.close()
            except NameError:
                pass
            # Reset parent agent context
            _parent_agent_var.reset(parent_token)
            # Call after_code_execution hook
            call_after_hook(
                "after_code_execution",
                hook_context,
                agent=self.agent,
                code=code,
                result=result,
                exception=result.error if result else None,
                execution_id=execution_id,
                tool_call_id=tool_call_id,  # LLM's tool call ID for trace correlation
            )

    async def execute_nested(
        self,
        strategy: Any,
        call: Any,
    ) -> Any:
        """Execute nested strategy within current generation session (RuntimeServices protocol).

        Use for composite strategies (Reflexion, PlanExecute) that wrap
        other strategies. Runs within the current session:
        - Inherits lock (won't deadlock)
        - Proper hook instrumentation
        - Events are shared

        Args:
            strategy: The nested GenerationStrategy to execute.
            call: The CurrentCall context.

        Returns:
            Result from the nested strategy.
        """
        from uuid import uuid4

        # Generate generation_id for nested strategy
        generation_id = str(uuid4())
        parent_generation_id = self._generation_id_stack[-1] if self._generation_id_stack else None
        # Push using copy-on-write semantics for parallel task safety
        _push_generation_id(generation_id)

        # Get strategy name for hooks
        strategy_name = getattr(strategy, "name", type(strategy).__name__)

        # Extract strategy config for tracing
        # Works for CodeActStrategy, PurePythonStrategy, PredictStrategy, ReflexionStrategy
        strategy_kwargs = {}
        if hasattr(strategy, "max_iterations"):
            strategy_kwargs["max_iterations"] = getattr(strategy, "max_iterations")  # noqa: B009
        if hasattr(strategy, "max_retries"):
            strategy_kwargs["max_retries"] = getattr(strategy, "max_retries")  # noqa: B009
        if hasattr(strategy, "max_reflections"):
            strategy_kwargs["max_reflections"] = getattr(strategy, "max_reflections")  # noqa: B009
        if hasattr(strategy, "prefill") and getattr(strategy, "prefill") is not None:  # noqa: B009
            strategy_kwargs["has_prefill"] = True

        # Call generation hooks (skip for non-traceable strategies like TemplateStrategy)
        should_trace = strategy.traceable
        hook_context = None
        if should_trace:
            hook_context = call_before_hook(
                "before_generation",
                agent=self.agent,
                method_name=call.method_name,
                strategy=strategy_name,
                generation_id=generation_id,
                parent_generation_id=parent_generation_id,
                agent_call_id=self._agent_call_id,
                **strategy_kwargs,  # Add strategy config parameters
            )

        result = None
        exception_caught = None

        # Start harness metrics + unifiedllm bridge (cleaned up in finally below)
        _hm_ctx = _harness_metrics_lifecycle(should_trace)
        _hm_ctx.__enter__()

        try:
            # Execute nested strategy directly (we're already in a generation session)
            result = await strategy.execute(self, call)
            return result
        except Exception as e:
            exception_caught = e
            raise
        finally:
            # Pop generation_id from stack using copy-on-write semantics
            _pop_generation_id()

            # Flush harness metrics BEFORE after_generation ends the span —
            # otherwise get_current_span() no longer returns the generation
            # span and harness.* attributes are dropped.
            if exception_caught is not None:
                _hm_ctx.__exit__(
                    type(exception_caught),
                    exception_caught,
                    exception_caught.__traceback__,
                )
            else:
                _hm_ctx.__exit__(None, None, None)

            # Call after generation hook
            if should_trace:
                call_after_hook(
                    "after_generation",
                    hook_context,
                    agent=self.agent,
                    method_name=call.method_name,
                    result=result,
                    exception=exception_caught,
                    generation_id=generation_id,
                )

    def get_generation_id(self) -> str | None:
        """Get the current generation session ID (RuntimeServices protocol).

        Returns the innermost generation_id from the stack, or None if
        not in a generation session. Used for correlating events with
        their generation context.

        Returns:
            Current generation ID or None if not in a generation session.

        Note:
            Never raises exceptions - returns None for empty stack.
        """
        stack = self._generation_id_stack
        return stack[-1] if stack else None

    def get_parent_generation_id(self) -> str | None:
        """Get the parent generation session ID (RuntimeServices protocol).

        Returns the second-to-last generation_id from the stack, or None
        if this is the root generation or not in a generation session.

        Returns:
            Parent generation ID or None if root or not in a session.

        Note:
            Never raises exceptions - returns None for stack with <2 entries.
        """
        stack = self._generation_id_stack
        return stack[-2] if len(stack) > 1 else None

    # --- End RuntimeServices Protocol ---
    # Note: Nested ellipsis method calls work implicitly via _call_plan() + _in_generation_session.
    # execute_nested() is for composite strategies (Reflexion, PlanExecute) to run sub-strategies.

    def get_code(self, method_name: str) -> str | None:
        """
        Get code for a method.

        Returns the original method definition with ... body.

        Args:
            method_name: Name of the method to inspect

        Returns:
            Code as string, or None if method doesn't exist

        Example:
            agent = MyAgent()
            code = agent.runtime.get_code("my_method")
            print(code)  # Shows: async def my_method(self): ... with docstring
        """
        import inspect

        # Return original definition with ... body
        original = getattr(type(self.agent), method_name, None)
        if original is None:
            return None

        # Build original definition
        sig = inspect.signature(original)
        doc = inspect.getdoc(original) or ""
        is_async = inspect.iscoroutinefunction(original)

        lines = []
        async_str = "async " if is_async else ""
        lines.append(f"{async_str}def {method_name}{sig}:")

        if doc:
            lines.append('    """')
            for line in doc.split("\n"):
                lines.append(f"    {line}")
            lines.append('    """')

        lines.append("    ...")

        return "\n".join(lines)

    async def evaluate_expression(
        self,
        expr: str,
        extra_context: dict[str, Any] | None = None,
        error_mode: str = "show",
    ) -> Any:
        """
        Evaluate a Python expression with full runtime context.

        Can handle both sync and async expressions (automatically awaits coroutines).

        The runtime has access to all state needed for evaluation:
        - Agent instance (self)
        - REPL state (if available)
        - Execution results
        - Standard Python builtins
        - Extra context (e.g., method arguments)

        Args:
            expr: Python expression to evaluate
            extra_context: Additional variables to include in namespace
            error_mode: How to handle errors:
                - "show": Return error as string (default)
                - "silent": Return None on error
                - "raise": Raise the exception

        Returns:
            Evaluation result, or error representation based on error_mode

        Example:
            >>> await runtime.evaluate_expression("len(self.tools)")
            3
            >>> await runtime.evaluate_expression("strategy.method(runtime)")
            "Result..."
        """

        # Build evaluation namespace with all runtime context
        from nooa.agentdoc import doc, pformat

        namespace = {
            "self": self.agent,
            # agentdoc introspection (doc respects hidden fields)
            "doc": doc,
            "pformat": pformat,
            "methods": methods,
            "variables": variables,
            **(extra_context or {}),
        }

        # Include REPL locals if available
        try:
            repl = getattr(self.agent, "repl", None)
            if repl:
                repl_locals = getattr(repl, "_data", None)
                if isinstance(repl_locals, dict):
                    namespace.update(repl_locals)
        except Exception:
            pass

        # Include execution result if available
        if hasattr(self.agent, "_last_execution_result"):
            namespace["result"] = self.agent._last_execution_result

        # Evaluate the expression
        try:
            # Compile as async expression
            code = compile(f"async def __eval_expr(): return {expr}", "<string>", "exec")
            # exec needs namespace as globals so the function can access variables
            exec(code, namespace)
            result = await namespace["__eval_expr"]()

            # If result is still a coroutine, await it
            if inspect.iscoroutine(result):
                result = await result

            # Handle subprocess.CompletedProcess - extract useful output
            import subprocess

            if isinstance(result, subprocess.CompletedProcess):
                if isinstance(result.stdout, str) and result.stdout:
                    return result.stdout
                if isinstance(result.stderr, str) and result.stderr:
                    return result.stderr
                # returncode is int per typeshed, but mocks/edge cases may pass None
                returncode: int | None = cast(int | None, result.returncode)
                if returncode is not None:
                    return f"[exit code: {returncode}]"
                return "[command completed]"

            return result

        except Exception as e:
            if error_mode == "raise":
                raise
            elif error_mode == "show":
                return f"{{ERROR: {e}}}"
            else:  # silent
                return None

    async def expand_variables(
        self,
        text: str,
        extra_context: dict[str, Any] | None = None,
        error_mode: str = "show",
    ) -> str:
        """
        Expand {expression} placeholders in text using runtime context.

        Args:
            text: Text with {expression} placeholders
            extra_context: Additional variables for evaluation
            error_mode: How to handle errors ("show", "silent", "raise")

        Returns:
            Text with expressions evaluated and substituted

        Example:
            >>> await runtime.expand_variables("Found {len(self.tools)} tools")
            "Found 3 tools"
            >>> await runtime.expand_variables("Result: {result.stdout}")
            "Result: output text"
        """
        import string

        if not text or "{" not in text:
            return text

        # Use Python's built-in formatter to parse the template
        # This handles format specs, conversions (!r, !s, !a), and escaped braces
        formatter = string.Formatter()
        result_parts = []

        for literal_text, field_name, format_spec, conversion in formatter.parse(text):
            # Add literal text (includes escaped {{ as {)
            result_parts.append(literal_text)

            # If there's a field to replace
            if field_name is not None:
                # Evaluate the expression
                value = await self.evaluate_expression(field_name, extra_context, error_mode)

                # Handle silent mode errors
                if error_mode == "silent" and value is None:
                    # Keep original placeholder
                    placeholder = f"{{{field_name}"
                    if conversion:
                        placeholder += f"!{conversion}"
                    if format_spec:
                        placeholder += f":{format_spec}"
                    placeholder += "}"
                    result_parts.append(placeholder)
                    continue

                # Apply conversion (!r, !s, !a)
                if conversion == "r":
                    value = repr(value)
                elif conversion == "s":
                    value = str(value)
                elif conversion == "a":
                    value = ascii(value)

                # Apply format spec
                if format_spec:
                    try:
                        result_parts.append(format(value, format_spec))
                    except Exception:
                        if error_mode == "silent":
                            # Keep original placeholder on format error
                            placeholder = f"{{{field_name}"
                            if conversion:
                                placeholder += f"!{conversion}"
                            placeholder += f":{format_spec}}}"
                            result_parts.append(placeholder)
                        else:
                            result_parts.append(f"{{{field_name}:{format_spec} | FORMAT ERROR}}")
                else:
                    result_parts.append(str(value))

        return "".join(result_parts)

    def list_methods(self) -> dict[str, dict[str, Any]]:
        """
        Get a complete listing of all methods on the agent.

        Returns a dict mapping method names to their metadata:
        - 'type': 'generator' | 'implemented' | 'generated'
        - 'ellipsis': True if ellipsis method | False otherwise
        - 'signature': Full signature string
        - 'docstring': First line of docstring
        - 'is_async': True if async method
        - 'strategy': GenerationStrategy if applicable
        - 'has_code': True if generated code is available

        Example:
            agent = MyAgent()
            methods = agent.runtime.list_methods()

            for name, info in methods.items():
                print(f"{name}: {info['type']} ({info['decorator']})")
        """
        import inspect

        methods_info = {}

        from nooa.agentdoc.visibility import is_hidden_method

        # Get all methods from the class
        for name in dir(type(self.agent)):
            if name.startswith("__"):
                continue

            attr = getattr(type(self.agent), name, None)
            if not callable(attr):
                continue

            # Skip properties and other descriptors
            # Note: property objects fail the callable() check above, so this
            # guard only fires for exotic descriptors that are callable.
            if isinstance(attr, property):
                continue

            # Skip hidden methods
            if is_hidden_method(attr):
                continue

            # Determine method type
            method_type = "implemented"
            decorator_type = None
            has_generated_code = False

            # Check if it's a generator method (has ... body)
            if hasattr(attr, "_needs_generation") and getattr(attr, "_needs_generation"):  # noqa: B009
                method_type = "generator"

            # Get decorator info
            if hasattr(attr, "_agent_decorator"):
                decorator_type = f"@{getattr(attr, '_agent_decorator')}"  # noqa: B009

            # Get signature
            try:
                sig = inspect.signature(attr)
                signature_str = f"{name}{sig}"
            except (ValueError, TypeError):
                signature_str = f"{name}(...)"

            # Get docstring (first line)
            doc = inspect.getdoc(attr) or ""
            doc_first_line = doc.split("\n")[0] if doc else ""

            # Get strategy
            strategy = getattr(attr, "_plan_strategy", None) or getattr(
                attr, "_signal_strategy", None
            )

            # Is async?
            is_async = inspect.iscoroutinefunction(attr)

            methods_info[name] = {
                "type": method_type,
                "decorator": decorator_type,
                "signature": signature_str,
                "docstring": doc_first_line,
                "is_async": is_async,
                "strategy": strategy,
                "has_code": has_generated_code,
            }

        return methods_info

    def print_methods(self) -> None:
        """
        Print a formatted listing of all agent methods.

        Groups methods by type and shows key information.
        Uses trace-view style formatting with │ for indentation.

        Example:
            agent = MyAgent()
            agent.runtime.print_methods()
        """
        methods = self.list_methods()

        # Group by type
        generators = {k: v for k, v in methods.items() if v["type"] == "generator"}
        implemented = {k: v for k, v in methods.items() if v["type"] == "implemented"}

        print(f"\n🎭 Agent Methods: {type(self.agent).__name__}")

        if generators:
            print("│ 🔮 Generator Methods")
            for _name, info in sorted(generators.items()):
                decorator = info["decorator"] or ""
                strategy = f"strategy={info['strategy'].name}" if info["strategy"] else ""
                if strategy:
                    decorator = f"{decorator}({strategy})"

                async_marker = "async " if info["is_async"] else ""
                signature = f"{async_marker}{info['signature']}"

                # Print with prefix (each line if multiline)
                for line in decorator.rstrip().split("\n"):
                    print(f"│ {line}")
                for line in signature.rstrip().split("\n"):
                    print(f"│ {line}")

                if info["docstring"]:
                    # Wrap long docstrings
                    doc = info["docstring"]
                    if len(doc) > 70:
                        doc = doc[:67] + "..."
                    print(f"│   → {doc}")

        if implemented:
            if generators:
                print("│")
            print("│ 📝 Implemented Methods")
            for _name, info in sorted(implemented.items()):
                decorator = info["decorator"] or ""
                if decorator:
                    for line in decorator.rstrip().split("\n"):
                        print(f"│ {line}")

                async_marker = "async " if info["is_async"] else ""
                signature = f"{async_marker}{info['signature']}"

                # Print with prefix (each line if multiline)
                for line in signature.rstrip().split("\n"):
                    print(f"│ {line}")

                if info["docstring"]:
                    # Wrap long docstrings
                    doc = info["docstring"]
                    if len(doc) > 70:
                        doc = doc[:67] + "..."
                    print(f"│   → {doc}")

        print(f"\nTotal: {len(generators)} generator, {len(implemented)} implemented\n")

    def _call_plan(
        self,
        method: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> "asyncio.Task[Any]":
        """
        Internal: Call an ellipsis method (returns immediately with asyncio.Task).

        Args:
            method: Method to call
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            asyncio.Task that will resolve to the method result
        """
        method_name = method.__name__
        task_name = f"{self.agent.__class__.__name__}.{method_name}"

        # NOTE: agent_call_id is pushed/popped by the decorator (decorators.py/metaclass.py)
        # before/after calling _call_plan. The ContextVar is inherited by the task.
        async def _execute_with_event() -> Any:
            return await self._execute_task(method, args, kwargs)

        # Return asyncio.Task directly (caller decides when to await)
        return asyncio.create_task(_execute_with_event(), name=task_name)

    async def _execute_task(
        self,
        method: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """
        Execute a method call.

        Args:
            method: Method to call
            args: Positional arguments
            kwargs: Keyword arguments

        Returns:
            Method result
        """
        method_name = method.__name__

        # Check if method needs LLM generation
        base_method = getattr(method, "__func__", method)
        needs_generation = getattr(base_method, "_needs_generation", False)

        if needs_generation:
            # Check if we're already in a generation session (nested call)
            in_session = _in_generation_session.get()

            if in_session:
                # We're inside a generation session - execute inline without acquiring lock
                # This prevents deadlock when generated code calls other @strategy methods
                return await self._execute_with_generation(method, args, kwargs, method_name)
            else:
                # Get strategy to check requires_lock
                from nooa.strategies import GenerationStrategy as GenerationStrategyABC
                from nooa.strategies import get_default_strategy

                call_strategy = kwargs.get("_strategy")
                decorator_strategy = getattr(base_method, "_plan_strategy", None)
                strategy = call_strategy or decorator_strategy or get_default_strategy()

                # Only acquire lock if strategy requires it
                if isinstance(strategy, GenerationStrategyABC) and strategy.requires_lock:
                    self._ensure_generation_lock_on_current_loop()
                    async with self._generation_lock:
                        return await self._execute_with_generation(
                            method, args, kwargs, method_name
                        )
                else:
                    # Lock-free execution (Methodic-style concurrent strategies)
                    return await self._execute_with_generation(method, args, kwargs, method_name)
        else:
            # Method has implementation - call directly with context vars set.
            # These context vars (_current_agent_var / _current_runtime_var)
            # are consumed by nooa.util._context._current_agent() and
            # by child-agent runtime inheritance; they are set only on this
            # implemented-method path, not for LLM-generated code.
            from nooa.util._context import _current_agent_var, _current_runtime_var

            agent_token = _current_agent_var.set(self.agent)
            runtime_token = _current_runtime_var.set(self)

            try:
                # method is the unwrapped function f (decorator always passes f, not wrapper)
                return await method(self.agent, *args, **kwargs)
            finally:
                _current_agent_var.reset(agent_token)
                _current_runtime_var.reset(runtime_token)

    async def _execute_with_generation(
        self,
        method: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        method_name: str,
    ) -> Any:
        """Execute a method that needs LLM generation."""
        # Extract framework parameters (don't pass to generated method)
        call_strategy = kwargs.pop("_strategy", None)
        call_llm = kwargs.pop("llm", None)
        call_session_locals = kwargs.pop("_session_locals", None)

        # Get strategy with priority: call-level > decorator > default
        base_method = getattr(method, "__func__", method)
        decorator_strategy = getattr(base_method, "_plan_strategy", None)

        # DEBUG: Log strategy retrieval for nested method debugging
        if logger.isEnabledFor(logging.DEBUG):
            has_plan_strategy = hasattr(base_method, "_plan_strategy")
            strategy_type = type(decorator_strategy).__name__ if decorator_strategy else "None"
            logger.debug(
                "[ACTOR] Strategy retrieval for %s: has_attr=%s, strategy=%s",
                method_name,
                has_plan_strategy,
                strategy_type,
            )

        # Two-level priority: call override > decorator > default fallback
        # Strategy can be: GenerationStrategy instance (new), enum (old), or None
        from nooa.strategies import GenerationStrategy as GenerationStrategyABC
        from nooa.strategies import get_default_strategy

        strategy = call_strategy or decorator_strategy or get_default_strategy()

        # Resolve LLM client with priority: call-level > @strategy decorator > agent's default.
        # A @strategy(llm=...) value may be a callable resolved against the agent
        # instance; only invoke it when it would actually be used, so a call-level
        # override doesn't trigger someone else's resolver side effects.
        if call_llm is not None:
            llm_client = call_llm
        else:
            plan_llm = getattr(base_method, "_plan_llm", None)
            if plan_llm is not None:
                from nooa.method_llm import resolve_method_llm

                plan_llm = resolve_method_llm(plan_llm, self.agent, method_name)
            llm_client = plan_llm or getattr(self.agent, "_llm", None)
        if llm_client is None:
            raise RuntimeError(f"No LLM client available for {method_name}")

        # Resolve truncation config: method-level @strategy(truncation=...) > agent-level
        method_truncation = getattr(base_method, "_strategy_truncation", None)
        resolved_truncation = self.agent._truncation.merge_with(method_truncation)

        # Mark that we're in a generation session
        # This allows nested ellipsis method calls to execute inline without deadlocking
        prev_in_session = _in_generation_session.get()
        _in_generation_session.set(True)

        # Generate generation_id for this session
        generation_id = str(uuid4())
        parent_generation_id = self._generation_id_stack[-1] if self._generation_id_stack else None
        # Push using copy-on-write semantics for parallel task safety
        _push_generation_id(generation_id)

        # Get strategy name for hooks
        strategy_name = getattr(strategy, "name", type(strategy).__name__)

        # Extract strategy config for tracing
        # Works for CodeActStrategy, PurePythonStrategy, PredictStrategy, ReflexionStrategy
        strategy_kwargs = {}
        if hasattr(strategy, "max_iterations"):
            strategy_kwargs["max_iterations"] = getattr(strategy, "max_iterations")  # noqa: B009
        if hasattr(strategy, "max_retries"):
            strategy_kwargs["max_retries"] = getattr(strategy, "max_retries")  # noqa: B009
        if hasattr(strategy, "max_reflections"):
            strategy_kwargs["max_reflections"] = getattr(strategy, "max_reflections")  # noqa: B009
        if hasattr(strategy, "prefill") and getattr(strategy, "prefill") is not None:  # noqa: B009
            strategy_kwargs["has_prefill"] = True

        # Call generation hooks (skip for non-traceable strategies like TemplateStrategy)
        should_trace = strategy.traceable if isinstance(strategy, GenerationStrategyABC) else True
        hook_context = None
        if should_trace:
            hook_context = call_before_hook(
                "before_generation",
                agent=self.agent,
                method_name=method_name,
                strategy=strategy_name,
                generation_id=generation_id,
                parent_generation_id=parent_generation_id,
                agent_call_id=self._agent_call_id,
                **strategy_kwargs,  # Add strategy config parameters
            )

        result = None
        exception_caught = None

        # Start harness metrics + unifiedllm bridge (cleaned up in finally below)
        _hm_ctx = _harness_metrics_lifecycle(should_trace)
        _hm_ctx.__enter__()

        # Dispatch to appropriate executor based on strategy
        try:
            # NOTE: agent_call_id is pushed by the decorator (decorators.py/metaclass.py)
            # before calling _call_plan or _execute_task. We don't push here anymore.

            # Strategy must be a GenerationStrategy instance
            if isinstance(strategy, GenerationStrategyABC):
                # Use strategy's execute() method directly
                from nooa.strategies.current_call import CurrentCall

                # Expand {placeholders} in method docstring using call arguments
                raw_docstring = getattr(method, "__doc__", None)
                expanded_docstring = None
                if raw_docstring:
                    # Build context: map parameter names to argument values
                    sig = inspect.signature(method)
                    param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
                    arg_context = dict(zip(param_names, args, strict=False))
                    arg_context.update(kwargs)
                    expanded_docstring = await self.expand_variables(
                        raw_docstring, extra_context=arg_context, error_mode="silent"
                    )

                # Build CurrentCall for the strategy
                # Map positional args to parameter names for kwargs (like from_method does)
                sig = inspect.signature(method)
                param_names = [p for p in sig.parameters.keys() if p != "self"]
                merged_kwargs = dict(kwargs)
                for i, value in enumerate(args):
                    if i < len(param_names):
                        param_name = param_names[i]
                        if param_name not in merged_kwargs:
                            merged_kwargs[param_name] = value

                # Extract return type annotation — use get_type_hints to
                # resolve PEP 563 stringified annotations.
                return_type = None
                try:
                    hints = get_type_hints(method, include_extras=True)
                    return_annotation = hints.get("return", inspect.Signature.empty)
                except (NameError, TypeError, AttributeError):
                    return_annotation = sig.return_annotation
                if return_annotation is not inspect.Signature.empty:
                    return_type = return_annotation

                # Extract pre-ellipsis code (setup code before ... marker)
                # Use _original if method was wrapped by metaclass
                from nooa.ellipsis_detection import get_pre_ellipsis_code

                original_method = getattr(method, "_original", method)
                pre_ellipsis_code = get_pre_ellipsis_code(original_method)

                # Use the call_id already pushed by the wrapper so events added
                # during this call have metadata["call_id"] matching the agent
                # call stack.  NOTE: strategies may later mutate call.id (e.g.
                # CodeActStrategy sets it to the task event tag), so
                # _prepare_context uses _agent_call_id (the stack value) for
                # EventQuery.current_call() filtering, not current_call.id.
                call_id = self._agent_call_id or str(uuid4())
                call = CurrentCall(
                    id=call_id,
                    method_name=method_name,
                    decorator="plan",
                    signature=str(sig),
                    docstring=expanded_docstring,
                    args=args,
                    kwargs=merged_kwargs,
                    parent_id=self._agent_call_stack[-2]
                    if len(self._agent_call_stack) >= 2
                    else None,
                    is_async=inspect.iscoroutinefunction(method),
                    return_type=return_type,
                    pre_ellipsis_code=pre_ellipsis_code,
                    session_locals=call_session_locals,
                    # Authoritative ordered names from the live signature (excludes
                    # 'self') so format_parameters_as_code never re-parses the string.
                    param_names=[p for p in sig.parameters if p != "self"],
                )

                # Store current call context in context vars for RuntimeServices.generate()
                # Using context vars (not instance attrs) allows parallel nested calls
                call_token = _current_call_var.set(call)
                method_token = _current_method_var.set(method)
                llm_token = _current_llm_var.set(llm_client)
                truncation_token = _current_truncation_config_var.set(resolved_truncation)
                event_format_token = _current_event_format_var.set(
                    resolved_truncation.event_format.model_dump()
                )

                # Propagate decorator context to nested calls:
                # merge parent's inherited context with this method's @strategy(context={...})
                parent_ctx = _decorator_context_var.get()
                own_ctx = getattr(getattr(method, "__func__", method), "_strategy_context", None)
                merged_ctx: dict[str, Any] | None = None
                if parent_ctx or own_ctx:
                    merged_ctx = {}
                    if parent_ctx:
                        merged_ctx.update(parent_ctx)
                    if own_ctx:
                        merged_ctx.update(own_ctx)
                decorator_ctx_token = _decorator_context_var.set(merged_ctx)

                # Propagate decorator events to nested calls:
                # Use this method's EventQuery if present, otherwise inherit parent's
                parent_evt = _decorator_events_var.get()
                own_evt = getattr(getattr(method, "__func__", method), "_strategy_events", None)
                # EventQuery: child overrides parent (not merged like context dicts)
                active_evt = own_evt if own_evt is not None else parent_evt
                decorator_evt_token = _decorator_events_var.set(active_evt)

                try:
                    # Set current strategy context var (_prepare_context reads it)
                    strategy_token = _current_strategy_var.set(strategy)

                    try:
                        result = await strategy.execute(self, call)
                        return result
                    finally:
                        _current_strategy_var.reset(strategy_token)
                finally:
                    # Restore previous context var values
                    _current_call_var.reset(call_token)
                    _current_method_var.reset(method_token)
                    _current_llm_var.reset(llm_token)
                    _current_truncation_config_var.reset(truncation_token)
                    _current_event_format_var.reset(event_format_token)
                    _decorator_context_var.reset(decorator_ctx_token)
                    _decorator_events_var.reset(decorator_evt_token)
            else:
                raise TypeError(f"Expected GenerationStrategy instance, got {type(strategy)}")
        except Exception as e:
            exception_caught = e
            raise
        finally:
            # NOTE: agent_call_id is popped by the decorator (decorators.py/metaclass.py)
            # after _call_plan or _execute_task returns. We don't pop here anymore.

            # Pop generation_id from stack using copy-on-write semantics
            _pop_generation_id()

            # Flush harness metrics BEFORE after_generation ends the span —
            # otherwise get_current_span() no longer returns the generation
            # span and harness.* attributes are dropped. Pass real exception
            # info so the context manager can chain correctly if its
            # teardown raises.
            if exception_caught is not None:
                _hm_ctx.__exit__(
                    type(exception_caught),
                    exception_caught,
                    exception_caught.__traceback__,
                )
            else:
                _hm_ctx.__exit__(None, None, None)

            # Call after generation hook
            if should_trace:
                call_after_hook(
                    "after_generation",
                    hook_context,
                    agent=self.agent,
                    method_name=method_name,
                    result=result,
                    exception=exception_caught,
                    generation_id=generation_id,
                )

            # Restore previous context variable value
            # (handles nested generation sessions correctly)
            _in_generation_session.set(prev_in_session)

    def _indent_code(self, code: str, indent: str) -> str:
        """Indent each line of code, preserving multiline string indentation."""
        lines = code.split("\n")
        try:
            # Tokenize to find string literals
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))

            # Track which lines are continuation lines of multiline strings
            # Uses 0-indexed line numbers to match enumerate(lines)
            string_ranges = set()
            fstring_middle_type = getattr(tokenize, "FSTRING_MIDDLE", None)

            for token in tokens:
                # Check for STRING tokens or FSTRING_MIDDLE (Python 3.12+)
                if token.type == tokenize.STRING or (
                    fstring_middle_type and token.type == fstring_middle_type
                ):
                    # Convert 1-based tokenizer positions to 0-indexed
                    start_line = token.start[0] - 1
                    end_line = token.end[0] - 1

                    # Check if this token spans multiple lines
                    if end_line > start_line:
                        # Mark continuation lines (not the first line, but include the last)
                        for line_idx in range(start_line + 1, end_line + 1):
                            string_ranges.add(line_idx)

            # Indent normal lines, preserve string continuation lines
            result_lines = []
            for idx, line in enumerate(lines):
                if idx in string_ranges:
                    # String continuation - keep as-is
                    result_lines.append(line)
                else:
                    # Normal code - add indent
                    result_lines.append(indent + line)

            return "\n".join(result_lines)
        except (tokenize.TokenError, IndentationError):
            # Fallback to simple indentation if tokenization fails
            return "\n".join(indent + line for line in lines)

    async def _prepare_context(
        self,
        method: Any,
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
    ) -> list[ResolvedBlock]:
        """Gather all blocks, resolve DynamicContext values, return list[ResolvedBlock].

        Thin wrapper around context_builder.build_context() that constructs
        the resolve function and strategy from the current runtime state.

        DynamicContext expression errors are displayed inline in the block content
        (not raised), so a single broken expression doesn't crash the whole
        context build.

        Args:
            method: Method being generated
            call_args: Current call positional arguments
            call_kwargs: Current call keyword arguments

        Returns:
            Ordered list of ResolvedBlock ready for render_context()
        """
        from nooa.runtime.context_builder import build_context

        tc = self.agent._truncation
        call_kwargs = call_kwargs or {}

        # Get current strategy
        strategy = getattr(method, "_plan_strategy", None)
        if strategy is None:
            strategy = _current_strategy_var.get()

        # Build evaluation context for DynamicContext expressions
        extra_context = {
            "method": method,
            "call_args": call_args,
            "call_kwargs": call_kwargs,
            "strategy": strategy,
            "datetime": datetime,
            "runtime": self,
        }

        # Hoist the format kwargs out of the resolve loop — same FormatConfig for
        # every block this turn, no need to re-dump per resolve.
        ctx_block_kwargs = tc.context_block_format.model_dump()

        async def _resolve_value(key: str, value: str | DynamicContext) -> str:
            """Resolve a value (static str or DynamicContext) to a string.

            Errors are displayed inline as "ExceptionType: message" so the
            LLM can see and fix the problem. Unlike codeact errors (which
            include full tracebacks), context block errors omit tracebacks
            because the source is a short expression, not user-written code.
            """
            if isinstance(value, DynamicContext):
                try:
                    result = await self.evaluate_expression(
                        value.expr, extra_context=extra_context, error_mode="raise"
                    )
                except Exception as e:
                    logger.warning(
                        "DynamicContext block %r failed to resolve: %s: %s (expr: %s)",
                        key,
                        type(e).__name__,
                        e,
                        value.expr,
                    )
                    return f"{type(e).__name__}: {e}"
                if result is None:
                    return "None"
                if isinstance(result, str):
                    return result
                from nooa.agentdoc import pformat as _pformat_value

                return _pformat_value(result, unquote_strings=True, **ctx_block_kwargs)
            return value

        build_result = await build_context(
            context_manager=self.agent.context_manager,
            event_manager=self.agent.event_manager,
            strategy=strategy,
            resolve_fn=_resolve_value,
            decorator_context=_decorator_context_var.get(),
            scoped_context=_scoped_blocks_var.get(),
            runtime_event_query=self.agent.event_manager.get_event_query(),
            decorator_event_query=_decorator_events_var.get(),
            scoped_event_query=_scoped_events_var.get(),
            agent_event_query=getattr(self.agent, "event_query", None),
            current_call_id=self._agent_call_id,
            context_block_format=tc.context_block_format,
        )

        # Apply the resolved cache (the only side effect)
        self.agent.context_manager._update_resolved(build_result.resolved_cache)

        return build_result.blocks

    async def _build_messages(
        self,
        method: Any,
        call_args: tuple[Any, ...] = (),
        call_kwargs: dict[str, Any] | None = None,
        *,
        tools: list[Any] | None = None,
        max_output_tokens: int | None = None,
    ) -> list[dict[str, Any]]:
        """Build messages for LLM API.

        Calls _prepare_context() to gather and resolve all blocks,
        then render_context() to format them using the agent's configured
        block and provider formatters.

        ``tools`` is accepted for call-site compatibility. This method no
        longer performs full-payload pre-call token estimation; context stats
        stay as render_context fallback until provider usage is written after a
        successful call.

        ``max_output_tokens`` is the completion budget of the upcoming call
        (``kwargs["max_tokens"]`` at the generate() call site, or the reduced
        value during context-window recovery). It is reserved out of the model
        window for budgeting and utilization: the provider rejects any request
        where prompt + completion budget exceeds the window, so the usable
        input window is ``context_window - reserve``. When the call does not
        set ``max_tokens`` explicitly, providers shrink the completion budget
        to fit and no hard reserve applies — we then fall back to the
        configured ``TruncationConfig.response_reserve_tokens`` as a planning
        reserve (0 disables).
        """
        hm = get_harness_metrics()
        with hm.timer("time_prepare_context"):
            blocks = await self._prepare_context(method, call_args, call_kwargs)
        tc = self.agent._truncation
        llm_client = _current_llm_var.get()

        effective_context_limit = tc.max_context_tokens
        ctx_window = getattr(llm_client, "context_window", None)

        # Output-token reserve (see docstring). Per-call max_tokens is the
        # binding constraint when set; otherwise the configured planning
        # reserve.
        reserved_output = max_output_tokens
        if not reserved_output:
            reserved_output = tc.response_reserve_tokens or None

        # Default (unconfigured) context budget: up to half the USABLE window
        # (model window minus the output reserve).
        if (
            effective_context_limit is None
            and ctx_window is not None
            and tc.response_reserve_tokens > 0
        ):
            usable_window = max(0, ctx_window - (reserved_output or 0))
            effective_context_limit = usable_window // 2

        # Token sizing for eviction uses the provider-calibrated chars→tokens
        # ratio (prompt_tokens / total_chars from the last response), not the
        # LLM's tokenizer. Every token number in the system is therefore
        # chars × the same ratio, anchored to real provider usage. No LLM
        # ``count_tokens`` method is required.
        ratio = self._tokens_per_char

        def count_tokens(text: str) -> int:
            # chars × ratio. No per-block floor: the budget is compared against
            # the SUM, so flooring each tiny block to ≥1 token would over-count
            # (and spuriously evict) agents with many small context blocks.
            return round(len(text) * ratio)

        with hm.timer("time_render_context"):
            provider_formatter = _resolve_provider_formatter(
                llm_client, self.agent.render_config.provider_formatter
            )
            result = render_context(
                blocks,
                block_formatter=self.agent.render_config.block_formatter,
                provider_formatter=provider_formatter,
                context_limit=effective_context_limit,
                count_tokens=count_tokens,
                event_format=tc.event_format,
                event_format_resolver=self._event_format_for_event,
                model_context_window=getattr(llm_client, "context_window", None),
                reserved_output_tokens=reserved_output,
            )

        # ``render_context`` is a framework-agnostic leaf and does not touch the
        # runtime's metrics. The runtime owns ``HarnessMetrics``, so it
        # increments the eviction counter here from the count the renderer
        # reported via its stats (see issue #330).
        if result.stats.context_blocks_dropped and isinstance(hm, HarnessMetrics):
            hm.context_limits_blocks_evicted += result.stats.context_blocks_dropped

        # Publish the rendered message list to the tracing sideband so
        # the litellm journal callback can compress block bodies into a
        # content-addressed sideband and ship a hash-only skeleton on
        # the wire.  Pure tracing concern -- the runtime hands over its
        # ``RenderedMessage``s and is otherwise oblivious to
        # ``JournalPayload`` / ``block_hash`` / SHA-256.
        from nooa.tracing._journal_builder import (
            set_journal_payload_from_messages,
        )

        set_journal_payload_from_messages(result.messages)

        # Do not pre-call estimate the full request size here. The only exact
        # prompt token count is provider usage from a successful API response;
        # generate() writes that actual value back into _last_context_stats after
        # the call. Until then, keep render_context's local estimate as a fallback
        # for diagnostics and context-window error recovery.
        messages = result.output
        self._last_context_stats = result.stats
        self._last_prompt_tokens_actual = None
        return messages
