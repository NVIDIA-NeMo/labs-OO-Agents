# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenInference instrumentation for NVIDIA OO Agents.

Composable exporter architecture with exporter-agnostic session routing.

Usage::

    from openinference_instrumentation_nooa import enable_tracing, exporters

    # Zero-config: OTLP to local viewer, falls back to JSONL files
    enable_tracing()

    # Multiple exporters
    enable_tracing(exporters=[exporters.jsonl("./traces"), exporters.langfuse()])

    # JSONL only (no viewer required)
    enable_tracing(exporters=[exporters.jsonl()])

    # Compact portable file (OTLP spans + content-addressed message journal)
    enable_tracing(exporters=[exporters.journal_file("./traces")])
"""

from __future__ import annotations

import contextlib
import os
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanLimits, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.sdk.trace.id_generator import RandomIdGenerator

from nooa._version import __version__
from nooa.tracing import exporters as exporters_mod

# NOTE: The old OTel-based ``AtifTrajectoryExporter`` was removed when ATIF
# export became event-subscriber-based; see
# ``nooa.atif.install_atif`` / ``atif_scope``.
from nooa.tracing._hooks_impl import OpenInferenceHooks, end_active_spans
from nooa.tracing._metadata import get_all_metadata
from nooa.tracing._otlp_file_exporter import OtlpJsonFileExporter
from nooa.tracing._otlp_http_exporter import OtlpJsonHttpExporter
from nooa.tracing._session import get_session, set_session
from nooa.tracing._session_processor import SessionSpanProcessor

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

_enabled: bool = False
_provider: TracerProvider | None = None
_hooks: OpenInferenceHooks | None = None  # retained so we can re-register per-Task


class _IsolatedIdGenerator(RandomIdGenerator):
    """Span/trace id generator backed by its own ``random.Random`` instance.

    OpenTelemetry's default ``RandomIdGenerator`` draws ids from the
    process-global ``random`` module. User code run via ``execute_python``
    routinely calls ``random.seed(...)`` (simulations, reproducible demos),
    which makes the global RNG deterministic — so two executions that reseed
    to the same state get handed *identical* span ids. The trace explorer then
    warns "Found N duplicate span_id(s) in trace" and may mis-attribute spans.

    Using a private RNG instance fully decouples id generation from whatever
    user code does to the global module.
    """

    def __init__(self) -> None:
        super().__init__()
        import secrets

        # SystemRandom (os.urandom-backed) is both cryptographically strong
        # and immune to global ``random.seed()`` — satisfies Ruff S311 for
        # security-sensitive trace ids while keeping the isolation guarantee.
        self._rng = secrets.SystemRandom()

    def generate_span_id(self) -> int:
        span_id = self._rng.getrandbits(64)
        while span_id == trace.INVALID_SPAN_ID:
            span_id = self._rng.getrandbits(64)
        return span_id

    def generate_trace_id(self) -> int:
        trace_id = self._rng.getrandbits(128)
        while trace_id == trace.INVALID_TRACE_ID:
            trace_id = self._rng.getrandbits(128)
        return trace_id


# True after a probe found the viewer unreachable (prevents repeated warnings)
_probe_failed: bool = False


# `exporters` is importable directly:
#   from openinference_instrumentation_nooa import exporters
exporters = exporters_mod


# ---------------------------------------------------------------------------
# Instrumentor
# ---------------------------------------------------------------------------


class NemoOOAgentsInstrumentor:
    """OpenTelemetry instrumentor for NVIDIA OO Agents.

    Instruments nooa framework to emit OpenTelemetry spans with
    OpenInference semantic conventions.
    """

    def __init__(self):
        self._hooks: OpenInferenceHooks | None = None
        self._is_instrumented = False

    def instrument(self, **kwargs: Any) -> None:
        """Enable nooa instrumentation."""
        if self._is_instrumented:
            return

        tracer_provider = kwargs.get("tracer_provider") or trace.get_tracer_provider()
        tracer = tracer_provider.get_tracer(__name__, __version__)

        self._hooks = OpenInferenceHooks(tracer=tracer)

        from nooa.runtime.hooks import set_hooks

        set_hooks(self._hooks)
        self._is_instrumented = True

    def uninstrument(self, **kwargs: Any) -> None:
        """Disable nooa instrumentation."""
        if not self._is_instrumented:
            return

        from nooa.runtime.hooks import set_hooks

        set_hooks(None)
        self._hooks = None
        self._is_instrumented = False


# ---------------------------------------------------------------------------
# Core entry point
# ---------------------------------------------------------------------------

# OtlpJsonFileExporter and ConsoleSpanExporter are truly local (in-process I/O),
# so SimpleSpanProcessor is appropriate.  OtlpJsonHttpExporter makes a network
# call (even to localhost) and must use BatchSpanProcessor so that:
#   1. Span exports are batched — far fewer HTTP POSTs under parallel load.
#   2. Exports happen on a background thread — the agent is never blocked.
#   3. BatchSpanProcessor has a built-in queue with drop detection.
_LOCAL_EXPORTER_TYPES: tuple[type, ...] = (
    OtlpJsonFileExporter,
    ConsoleSpanExporter,
)


def probe_otlp_endpoint(endpoint: str, timeout: float | None = None) -> bool:
    """Check whether the OTLP endpoint is reachable.

    Returns ``True`` if the server responds (any HTTP status), ``False`` on
    connection refused or timeout.

    *timeout* defaults to the ``OTLP_PROBE_TIMEOUT`` env var (seconds) if set,
    otherwise **2.0 s**.

    Uses GET /api/eval/health instead of POSTing to the traces endpoint so we
    don't accidentally create ``unknown_*`` sessions in the viewer's store.

    Useful for external callers (e.g. the Harbor runner) that need to decide
    whether to send traces live vs. fall back to file export.
    """
    if timeout is None:
        raw = os.getenv("OTLP_PROBE_TIMEOUT", "2.0")
        try:
            timeout = float(raw)
        except (ValueError, TypeError):
            timeout = 2.0
    base = endpoint.rstrip("/").removesuffix("/v1/traces").removesuffix("/v1")
    health_url = f"{base}/api/eval/health"
    try:
        from nooa.tracing._viewer_auth import apply_viewer_auth

        req = urllib.request.Request(health_url, headers=apply_viewer_auth({}), method="GET")
        with urllib.request.urlopen(req, timeout=timeout):
            pass
        return True
    except urllib.error.HTTPError:
        # Server is up but returned an error (e.g. 400, 405) — still reachable
        return True
    except Exception:
        return False


def enable_tracing(
    exporters: list[SpanExporter] | None = None,
    *,
    experiment: str | None = None,
    extra_resource_attrs: dict[str, Any] | None = None,
) -> None:
    """Configure tracing with one or more exporters.

    **Reconfigure behaviour:** When called with explicit *exporters* on an
    already-enabled provider, the old exporter processors are shut down and
    **replaced** by the new ones.  The ``SessionSpanProcessor`` and
    instrumentation hooks are preserved.  This lets the auto-probe set up a
    default exporter at import time which is then cleanly replaced when the
    eval pipeline (or user code) specifies the real target.

    When called with ``exporters=None`` after tracing is already enabled the
    call is a no-op — the auto-probe runs at most once.

    When called with no *exporters* argument the default behaviour is:

    1. Probe the local OTLP endpoint (``localhost:5001``).
    2. If reachable → send spans via OTLP HTTP to the viewer.
    3. If unreachable → print a warning and disable tracing entirely.

    A log message is printed in both cases so you always know where traces go.

    Args:
        exporters: List of :class:`SpanExporter` instances.  When ``None`` the
            fallback logic described above is used.
        experiment: Optional experiment name for grouping traces.  Added as a
            resource attribute.  Defaults to ``TRACE_EXPERIMENT`` env var.
        extra_resource_attrs: Optional dict of additional resource attributes
            stored with traces (e.g. ``{"eval.model": "gpt-4o"}``).
    """
    global _enabled, _provider, _probe_failed, _hooks

    # --- Fast paths for no-arg (auto-probe) calls --------------------------
    if exporters is None:
        if _enabled and _provider is not None:
            _re_register_hooks()
            return
        if _probe_failed:
            return
        exporters = _default_exporters()
        if exporters is None:
            return

    # --- Already enabled: replace exporters on existing provider -----------
    if _enabled and _provider is not None:
        _replace_exporters(_provider, exporters)
        _print_trace_target(exporters, experiment or os.getenv("TRACE_EXPERIMENT"))
        # Re-register hooks in the current async context.
        # Hooks are stored in a ContextVar; loop.run_until_complete() creates a fresh
        # Task context from the calling thread each time, so hooks set in a previous
        # Task are invisible here.  Without this, task 2+ in a persistent subprocess
        # worker have no hooks → no AGENT/GENERATION spans.
        _re_register_hooks()
        return

    # --- First-time setup --------------------------------------------------
    # Metadata
    metadata = get_all_metadata()

    # Experiment name (optional)
    experiment = experiment or os.getenv("TRACE_EXPERIMENT")
    if experiment:
        metadata["experiment"] = experiment

    # Merge extra resource attributes
    if extra_resource_attrs:
        for k, v in extra_resource_attrs.items():
            if v is not None:
                metadata[k] = v

    resource = Resource(attributes=metadata)

    # Create or reuse TracerProvider
    existing_provider = trace.get_tracer_provider()

    if isinstance(existing_provider, TracerProvider):
        tracer_provider = existing_provider
    else:
        tracer_provider = TracerProvider(
            resource=resource,
            # User code run via execute_python may call random.seed(); OTel's
            # default id generator draws from the global random module, so a
            # private-RNG generator keeps span ids unique regardless.
            id_generator=_IsolatedIdGenerator(),
            # No attribute cap. LiteLLM's instrumentor emits one attribute
            # per message in the conversation history
            # (``llm.input_messages.N.message.role`` / ``.content`` /
            # ``.tool_call_id`` / …). A long conversation overflows any
            # finite cap and OTel SDK's ``BoundedAttributes`` evicts the
            # OLDEST entries FIFO — which is exactly where ``session.id``
            # lives (stamped by :class:`SessionSpanProcessor.on_start`
            # *before* LiteLLM adds anything else). Evicted ``session.id``
            # → span exports with no session → viewer buckets it as
            # ``unknown_*``. The OTLP exporter already handles multi-MB
            # spans, so drop the cap.
            span_limits=SpanLimits(max_span_attributes=SpanLimits.UNSET),
        )
        trace.set_tracer_provider(tracer_provider)

    # Session processor first -- stamps session.id on span attrs before export
    tracer_provider.add_span_processor(SessionSpanProcessor())

    # Add one processor per exporter
    _add_exporters(tracer_provider, exporters)

    # Default session ID so spans always have one
    default_session = datetime.now(UTC).strftime("%Y%m%d_%H%M%S") + "_" + uuid4().hex[:8]
    set_session(default_session)

    # Store provider for flush/shutdown
    _provider = tracer_provider

    # Instrument nooa hooks; capture the instance for re-registration
    # in future asyncio task contexts (see _re_register_hooks).
    with contextlib.suppress(ImportError):
        NemoOOAgentsInstrumentor().instrument(tracer_provider=tracer_provider)
        from nooa.runtime.hooks import get_hooks

        _hooks = get_hooks()  # type: ignore[assignment]
        assert _hooks is not None, (
            "NemoOOAgentsInstrumentor.instrument() should have called set_hooks()"
        )

    # Instrument litellm if available
    _instrument_litellm(tracer_provider)

    # Print trace target
    _print_trace_target(exporters, experiment)

    _enabled = True


def _re_register_hooks() -> None:
    """Re-register instrumentation hooks in the current async context.

    Hooks are stored in a ContextVar (``_instrumentation_hooks_var``).  When
    the eval pipeline uses a persistent subprocess worker, each task is run via
    ``loop.run_until_complete(run_task(...))``, which creates a **new** asyncio
    Task whose context is copied from the calling (main) thread.  The main
    thread never has hooks registered, so every task-2+ Task inherits
    ``hooks = None``.  Without this call, no AGENT/GENERATION spans are emitted
    for any task after the first.
    """
    if _hooks is None:
        return
    with contextlib.suppress(ImportError):
        from nooa.runtime.hooks import set_hooks

        set_hooks(_hooks)


def _add_exporters(provider: TracerProvider, exporters: list[SpanExporter]) -> None:
    """Attach span processors for each exporter to *provider*.

    Exporters that are known-local (file, console) or that set
    ``synchronous = True`` use SimpleSpanProcessor (immediate, no queue).
    All others use BatchSpanProcessor (async background thread).

    All processors are wrapped with ``SecretScrubSpanProcessor`` to redact
    API keys, tokens, and other secrets from span attributes before export.
    """
    from nooa.tracing._secret_scrubber import SecretScrubSpanProcessor

    for exp in exporters:
        if isinstance(exp, _LOCAL_EXPORTER_TYPES) or getattr(exp, "synchronous", False):
            inner = SimpleSpanProcessor(exp)
        else:
            inner = BatchSpanProcessor(
                exp,
                schedule_delay_millis=1000,
                max_queue_size=8192,
            )
        provider.add_span_processor(SecretScrubSpanProcessor(inner))


def _replace_exporters(provider: TracerProvider, exporters: list[SpanExporter]) -> None:
    """Replace all exporter processors on *provider*, keeping SessionSpanProcessor.

    Shuts down the old exporter processors (flushing pending spans), then
    installs new ones for the given *exporters*.
    """
    multi = provider._active_span_processor
    old_processors = list(getattr(multi, "_span_processors", []))

    # Partition: keep SessionSpanProcessors, shut down the rest
    keep: list = []
    for proc in old_processors:
        if isinstance(proc, SessionSpanProcessor):
            keep.append(proc)
        else:
            with contextlib.suppress(Exception):
                proc.shutdown()

    # Replace the processor list (SessionSpanProcessor stays)
    with getattr(multi, "_lock", contextlib.nullcontext()):
        multi._span_processors = tuple(keep)

    # Add new exporter processors
    _add_exporters(provider, exporters)


def _default_exporters() -> list[SpanExporter] | None:
    """Select default exporters: OTLP if viewer is running, else silently disable.

    When ``OTLP_ENDPOINT`` is explicitly set the user opted in — warn on failure.
    When using the default endpoint, stay silent so tracing is invisible until
    ``nooa start-dev`` is running.
    """
    global _probe_failed

    explicit_endpoint = os.getenv("OTLP_ENDPOINT")
    endpoint = explicit_endpoint or "http://localhost:5001/v1/traces"

    if probe_otlp_endpoint(endpoint):
        return [exporters_mod.journal(endpoint=endpoint)]

    _probe_failed = True

    if explicit_endpoint is not None:
        print(
            f"OTLP_ENDPOINT ({endpoint}) is not reachable — tracing disabled.",
            file=sys.stderr,
        )

    return None


def _instrument_litellm(tracer_provider: TracerProvider) -> None:
    """Instrument LiteLLM with the OpenInference auto-instrumentor.

    Required dep: ``openinference-instrumentation-litellm`` is what stamps
    ``llm.input_messages.*`` / ``llm.output_messages.*`` onto LLM spans
    at the source.  Without it, file exports silently lack message
    content (the wire-strip-and-reconstruct path on the viewer side
    can't recover what was never produced).  Listed in
    ``pyproject.toml::project.dependencies`` so install failures surface
    here rather than as missing message attrs at runtime.
    """
    from openinference.instrumentation.litellm import LiteLLMInstrumentor

    from nooa.tracing._litellm_patch import apply_litellm_patch

    LiteLLMInstrumentor().instrument(tracer_provider=tracer_provider)
    apply_litellm_patch()


def _describe_exporter(exp: SpanExporter) -> str:
    """Return a human-readable description of an exporter."""
    describe = getattr(exp, "describe", None)
    if describe is not None:
        return describe()
    return type(exp).__name__


def _print_trace_target(exporters: list[SpanExporter], experiment: str | None) -> None:
    """Print where traces are being sent."""
    session_id = get_session()
    descriptions: list[str] = []
    has_file_exporter = False
    for exp in exporters:
        if isinstance(exp, OtlpJsonFileExporter):
            has_file_exporter = True
            descriptions.append(exp.describe(session_id))
        else:
            descriptions.append(_describe_exporter(exp))

    suffix = f" (experiment={experiment})" if experiment else ""
    print(f"OTel tracing enabled: {', '.join(descriptions)}{suffix}")

    if has_file_exporter and len(exporters) == 1:
        print("  Tip: Run `nooa start-dev` to launch the trace viewer for interactive exploration.")


# ---------------------------------------------------------------------------
# Lifecycle helpers
# ---------------------------------------------------------------------------


def flush_traces(timeout_millis: int = 30000) -> None:
    """Force-flush all pending spans across all exporters."""
    if _provider:
        _provider.force_flush(timeout_millis)


def shutdown_traces() -> None:
    """Graceful shutdown: flush pending spans + release resources."""
    if _provider:
        _provider.force_flush()
        _provider.shutdown()


# ---------------------------------------------------------------------------
# Public exports
# ---------------------------------------------------------------------------

__all__ = [
    "NemoOOAgentsInstrumentor",
    "OtlpJsonFileExporter",
    "OtlpJsonHttpExporter",
    "enable_tracing",
    "end_active_spans",
    "exporters",
    "probe_otlp_endpoint",
    "set_session",
    "get_session",
    "flush_traces",
    "shutdown_traces",
]
