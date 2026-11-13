# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF exporter activation API.

Three surfaces, in order of increasing ceremony:

1. :func:`enable_atif` — zero-config global enable. Auto-instruments
   every subsequently-constructed :class:`Agent`; writes trajectories
   under ``./logs/atif/<agent_name>/<session_id>.json``. Mirrors the
   shape of :func:`nooa.tracing.enable_tracing` so users who
   already use tracing have a familiar pattern.

   .. code-block:: python

       enable_atif()
       agent = InventoryAgent()
       await agent.run("…")    # trajectory auto-generated

2. :func:`atif_scope` — async context manager scoped to a single agent
   run. ``async with atif_scope(agent)`` is enough for most uses;
   path / session_id / agent_name / version all derive from sensible
   defaults if omitted.

3. :func:`install_atif` — low-level: register subscribers on a specific
   :class:`EventManager` and return an explicit uninstall callable.
   Use when you need precise lifecycle control.

The exporter is a pure read-only event subscriber — no
``EventManager.intercept(...)`` is used.
"""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from nooa.atif.exporter import AtifExporter
from nooa.standalone import _atif_exporter_var

if TYPE_CHECKING:
    from nooa.runtime.event_manager import EventManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def _default_path(agent: Any, session_id: str) -> Path:
    """Default trajectory file path: ``./logs/atif/<agent_name>/<session_id>.json``.

    Honours the ``ATIF_OUTPUT_DIR`` env var if set; otherwise writes
    under the current working directory.  Both segments are sanitized
    to filesystem-safe characters.
    """
    base = Path(os.environ.get("ATIF_OUTPUT_DIR", "logs/atif"))
    agent_name = _safe_path_segment(type(agent).__name__)
    return base / agent_name / f"{_safe_path_segment(session_id)}.json"


def _default_session_id() -> str:
    """Default session id: ISO-8601 UTC timestamp + short UUID.

    Stable across nested generations within one ``atif_scope`` call but
    new for each invocation of the scope.
    """
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{now}-{uuid4().hex[:8]}"


def _safe_path_segment(value: str) -> str:
    """Replace filesystem-unsafe characters with ``_``."""
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value) or "trajectory"


# ---------------------------------------------------------------------------
# Low-level: install_atif
# ---------------------------------------------------------------------------


def install_atif(
    event_manager: EventManager,
    *,
    path: str | Path,
    session_id: str | None,
    agent_name: str,
    agent_version: str,
    agent_model_name: str | None = None,
    agent_tool_definitions: list[dict[str, Any]] | None = None,
    agent_extra: dict[str, Any] | None = None,
    trajectory_id: str | None = None,
    cascade_to_standalones: bool = False,
) -> Callable[[], None]:
    """Register an :class:`AtifExporter` on *event_manager*.

    Subscribes to every event the exporter cares about. Returns an
    ``uninstall()`` callable that removes the subscription (and resets
    the ContextVar if it was set). ``uninstall.exporter`` exposes the
    :class:`AtifExporter` for callers that want to introspect the
    trajectory afterwards.

    Args:
        cascade_to_standalones: when ``True``, bind the cascade ContextVar at
            install time so standalone generation functions in this async
            context embed under this exporter. Use only when the caller owns one
            logical run (``atif_scope``); ``enable_atif`` leaves it ``False`` and
            binds run-scoped instead, since its agents share an async context and
            an install-time bind would cross-attribute their trajectories.
    """
    exporter = AtifExporter(
        path=path,
        session_id=session_id,
        agent_name=agent_name,
        agent_version=agent_version,
        agent_model_name=agent_model_name,
        agent_tool_definitions=agent_tool_definitions,
        agent_extra=agent_extra,
        trajectory_id=trajectory_id,
    )

    # Wildcard subscription so any event type flows into the trajectory.
    # Standalone-cascade arming happens off this subscription, via the
    # exporter's BeforeAgentCall/AfterAgentCall (and BeforeTurn) handlers.
    unsubscribers: list[Callable[[], None]] = [
        event_manager.on("*", exporter._dispatch_event),
    ]
    # atif_scope opts into an install-time binding so standalone calls outside
    # any agent-call frame also cascade; enable_atif does not (it binds
    # run-scoped instead — see cascade_to_standalones).
    if cascade_to_standalones:
        exporter._install_token = _atif_exporter_var.set(exporter)

    def uninstall() -> None:
        for unsub in unsubscribers:
            try:
                unsub()
            except Exception:  # noqa: BLE001
                logger.debug("atif: failed to unsubscribe handler", exc_info=True)
        # Defensively release any still-held cascade binding.
        if exporter._run_token is not None:
            try:
                _atif_exporter_var.reset(exporter._run_token)
            except (LookupError, ValueError):
                pass
            exporter._run_token = None
        if exporter._entrypoint_token is not None:
            try:
                _atif_exporter_var.reset(exporter._entrypoint_token)
            except (LookupError, ValueError):
                pass
            exporter._entrypoint_token = None
        if exporter._install_token is not None:
            try:
                _atif_exporter_var.reset(exporter._install_token)
            except (LookupError, ValueError):
                # Token-out-of-context (the var was rebound in a child task);
                # best-effort cleanup.
                pass
            exporter._install_token = None
        # Flush buffered events and release any remaining exporter state.
        exporter.close()

    # Expose the exporter on the uninstall callable so callers (e.g.
    # atif_scope) can read final state without re-installing.
    uninstall.exporter = exporter  # type: ignore[attr-defined]

    return uninstall


# ---------------------------------------------------------------------------
# Async context manager: atif_scope
# ---------------------------------------------------------------------------


@asynccontextmanager
async def atif_scope(
    agent: Any,
    *,
    inherit_parent: bool = False,
    path: str | Path | None = None,
    session_id: str | None = None,
    agent_name: str | None = None,
    agent_version: str | None = None,
    agent_model_name: str | None = None,
    agent_tool_definitions: list[dict[str, Any]] | None = None,
    agent_extra: dict[str, Any] | None = None,
    trajectory_id: str | None = None,
) -> AsyncIterator[AtifExporter | None]:
    """Async context manager: install + uninstall ATIF export around a block.

    All parameters are optional. With zero overrides::

        async with atif_scope(agent):
            await agent.run(...)

    writes the trajectory to
    ``./logs/atif/<AgentClass>/<timestamp>-<uuid8>.json`` (override the
    base dir with the ``ATIF_OUTPUT_DIR`` env var).

    With ``inherit_parent=True``, subscribe to this agent while retaining the
    active parent until an agent call starts. The existing call handlers embed
    the child trajectory in that parent; no separate child file is written.
    Without an active parent this mode is a no-op and yields ``None``.
    Newly installed exporters default to the parent's path and session ID;
    exporters already installed by ``enable_atif`` retain their metadata::

        async with atif_scope(parent, path="trajectory.json"):
            async with atif_scope(child, inherit_parent=True):
                await child.run(...)

    Defaults:
        - ``path`` → ``_default_path(agent, session_id)``
        - ``session_id`` → ``_default_session_id()`` (ISO timestamp + UUID)
        - ``agent_name`` → ``type(agent).__name__``
        - ``agent_version`` → ``"0.0.0"`` (or ``getattr(agent, '__version__', ...)``)
        - ``trajectory_id`` → fresh UUID4 (set by :class:`AtifExporter`)

    On exception inside the wrapped block, the exporter records
    ``extra.crashed = True`` plus the exception type before propagating
    (see :meth:`AtifExporter.finalize_on_exception`).

    Yields the active :class:`AtifExporter` for tests that want to
    introspect ``get_trajectory()`` after the block exits.
    """
    parent = _atif_exporter_var.get() if inherit_parent else None
    if inherit_parent and parent is None:
        yield None
        return

    resolved_agent_name: str = agent_name if agent_name is not None else type(agent).__name__
    resolved_agent_version: str = (
        agent_version
        if agent_version is not None
        else getattr(agent, "__version__", None) or "0.0.0"
    )
    resolved_session_id = (
        session_id
        if session_id is not None
        else parent._trajectory.session_id
        if parent is not None
        else _default_session_id()
    )
    resolved_path: str | Path = (
        path
        if path is not None
        else parent.path
        if parent is not None
        else _default_path(agent, resolved_session_id)
    )

    # enable_atif already subscribed this agent. A second exporter would
    # mistake the first for its parent and duplicate the child trajectory.
    exporter = getattr(agent, "_atif_auto_exporter", None) if inherit_parent else None
    if exporter is parent and parent is not None:
        yield exporter
        return
    uninstall = None
    if exporter is None:
        uninstall = install_atif(
            agent.event_manager,
            path=resolved_path,
            session_id=resolved_session_id,
            agent_name=resolved_agent_name,
            agent_version=resolved_agent_version,
            agent_model_name=agent_model_name,
            agent_tool_definitions=agent_tool_definitions,
            agent_extra=agent_extra,
            trajectory_id=trajectory_id,
            # Inherited scopes leave the parent bound until the child call.
            cascade_to_standalones=not inherit_parent,
        )
        exporter = uninstall.exporter  # type: ignore[attr-defined]
    writes_to_disk = exporter._writes_to_disk
    if inherit_parent:
        exporter._writes_to_disk = False
    try:
        yield exporter
    except BaseException as exc:
        # Mark the trajectory as crashed and write before propagating.
        exporter.finalize_on_exception(exc)
        raise
    finally:
        if uninstall is not None:
            uninstall()
        else:
            exporter.close()
        exporter._writes_to_disk = writes_to_disk
        if parent is not None:
            # Call handlers already emitted the handoff. Refresh its snapshot
            # after exception metadata and buffered dispatches have been flushed.
            trajectory = exporter.get_trajectory()
            with parent._lock:
                if any(
                    child.trajectory_id == trajectory.trajectory_id
                    for child in parent._trajectory.subagent_trajectories or []
                ):
                    parent._upsert_subagent(trajectory)
                    parent._write()


# ---------------------------------------------------------------------------
# Zero-config global: enable_atif
# ---------------------------------------------------------------------------


_AGENT_INIT_PATCHED = False
_ENABLE_ATIF_CONFIG: dict[str, Any] = {}


def enable_atif(
    *,
    output_dir: str | Path | None = None,
    agent_version: str = "0.0.0",
) -> None:
    """Globally enable ATIF trajectory export for every agent created
    afterwards.

    Patches :meth:`Agent.__init__` so each freshly-instantiated Agent
    has the ATIF subscribers attached to its :class:`EventManager`
    automatically. Trajectories are written under
    ``<output_dir>/<AgentClass>/<session_id>.json``. The session id is
    generated per Agent instance.

    Idempotent — calling :func:`enable_atif` more than once just updates
    the configuration (output_dir, default version) without
    re-patching.

    Args:
        output_dir: base directory for generated trajectories. Defaults
            to the ``ATIF_OUTPUT_DIR`` env var or ``./logs/atif``.
        agent_version: ``agent.version`` value to embed.

    Example::

        from nooa.atif import enable_atif

        enable_atif()
        agent = MyAgent()
        await agent.run(...)   # trajectory auto-generated

    For per-call scoping (recommended in tests / library code), prefer
    :func:`atif_scope` instead.
    """
    global _AGENT_INIT_PATCHED

    _ENABLE_ATIF_CONFIG["output_dir"] = (
        Path(output_dir)
        if output_dir is not None
        else Path(os.environ.get("ATIF_OUTPUT_DIR", "logs/atif"))
    )
    _ENABLE_ATIF_CONFIG["agent_version"] = agent_version

    if _AGENT_INIT_PATCHED:
        return

    from nooa.agent import Agent

    _orig_init = Agent.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        """Attach an exporter after initialization and retain it for inherited scopes."""
        _orig_init(self, *args, **kwargs)
        # Auto-install on this fresh agent's EventManager.
        try:
            session_id = _default_session_id()
            agent_name = type(self).__name__
            cfg_dir = _ENABLE_ATIF_CONFIG["output_dir"]
            path = (
                Path(cfg_dir)
                / _safe_path_segment(agent_name)
                / f"{_safe_path_segment(session_id)}.json"
            )
            uninstall = install_atif(
                self.event_manager,
                path=path,
                session_id=session_id,
                agent_name=agent_name,
                agent_version=_ENABLE_ATIF_CONFIG["agent_version"],
            )
            self._atif_auto_exporter = uninstall.exporter  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            # Auto-install is best-effort; never block agent construction.
            logger.exception("atif: enable_atif auto-install failed")

    Agent.__init__ = _patched_init  # type: ignore[method-assign]
    _AGENT_INIT_PATCHED = True
