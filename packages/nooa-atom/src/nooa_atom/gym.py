# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Atom as a NeMo Gym harness.

Gym constructs one ``AtomHarness`` per rollout with the rollout's model client as ``llm`` and
awaits :meth:`AtomHarness.solve` with the task text. The harness runs one Atom session the way
the TUI does: a ``SessionRegistry`` creates the root session with the configured turn method
(``handle_batch`` by default: one agent works the task unattended; ``handle_orchestrate`` lets the
root delegate to child agents), the task is the first prompt, and the session's ``Done`` outcome
carries the :class:`TaskResult`.

Every model call, the root's and the children's, goes through the injected client, so Gym's
model server records all of them; alias lookups are refused so no other model can be reached.
The instrumentation hooks active when ``solve`` is called are carried into each session loop,
so a harness observer sees Atom's code executions and tool calls as it would any agent's.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from nooa import Agent
from nooa.agentdoc import hidden
from nooa.interactive import Done, TaskResult
from nooa.runtime.hooks import compose_hooks, get_hooks, set_hooks
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.loader import ATOM_AGENT
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore, sessions_root

__all__ = ["TURN_METHODS", "AtomHarness"]

TURN_METHODS = ("handle_batch", "handle_orchestrate")


class AtomHarness(Agent):
    """One Atom session per Gym rollout; the task agent is Atom's own session agent.

    This class is an ``Agent`` only so a Gym adapter can construct it like any other NOOA
    agent class; it makes no model calls itself and exposes no methods to a model.
    """

    def __init__(
        self,
        llm: Any,
        *,
        turn_method: str = "handle_batch",
        workdir: str | Path | None = None,
        sessions_dir: str | Path | None = None,
        **_: Any,  # run_command and other harness-supplied arguments Atom does not need
    ) -> None:
        if turn_method not in TURN_METHODS:
            raise ValueError(f"turn_method must be one of {TURN_METHODS}, got {turn_method!r}")
        super().__init__(llm=llm)
        self.turn_method = turn_method
        self.workspace = Path(workdir or Path.cwd()).resolve()
        # Session records (<id>.db) live under sessions_dir. A directory the harness created for
        # one rollout is removed after solve(); a caller-supplied one is left alone.
        self._owned_root: Path | None = None
        if sessions_dir:
            self.sessions_dir = Path(sessions_dir)
        else:
            self._owned_root = Path(mkdtemp(prefix="gym-atom-"))
            self.sessions_dir = self._owned_root / "sessions"

    @hidden
    async def solve(self, task: str) -> TaskResult:
        """Run the task to Atom's ``Done`` and return its result."""
        outer_hooks = get_hooks()

        class HookedRegistry(SessionRegistry):
            async def _build(self, options, handle, *, restore=False):
                session = await super()._build(options, handle, restore=restore)
                # Atom starts each session loop in a fresh Context; carry the
                # harness hooks (tool records, invocation ids) into it.
                session.add_loop_context_hook(
                    lambda: set_hooks(compose_hooks(get_hooks(), outer_hooks))
                )
                return session

        def refuse_alias(alias: str | None, workspace: Path) -> Any:
            raise ValueError("A Gym rollout permits only its registered policy model")

        options = SessionOptions(
            workspace=self.workspace,
            agent_spec=ATOM_AGENT,
            llm=self.llm,
            turn_method=self.turn_method,
            sessions_dir=self.sessions_dir,
        )
        store = SessionStore(sessions_root(options.workspace, options.sessions_dir))
        registry = HookedRegistry(
            store, agent_factory=create_session_agent, llm_factory=refuse_alias
        )
        try:
            root = await registry.create(options)
            outcome = await root.prompt(task)
            await self._settle_children(registry, root)
        finally:
            try:
                await registry.close_all()
            finally:
                if self._owned_root is not None:
                    shutil.rmtree(self._owned_root, ignore_errors=True)
        if not isinstance(outcome, Done) or not isinstance(outcome.result, TaskResult):
            ended_with = (
                f"{type(outcome).__name__} carrying {type(outcome.result).__name__}"
                if isinstance(outcome, Done)
                else type(outcome).__name__
            )
            raise RuntimeError(f"Atom ended without a TaskResult: {ended_with}")
        return outcome.result

    @hidden
    @staticmethod
    async def _settle_children(registry: Any, root: Any) -> None:
        """Refuse a root outcome that leaves child turns executing.

        ``Session.close`` will not close a session whose agent is mid-turn, so an orchestrator that
        answers while a child still works would leave that child's loop, agent and model client alive
        past ``solve()``. Such children are cancelled (their turn settles before ``cancel`` returns) and
        the outcome is rejected: Atom's protocol keeps the root waiting until its children finish.
        """
        executing = [
            session
            for session in list(registry.sessions.values())
            if session is not root and registry.live_info(session).status in ("running", "working")
        ]
        for session in executing:
            await session.cancel(by="harness")
        if executing:
            ids = ", ".join(str(session.id) for session in executing)
            raise RuntimeError(
                f"Atom returned while {len(executing)} child turn(s) were still executing: {ids}"
            )
