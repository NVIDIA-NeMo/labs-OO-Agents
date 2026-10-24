# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run Atom as a NeMo Gym harness through ``nooa_bench.gym``.

One native Atom session per episode. The root turn method is configurable
(``handle_batch`` by default); delegated children always run ``handle_batch``.
The Gym-injected policy client is the only model: alias lookups are refused.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from nooa.interactive import Done, TaskResult
from nooa.runtime.hooks import compose_hooks, get_hooks, set_hooks
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.session.loader import ATOM_AGENT
from nooa_atom.session.options import SessionOptions
from nooa_atom.session.registry import SessionRegistry
from nooa_atom.session.store import SessionStore, sessions_root

TURN_METHODS = ("handle_batch", "handle_orchestrate")


class AtomHarness:
    """Construction shell for ``nooa_bench.gym_worker``; the task agent is Atom's own session."""

    def __init__(
        self,
        llm: Any,
        *,
        turn_method: str = "handle_batch",
        workdir: str | Path | None = None,
        sessions_dir: str | Path | None = None,
        **_: Any,  # run_command and other worker-supplied arguments Atom does not need
    ) -> None:
        if turn_method not in TURN_METHODS:
            raise ValueError(f"turn_method must be one of {TURN_METHODS}, got {turn_method!r}")
        self.llm = llm
        self.turn_method = turn_method
        self.workspace = Path(workdir or Path.cwd()).resolve()
        self.sessions_dir = (
            Path(sessions_dir) if sessions_dir else Path(mkdtemp(prefix="gym-atom-")) / "sessions"
        )

    async def solve(self, task: str) -> TaskResult:
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
        finally:
            await registry.close_all()
        if not isinstance(outcome, Done) or outcome.result is None:
            raise RuntimeError(f"Atom ended without a task result: {type(outcome).__name__}")
        return outcome.result
