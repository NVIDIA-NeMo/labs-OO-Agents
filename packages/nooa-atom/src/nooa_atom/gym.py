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

import asyncio
import shutil
from pathlib import Path
from tempfile import mkdtemp
from typing import Any

from nooa import Agent
from nooa.agentdoc import hidden
from nooa.interactive import Done, TaskResult
from nooa.runtime.hooks import compose_hooks, get_hooks, set_hooks
from nooa_atom.agent.factory import create_session_agent
from nooa_atom.benchmark_policy import BenchmarkPolicyGuard, BenchmarkPolicyViolation
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

    benchmark_policy_required = True

    def __init__(
        self,
        llm: Any,
        *,
        turn_method: str = "handle_batch",
        workdir: str | Path | None = None,
        sessions_dir: str | Path | None = None,
        benchmark_policy: str | None = None,
        experimental_done_reflection: bool | None = None,
        benchmark_allow_session_titles: bool | None = None,
        benchmark_policy_path: str | Path | None = None,
        experimental_reflection_min_turns: int = 50,
        experimental_reflection_min_seconds: float = 3600.0,
        session_name: str | None = None,
        **extra: Any,
    ) -> None:
        if turn_method not in TURN_METHODS:
            raise ValueError(f"turn_method must be one of {TURN_METHODS}, got {turn_method!r}")
        super().__init__(llm=llm)
        unknown = [key for key in extra if key.startswith(("experimental_", "benchmark_"))]
        # The benchmark policy guard is opt-in: any benchmark key (or a receipt path)
        # selects it, and then every key must be explicit and valid (fail closed).
        # With none, a plain Gym adapter runs Atom on its defaults: reflection is
        # off and this harness never requests a session title. No receipt is written.
        guarded = any(
            value is not None
            for value in (
                benchmark_policy,
                experimental_done_reflection,
                benchmark_allow_session_titles,
                benchmark_policy_path,
            )
        )
        self.benchmark_guard: BenchmarkPolicyGuard | None = None
        config: dict[str, Any] = {}
        if not guarded:
            if unknown:
                raise ValueError(f"unknown benchmark/experiment keys: {sorted(unknown)}")
        else:
            self.benchmark_guard = BenchmarkPolicyGuard(benchmark_policy_path)
            config = {
                "benchmark_policy": benchmark_policy,
                "experimental_done_reflection": experimental_done_reflection,
                "benchmark_allow_session_titles": benchmark_allow_session_titles,
            }
            try:
                bind = getattr(llm, "bind_episode_policy", None)
                if not callable(bind):
                    raise ValueError("Atom benchmark requires the instrumented Gym policy client")
                bind(self.benchmark_guard)
                self.benchmark_guard.configure(config)
                if unknown:
                    raise ValueError(f"unknown benchmark/experiment keys: {sorted(unknown)}")
            except BaseException:
                self.benchmark_guard.violate("construction_configuration_invalid")
                self.benchmark_guard.finish(None, "construction_error")
                raise
        self._used = False
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
        self.options = SessionOptions(
            workspace=self.workspace,
            agent_spec=ATOM_AGENT,
            llm=self.llm,
            turn_method=self.turn_method,
            sessions_dir=self.sessions_dir,
            name=session_name,
            experimental_reflection_min_turns=experimental_reflection_min_turns,
            experimental_reflection_min_seconds=experimental_reflection_min_seconds,
            **config,
        )

    @hidden
    def benchmark_policy_receipt(self) -> dict[str, Any] | None:
        """The final policy receipt, or None for a plain (unguarded) harness."""
        if self.benchmark_guard is None:
            return None
        return self.benchmark_guard.final_receipt()

    @hidden
    async def solve(self, task: str) -> TaskResult:
        guard = self.benchmark_guard
        if guard is None:
            return await self._solve_plain(task)
        if self._used:
            guard.refuse("harness_episode_reused")
        self._used = True
        outer_hooks = get_hooks()

        class HookedRegistry(SessionRegistry):
            def _publish(self, session, reservation):
                # Closed sessions leave registry.sessions; preserve publication
                # identities for final coverage of the entire episode tree.
                guard.published_ids.add(session.id)
                if session.id not in guard.sessions:
                    guard.violate("published_session_uncovered", session_id=session.id)
                return super()._publish(session, reservation)

            async def _build(self, options, handle, *, restore=False):
                guard.expect(handle, options)
                session = await super()._build(options, handle, restore=restore)
                try:
                    guard.install(session)
                    guard.check_session(session, require_ready=False)

                    # Each loop starts in a fresh Context, after child preparation.
                    # Preserve the outer _ToolHooks invocation/header correlation.
                    def ready():
                        set_hooks(compose_hooks(get_hooks(), outer_hooks, guard))
                        guard.ready(session)

                    session.add_loop_context_hook(ready)
                    return session
                except BaseException:
                    # Base create() has not received this Session yet.
                    await self._close_half_built(session)
                    raise

        def refuse_alias(alias: str | None, workspace: Path) -> Any:
            raise ValueError("A Gym rollout permits only its registered policy model")

        registry = HookedRegistry(
            SessionStore(sessions_root(self.workspace, self.sessions_dir)),
            agent_factory=create_session_agent,
            llm_factory=refuse_alias,
        )
        outcome, error = None, None
        try:
            if (
                guard._client is not self.llm
                or getattr(self.llm, "_episode_policy_guard", None) is not guard
            ):
                guard.refuse("client_policy_guard_missing")
            root = await registry.create(self.options)
            outcome = await root.prompt(task)
            await self._settle_children(registry, root)
        except BaseException as exc:
            error = exc
        finally:
            # Own teardown to completion even if the caller cancels again.
            cleanup = asyncio.create_task(registry.close_all())
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError as exc:
                        error = error or exc
                cleanup.result()
            except BaseException as exc:
                guard.violate("close_failed")
                error = error or exc
            try:
                receipt = guard.finish(
                    registry,
                    "cancelled"
                    if isinstance(error, asyncio.CancelledError)
                    else "error"
                    if error
                    else "completed",
                )
            except BaseException as verdict_error:
                if isinstance(error, asyncio.CancelledError):
                    raise error from verdict_error
                raise
            finally:
                guard.detach()
        # Keep f497's owned-directory cleanup after verified session teardown.
        # Failed policy/cleanup evidence is retained rather than deleted.
        if self._owned_root is not None and receipt["verdict"] == "pass":
            shutil.rmtree(self._owned_root, ignore_errors=True)
        # Keep cancellation semantics; its final receipt still carries failures.
        if isinstance(error, asyncio.CancelledError):
            raise error
        if receipt["verdict"] != "pass":
            raise BenchmarkPolicyViolation("benchmark policy/coverage verdict failed") from error
        if error is not None:
            raise error
        guard.final_receipt()  # successful Done cannot replace a missing/unwritten verdict
        if not isinstance(outcome, Done) or not isinstance(outcome.result, TaskResult):
            ended_with = (
                f"{type(outcome).__name__} carrying {type(outcome.result).__name__}"
                if isinstance(outcome, Done)
                else type(outcome).__name__
            )
            raise RuntimeError(f"Atom ended without a TaskResult: {ended_with}")
        return outcome.result

    @hidden
    async def _solve_plain(self, task: str) -> TaskResult:
        """No benchmark policy: one Atom session, the harness hooks carried into its loops."""
        if self._used:
            raise RuntimeError("an AtomHarness runs one episode; construct another")
        self._used = True
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

        registry = HookedRegistry(
            SessionStore(sessions_root(self.workspace, self.sessions_dir)),
            agent_factory=create_session_agent,
            llm_factory=refuse_alias,
        )
        try:
            root = await registry.create(self.options)
            outcome = await root.prompt(task)
            await self._settle_children(registry, root)
        finally:
            # Own teardown to completion even if the caller cancels again.
            cleanup = asyncio.create_task(registry.close_all())
            cancelled: asyncio.CancelledError | None = None
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError as exc:
                        cancelled = cancelled or exc
                cleanup.result()
            finally:
                if self._owned_root is not None:
                    shutil.rmtree(self._owned_root, ignore_errors=True)
            if cancelled is not None:
                raise cancelled
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
        """Refuse a root outcome that leaves child sessions with unfinished work.

        ``Session.close`` will not close a session whose agent is mid-turn, so an orchestrator that
        answers while a child still works would leave that child's loop, agent and model client alive
        past ``solve()``. Such children are cancelled (their turn settles before ``cancel`` returns) and
        the outcome is rejected: Atom's protocol keeps the root waiting until its children finish.
        """
        # A child is unfinished while its turn runs, while its last turn ended Waiting, or while its
        # prompt is still queued (a freshly delegated child before its turn starts); the registry's
        # idle check covers all three.
        unfinished = [
            session
            for session in list(registry.sessions.values())
            if session is not root and not registry.is_idle(session)
        ]
        # Fence every unfinished child before the first await: cancelling one child yields, and a
        # sibling whose prompt is still queued would otherwise start a turn (and change the
        # workspace) in that gap. Pausing dispatch keeps queued input queued until close.
        for session in unfinished:
            session.pause_dispatch()
        for session in unfinished:
            await session.cancel(by="harness")
        if unfinished:
            ids = ", ".join(str(session.id) for session in unfinished)
            raise RuntimeError(
                f"Atom returned while {len(unfinished)} child session(s) had unfinished work: {ids}"
            )
