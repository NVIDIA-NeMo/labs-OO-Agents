# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch the ARC-AGI-3 solver agent in headless mode (no TUI / tmux required).

Runs the agent directly via the ``InteractiveAgent`` dispatcher loop — no
``nooa_tui`` package needed. All agent behaviours (memory, reflection,
summarization, effort ladder, IPC tail, context blocks) are preserved; the only
thing removed is the terminal-drawing layer.

Usage (normally started by run_solver.py):

    .venv/bin/python examples/arc_agi_3/launcher.py \
        --run-dir results/arc_agi_3/nemo_solver/<run> --game ls20 --variant memory
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))

from arc_llm import build_embedding_config, build_llm  # noqa: E402
from solver_agent import MdArcSolverAgent, MemArcSolverAgent  # noqa: E402

DEFAULT_SKILL = "grid-game-solver"


def _skill_path(name: str) -> Path | None:
    """Resolve a skill by directory name under ``skills/`` (e.g. ``grid-game-solver``).
    Falls back to the default skill, then to None if neither SKILL.md exists."""
    skills_dir = EXAMPLE_DIR / "skills"
    requested = name or DEFAULT_SKILL
    for candidate in (requested, DEFAULT_SKILL):
        path = skills_dir / candidate / "SKILL.md"
        if path.exists():
            if candidate != requested:
                print(
                    f"[launcher] WARNING: skill '{requested}' not found at {skills_dir / requested};"
                    f" falling back to '{candidate}'",
                    file=sys.stderr,
                )
            return path
    return None


def build_memory_config(store_path: str | Path):
    """The memory variant's MemoryConfig (extracted for testable wiring).

    ``reflection.background=True`` keeps manual ``self.memory.reflect()``
    consolidation OFF the turn's critical path — inline it cost 600-1,500s/game
    on the 20260716 fleet and triggered sandbox cell-deadline kills.
    """
    from nooa_memory import MemoryConfig
    from nooa_memory.config import ReflectionPolicy, RetrievalConfig, SpontaneousConfig

    return MemoryConfig(
        enabled=True,
        path=str(store_path),
        owner="ArcSolver",
        embedding=build_embedding_config(),
        retrieval=RetrievalConfig(top_k=8, hops=1),
        spontaneous=SpontaneousConfig(enabled=True, top_k=6),
        reflection=ReflectionPolicy(trigger="manual", background=True),
    )


def parse_effort_ladder(
    spec: str | None, primary_effort: str | None
) -> list[tuple[float, str]] | None:
    """Parse ``"600:medium,900:low"`` into an ascending effort ladder.

    Rung 0 is the client's primary effort (``primary_effort``; may be None =
    model default); each ``after_seconds:effort`` pair is a downshift rung the
    agent applies once a turn has run that long. Returns None when there is no
    downshift rung to apply (so the agent keeps a single fixed effort).
    """
    rungs: list[tuple[float, str]] = [(0.0, primary_effort or "")]
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        after, _, effort = part.partition(":")
        rungs.append((float(after), effort.strip()))
    return sorted(rungs) if len(rungs) > 1 else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--game", required=True)
    p.add_argument("--variant", choices=["memory", "mdfiles"], required=True)
    p.add_argument("--reflect-every", type=int, default=8)
    p.add_argument(
        "--alias", default="", help="opaque handle the agent sees instead of the game id"
    )
    p.add_argument(
        "--model",
        default=None,
        help="gateway model id override, e.g. openai/openai/gpt-5.5 "
        "(default: ARC_LLM_MODEL from .env)",
    )
    p.add_argument(
        "--reasoning-effort", default=None, help="primary reasoning effort (ladder rung 0)"
    )
    p.add_argument(
        "--effort-ladder",
        default=None,
        help="downshift rungs 'after_s:effort[,after_s:effort…]', e.g. "
        "'600:medium' — reasoning_effort drops to that level once a "
        "turn has run that long (single model, no second client)",
    )
    p.add_argument(
        "--visual",
        choices=["off", "only", "additive"],
        default="off",
        help="grid-as-image mode: off | only (image replaces the hex grid) "
        "| additive (image + hex grid)",
    )
    p.add_argument(
        "--png-scale",
        dest="png_scale",
        type=int,
        default=8,
        help="pixels per grid cell for the visual PNG (grid is "
        "(scale*64)x(scale*64); typically 8-16)",
    )
    p.add_argument(
        "--seed-knowledge",
        default=None,
        help="Prior run's workspace to seed knowledge from "
        "(memory.sqlite / knowledge/*.md are copied in)",
    )
    p.add_argument(
        "--skill",
        default=DEFAULT_SKILL,
        help="skill directory under skills/ to load as the agent's "
        "<arc_skill> block (e.g. grid-game-solver, interactive-game-solver)",
    )
    p.add_argument(
        "--max-actions-per-turn",
        type=int,
        default=20,
        help="submit_actions per-turn batch cap; <=0 = unlimited "
        "(run_solver forwards the same value to the harness)",
    )
    return p.parse_args()


def seed_knowledge(src: Path, workspace: Path, variant: str) -> None:
    """Copy accumulated knowledge from a prior run's workspace into this one."""
    import shutil

    if variant == "memory":
        db = src / "memory.sqlite"
        if db.exists():
            shutil.copy2(db, workspace / "memory.sqlite")
    else:
        src_knowledge = src / "knowledge"
        if src_knowledge.is_dir():
            (workspace / "knowledge").mkdir(parents=True, exist_ok=True)
            for f in src_knowledge.glob("*.md"):
                shutil.copy2(f, workspace / "knowledge" / f.name)
    src_helpers = src / "helpers"
    if src_helpers.is_dir():
        (workspace / "helpers").mkdir(parents=True, exist_ok=True)
        for f in src_helpers.glob("*.py"):
            shutil.copy2(f, workspace / "helpers" / f.name)


async def _headless_dispatch(agent) -> None:
    """Drive the agent via its queue dispatcher — no TUI session required.

    Races all registered queues (game_states, user_messages, system_messages)
    each turn and calls ``agent.handle(notification)`` — the dispatcher contract
    described on ``RespondResult``: every stop reason (including DONE) re-enters
    the race. The loop runs until run_solver terminates this process when the
    harness finishes; DONE must NOT exit here, or a premature DONE from the
    agent would tear down a still-running game (run_solver kills the harness
    when the launcher dies). Errors are logged and retried with backoff — the
    harness's nudge/force-advance ladder and the fleet wall-clock cap already
    bound a wedged agent, and a transient LLM outage must not kill the game.
    """
    from nooa.interactive import RespondReason

    consecutive_errors = 0

    while True:
        try:
            wins = await agent.queue_manager.race()
        except (asyncio.CancelledError, ValueError):
            break
        notification: dict[str, list] = {}
        for name, item in wins:
            notification.setdefault(name, []).append(item)
        try:
            result = await agent.handle(notification)
            consecutive_errors = 0
        except asyncio.CancelledError:
            break
        except Exception as exc:
            consecutive_errors += 1
            backoff = min(2 ** min(consecutive_errors, 6), 60)
            print(
                f"[launcher] handle() raised {exc!r} "
                f"(consecutive={consecutive_errors}) — retrying in {backoff}s",
                file=sys.stderr,
            )
            await asyncio.sleep(backoff)
            continue
        if getattr(result, "kind", None) == RespondReason.DONE:
            print(f"[launcher] agent returned DONE: {result.explanation}", file=sys.stderr)


async def run() -> None:
    args = parse_args()
    import os

    if args.model:
        os.environ["ARC_LLM_MODEL"] = args.model
    run_dir = Path(args.run_dir).resolve()
    workspace = run_dir / "team_nemo" / "shared"
    workspace.mkdir(parents=True, exist_ok=True)

    from viewer_event_exporter import ViewerEventExporter
    from viewer_trace_exporter import ViewerMessageExporter

    from nooa.tracing import enable_tracing, exporters

    trace_exporters = [
        exporters.jsonl(trace_dir=str(run_dir / "traces")),
        ViewerMessageExporter(run_dir),
        ViewerEventExporter(run_dir),
    ]
    otlp_endpoint = os.environ.get("OTLP_ENDPOINT")
    if otlp_endpoint:
        trace_exporters.append(exporters.otlp(otlp_endpoint))
    enable_tracing(exporters=trace_exporters)

    if args.seed_knowledge:
        seed_knowledge(Path(args.seed_knowledge).resolve(), workspace, args.variant)

    import shutil

    # ARC_HTTP_LOG_DIR: capture raw gateway request/response bodies (cache-awareness
    # debugging). Must run BEFORE build_llm() so the patched httpx transport is used.
    http_log_dir = os.environ.get("ARC_HTTP_LOG_DIR")
    if http_log_dir:
        from nooa.unifiedllm.http_logging import enable_http_request_logging

        enable_http_request_logging(
            output_dir=http_log_dir,
            save_responses=True,
            verbose=False,
        )

    llm = build_llm(reasoning_effort=args.reasoning_effort)
    effort_ladder = parse_effort_ladder(args.effort_ladder, args.reasoning_effort)
    agent_cls = MemArcSolverAgent if args.variant == "memory" else MdArcSolverAgent
    # The agent process never receives the real game id — only the opaque alias
    # (passed as both game_id and alias so no self.game_id path can leak identity).
    _alias = args.alias or "the game"
    agent = agent_cls(
        llm=llm,
        run_dir=run_dir,
        game_id=_alias,
        alias=_alias,
        reflect_every=args.reflect_every,
        effort_ladder=effort_ladder,
        visual=args.visual,
        png_scale=args.png_scale,
        skill_path=_skill_path(args.skill),
        max_actions_per_turn=args.max_actions_per_turn,
    )

    # Memory store lives INSIDE the neutral run dir (run_solver puts the whole
    # run dir on a game-name-free /tmp path), so the memory-skill guide can show
    # its path without leaking the benchmark/game id. Keeping it under run_dir
    # (not the shared /tmp/agent_stores) also means --sandbox drop's carve_own
    # gives the agent uid sole ownership of its own store — no cross-uid perms.
    # Copied back into the workspace at run end for the viewer / seeding / analysis.
    store_path = run_dir / "store" / "memory.sqlite"
    ws_store = workspace / "memory.sqlite"

    if args.variant == "memory":
        from nooa_memory.generative import llm_reasoner, llm_reconciler
        from nooa_memory.memory_skill import MemorySkill

        store_path.parent.mkdir(parents=True, exist_ok=True)
        store_path.unlink(missing_ok=True)
        # If seeded, the seed's memory.sqlite was copied into the workspace; move
        # it to the neutral live path so the run starts seeded but the agent never
        # sees the workspace path.
        if ws_store.exists():
            shutil.move(str(ws_store), str(store_path))

        def _session_llm() -> object:
            return agent._llm

        memory_config = build_memory_config(store_path)
        agent.skills.register(
            "nemo.memory",
            MemorySkill(
                memory_config,
                reasoner=llm_reasoner(_session_llm),
                reconciler=llm_reconciler(_session_llm),
            ),
        )
        agent.skills.activate(["nemo.memory"])

    # Kickoff parity with the TUI runner: run_solver used to type this message
    # into the TUI once the first game state existed. Deliver the same message
    # on the user_messages channel so turn 0 matches the TUI-driven agent.
    states_path = run_dir / "ipc" / "states.jsonl"

    async def _kickoff() -> None:
        while True:
            try:
                if states_path.stat().st_size > 0:
                    break
            except OSError:
                pass
            await asyncio.sleep(2)
        agent._user_messages_in.put(
            f"Start solving {_alias}. The first game state is on your "
            "game_states queue. Follow the arc_skill context block."
        )
        print("[launcher] kickoff message sent to the agent", file=sys.stderr)

    kickoff_task = asyncio.create_task(_kickoff())

    try:
        await _headless_dispatch(agent)
    finally:
        kickoff_task.cancel()
        # Consolidate the neutral store and copy it into the workspace so the
        # viewer, analysis tools, and seeded runs find it where they expect.
        if args.variant == "memory" and store_path.exists():
            try:
                import sqlite3

                con = sqlite3.connect(str(store_path))
                con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                con.close()
            except Exception:
                pass
            try:
                shutil.copy2(store_path, ws_store)
            except Exception:
                pass


if __name__ == "__main__":
    asyncio.run(run())
