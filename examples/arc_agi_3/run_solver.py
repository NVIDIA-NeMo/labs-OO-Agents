# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestrate one ARC-AGI-3 solving run: headless agent + environment harness.

Stdlib-only — run with any python:

    python examples/arc_agi_3/run_solver.py --game ls20 --variant memory

What it does:
1. creates ``results/arc_agi_3/nemo_solver/<ts>_<game>_<variant>/`` (+ ipc/),
2. starts launcher.py as a headless subprocess (no TUI / tmux required),
3. starts harness.py in this venv (the `arc` extra provides the arcade SDK),
4. waits for the harness to finish and prints the result summary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_DIR))  # for `from sandbox import ...`
MAIN_PY = sys.executable
# The harness runs from this directory; the ARC-AGI SDK looks for (and downloads)
# offline game files AND the per-level RHAE baselines under
# <cwd>/environment_files/ — in EVERY operation mode (competition included: the
# vendored SDK wrapper scans local files for env info/baselines). Resolution:
# ARC_DATA_DIR env override > a progressive-learning checkout inside the repo
# (ships environment_files/) > the repo root. Without environment_files the run
# still plays, but baselines are empty → RHAE shows 0/100 live and 0 in
# result.json.
_PL_DIR = REPO_ROOT / "progressive-learning"
DATA_DIR = Path(
    os.environ.get("ARC_DATA_DIR")
    or (_PL_DIR if (_PL_DIR / "environment_files").is_dir() else REPO_ROOT)
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--game", required=True, help="game id, e.g. ls20")
    p.add_argument("--variant", choices=["memory", "mdfiles"], required=True)
    p.add_argument("--results-root", default=str(REPO_ROOT / "results" / "arc_agi_3"))
    p.add_argument(
        "--group", default="nemo_solver", help="grouping dir under results-root (viewer type_dir)"
    )
    p.add_argument(
        "--max-turns",
        default=None,
        help="max agent turns; None/unlimited by default (pass an int to cap)",
    )
    p.add_argument(
        "--max-env-steps",
        type=int,
        default=5000,
        help="max total environment actions — the main run bound",
    )
    p.add_argument(
        "--max-actions-per-turn",
        type=int,
        default=20,
        help="per-turn action batch cap, enforced agent-side (submit_actions) "
        "and harness-side (truncation backstop); <=0 = NO cap on either side",
    )
    p.add_argument(
        "--allowed-game-overs",
        type=int,
        default=-1,
        help="harness auto-RESET budget after GAME_OVER (-1 unlimited)",
    )
    p.add_argument(
        "--agent-turn-timeout",
        type=float,
        default=1200.0,
        help="agent silence before agent_timeout (kill rung)",
    )
    p.add_argument(
        "--nudge-after",
        type=float,
        default=900.0,
        help="agent silence before the harness writes one reminder state",
    )
    p.add_argument(
        "--fallback-window",
        type=float,
        default=90.0,
        help="seconds after the urgent nudge before the harness "
        "force-advances with a default action (never terminates)",
    )
    p.add_argument(
        "--effort-ladder",
        default="600:medium",
        help="reasoning-effort downshift rungs 'after_s:effort', e.g. "
        "'600:medium' (single model; '' disables)",
    )
    p.add_argument("--reflect-every", type=int, default=8)
    p.add_argument(
        "--model", default=None, help="gateway model id override, e.g. openai/openai/gpt-5.5"
    )
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument(
        "--visual",
        choices=["off", "only", "additive"],
        default="off",
        help="grid-as-image mode: off | only (image replaces hex grid) | "
        "additive (image + hex grid)",
    )
    p.add_argument(
        "--png-scale",
        dest="png_scale",
        type=int,
        default=8,
        help="pixels per grid cell for the visual PNG (typically 8-16)",
    )
    p.add_argument(
        "--seed-knowledge",
        default=None,
        help="prior run's team_nemo/shared dir to seed knowledge from",
    )
    p.add_argument("--operation-mode", default="offline")
    p.add_argument(
        "--scorecard-id",
        default="",
        help="competition shared scorecard id (forwarded to the harness)",
    )
    p.add_argument(
        "--event-prompt",
        choices=["off", "count"],
        default="count",
        help="animation-event summary in the agent's state (off|count)",
    )
    p.add_argument("--tag", default="", help="extra tag appended to the run dir name")
    p.add_argument(
        "--skill",
        default="grid-game-solver",
        help="skill dir under examples/arc_agi_3/skills/ to load "
        "(e.g. grid-game-solver, interactive-game-solver)",
    )
    p.add_argument(
        "--agent-uid",
        type=int,
        default=0,
        help="uid to drop the agent to under --sandbox drop; 0 = derive a distinct "
        "uid per run so concurrent games can't read each other",
    )
    p.add_argument(
        "--sandbox",
        choices=["off", "inproc", "full", "drop"],
        default="drop",
        help="drop (DEFAULT): in-process L1+L2 guards PLUS run the agent as a "
        "distinct per-run unprivileged uid so the game source + other runs are "
        "unreadable (Docker-friendly, no namespaces; fails closed if setpriv/root "
        "absent); inproc: only the in-process guards; full: L3 OS namespace "
        "sandbox (needs user namespaces — fails closed if absent); off: no guards",
    )
    # Per-cell OS sandbox (execution_backend="sandbox") for the CodeAct cells. This
    # is ORTHOGONAL to --sandbox above (which isolates the whole agent process): it
    # runs each generated cell in a guarded worker with a hard cell_timeout kill plus
    # kernel-enforced memory/CPU/filesystem/network limits. Configured here (or via a
    # run_multi `sandbox:` config block) rather than a hand-exported env var; the
    # legacy ARC_SANDBOX_* env vars still work when these flags are absent.
    p.add_argument(
        "--sandbox-cells",
        action="store_true",
        help="run each CodeAct cell in a guarded worker process (hard cell_timeout, "
        "memory/CPU caps, Landlock FS confinement, seccomp network block)",
    )
    p.add_argument(
        "--sandbox-workspace",
        default=None,
        help="writable workspace for sandboxed cells (default: the agent cwd)",
    )
    p.add_argument("--sandbox-mem-mb", type=int, default=4096, help="per-cell memory cap (MiB)")
    p.add_argument("--sandbox-cpu-s", type=int, default=90, help="per-cell CPU-time cap (seconds)")
    p.add_argument(
        "--sandbox-require",
        action="store_true",
        help="fail closed if Landlock/seccomp can't be enforced on this host",
    )
    return p.parse_args()


def _set_parent_death_signal() -> None:
    """Linux best-effort: get SIGTERM if our parent (run_multi) dies — even on an
    unhandleable SIGKILL of the orchestrator — so a game never outlives its runner.
    Our SIGTERM handler then reaps the harness + launcher agent."""
    try:
        import ctypes

        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG = 1
    except Exception:
        pass


def _load_dotenv() -> None:
    """Minimal .env loader so ARC_API_KEY / ARC_SDK_SESSION_COOKIES reach the
    harness subprocess (competition mode) even on direct invocation. Real env wins."""
    f = REPO_ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def main() -> int:
    _set_parent_death_signal()  # die with run_multi (covers its SIGKILL)
    _load_dotenv()
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    run_name = f"{ts}_{args.game}_{args.variant}{tag}"
    run_dir = Path(args.results_root) / args.group / run_name
    # Opaque per-run handle the agent sees instead of the real game id. Derived
    # from run metadata only (timestamp/variant) — never from the game name — so
    # it carries no game identity.
    import hashlib

    alias = "game-" + hashlib.sha1(f"{ts}{args.variant}{tag}".encode()).hexdigest()[:6]

    # Neutral work dir: the agent AND the harness run here, so NO path the agent
    # can reach (helper tracebacks, ipc/log paths, trajectory() event files) ever
    # embeds the real game name. The results dir keeps only the run_multi/viewer
    # structure and receives the durable artifacts at run end. The subdirs the
    # live viewer tails (agent_logs/traces/steps) and the agent+harness share
    # (ipc, team_nemo, agent_logs) are symlinked from results -> neutral so the
    # dashboard reads them live; the symlinks are replaced by real copies at the
    # end (see _copy_back). The agent itself only ever sees the neutral path.
    import re as _re

    safe_alias = _re.sub(r"[^a-z0-9_-]", "_", alias.lower())
    # Neutral dirs live under a dedicated /tmp/arc_runs parent (not /tmp itself),
    # so --sandbox drop can chmod that parent 0711 (non-enumerable) without
    # touching shared /tmp.
    neutral_parent = Path(tempfile.gettempdir()) / "arc_runs"
    neutral_parent.mkdir(parents=True, exist_ok=True)
    neutral = neutral_parent / f"arc_run_{safe_alias}"
    if neutral.exists():
        shutil.rmtree(neutral, ignore_errors=True)
    for sub in ("ipc", "team_nemo/shared", "agent_logs/nemo/team_leader", "traces", "steps"):
        (neutral / sub).mkdir(parents=True, exist_ok=True)
    (neutral / "ipc" / "states.jsonl").touch()
    (neutral / "ipc" / "actions.jsonl").touch()
    # Pre-create the events.jsonl the harness recorder (root) AND the launcher
    # exporters (dropped uid, under --sandbox drop) BOTH append to. Creating it
    # now means carve_own hands it to the agent uid, so the agent can write it and
    # root always can — otherwise whichever writes first owns it and the other
    # EACCESes. (No-op cost outside drop mode.)
    (neutral / "agent_logs" / "nemo" / "team_leader" / "events.jsonl").touch()

    run_dir.mkdir(parents=True, exist_ok=True)
    _LINKS = ("ipc", "team_nemo", "agent_logs", "traces", "steps")
    for name in _LINKS:
        link = run_dir / name
        if link.is_symlink() or link.exists():
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link, ignore_errors=True)
        link.symlink_to(neutral / name)

    launcher_cmd = [
        MAIN_PY,
        str(REPO_ROOT / "examples" / "arc_agi_3" / "launcher.py"),
        "--run-dir",
        str(neutral),
        "--game",
        args.game,
        "--variant",
        args.variant,
        "--alias",
        alias,
        "--reflect-every",
        str(args.reflect_every),
        "--skill",
        args.skill,
        "--visual",
        args.visual,
        "--png-scale",
        str(args.png_scale),
        "--max-actions-per-turn",
        str(args.max_actions_per_turn),
    ]
    if args.model:
        launcher_cmd += ["--model", args.model]
    if args.reasoning_effort:
        launcher_cmd += ["--reasoning-effort", args.reasoning_effort]
    if args.effort_ladder:
        launcher_cmd += ["--effort-ladder", args.effort_ladder]
    if args.seed_knowledge:
        launcher_cmd += ["--seed-knowledge", args.seed_knowledge]

    # L3: wrap the launcher in the OS sandbox when requested. Fails closed — if
    # namespaces are unavailable, run_solver aborts rather than run unsandboxed.
    if args.sandbox == "full":
        from sandbox import SandboxSpec, SandboxUnavailable, wrap

        spec = SandboxSpec(
            run_dir=neutral,
            repo_root=REPO_ROOT,
            llm_socket=neutral / "ipc" / "llm.sock",
            tmp_dir=neutral / "agent-tmp",
        )
        (neutral / "agent-tmp").mkdir(exist_ok=True)
        try:
            launcher_cmd = wrap(launcher_cmd, spec)
        except SandboxUnavailable as e:
            print(f"[run] {e}", file=sys.stderr)
            return 3
        print("[run] launcher wrapped in L3 OS sandbox")

    elif args.sandbox == "drop":
        # Docker-friendly external isolation (no user namespaces; fails closed).
        # HARD filesystem boundary: the agent runs as a distinct per-run
        # unprivileged uid, so the game source and every OTHER run are unreadable
        # by POSIX perms no matter what a generated cell imports. The harness stays
        # root (it needs the SDK + writes the shared ipc/events). The model gateway
        # is reached DIRECTLY (Responses API + gateway embeddings unchanged);
        # network *misuse from cells* is handled in-process by the B-lite cell
        # guard in solver_agent. Everything the agent touches is the neutral dir.
        import uid_sandbox

        ok, why = uid_sandbox.available()
        if not ok:
            print(f"[run] --sandbox drop unavailable: {why}", file=sys.stderr)
            return 3
        # uid 0 => derive a distinct per-run uid (40000..59999) so concurrent games
        # run as different users and cannot read each other's neutral dir.
        uid = gid = args.agent_uid or (
            40000 + int(hashlib.sha1(run_name.encode()).hexdigest(), 16) % 20000
        )
        # Block the game source from the dropped uid (root-owned 0700).
        uid_sandbox.block(
            [DATA_DIR / "environment_files", DATA_DIR / "environment_files_generated"]
        )
        # Give the uid exclusive ownership of its neutral dir (0700) and make the
        # /tmp/arc_runs parent traverse-only (0711, non-enumerable) so it can reach
        # its own dir by exact path but cannot list sibling runs.
        uid_sandbox.carve_own(neutral, uid, gid, up_to=neutral_parent)
        # The memory store lives inside the neutral dir (owned by this uid via
        # carve_own), so no shared cross-uid store dir is needed.
        launcher_cmd = uid_sandbox.drop_prefix(uid, gid) + launcher_cmd
        print(
            f"[run] launcher dropped to uid {uid} "
            "(game source + other runs unreadable; LLM direct; cell guard active)"
        )

    launcher_env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    otlp = os.environ.get("OTLP_ENDPOINT", "")
    if otlp:
        launcher_env["OTLP_ENDPOINT"] = otlp
    # Translate the per-cell sandbox flags into the ARC_SANDBOX_* env the agent
    # reads at import (solver_agent._arc_cell_config). Flags win over any inherited
    # env; when no flag is given the inherited env (legacy path) passes through.
    if args.sandbox_cells:
        launcher_env["ARC_SANDBOX_CELLS"] = "1"
        launcher_env["ARC_SANDBOX_MEM_MB"] = str(args.sandbox_mem_mb)
        launcher_env["ARC_SANDBOX_CPU_S"] = str(args.sandbox_cpu_s)
        if args.sandbox_workspace:
            launcher_env["ARC_SANDBOX_WORKSPACE"] = args.sandbox_workspace
        if args.sandbox_require:
            launcher_env["ARC_SANDBOX_REQUIRE"] = "1"

    launcher_err = None  # opened here so the finally below can always close it safely
    launcher_err = open(run_dir / "launcher.err", "w")
    launcher = subprocess.Popen(
        launcher_cmd,
        cwd=str(REPO_ROOT),
        stdout=launcher_err,
        stderr=subprocess.STDOUT,
        env=launcher_env,
        start_new_session=True,
    )
    print(f"[run] headless agent started (pid {launcher.pid})")
    print(f"[run]   run dir: {run_dir}")

    llm_uri = args.model or os.environ.get("ARC_LLM_MODEL", "")
    if args.reasoning_effort:
        llm_uri = f"{llm_uri}#{args.reasoning_effort}"
    if not llm_uri and (REPO_ROOT / ".env").exists():
        for line in (REPO_ROOT / ".env").read_text().splitlines():
            if line.startswith("ARC_LLM_MODEL="):
                llm_uri = line.split("=", 1)[1].strip()
                break
    harness_cmd = [
        sys.executable,
        str(EXAMPLE_DIR / "harness.py"),
        "--run-dir",
        str(neutral),
        "--game",
        args.game,
        "--variant",
        args.variant,
        "--alias",
        alias,
        "--llm-uri",
        llm_uri,
        "--operation-mode",
        args.operation_mode,
        "--max-turns",
        str(args.max_turns),
        "--max-env-steps",
        str(args.max_env_steps),
        "--max-actions-per-turn",
        str(args.max_actions_per_turn),
        "--allowed-game-overs",
        str(args.allowed_game_overs),
        "--agent-turn-timeout",
        str(args.agent_turn_timeout),
        "--nudge-after",
        str(args.nudge_after),
        "--fallback-window",
        str(args.fallback_window),
        "--scorecard-id",
        args.scorecard_id,
        "--event-prompt",
        args.event_prompt,
        "--visual",
        args.visual,
    ]
    # harness.log lives OUTSIDE the run dir: it carries the arcade SDK's boot
    # line ("loaded ... from environment_files/<game>/.../<game>.py"), which would
    # leak the game identity if an agent tailed it. Kept in a sibling _harness/
    # dir the agent's sandbox never mounts.
    harness_log_dir = Path(args.results_root) / args.group / "_harness"
    harness_log_dir.mkdir(parents=True, exist_ok=True)
    harness_log = (harness_log_dir / f"{run_name}.log").open("w")
    harness = subprocess.Popen(
        harness_cmd,
        cwd=str(DATA_DIR),
        stdout=harness_log,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # End-of-run copy-back: the agent+harness worked entirely in the neutral /tmp
    # dir (no game name in any path). Move the durable artifacts into the results
    # dir — replacing the results->neutral symlinks with real copies — so the
    # viewer, analysis, and seeding find memory.sqlite / helpers / knowledge /
    # agent_logs / traces / steps / ipc / result.json where they expect. Then the
    # neutral dir is deleted so nothing lingers under /tmp. Idempotent.
    def _copy_back() -> None:
        # 1) Memory fallback: the launcher normally checkpoints its store (under
        #    neutral/store, owned by the agent) into neutral/team_nemo/shared at
        #    its own teardown. If that didn't run (killed mid-write), copy the raw
        #    store files here so the memory is never lost. root reads the agent-uid
        #    files (bypasses DAC); a raw db+wal+shm copy is a valid database.
        if args.variant == "memory":
            ws_store = neutral / "team_nemo" / "shared" / "memory.sqlite"
            live = neutral / "store" / "memory.sqlite"
            if not ws_store.exists() and live.exists():
                ws_store.parent.mkdir(parents=True, exist_ok=True)
                for suffix in ("", "-wal", "-shm"):
                    src = Path(f"{live}{suffix}")
                    if src.exists():
                        try:
                            shutil.copy2(src, f"{ws_store}{suffix}")
                        except Exception as e:
                            print(f"[run] warning: memory fallback copy failed: {e}")
        # 2) move neutral -> results, swapping each symlink for a real copy.
        if neutral.exists():
            for name in ("team_nemo", "agent_logs", "traces", "steps", "ipc"):
                src, dst = neutral / name, run_dir / name
                if dst.is_symlink() or dst.is_file():
                    dst.unlink()
                elif dst.is_dir():
                    shutil.rmtree(dst, ignore_errors=True)
                if src.is_dir():
                    try:
                        shutil.copytree(src, dst)
                    except Exception as e:
                        print(f"[run] warning: copy-back {name} failed: {e}")
            for fname in ("result.json", "gameplay.json"):
                src = neutral / fname
                if src.is_file():
                    try:
                        shutil.copy2(src, run_dir / fname)
                    except Exception as e:
                        print(f"[run] warning: copy-back {fname} failed: {e}")
            print("[run] artifacts copied back to results dir")
        # 3) delete the neutral dir (nothing left in /tmp).
        shutil.rmtree(neutral, ignore_errors=True)

    # Reliable teardown on ANY abnormal exit (SIGTERM/SIGHUP/SIGINT, incl. the
    # pdeathsig SIGTERM when run_multi dies): reap the harness AND launcher so
    # nothing is orphaned, then preserve the memory store (killed games used to
    # lose it — the copy-back only ran on the normal path).
    _torn = {"done": False}

    def _teardown() -> None:
        if _torn["done"]:
            return
        _torn["done"] = True
        for proc in (harness, launcher):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except Exception:
                        proc.kill()
            except Exception:
                pass
        _copy_back()

    def _sig_teardown(signum, _frame):
        print(f"[run] signal {signum} — tearing down game {run_name}")
        _teardown()
        raise SystemExit(128 + signum)

    for _s in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(_s, _sig_teardown)
        except (ValueError, OSError):
            pass

    # Wait for the harness to finish. The launcher's game_states tail producer picks
    # up new states automatically — no explicit kickoff message needed.
    try:
        while harness.poll() is None:
            if launcher.poll() is not None:
                print("[run] launcher exited early — terminating harness")
                _teardown()
                break
            time.sleep(2)
    except KeyboardInterrupt:
        print("[run] interrupted — tearing down game (harness + launcher)")
        _teardown()
    finally:
        # Normal completion included: the headless launcher never exits on its
        # own (it races the agent queues until killed), and it runs in its own
        # session — without this it outlives every finished game as an orphan.
        _teardown()
        harness_log.close()
        if launcher_err is not None:
            launcher_err.close()

    result_path = run_dir / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        print(
            f"[run] finished: termination={result.get('termination_reason')} "
            f"levels={result.get('levels_completed')} "
            f"steps={result.get('total_steps')} "
            f"wall={result.get('wall_time_seconds')}s"
        )
    else:
        print(
            "[run] harness exited without result.json — check "
            f"{run_dir}/harness.log and {run_dir}/launcher.err"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
