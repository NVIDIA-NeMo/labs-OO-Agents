# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-driven multi-run orchestrator for the ARC-AGI-3 single-agent solver.

Runs many (game x variant) solves in parallel:
reads a YAML config (or CLI flags), runs many (game × variant) solves with a
concurrency cap, and lays the results out as a **multi-run container** that both
viewers discover natively:

    results/arc_agi_3/nemo_solver/<ts>_<name>/          <- container
      manifest.json  status.json  summary.json          <- container metadata
      <game>/<ts>_<game>_<variant>/                      <- each run (team_nemo/, steps/, agent_logs/…)
      ...

``manifest.json`` + ``status.json`` make ``_detect_run_type`` classify the
container as a multi-run, and ``viewer.py`` reads
``status.json`` directly. Real LLM messages stream into each run's
``agent_logs/.../messages`` live (ViewerMessageExporter).

Usage:
    python examples/arc_agi_3/run_multi.py --config examples/arc_agi_3/configs/offline_focused.yaml
    python examples/arc_agi_3/run_multi.py --games cd82 ka59 --variants memory mdfiles --parallel 2
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = REPO_ROOT / "results" / "arc_agi_3" / "nemo_solver"
# Progressive-learning checkout (optional): provides the interactive multi-game
# TUI (tui_multi_game_viewer.py reuses its reference panels). When absent, the
# self-contained viewer.py status table is used instead.
PL_DIR = REPO_ROOT / "progressive-learning"
PL_PY = PL_DIR / ".venv" / "bin" / "python"


def _load_dotenv() -> None:
    """Minimal .env loader so ARC_API_KEY reaches the broker + game subprocesses
    (competition mode) without a manual export. Real environment wins."""
    f = REPO_ROOT / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


DEFAULTS = {
    "name": "run",
    "operation_mode": "offline",
    "model": None,
    "reasoning_effort": None,
    # Skill selection: which skills/<name>/SKILL.md the agent loads as its
    # <arc_skill> block. e.g. "grid-game-solver" or "interactive-game-solver".
    "skill": "grid-game-solver",
    "variants": ["memory", "mdfiles"],
    "seeded": False,
    "parallel": 4,
    "max_turns": None,
    "max_env_steps": 5000,
    # Per-turn action batch cap, enforced BOTH agent-side (submit_actions REJECTED)
    # and harness-side (truncation backstop). 0 = NO cap at all on either side.
    "max_actions_per_turn": 20,
    # Wall-clock cap for the WHOLE fleet, in seconds. None/0 = no cap. When reached,
    # every still-running game is stopped and the summary is written.
    "max_wall_seconds": None,
    "turn_timeout": 1200,
    # Timeout ladder (seconds of agent silence): effort downshift (agent-side, in
    # effort_ladder) < nudge_after (harness reminder) < turn_timeout (kill).
    "nudge_after": 900,
    "fallback_window": 90,
    "effort_ladder": "600:medium",
    "event_prompt": "count",  # animation-event summary in the agent's state
    "visual": "off",
    "png_scale": 8,  # grid-as-image: off | only | additive
    "reflect_every": 8,
    "allowed_game_overs": -1,
    # Per-cell OS sandbox for CodeAct cells (execution_backend="sandbox"), applied
    # to every agent in the fleet. Configured here instead of a hand-exported
    # ARC_SANDBOX_* env var. `cells: true` turns it on; the rest are the guardrail
    # knobs. workspace null -> the agent cwd.
    "sandbox": {
        "cells": False,
        "workspace": None,
        "max_memory_mb": 4096,
        "max_cpu_seconds": 90,
        "require": False,
    },
    "games": [],
    # Live TUI dashboard (viewer.py). On by default; pass --no-tui for automated runs.
    "no_tui": False,
    "watch": "team_leader",
}


def load_config(path: str | None, args: argparse.Namespace) -> dict:
    cfg = dict(DEFAULTS)
    if path:
        import yaml

        raw = {k: v for k, v in yaml.safe_load(Path(path).read_text()).items() if v is not None}
        # Merge the nested `sandbox:` block over its defaults so a partial block
        # (e.g. just `cells: true`) keeps the default guardrail knobs.
        if isinstance(raw.get("sandbox"), dict):
            raw["sandbox"] = {**DEFAULTS["sandbox"], **raw["sandbox"]}
        cfg.update(raw)
    # CLI overrides
    for key in (
        "model",
        "reasoning_effort",
        "skill",
        "parallel",
        "max_turns",
        "max_env_steps",
        "max_actions_per_turn",
        "max_wall_seconds",
        "turn_timeout",
        "nudge_after",
        "fallback_window",
        "effort_ladder",
        "event_prompt",
        "visual",
        "png_scale",
        "reflect_every",
        "allowed_game_overs",
        "name",
        "operation_mode",
        "watch",
    ):
        v = getattr(args, key, None)
        if v is not None:
            cfg[key] = v
    if args.games:
        cfg["games"] = args.games
    if args.variants:
        cfg["variants"] = args.variants
    if args.seeded:
        cfg["seeded"] = True
    if args.no_tui:
        cfg["no_tui"] = True
    if not cfg["games"]:
        raise SystemExit("no games specified (config `games:` or --games)")
    return cfg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None)
    p.add_argument("--games", nargs="+", default=None)
    p.add_argument("--variants", nargs="+", default=None, choices=["memory", "mdfiles"])
    p.add_argument("--seeded", action="store_true")
    p.add_argument("--parallel", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--reasoning-effort", dest="reasoning_effort", default=None)
    p.add_argument(
        "--skill",
        default=None,
        help="skill dir under examples/arc_agi_3/skills/ to load "
        "(e.g. grid-game-solver, interactive-game-solver)",
    )
    p.add_argument("--max-turns", dest="max_turns", type=int, default=None)
    p.add_argument("--max-env-steps", dest="max_env_steps", type=int, default=None)
    p.add_argument(
        "--max-actions-per-turn",
        dest="max_actions_per_turn",
        type=int,
        default=None,
        help="per-turn action batch cap (agent + harness); 0 = NO cap",
    )
    p.add_argument(
        "--max-wall-seconds",
        dest="max_wall_seconds",
        type=float,
        default=None,
        help="stop the whole fleet after this many seconds (e.g. 7200 = 2h). "
        "Default: no wall-clock cap.",
    )
    p.add_argument("--turn-timeout", dest="turn_timeout", type=float, default=None)
    p.add_argument("--nudge-after", dest="nudge_after", type=float, default=None)
    p.add_argument("--fallback-window", dest="fallback_window", type=float, default=None)
    p.add_argument("--effort-ladder", dest="effort_ladder", default=None)
    p.add_argument("--event-prompt", dest="event_prompt", choices=["off", "count"], default=None)
    p.add_argument("--visual", dest="visual", choices=["off", "only", "additive"], default=None)
    p.add_argument("--png-scale", dest="png_scale", type=int, default=None)
    p.add_argument("--reflect-every", dest="reflect_every", type=int, default=None)
    p.add_argument("--allowed-game-overs", dest="allowed_game_overs", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--operation-mode", dest="operation_mode", default=None)
    p.add_argument(
        "--no-tui",
        dest="no_tui",
        action="store_true",
        help="disable the live TUI dashboard (for non-interactive runs)",
    )
    p.add_argument(
        "--watch",
        dest="watch",
        default=None,
        help="agent role shown in the TUI game tabs (default: team_leader)",
    )
    return p.parse_args()


class MultiRunner:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._ts = ts
        self.container = RESULTS_ROOT / f"{ts}_{cfg['name']}"
        self.container.mkdir(parents=True, exist_ok=True)
        # (game, variant, phase) work items
        self.items = [(g, v, "fresh") for g in cfg["games"] for v in cfg["variants"]]
        self.status: dict[str, dict] = {}
        self._lock = threading.Lock()
        # Competition shared scorecard (minted by the broker in run()).
        self.scorecard_id: str | None = None
        self._broker: subprocess.Popen | None = None
        self._broker_err = None
        # Per-game run_solver process, for teardown on ANY exit.
        self._procs: dict[str, subprocess.Popen] = {}
        self._stop = threading.Event()
        self._cleaned = False
        self._cleanup_lock = threading.Lock()
        self._write_manifest(ts)
        self._write_status()

    def _write_manifest(self, ts: str) -> None:
        (self.container / "manifest.json").write_text(
            json.dumps(
                {
                    "execution_mode": "nemo_single_agent",
                    "command": "examples/arc_agi_3/run_multi.py",
                    "started_at": ts,
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    # Top-level fields the TUI banner reads (viewer.py).
                    "operation_mode": self.cfg["operation_mode"],
                    "parallel": self.cfg["parallel"],
                    "launch_stagger_seconds": 3,
                    "scorecard_id": self.scorecard_id,
                    "config": {
                        k: self.cfg[k]
                        for k in (
                            "name",
                            "operation_mode",
                            "model",
                            "reasoning_effort",
                            "skill",
                            "variants",
                            "seeded",
                            "parallel",
                            "max_turns",
                            "max_env_steps",
                            "max_actions_per_turn",
                            "max_wall_seconds",
                            "turn_timeout",
                            "nudge_after",
                            "effort_ladder",
                        )
                    },
                },
                indent=2,
            )
        )

    def _key(self, game: str, variant: str) -> str:
        return f"{game}_{variant}"

    def _write_status(self) -> None:
        with self._lock:
            counts = {"running": 0, "completed": 0, "failed": 0, "queued": 0}
            for info in self.status.values():
                counts[info.get("status", "queued")] = (
                    counts.get(info.get("status", "queued"), 0) + 1
                )
            queued = len(self.items) - len(self.status)
            (self.container / "status.json").write_text(
                json.dumps(
                    {
                        "total": len(self.items),
                        "queued": max(0, queued) + counts["queued"],
                        "running": counts["running"],
                        "completed": counts["completed"],
                        "failed": counts["failed"],
                        "terminated": 0,
                        "killed": 0,
                        "runs": self.status,
                    },
                    indent=2,
                )
            )

    def _sync_once(self) -> None:
        """One pass: refresh each run's live status from its result/states files."""
        with self._lock:
            for _key, info in self.status.items():
                rd = info.get("_run_dir")
                if not rd:
                    continue
                rd = Path(rd)
                res = rd / "result.json"
                if res.exists():
                    try:
                        r = json.loads(res.read_text())
                        info.update(
                            status="completed",
                            levels=r["levels_completed"],
                            steps=r["total_steps"],
                            wall_time=r.get("wall_time_seconds"),
                            outcome=r["termination_reason"],
                            rhae=r.get("rhae_game_score"),
                        )
                    except (OSError, json.JSONDecodeError, KeyError):
                        pass
                elif info.get("status") == "running":
                    sf = rd / "ipc" / "states.jsonl"
                    try:
                        last = [ln for ln in sf.read_text().splitlines() if ln.strip()][-1]
                        s = json.loads(last)
                        info.update(levels=s.get("levels_completed"), steps=s.get("step"))
                    except (OSError, IndexError, json.JSONDecodeError):
                        pass
        self._write_status()

    def _refresh_live(self, stop: threading.Event) -> None:
        """Poll each running run to keep status.json fresh until stopped."""
        while not stop.is_set():
            self._sync_once()
            stop.wait(10)

    def _run_one(self, game: str, variant: str, sem: threading.Semaphore) -> None:
        with sem:
            if self._stop.is_set():
                return  # fleet stopped before this game started
            run_group_root = self.container  # results-root for run_solver
            key = self._key(game, variant)  # "<game>_<variant>" — slash-free
            group = f"{game}/{variant}"  # nested on-disk hierarchy
            with self._lock:
                self.status[key] = {
                    "title": key,
                    "status": "running",
                    "game": game,
                    "variant": variant,
                }
            cmd = [
                sys.executable,
                str(EXAMPLE_DIR / "run_solver.py"),
                "--game",
                game,
                "--variant",
                variant,
                "--results-root",
                str(run_group_root),  # -> <container>/<game>/<variant>/<run>
                "--group",
                group,
                "--max-turns",
                str(self.cfg["max_turns"]),
                "--max-env-steps",
                str(self.cfg["max_env_steps"]),
                "--max-actions-per-turn",
                str(self.cfg["max_actions_per_turn"]),
                "--agent-turn-timeout",
                str(self.cfg["turn_timeout"]),
                "--nudge-after",
                str(self.cfg["nudge_after"]),
                "--fallback-window",
                str(self.cfg["fallback_window"]),
                "--effort-ladder",
                str(self.cfg["effort_ladder"]),
                "--event-prompt",
                str(self.cfg["event_prompt"]),
                "--visual",
                str(self.cfg["visual"]),
                "--png-scale",
                str(self.cfg["png_scale"]),
                "--reflect-every",
                str(self.cfg["reflect_every"]),
                "--allowed-game-overs",
                str(self.cfg["allowed_game_overs"]),
                "--skill",
                str(self.cfg["skill"]),
                "--operation-mode",
                self.cfg["operation_mode"],
            ]
            if self.scorecard_id:
                cmd += ["--scorecard-id", self.scorecard_id]
            if self.cfg["model"]:
                cmd += ["--model", str(self.cfg["model"])]
            if self.cfg["reasoning_effort"]:
                cmd += ["--reasoning-effort", self.cfg["reasoning_effort"]]
            # Per-cell OS sandbox — driven by the config `sandbox:` block, forwarded
            # to run_solver as flags (no ARC_SANDBOX_* env var needed at launch).
            sb = self.cfg.get("sandbox") or {}
            if sb.get("cells"):
                cmd += ["--sandbox-cells"]
                if sb.get("workspace"):
                    cmd += ["--sandbox-workspace", str(sb["workspace"])]
                cmd += ["--sandbox-mem-mb", str(sb.get("max_memory_mb", 4096))]
                cmd += ["--sandbox-cpu-s", str(sb.get("max_cpu_seconds", 90))]
                if sb.get("require"):
                    cmd += ["--sandbox-require"]
            # Own process group so _shutdown_games() can killpg the run_solver +
            # its harness child together on TUI quit.
            proc = subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, start_new_session=True
            )
            with self._lock:
                self._procs[key] = proc
            # discover the run dir this spawn creates (newest under <container>/<game>/<variant>/)
            game_dir = run_group_root / group
            before = set(game_dir.glob("*")) if game_dir.exists() else set()
            for _ in range(30):
                time.sleep(1)
                new = (set(game_dir.glob("*")) - before) if game_dir.exists() else set()
                new = [d for d in new if d.is_dir() and d.name != "_harness"]
                if new:
                    with self._lock:
                        self.status[key]["_run_dir"] = str(sorted(new)[-1])
                    break
            proc.wait()
            print(f"[multi] finished {key}")

    def _start_broker(self) -> None:
        """Competition only: mint ONE shared scorecard (broker holds the session +
        keep-alive) and export its ALB cookies so every game subprocess lands on
        the replica that owns the card. No-op for offline/online/normal."""
        if self.cfg["operation_mode"] != "competition":
            return
        if not os.environ.get("ARC_API_KEY"):
            raise SystemExit("competition mode needs ARC_API_KEY (set it in .env)")
        tags = ",".join(["nemo_solver", self.cfg["name"], "competition"])
        # Broker stderr -> a log file (the ARC SDK is chatty); stdout carries the
        # JSON handshake, but the SDK also logs to stdout, so SCAN for JSON rather
        # than trusting the first line.
        self._broker_err = (self.container / "_broker.log").open("w")
        self._broker = subprocess.Popen(
            [sys.executable, str(EXAMPLE_DIR / "scorecard_broker.py"), "--tags", tags],
            cwd=str(REPO_ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._broker_err,
            text=True,
            env={**os.environ},
        )
        info = self._read_broker_json() or {"error": "broker gave no JSON handshake"}
        if info.get("error") or not info.get("scorecard_id"):
            self._stop_broker()
            raise SystemExit(f"[multi] scorecard broker failed: {info.get('error')}")
        self.scorecard_id = info["scorecard_id"]
        cookies = info.get("cookies") or {}
        if cookies:
            os.environ["ARC_SDK_SESSION_COOKIES"] = json.dumps(cookies)
        self._write_manifest(self._ts)  # refresh manifest with the scorecard_id
        print(
            f"[multi] competition shared scorecard: {self.scorecard_id} "
            f"({len(cookies)} session cookies propagated)"
        )

    def _read_broker_json(self) -> dict | None:
        """Read the broker's stdout until a JSON object line appears (the SDK also
        logs to stdout, so skip any non-JSON noise)."""
        for line in self._broker.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and ("scorecard_id" in obj or "error" in obj):
                return obj
        return None

    def _stop_broker(self) -> None:
        """Signal the broker to close the scorecard; record its final summary.

        Bounded: the read of the close summary runs in a watchdog thread so a slow
        or hung ``close_scorecard`` network call can never wedge the whole run — we
        terminate the broker and move on."""
        if self._broker is None:
            return
        try:
            self._broker.stdin.write("close\n")
            self._broker.stdin.flush()
            self._broker.stdin.close()
        except Exception:
            pass
        holder: dict = {}
        reader = threading.Thread(
            target=lambda: holder.__setitem__("summary", self._read_broker_json()), daemon=True
        )
        reader.start()
        reader.join(timeout=45)
        if reader.is_alive():
            print("[multi] broker close timed out — terminating broker")
            try:
                self._broker.terminate()
            except Exception:
                pass
        try:
            self._broker.wait(timeout=10)
        except Exception:
            try:
                self._broker.kill()
            except Exception:
                pass
        summary = holder.get("summary")
        if summary:
            (self.container / "scorecard.json").write_text(json.dumps(summary, indent=2))
            print(f"[multi] competition scorecard closed -> {self.container / 'scorecard.json'}")
        try:
            self._broker_err.close()
        except Exception:
            pass
        self._broker = None

    def _orchestrate(self) -> None:
        """Spawn all (game×variant) runs (staggered, capped) and wait for them."""
        refresher = threading.Thread(target=self._refresh_live, args=(self._stop,), daemon=True)
        refresher.start()
        sem = threading.Semaphore(self.cfg["parallel"])
        threads = [
            threading.Thread(target=self._run_one, args=(g, v, sem)) for g, v, _ in self.items
        ]
        started: list[threading.Thread] = []
        for t in threads:
            if self._stop.is_set():
                break  # TUI quit before fan-out finished
            t.start()
            started.append(t)
            time.sleep(3)  # stagger startup to avoid simultaneous env downloads
        for t in started:  # join only threads we actually started
            t.join()
        self._stop.set()  # stop the refresher
        self._sync_once()  # one final status sync

    def _install_shutdown_hooks(self) -> None:
        """Guarantee teardown on ANY exit: atexit (normal exit / unhandled
        exception / sys.exit) + SIGINT/SIGTERM/SIGHUP handlers that convert the
        signal into a SystemExit so both the run() ``finally`` and atexit run the
        (idempotent) cleanup. Must run in the main thread (signal.signal)."""
        import atexit
        import signal

        atexit.register(self._cleanup_all)

        def _handler(signum, _frame):
            raise SystemExit(128 + signum)

        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            try:
                signal.signal(s, _handler)
            except (ValueError, OSError):
                pass  # not main thread / unsupported platform

    def _cleanup_all(self, *_a) -> None:
        """Idempotent teardown for ANY exit: stop launching, kill every
        still-running game (its run_solver process group — run_solver + harness),
        close the shared scorecard, and write the summary. Registered via atexit +
        signal handlers + run()'s finally, so games are never orphaned."""
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True
        import signal

        self._stop.set()  # stop the orchestrator from launching more games
        with self._lock:
            procs = list(self._procs.items())
        live = [(k, p) for k, p in procs if p.poll() is None]
        if live:
            print(f"[multi] cleanup: terminating {len(live)} running game(s)...")
            for _, p in live:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline and any(p.poll() is None for _, p in live):
                time.sleep(0.2)
            for _, p in live:
                if p.poll() is None:
                    try:
                        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except (ProcessLookupError, PermissionError):
                        pass
        self._stop_broker()  # close the shared scorecard (idempotent)
        try:
            self._sync_once()
            self._write_summary()
        except Exception:
            pass

    def _deadline_watch(self, seconds: float) -> None:
        """Wall-clock guard: stop the whole fleet once ``seconds`` elapse.

        Waits on ``self._stop`` so it exits immediately (doing nothing) if the run
        finishes on its own first. When the deadline is reached it tears down all
        games, closes the shared scorecard, and writes the summary."""
        if self._stop.wait(timeout=seconds):
            return  # run finished before the deadline
        print(f"[multi] wall-clock deadline reached ({int(seconds)}s) — stopping fleet")
        self._deadline_hit = True
        self._stop.set()
        self._cleanup_all()
        tui = getattr(self, "_tui_proc", None)  # unblock the foreground viewer, if up
        if tui is not None and tui.poll() is None:
            try:
                tui.terminate()
            except OSError:
                pass

    # Back-compat name.
    def _shutdown_games(self) -> None:
        self._cleanup_all()

    def _run_tui(self) -> tuple[int | None, float]:
        """Foreground live dashboard. Blocks until the user quits. Returns
        (returncode, seconds_ran) so run() can distinguish a real quit from a
        failed launch (missing venv/tty).

        Prefers the interactive multi-game TUI (tui_multi_game_viewer.py — the
        reference per-game panels: grid, reasoning, REPL, world-model; needs the
        progressive-learning venv). Falls back to the self-contained viewer.py
        status table (this venv) when that checkout is absent."""
        # Wait briefly for the container's status.json so the TUI opens populated.
        for _ in range(20):
            if (self.container / "status.json").exists():
                break
            time.sleep(0.2)
        if PL_PY.exists():
            tui_cmd = [
                str(PL_PY),
                str(EXAMPLE_DIR / "tui_multi_game_viewer.py"),
                str(self.container),
                "--watch",
                str(self.cfg["watch"]),
            ]
            tui_cwd = PL_DIR  # arc_agi_3 reference package imports relative to here
        else:
            tui_cmd = [
                sys.executable,
                str(EXAMPLE_DIR / "viewer.py"),
                str(self.container),
                "--watch",
                str(self.cfg["watch"]),
            ]
            tui_cwd = REPO_ROOT
        t0 = time.monotonic()
        try:
            # Popen (not run) so the wall-clock deadline watcher can terminate the
            # TUI and unblock run() when the fleet is force-stopped.
            self._tui_proc = subprocess.Popen(tui_cmd, cwd=str(tui_cwd))
        except (FileNotFoundError, OSError) as e:
            print(f"[multi] TUI unavailable ({e}); continuing headless (--no-tui to hide)")
            return 1, 0.0
        rc = self._tui_proc.wait()
        return rc, time.monotonic() - t0

    def run(self) -> None:
        print(f"[multi] container: {self.container}")
        print(f"[multi] {len(self.items)} runs, parallel={self.cfg['parallel']}")
        self._install_shutdown_hooks()  # teardown guaranteed from here on
        self._start_broker()
        # Optional wall-clock cap for the whole fleet.
        self._deadline_hit = False
        max_wall = self.cfg.get("max_wall_seconds")
        if max_wall and max_wall > 0:
            print(
                f"[multi] wall-clock deadline: {int(max_wall)}s "
                f"({max_wall / 3600:.1f}h) — fleet auto-stops when reached"
            )
            threading.Thread(
                target=self._deadline_watch, args=(float(max_wall),), daemon=True, name="deadline"
            ).start()
        try:
            if self.cfg["no_tui"]:
                self._orchestrate()
            else:
                orch = threading.Thread(target=self._orchestrate, daemon=True, name="orchestrator")
                orch.start()
                rc, secs = self._run_tui()
                if secs < 3 and rc not in (0, None):
                    # TUI never really started (no PL venv / no tty) — don't tear
                    # the fleet down; just run headless to completion.
                    print(
                        "[multi] TUI did not start — continuing headless "
                        "(use --no-tui for automated runs)"
                    )
                    orch.join()
                else:
                    self._stop.set()
                    self._cleanup_all()  # TUI quit -> stop running games
                    orch.join(timeout=30)
        finally:
            # Normal end, TUI quit, exception, or a signal-driven SystemExit all
            # land here; _cleanup_all is idempotent (on a clean finish the games are
            # already done, so it just closes the scorecard + writes the summary).
            self._cleanup_all()
        print(f"[multi] done. container: {self.container}")

    def _write_summary(self) -> None:
        results = []
        for _key, info in self.status.items():
            rhae = info.get("rhae")
            results.append(
                {
                    "game_id": info.get("game"),
                    "variant": info.get("variant"),
                    "levels": info.get("levels"),
                    "steps": info.get("steps"),
                    "outcome": info.get("outcome"),
                    "wall_time": info.get("wall_time"),
                    # RHAE % — the viewer's dashboard + summary read this.
                    "rhae_game_pct": round(rhae * 100, 2) if rhae is not None else None,
                }
            )
        # Aggregate block the viewer's summary line reads (avg over completed runs).
        done = [r for r in results if r.get("steps") is not None]
        n = len(done) or 1
        aggregate = {
            "total_games": len(results),
            "completed": len(done),
            "total_levels_completed": sum(r["levels"] or 0 for r in done),
            "avg_steps_per_game": round(sum(r["steps"] or 0 for r in done) / n, 1),
            "avg_wall_time_per_game": round(sum(r["wall_time"] or 0 for r in done) / n, 1),
        }
        scores = [r["rhae_game_pct"] for r in done if r.get("rhae_game_pct") is not None]
        if scores:
            aggregate["avg_rhae_pct"] = round(sum(scores) / len(scores), 2)
        payload: dict = {
            "container": self.container.name,
            "aggregate": aggregate,
            "results": results,
        }
        if getattr(self, "_deadline_hit", False):
            payload["shutdown_reason"] = "wall_clock_deadline"
        (self.container / "summary.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    _load_dotenv()
    args = parse_args()
    cfg = load_config(args.config, args)
    MultiRunner(cfg).run()


if __name__ == "__main__":
    main()
