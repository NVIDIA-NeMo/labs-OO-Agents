# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Self-contained live TUI for the ARC-AGI-3 nemo-oo solver multi-run.

This viewer reads ONLY a run container's on-disk files -- it has no dependency
on the solver's internal Python package or any reference-dashboard tree, so it
runs in the example's own (main) venv via ``sys.executable``.

Data sources (polled ~every 1.5s):

* ``<container>/status.json`` -- top-level ``total/running/completed/failed``
  counts plus a ``runs`` map with per-game ``status``, ``_run_dir``, ``levels``,
  ``steps``, ``outcome``, ``wall_time`` and ``rhae``.
* ``<run_dir>/agent_logs/nemo/<role>/events.jsonl`` -- one JSON object per line,
  written live by the harness recorder and the trace->event exporter. The
  handled event types are ``solver_start`` / ``env_step`` / ``llm_call`` /
  ``repl_execute`` / ``round_complete`` / ``step_complete`` / ``level_score``.

Per game the dashboard shows a row with: state, levels/steps, wall time, ``$``
cost (priced from the ``llm_call`` token counts x per-model pricing), LLM call
count, input->output tokens, reasoning tokens, tool-call / python-block counts,
REPL (code-cell) success/total, round count + average round seconds, average
step wall time, and live RHAE lower/upper bounds with per-level scores. An
aggregate header summarises the whole container.

Usage::

    python viewer.py --results-dir <container_dir> [--watch team_leader]
    python viewer.py <container_dir> [--watch team_leader]

``--results-dir`` may point either at a container (a directory containing
``status.json``) or at a parent results directory, in which case the newest
container beneath it is selected.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Pricing -- per-1M-token literals mirrored from the solver's llm_configs so this
# file needs no import from the solver package. The keys match the ``agent_id``
# the event exporter writes on each ``llm_call`` event.
# ---------------------------------------------------------------------------
_ANTHROPIC_OPUS = {"input": 5.0, "output": 25.0, "cached": 0.50, "cache_write": 6.25}
_OPENAI_GPT55 = {"input": 5.0, "output": 30.0, "cached": 0.50, "cache_write": 5.0}
_GEMINI_FLASH = {"input": 0.50, "output": 3.0, "cached": 0.05, "cache_write": 0.50}

LLM_PRICING = {
    "opus": _ANTHROPIC_OPUS,
    "gpt5.5": _OPENAI_GPT55,
    "gpt5.6": _OPENAI_GPT55,  # gpt-5.6(-sol): same published figures as gpt-5.5
    "flash": _GEMINI_FLASH,
}
_ZERO_PRICING = {"input": 0.0, "output": 0.0, "cached": 0.0}


def get_pricing(llm_key: str) -> dict:
    """Return per-million-token pricing for a short or ``arc3-*`` name."""
    pricing = LLM_PRICING.get(llm_key)
    if pricing:
        return pricing
    key = llm_key[5:] if llm_key.startswith("arc3-") else llm_key
    while key:
        pricing = LLM_PRICING.get(key)
        if pricing:
            return pricing
        if "-" not in key:
            break
        key = key.rsplit("-", 1)[0]
    return _ZERO_PRICING


def _model_cost(mid: str, mt: dict) -> float:
    """Dollar cost for one model's accumulated token counts."""
    p = get_pricing(mid)
    uncached = mt["input"] - mt["cache"] - mt["cache_write"]
    in_cost = (
        max(uncached, 0) / 1e6 * p["input"]
        + mt["cache"] / 1e6 * p["cached"]
        + mt["cache_write"] / 1e6 * p.get("cache_write", p["input"])
    )
    return in_cost + mt["output"] / 1e6 * p["output"]


def _pricing_key(model: str) -> str:
    """Map a litellm model id to a pricing key (``opus`` / ``gpt5.5`` / ``flash``)."""
    m = (model or "").lower()
    if "gpt-5.5" in m or "gpt5.5" in m:
        return "gpt5.5"
    if "gpt-5.6" in m or "gpt5.6" in m:
        return "gpt5.6"
    if "opus" in m:
        return "opus"
    if "flash" in m:
        return "flash"
    return m.rsplit("/", 1)[-1] if m else "unknown"


# ---------------------------------------------------------------------------
# RHAE scoring -- pure functions mirrored from the solver's rhae module.
# ---------------------------------------------------------------------------
LEVEL_SCORE_CAP = 1.15
PERFECT_LEVEL_SCORE = 1.0


def rhae_level_score(baseline: float, ai_actions: float) -> float:
    """Squared human-relative action efficiency, capped at 115%; 0 if invalid."""
    if baseline <= 0 or ai_actions <= 0:
        return 0.0
    return min(LEVEL_SCORE_CAP, (float(baseline) / float(ai_actions)) ** 2)


def _weighted_fraction(n_levels: int, completed_levels: int) -> float:
    if n_levels <= 0:
        return 0.0
    completed = max(0, min(int(completed_levels), n_levels))
    return (completed * (completed + 1) / 2) / (n_levels * (n_levels + 1) / 2)


def _bounded_level_score(score: float) -> float:
    return min(LEVEL_SCORE_CAP, max(0.0, float(score)))


def rhae_score_from_level_scores(
    level_scores: list[float], n_levels: int, completed_levels: int
) -> float:
    """Capped environment RHAE from per-level scores."""
    if n_levels <= 0:
        return 0.0
    completed = max(0, min(int(completed_levels), n_levels))
    total_weight = n_levels * (n_levels + 1) / 2
    weighted_score = 0.0
    for i in range(completed):
        score = level_scores[i] if i < len(level_scores) else 0.0
        weighted_score += (i + 1) * _bounded_level_score(score)
    env_score = weighted_score / total_weight
    env_cap = _weighted_fraction(n_levels, completed)
    return min(env_cap, env_score)


def rhae_bounds(
    baseline_actions: list[int],
    completed_level_scores: list[float],
    current_level: int,
    actions_this_level: int,
    *,
    done: bool = False,
) -> tuple[float, float]:
    """Live ``(RHAE_L, RHAE_U)`` bounds: score if it ends now vs. optimistic."""
    n_levels = len(baseline_actions)
    if n_levels <= 0:
        return 0.0, 0.0
    completed = max(0, min(int(current_level), n_levels))
    lower_scores = [0.0] * n_levels
    for i in range(min(completed, len(completed_level_scores))):
        lower_scores[i] = _bounded_level_score(completed_level_scores[i])
    lower = rhae_score_from_level_scores(lower_scores, n_levels, completed)
    if done or completed >= n_levels:
        return lower, lower
    optimistic_scores = list(lower_scores)
    optimistic_scores[completed] = rhae_level_score(
        baseline_actions[completed], max(0, int(actions_this_level)) + 1
    )
    for i in range(completed + 1, n_levels):
        optimistic_scores[i] = PERFECT_LEVEL_SCORE
    upper = rhae_score_from_level_scores(optimistic_scores, n_levels, n_levels)
    return lower, max(lower, upper)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def _fmt_tokens(n: float) -> str:
    n = int(n or 0)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def _fmt_dur(seconds: float) -> str:
    s = int(seconds or 0)
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    if m < 60:
        return f"{m}m{sec:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


_DONE_STATES = {"completed", "failed", "crashed", "terminated", "killed", "error"}
_ERROR_STATES = {"failed", "crashed", "terminated", "killed", "error"}


def _state_label(info: dict) -> tuple[str, str]:
    """(display label, rich style) for a run's status.json entry."""
    st = (info.get("status") or "?").lower()
    outcome = (info.get("outcome") or "").lower()
    if st == "running":
        return "running", "cyan"
    if st == "queued":
        return "queued", "yellow"
    if st == "completed":
        if outcome == "win":
            return "win", "bold green"
        return outcome or "done", "green"
    if st in _ERROR_STATES:
        return "agent-error", "bold red"
    return st, "white"


# ---------------------------------------------------------------------------
# Per-game accumulator (folds events.jsonl into aggregate stats)
# ---------------------------------------------------------------------------
@dataclass
class GameStats:
    game_id: str
    total_llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_reasoning: int = 0
    total_cache_read: int = 0
    num_tool_calls: int = 0
    num_python_blocks: int = 0
    repl_total: int = 0
    repl_success: int = 0
    total_rounds: int = 0
    total_round_time: float = 0.0
    llm_s_sum: float = 0.0
    repl_s_sum: float = 0.0
    compaction_s_sum: float = 0.0
    completed_steps: int = 0
    total_step_time: float = 0.0
    step: int = 0
    level: int = 0
    actions_this_level: int = 0
    max_steps: int = 0
    solver_game_id: str = ""
    first_ts: float = 0.0
    last_ts: float = 0.0
    model_tokens: dict = field(default_factory=dict)
    baseline_actions: list = field(default_factory=list)
    rhae_level_scores: list = field(default_factory=list)
    ai_actions_per_level: list = field(default_factory=list)

    def update(self, ev: dict) -> None:
        etype = ev.get("event")
        ts = ev.get("unix_time_s")
        if isinstance(ts, (int, float)):
            self.first_ts = ts if not self.first_ts else min(self.first_ts, ts)
            self.last_ts = max(self.last_ts, ts)

        if etype == "llm_call":
            inp = int(ev.get("input_tokens", 0) or 0)
            out = int(ev.get("output_tokens", 0) or 0)
            self.total_llm_calls += 1
            self.input_tokens += inp
            self.output_tokens += out
            self.total_reasoning += int(ev.get("reasoning_tokens", 0) or 0)
            self.total_cache_read += int(ev.get("cache_read_tokens", 0) or 0)
            self.num_tool_calls += int(ev.get("num_tool_calls", 0) or 0)
            self.num_python_blocks += int(ev.get("num_python_blocks", 0) or 0)
            key = ev.get("agent_id", "") or ""
            if key not in LLM_PRICING:
                key = _pricing_key(ev.get("model", "") or "")
            mt = self.model_tokens.setdefault(
                key, {"input": 0, "output": 0, "cache": 0, "cache_write": 0}
            )
            mt["input"] += inp
            mt["output"] += out
            mt["cache"] += int(ev.get("cache_read_tokens", 0) or 0)
            mt["cache_write"] += int(ev.get("cache_creation_tokens", 0) or 0)
        elif etype == "repl_execute":
            self.repl_total += 1
            if ev.get("success", True):
                self.repl_success += 1
        elif etype == "round_complete":
            self.total_rounds += 1
            self.total_round_time += float(ev.get("round_total_s", 0) or 0)
            self.llm_s_sum += float(ev.get("llm_s", 0) or 0)
            self.repl_s_sum += float(ev.get("repl_s", 0) or 0)
            self.compaction_s_sum += float(ev.get("compaction_s", 0) or 0)
        elif etype == "step_complete":
            self.completed_steps += 1
            self.total_step_time += float(ev.get("wall_time_s", 0) or 0)
        elif etype == "level_score":
            ai = int(ev.get("ai_actions", 0) or 0)
            bl = int(ev.get("baseline_actions", 0) or 0)
            score = (
                rhae_level_score(bl, ai) if bl and ai else float(ev.get("rhae_score", 0.0) or 0.0)
            )
            lvl = int(ev.get("level", len(self.rhae_level_scores)) or 0)
            while len(self.rhae_level_scores) <= lvl:
                self.rhae_level_scores.append(0.0)
            self.rhae_level_scores[lvl] = score
            while len(self.ai_actions_per_level) <= lvl:
                self.ai_actions_per_level.append(0)
            self.ai_actions_per_level[lvl] = ai
        elif etype == "solver_start":
            baseline = ev.get("baseline_actions")
            if baseline:
                self.baseline_actions = list(baseline)
            if ev.get("game_id"):
                self.solver_game_id = ev["game_id"]
            self.max_steps = int(ev.get("max_steps", self.max_steps) or self.max_steps)
        elif etype == "env_step":
            self.step = int(ev.get("step", self.step) or self.step)
            new_level = int(ev.get("levels_completed", self.level) or self.level)
            if ev.get("action_name") or ev.get("action"):
                self.actions_this_level += 1
            if new_level > self.level:
                self.actions_this_level = 0
            self.level = new_level

    def cost(self) -> float:
        return sum(_model_cost(mid, mt) for mid, mt in self.model_tokens.items())

    def rhae(self, done: bool) -> tuple[float, float] | None:
        if not self.baseline_actions:
            return None
        return rhae_bounds(
            [int(x) for x in self.baseline_actions],
            [float(x) for x in self.rhae_level_scores],
            self.level,
            self.actions_this_level,
            done=done,
        )


# ---------------------------------------------------------------------------
# Incremental events.jsonl tailer + status.json binding
# ---------------------------------------------------------------------------
class GameTracker:
    """Folds a game's events.jsonl into a ``GameStats``, reading only new bytes."""

    def __init__(self, game_id: str, run_dir: Path | None, watch: str):
        self.game_id = game_id
        self.run_dir = run_dir
        self.watch = watch
        self.info: dict = {}
        self.stats = GameStats(game_id)
        self.path: Path | None = None
        self._offset = 0
        self._buf = b""

    def _resolve_path(self) -> Path | None:
        if self.path is not None and self.path.exists():
            return self.path
        if not self.run_dir or not self.run_dir.exists():
            return None
        direct = self.run_dir / "agent_logs" / "nemo" / self.watch / "events.jsonl"
        if direct.exists():
            self.path = direct
            return direct
        for pattern in (f"agent_logs/*/{self.watch}/events.jsonl", "agent_logs/*/*/events.jsonl"):
            matches = sorted(self.run_dir.glob(pattern))
            if matches:
                self.path = matches[0]
                return self.path
        return None

    def poll(self) -> None:
        path = self._resolve_path()
        if path is None:
            return
        try:
            size = path.stat().st_size
        except OSError:
            return
        if size < self._offset:  # truncated / rotated -- restart the fold
            self.stats = GameStats(self.game_id)
            self._offset = 0
            self._buf = b""
        if size == self._offset:
            return
        try:
            with path.open("rb") as f:
                f.seek(self._offset)
                chunk = f.read()
        except OSError:
            return
        self._offset = size
        data = self._buf + chunk
        lines = data.split(b"\n")
        self._buf = lines.pop()  # trailing partial line (may be empty)
        for raw in lines:
            if not raw.strip():
                continue
            try:
                self.stats.update(json.loads(raw))
            except (json.JSONDecodeError, ValueError):
                continue

    @property
    def done(self) -> bool:
        return (self.info.get("status") or "").lower() in _DONE_STATES

    def wall_time(self) -> float:
        wt = self.info.get("wall_time")
        if isinstance(wt, (int, float)) and wt:
            return float(wt)
        if self.stats.first_ts and self.stats.last_ts:
            return self.stats.last_ts - self.stats.first_ts
        return 0.0


# ---------------------------------------------------------------------------
# Container discovery + status.json reading
# ---------------------------------------------------------------------------
def resolve_container(path: Path) -> Path | None:
    """Return the container dir: ``path`` itself if it has a status.json, else
    the newest immediate child that does."""
    if (path / "status.json").exists():
        return path
    candidates = [d for d in path.iterdir() if d.is_dir() and (d / "status.json").exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


def read_status(container: Path) -> dict:
    try:
        return json.loads((container / "status.json").read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _rhae_cell(tracker: GameTracker) -> Text:
    bounds = tracker.stats.rhae(tracker.done)
    if bounds is None and not tracker.stats.rhae_level_scores:
        return Text("-", style="dim")
    lower, upper = bounds if bounds else (0.0, 0.0)
    scores = tracker.stats.rhae_level_scores[
        : max(tracker.stats.level, len(tracker.stats.rhae_level_scores))
    ]
    levels = ",".join(f"{s * 100:.0f}" for s in scores) if scores else "-"
    style = "green" if lower >= 1.0 else ("bright_yellow" if upper > 0 else "dim")
    return Text(f"{lower * 100:.0f}/{upper * 100:.0f}% [{levels}]", style=style)


def build_view(container: Path, status: dict, trackers: dict[str, GameTracker]) -> Group:
    table = Table(expand=True, header_style="bold", border_style="dim", pad_edge=False)
    table.add_column("Game", no_wrap=True)
    table.add_column("State", no_wrap=True)
    table.add_column("Lv/Step", justify="right", no_wrap=True)
    table.add_column("Wall", justify="right", no_wrap=True)
    table.add_column("$", justify="right", no_wrap=True)
    table.add_column("LLM", justify="right", no_wrap=True)
    table.add_column("In→Out", justify="right", no_wrap=True)
    table.add_column("Reason", justify="right", no_wrap=True)
    table.add_column("Tools/py", justify="right", no_wrap=True)
    table.add_column("REPL", justify="right", no_wrap=True)
    table.add_column("Rounds", justify="right", no_wrap=True)
    table.add_column("Step s", justify="right", no_wrap=True)
    table.add_column("RHAE L/U [levels]", no_wrap=True)

    # Aggregates
    agg_in = agg_out = agg_reason = 0
    agg_llm = agg_tools = agg_py = agg_repl_ok = agg_repl_tot = agg_rounds = 0
    agg_model: dict[str, dict] = {}
    rhae_completed: list[float] = []

    order = sorted(
        trackers.values(),
        key=lambda t: (0 if (t.info.get("status") or "") == "running" else 1, t.game_id),
    )
    for t in order:
        st = t.stats
        label, style = _state_label(t.info)
        levels = t.info.get("levels", st.level)
        steps = t.info.get("steps", st.step)
        cost = st.cost()
        reason_avg = st.total_reasoning / max(st.completed_steps, 1)
        avg_round = st.total_round_time / max(st.total_rounds, 1)
        avg_step = st.total_step_time / max(st.completed_steps, 1)

        agg_in += st.input_tokens
        agg_out += st.output_tokens
        agg_reason += st.total_reasoning
        agg_llm += st.total_llm_calls
        agg_tools += st.num_tool_calls
        agg_py += st.num_python_blocks
        agg_repl_ok += st.repl_success
        agg_repl_tot += st.repl_total
        agg_rounds += st.total_rounds
        for mid, mt in st.model_tokens.items():
            acc = agg_model.setdefault(mid, {"input": 0, "output": 0, "cache": 0, "cache_write": 0})
            for k in acc:
                acc[k] += mt[k]
        if t.done and (b := st.rhae(True)):
            rhae_completed.append(b[0])

        table.add_row(
            t.game_id,
            Text(label, style=style),
            f"{levels}/{steps}",
            _fmt_dur(t.wall_time()),
            f"${cost:.2f}",
            str(st.total_llm_calls),
            f"{_fmt_tokens(st.input_tokens)}→{_fmt_tokens(st.output_tokens)}",
            f"{_fmt_tokens(st.total_reasoning)}|{reason_avg:.0f}",
            f"{st.num_tool_calls}/{st.num_python_blocks}",
            f"{st.repl_success}/{st.repl_total}",
            f"{st.total_rounds}@{avg_round:.0f}s",
            f"{avg_step:.0f}s",
            _rhae_cell(t),
        )

    agg_cost_check = sum(_model_cost(mid, mt) for mid, mt in agg_model.items())
    table.add_section()
    table.add_row(
        Text("TOTAL", style="bold"),
        Text(f"{len(trackers)} games", style="bold"),
        "",
        "",
        Text(f"${agg_cost_check:.2f}", style="bold green"),
        str(agg_llm),
        f"{_fmt_tokens(agg_in)}→{_fmt_tokens(agg_out)}",
        _fmt_tokens(agg_reason),
        f"{agg_tools}/{agg_py}",
        f"{agg_repl_ok}/{agg_repl_tot}",
        str(agg_rounds),
        "",
        Text(
            f"avg(completed) {sum(rhae_completed) / len(rhae_completed) * 100:.1f}%"
            if rhae_completed
            else "-",
            style="bold bright_yellow",
        ),
    )

    # Header banner
    by_model = " ".join(
        f"{mid}:${_model_cost(mid, mt):.1f}" for mid, mt in sorted(agg_model.items())
    )
    header = Text()
    header.append(f"{container.name}\n", style="bold white")
    header.append(
        f"total={status.get('total', len(trackers))} "
        f"running={status.get('running', 0)} "
        f"completed={status.get('completed', 0)} "
        f"failed={status.get('failed', 0)} "
        f"queued={status.get('queued', 0)}\n",
        style="cyan",
    )
    header.append(f"$ {agg_cost_check:.2f}", style="bold green")
    if by_model:
        header.append(f"  ({by_model})", style="dim")
    header.append(
        f"   tokens {_fmt_tokens(agg_in)}→{_fmt_tokens(agg_out)}"
        f"   reasoning {_fmt_tokens(agg_reason)}"
        f"   repl {agg_repl_ok}/{agg_repl_tot}",
        style="dim",
    )
    return Group(Panel(header, border_style="blue", title="ARC-AGI-3 live viewer"), table)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(container: Path, watch: str, interval: float = 1.5) -> None:
    console = Console()
    trackers: dict[str, GameTracker] = {}
    with Live(console=console, screen=False, auto_refresh=False) as live:
        while True:
            status = read_status(container)
            runs = status.get("runs", {})
            for game_id, info in runs.items():
                tracker = trackers.get(game_id)
                run_dir = info.get("_run_dir")
                run_path = Path(run_dir) if run_dir else None
                if tracker is None:
                    tracker = GameTracker(game_id, run_path, watch)
                    trackers[game_id] = tracker
                elif tracker.run_dir is None and run_path is not None:
                    tracker.run_dir = run_path
                tracker.info = info
                tracker.poll()
            live.update(build_view(container, status, trackers), refresh=True)
            time.sleep(interval)


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-contained live ARC-AGI-3 multi-run viewer")
    ap.add_argument(
        "results_dir_pos",
        nargs="?",
        help="run container (has status.json) or a parent results directory",
    )
    ap.add_argument(
        "--results-dir",
        dest="results_dir",
        default=None,
        help="run container (has status.json) or a parent results directory",
    )
    ap.add_argument(
        "--watch",
        default="team_leader",
        help="agent role whose events.jsonl is read (default: team_leader)",
    )
    ap.add_argument("--interval", type=float, default=1.5, help="poll seconds (default: 1.5)")
    args = ap.parse_args()

    target = args.results_dir or args.results_dir_pos
    if not target:
        ap.error("provide a container via positional arg or --results-dir")
    root = Path(target)
    if not root.exists():
        print(f"Results directory not found: {root}")
        sys.exit(1)
    container = resolve_container(root)
    if container is None:
        print(f"No status.json found under: {root}")
        sys.exit(1)
    try:
        run(container, args.watch, args.interval)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
