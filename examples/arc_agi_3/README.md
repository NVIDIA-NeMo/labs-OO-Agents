# ARC-AGI-3 solving agent

A worked example of building a capable agent with **nooa**: a single agent that
plays [ARC-AGI-3](https://arcprize.org/arc-agi/3) interactive grid games —
discovering each game's hidden rules by experimenting, then solving every level.

It shows a realistic agent end-to-end:

- **One `nooa` agent** (`solver_agent.py`) whose generation methods *are* its
  capabilities — it reads the grid, forms hypotheses, writes reusable Python
  helpers (its "world model"), records what it learns, and submits action batches.
- **Two knowledge variants** you can compare head-to-head:
  - `memory` — the nooa long-term **memory** system (`self.memory.recall/remember/reflect`).
  - `mdfiles` — plain Markdown **knowledge files** the agent reads/writes.
- **In-process isolation** so the game it is solving cannot leak into its context
  (an `open()` jail + code restrictions + path redaction), plus an optional OS
  sandbox (`sandbox.py`).
- **A live TUI dashboard** (`viewer.py`) with per-game cost, tokens, reasoning,
  tool-call, REPL, round and RHAE-score panels.

## How it fits together

```
run_multi.py ──┬─ run_solver.py ── launcher.py ── solver_agent.py   (the nooa agent)
   (fleet)     │        │                              ▲  │
               │        └── harness.py ── arc_agi_3/ ──┘  │  ipc/states.jsonl  (harness → agent)
               │            (the game side)               ▼  ipc/actions.jsonl (agent → harness)
               └─ viewer.py (live dashboard)  ·  scorecard_broker.py (competition scorecard)
```

The **agent** and the **environment harness** run as separate processes that talk
only through two append-only files (`ipc/states.jsonl`, `ipc/actions.jsonl`) — so
the agent never imports or sees the game's code. The harness drives the game
through `arc_agi_3/`, a small vendored wrapper over the public ARC-AGI-3 SDK.

## Install

From a checkout of this repository:

```bash
uv sync --extra arc        # or: pip install -e ".[arc]"
```

The `arc` extra pulls the public **ARC-AGI-3 SDK** (`arc-agi`, `arcengine`) plus
`pillow` (visual grid input) and `rich` (the viewer). Everything runs
in this one environment — there is no second venv.

## Configure

The agent talks to any **OpenAI-compatible LLM gateway**. Put your settings in a
`.env` file at the repo root (or export them):

```bash
# LLM (required)
ARC_LLM_MODEL=openai/your-model            # an OpenAI/Anthropic model on your gateway
ARC_LLM_BASE_URL=https://your-gateway/v1
ARC_LLM_API_KEY=sk-...

# Embeddings — required only for the `memory` variant
MEM_EMBED_MODEL=openai/your-embedding-model
MEM_EMBED_BASE_URL=https://your-gateway/v1
MEM_EMBED_API_KEY=sk-...
MEM_EMBED_DIMS=1024

# ARC-AGI-3 backend key — from https://arcprize.org
# Needed to DOWNLOAD games for offline play (one time) and to play competition mode.
ARC_API_KEY=...
```

Notes:
- `ARC_LLM_MODEL` is passed to the gateway with an `openai/` provider prefix; set
  `ARC_LLM_USE_RESPONSES=1` if your gateway needs the OpenAI *Responses* API for a
  given model (auto-on for `gpt-5.5`).
- Offline mode replays games from a local `environment_files/` directory. The SDK
  **downloads** them on first use with your `ARC_API_KEY` (the games are ARC
  Prize's, fetched per user — they are not shipped with this example).

## Run

### Offline — a single game

```bash
python examples/arc_agi_3/run_solver.py \
    --game ls20 --variant memory --operation-mode offline \
    --model openai/your-model --reasoning-effort high --max-env-steps 200
```

This opens a tmux session with the live agent (attach with `tmux attach -t <name>`),
runs the game to completion, and writes results under `results/arc_agi_3/`.

### Offline — a fleet (both configs)

```bash
python examples/arc_agi_3/run_multi.py --config examples/arc_agi_3/configs/offline.yaml
```

Runs several games × variants in parallel with the live dashboard (add `--no-tui`
for headless). `configs/offline.yaml` runs both `memory` and `mdfiles` variants on
a handful of games.

### Competition

```bash
python examples/arc_agi_3/run_multi.py --config examples/arc_agi_3/configs/competition.yaml
```

Competition mode plays live against `three.arcprize.org` and needs `ARC_API_KEY`.
All games in a fleet share **one scorecard** (`scorecard_broker.py` mints it and
propagates the session so the whole run is one leaderboard entry). Per-game tmux
sessions live on a private socket so they don't clutter `tmux ls`.

## Where results appear

Each run creates a container directory:

```
results/arc_agi_3/nemo_solver/<timestamp>_<config-name>/
├── manifest.json          # run metadata (model, parallelism, scorecard id, …)
├── status.json            # live per-game status (the viewer reads this)
├── scorecard.json         # competition scorecard summary (written at the end)
└── <game>/<variant>/<ts>_<game>_<variant>/
    ├── gameplay.json       # steps, levels completed, outcome
    ├── ipc/                # states.jsonl / actions.jsonl (the agent↔harness channel)
    ├── agent_logs/…/events.jsonl   # llm_call / repl_execute / round / RHAE events
    ├── team_nemo/shared/   # the agent's workspace: helpers/, knowledge/, memory.sqlite
    └── traces/             # JSONL traces
```

Watch a run live (also launched automatically by `run_multi`):

```bash
python examples/arc_agi_3/viewer.py results/arc_agi_3/nemo_solver/<container>
```

The dashboard shows, per game: state, level/step, wall time, **`$` cost**, LLM call
count and tokens, **reasoning** tokens, **tool-call**/REPL counts, rounds, and
**RHAE** score bounds (lower/upper) with per-level detail — plus an aggregate header
and totals.

## The two configs

| Config | Mode | Variants | Notes |
|---|---|---|---|
| `configs/offline.yaml` | offline | `memory` + `mdfiles` | a few games, `parallel: 4` — good for local comparison |
| `configs/competition.yaml` | competition | `memory` | all 25 games, `parallel: 25`, one shared scorecard |

Both feed the grid to the model **visually** (`visual: additive` — the settled
grid is rendered as a color PNG alongside the hex grid each turn), use an effort
ladder (`reasoning_effort: xhigh`, downshifting to `medium` after 600 s in a
turn), and never deliberately terminate a game — the harness force-advances a
default action after prolonged agent silence.

## Layout

| File | Role |
|---|---|
| `solver_agent.py` | the nooa agent (generation methods, world-model helpers, isolation) |
| `launcher.py` | wires the agent into the runtime + memory skill, starts the session |
| `harness.py` | the game side of the IPC loop (drives `arc_agi_3/`) |
| `arc_agi_3/` | vendored wrapper over the public ARC-AGI-3 SDK (`environment`, `rhae`, …) |
| `run_solver.py` / `run_multi.py` | single-game / fleet orchestration |
| `viewer.py` | live TUI dashboard |
| `scorecard_broker.py` | competition shared-scorecard broker |
| `sandbox.py` | optional OS sandbox for the agent process |
| `skills/grid-game-solver/SKILL.md` | the agent's guiding skill |
| `configs/` | the two run configs |
