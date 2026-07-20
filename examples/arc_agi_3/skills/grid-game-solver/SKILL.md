---
name: grid-game-solver
description: Solve an interactive grid game iteratively — observe, hypothesize, experiment, record knowledge, and submit action sequences until every level is won.
---

# Interactive Grid-Game Solver

You are solving an **interactive grid game**. You do not know the rules — you must
discover them by experimenting, forming hypotheses, and testing them. The screen is a
64×64 grid of colors (hex digits 0–f). Games have multiple levels; completing all levels
wins the game (state becomes WIN).

## The loop

Each turn a new game state arrives on your `game_states` queue (see `<arc_game>` for the
live status header). You analyze it, update your knowledge, and answer with
`self.submit_actions([...], rationale)`. The harness executes the sequence, then sends
the next state. Repeat until WIN.

### Actions

| Action | Meaning |
|---|---|
| `UP` / `DOWN` / `LEFT` / `RIGHT` | Directional input (often movement — semantics are game-specific) |
| `USE` | Interact / confirm (game-specific) |
| `CLICK x y` | Click cell at column x, row y (0-indexed) — only if CLICK is available |
| `RESET` | Restart the current level (loses level progress — use deliberately) |
| `UNDO` | Undo the last action — only if available |

Only actions in the state's `available_actions` are legal. Keep sequences **short (1–5)**
while your understanding is weak; go long (up to 20) only when executing a confident,
tested plan. After a level completes or a GAME_OVER, the harness cancels the rest of
your sequence so you can re-plan.

## How to play well

1. **Compute, don't squint.** Parse the grid with `self.grid_array(state["grid_rows"])`
   and analyze with numpy: connected components, bounding boxes, color counts, symmetry,
   what changed (`diff_summary` tells you which cells moved). Use
   `self.render_grid(...)` when you need to *look* at a region with coordinates.
   `self.trajectory()` returns the whole game history (every executed action with the
   grid after it) — use it to diff distant states, spot periodic behavior, or re-check
   what an earlier action really did instead of trusting recollection.
2. **One experiment per weak hypothesis.** Early actions are probes. In `rationale`,
   write your *prediction*; next turn, check it against `action_results` and
   `diff_summary`. A wrong prediction is information — record it.
3. **Build a world model in code.** Persist parsing/prediction functions with
   `self.write_helper("world_model.py", source)`; reload each turn with
   `self.load_helpers()` and call via `self.h.world_model.<fn>(...)`. Refine it whenever
   a prediction fails. Good first helper: `parse(grid) -> dict` extracting the player,
   objects, walls, counters.
4. **Track the goal.** Look for win conditions: target patterns, doors/keys, counters,
   matching shapes, an exit. Levels reuse mechanics with rising complexity — knowledge
   from level 1 transfers; recall it when a new level starts.
5. **Don't grind.** If your last few batches produced no meaningful change (same
   diff pattern, no progress), stop: you're missing a mechanic. Re-examine the grid,
   try an untried action, click salient objects, or RESET with a better plan.
6. **RESET strategically.** If a level looks unwinnable (consumed resource, dead end),
   first record what you learned and the exact improved action plan, then RESET and
   execute it.
7. **Respect the human.** Anything on `user_messages` overrides your plan — a
   supervisor may be watching the tmux session. Acknowledge with `message()`.

## Knowledge discipline

Your knowledge store API is in `<knowledge_api>` (memory tools or markdown files,
depending on configuration). Every turn, AFTER observing the results:

- Record **observations** (confirmed facts: "walls are color 3", "USE toggles the
  switch under the player") with the evidence.
- Record **hypotheses** with status (untested / confirmed / contradicted) and the
  concrete test that would decide them. Update statuses as evidence arrives.
- Record **action semantics** — what each action actually does in this game.
- On level completion: record the level's mechanic, the winning sequence, and what
  generalizes. Then **reflect/curate** per `<knowledge_api>`.
- Record **failures** too — what did NOT work, so you never repeat it.

Before deciding, ALWAYS consult the store first (recall / read the files). At the start
of a new level, retrieve everything relevant from previous levels.

## Turn contract

Every turn ends in exactly one of:
- `self.submit_actions([...], rationale="prediction: ...")` then
  `return_result(RespondReason.WAIT, explanation="waiting for the harness ...")` — the
  normal case; or
- if the state is `WIN` (or the harness note says the run stopped): write the final
  reflection to the knowledge store, `message()` a short summary of the solution, and
  `return_result(RespondReason.DONE, explanation="...")`.

Never end a turn without submitting actions on an active game. Never fabricate a state —
if `game_states` brought nothing and `<arc_game>` shows you already submitted for this
turn, just WAIT again.
