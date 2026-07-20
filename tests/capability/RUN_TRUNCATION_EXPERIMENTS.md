# Reproducing the Truncation Marker Experiments

This document is a step-by-step guide for re-running the truncation-comprehension experiment series from scratch. Companion to [`EXPERIMENT_truncation_markers.md`](EXPERIMENT_truncation_markers.md), which has the methodology and findings.

---

## Prerequisites

1. **Repo checkout** on the experiment branch:
   ```bash
   git fetch origin test/truncation-comprehension
   git checkout test/truncation-comprehension
   uv sync --all-extras --no-extra sandbox
   ```

2. **API access** — these models route through the NVIDIA internal inference gateway. You need:
   - `NVIDIA_INTERNAL_API_KEY` in your `.env` (or shell environment) for the Nemotron / Qwen / GPT-OSS / GPT-5 / Claude / Gemini routes.
   - Optionally `NVIDIA_API_KEY` for the public NIM endpoint (a few `qwen` aliases use this).

3. **Model aliases** (claude-haiku, nemotron3-nano-30b, …) ship in the `nemo-oo-agents-nvidia` package — install it (or the `[nvidia]` extra) and verify they're loaded:
   ```bash
   uv run nooa config show
   ```

4. **Optional: trace viewer** for live trace inspection:
   ```bash
   uv run python -m nooa.viewer &  # http://localhost:5001
   ```

5. **Source your env**:
   ```bash
   set -a; source .env; set +a
   ```

---

## What's in this directory

```
tests/capability/
  agents/
    truncation_comprehension.py    # 4 agents:
                                   #   TruncationComprehensionAgent       (PredictStrategy + Answer schema)
                                   #   TruncationComprehensionAgentBare   (PredictStrategy + bare int|None — A/B control)
                                   #   TruncationComprehensionAgentCodeAct (CodeActStrategy + Answer schema)
                                   #   TruncationRealDataAgent{Predict,CodeAct} (real-list parameter)
  config_truncation.yaml           # Test suite definition: ~30 fixtures, ~12 models
  data/
    truncation_aware_*.jsonl       # The 7-question awareness fixtures (lower/xml/pascal/.../bare/...)
    truncation_size_*.jsonl        # Round-2: marker-shape comparison
    truncation_str_v*.jsonl        # String-truncation shape comparison
    truncation_realdata.jsonl      # Real-list fixture for CodeAct vs Predict A/B
    ...
  EXPERIMENT_truncation_markers.md # Findings writeup
  RUN_TRUNCATION_EXPERIMENTS.md    # This file
  analyze_truncation.py            # Result-analysis script (per-model, per-question tables)
```

---

## The experiment rounds

Each round below is a self-contained step. You can run all of them, or pick the ones you want to verify.

### Round 1 — Apples-to-apples: today's pformat vs proposed shapes

**Goal:** Compare four marker styles on the same 7-question awareness fixture.

```bash
uv run python -m eval_pipeline \
  --config tests/capability/config_truncation.yaml \
  --test truncation_aware_today_verbose,truncation_aware_xml,truncation_aware_pascal,truncation_aware_lower \
  --runs 3 --parallel 30 --timeout 240 \
  --output-dir results/round1
```

Then analyze:
```bash
uv run python tests/capability/analyze_truncation.py results/round1 --pivot style-x-question
```

Expected: `lower` style (`list(len=N, items=[…])`) wins at ~84%; today's verbose form ~74%.

### Round 2 — Schema A/B: with-reason vs bare

**Goal:** Show the Pydantic `Answer(answer, reason)` schema is the biggest single lever for truncation awareness.

```bash
uv run python -m eval_pipeline \
  --config tests/capability/config_truncation.yaml \
  --test truncation_aware_today_verbose,truncation_aware_xml,truncation_aware_pascal,truncation_aware_lower,truncation_aware_today_verbose_bare,truncation_aware_xml_bare,truncation_aware_pascal_bare,truncation_aware_lower_bare \
  --runs 3 --parallel 30 --timeout 240 \
  --output-dir results/round2
```

Then:
```bash
uv run python tests/capability/analyze_truncation.py results/round2 --pivot ab-bare-vs-reason
```

Expected: bare agent caps at ~50% (can't return None on awareness questions); with-reason agent reaches ~84%. **+30pp gap.**

### Round 3 — Container generalization

**Goal:** Validate the `lower` shape works across container types (dict, tuple, pydantic, dataclass, json, records-of-dicts).

```bash
uv run python -m eval_pipeline \
  --config tests/capability/config_truncation.yaml \
  --test truncation_aware_lower,truncation_aware_lower_dict,truncation_aware_lower_tuple,truncation_aware_lower_pydantic,truncation_aware_lower_dataclass,truncation_aware_lower_json,truncation_aware_records \
  --runs 3 --parallel 30 --timeout 240 \
  --output-dir results/round3
```

```bash
uv run python tests/capability/analyze_truncation.py results/round3 --pivot fixture-totals
```

Expected: dict 92%, list 84%, tuple 82%, pydantic 79%, dataclass 79%, json 82%, records 92%.

### Round 4 — Other pprint mechanics: depth, cycle, generator, string

**Goal:** Validate the rest of the marker family.

```bash
uv run python -m eval_pipeline \
  --config tests/capability/config_truncation.yaml \
  --test truncation_aware_depth,truncation_aware_cycle,truncation_aware_generator,truncation_str_v1_plus_n,truncation_str_v2_str_len,truncation_str_v3_head_tail,truncation_str_v4_start_end,truncation_str_v5_prefix_suffix,truncation_str_v6_slice_keys \
  --runs 3 --parallel 30 --timeout 240 \
  --output-dir results/round4
```

```bash
uv run python tests/capability/analyze_truncation.py results/round4 --pivot fixture-totals
```

Expected:
- depth (`{Type: N items}`): 98%
- cycle (`<cycle>`): 94%
- generator (`<generator>`): 99%
- string `'foo'+N` (rich legacy): 66%
- string `str(len=N, head=…, tail=…)`: 98%
- string slice-key form `str(len=N, [:50]=…, [-50:]=…)`: 98%

### Round 5 — Flagship-model validation

**Goal:** Confirm the recommendation holds at flagship scale.

```bash
uv run python -m eval_pipeline \
  --config tests/capability/config_truncation.yaml \
  --models claude-sonnet,gpt-5.2,gemini-2.5-pro,nemotron-3-super-v3,gemini-3.1-pro-preview \
  --test truncation_aware_lower,truncation_aware_lower_dict,truncation_aware_lower_tuple,truncation_aware_lower_pydantic,truncation_aware_lower_dataclass,truncation_aware_lower_json,truncation_aware_depth,truncation_aware_string_slice \
  --runs 3 --parallel 30 --timeout 240 \
  --output-dir results/round5
```

```bash
uv run python tests/capability/analyze_truncation.py results/round5 --pivot per-model
```

Expected: claude-sonnet 100%, nemotron-3-super-v3 ~98%, gemini-2.5-pro ~95%, gpt-5.2 ~88%.

### Round 6 — CodeAct + real data (the headline)

**Goal:** Show that when the agent has direct access to the underlying data, CodeAct unlocks elided-position questions that PredictStrategy literally cannot answer.

```bash
uv run python -m eval_pipeline \
  --config tests/capability/config_truncation.yaml \
  --test truncation_realdata_predict,truncation_realdata_codeact \
  --runs 3 --parallel 30 --timeout 240 \
  --output-dir results/round6
```

```bash
uv run python tests/capability/analyze_truncation.py results/round6 --pivot predict-vs-codeact
```

Expected per-question gap (predict → codeact):
- count: 66% → 79%
- min: 48% → 76%   (+28pp)
- 50th item: **5% → 61%**   (**+56pp** — the headline)
- 99th item: 61% → 84%   (+23pp)
- visible-position questions (1st, 3rd, 9th): roughly tied or slight Predict advantage

---

## Run everything

For a full reproduction (all rounds, ~30 minutes wall-clock at parallel=30):

```bash
mkdir -p results
for round in round1 round2 round3 round4 round5 round6; do rm -rf results/$round; done

# Round 1-4 use the small/mini matrix (8 models)
uv run python -m eval_pipeline --config tests/capability/config_truncation.yaml \
  --runs 3 --parallel 30 --timeout 240 \
  --output-dir results/all

uv run python tests/capability/analyze_truncation.py results/all --pivot all
```

This runs every fixture × every model in `agent_models` × 3 runs. Total ≈ 7000-9000 samples.

---

## Interpreting the analysis output

`analyze_truncation.py` produces several pivots. Common ones:

- `--pivot style-x-question`: matrix of marker style × question; useful for round 1.
- `--pivot ab-bare-vs-reason`: side-by-side of bare and with-reason agents; useful for round 2.
- `--pivot fixture-totals`: one row per fixture, one cell per model; quick health check.
- `--pivot per-model`: one row per model, total across all fixtures.
- `--pivot predict-vs-codeact`: Predict vs CodeAct split per question; useful for round 6.
- `--pivot all`: everything (verbose).

Each row shows: `passed/total (%)`. The `passed` count is the sum across `runs` per cell.

---

## Troubleshooting

- **`ValueError: Model 'X' not found.`** — run `nooa config show` to confirm the `nemo-oo-agents-nvidia` package is installed and its bundled defaults are loading.
- **`AuthenticationError: key not allowed to access model`** — the API key doesn't have permission for that model. `gemini-3-pro` is one example; substitute `gemini-2.5-pro` or `gemini-3.1-pro-preview`.
- **Empty `actual: None` for many results** — that model is failing the output-format protocol (PredictStrategy JSON or CodeAct tool-calling). See the limitations section of `EXPERIMENT_truncation_markers.md` for known cases (`llama-3.1-8b`, `gpt-oss-20b`, `qwen3.5-35b`).
- **`Spec.__call__() got an unexpected keyword argument 'max_string'`** — the agent module is on a branch where `spec()` doesn't yet accept pformat hints. The current `truncation_comprehension.py` doesn't use those kwargs; if you're modifying it, leave the spec annotations off the parameters.

---

## Adding new models

1. Register the alias in `packages/nemo-oo-agents-nvidia/src/nooa_nvidia/data/llm_config_default.yaml` (the bundled default that ships in the NVIDIA package):
   ```yaml
   my-new-model:
     model_name: openai/path/to/model
     api_base: https://inference-api.nvidia.com/v1
     api_key_env: NVIDIA_INTERNAL_API_KEY
     context_window: 131072
     max_tokens: 16384
   ```

2. Add to `agent_models` in `tests/capability/config_truncation.yaml`.

3. Run any single round to verify, e.g.:
   ```bash
   uv run python -m eval_pipeline --config tests/capability/config_truncation.yaml \
     --models my-new-model \
     --test truncation_aware_lower \
     --runs 3 --parallel 5 --timeout 240 \
     --output-dir results/sanity
   ```

4. If the run shows lots of `actual: None` with errors, the model probably can't follow the output protocol — note it in the writeup limitations and exclude.

---

## Adding new fixtures

1. Create a JSONL file in `tests/capability/data/`:
   ```jsonl
   {"args": [], "kwargs": {"context": "...", "question": "..."}, "expected": 100}
   ```

2. Register a test entry in `tests/capability/config_truncation.yaml`:
   ```yaml
   - name: truncation_my_test
     description: "..."
     tier: stable
     agent:
       module: tests.capability.agents.truncation_comprehension
       class: TruncationComprehensionAgent
     method: answer
     data_file: tests/capability/data/truncation_my_test.jsonl
     scorers:
       - name: exact_match
         class: ExactMatchScorer
         weight: 1.0
   ```

3. Run with `--test truncation_my_test` to validate.
