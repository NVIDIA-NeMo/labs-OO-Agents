# eval_pipeline

A flexible evaluation framework for nooa agents. Use it from **Python** or via **YAML config**.

## Installation

```bash
pip install -e ./util/eval_pipeline
```

---

## Two Ways to Use eval_pipeline

### Option 1: Python API (Recommended for flexibility)

```python
from eval_pipeline import Evaluator, ExactMatchScorer
from nooa.unifiedllm import CompletionClient

# Create your LLM clients
gpt4 = CompletionClient(model="openai/gpt-4", api_key="...")
claude = CompletionClient(model="anthropic/claude-3", api_key="...")

# Create evaluator
evaluator = Evaluator(
    models={"gpt-4": gpt4, "claude": claude},
    output_dir="experiments/results",
    name="my_eval",
)

# Add tests programmatically
evaluator.add_test(
    name="sentiment",
    agent_class=SentimentAgent,
    method="classify_single",
    data=[
        {"kwargs": {"text": "I love this!"}, "expected": "positive"},
        {"kwargs": {"text": "I hate this!"}, "expected": "negative"},
    ],
    scorers=[ExactMatchScorer()],
)

# Run evaluation
results = await evaluator.run(
    models=["gpt-4"],  # Which models to use
    runs=3,            # Self-consistency runs
)

print(results.summary())  # "15/18 passed (83.3%)"
print(results.output_file)  # Path to .noo-eval.jsonl
```

### Option 2: YAML Config (Recommended for reproducibility)

**config.yaml:**
```yaml
name: capability
description: "Agent capability tests"

# Define all models (referenced by ID elsewhere)
models:
  gpt-4:
    model_name: openai/gpt-4
    endpoint: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    max_tokens: 8192
  claude:
    model_name: anthropic/claude-3-haiku
    endpoint: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    max_tokens: 4096

# Which models to run agents with
agent_models:
  - gpt-4
  - claude

output_dir: experiments/results

test_suite:
  - name: sentiment
    agent:
      module: my_agents
      class: SentimentAgent
    method: classify_single
    data_file: data/sentiment.jsonl
    scorers:
      - name: exact_match
        class: ExactMatchScorer
        weight: 1.0
```

**Run from Python:**
```python
from eval_pipeline import Evaluator

evaluator = Evaluator.from_config("config.yaml")
results = await evaluator.run(runs=3, quiet=True)
```

**Run from CLI:**
```bash
python -m eval_pipeline --config config.yaml --runs 3 -q
```

---

## Quick Reference

### Python API

```python
from eval_pipeline import (
    Evaluator,           # High-level API
    ExactMatchScorer,    # Built-in scorer
    LLMJudgeScorer,      # LLM-as-judge scorer
)

# From scratch
evaluator = Evaluator(models={"gpt-4": client}, output_dir="results")
evaluator.add_test(name="test1", agent_class=MyAgent, method="run", data=[...], scorers=[...])
results = await evaluator.run()

# From config
evaluator = Evaluator.from_config("config.yaml")
results = await evaluator.run(tests=["test1"], models=["gpt-4"], runs=3, limit=10, quiet=True)

# Results
results.passed       # Number passed
results.total        # Total samples
results.pass_rate    # Percentage (0-100)
results.summary()    # "15/18 passed (83.3%)"
results.output_file  # Path to .noo-eval.jsonl
results.results      # List of all result dicts
```

### CLI

```bash
python -m eval_pipeline --config config.yaml                 # Run all
python -m eval_pipeline --config config.yaml --test test1    # One test
python -m eval_pipeline --config config.yaml --runs 3        # Self-consistency
python -m eval_pipeline --config config.yaml --limit 5       # Limit samples
python -m eval_pipeline --config config.yaml -q              # Quiet mode
```

**Quiet mode output:**
```text
sentiment gpt-4: 4/4 ✓
sentiment claude: 3/4 ✗
TOTAL: 7/8 passed (87.5%) → experiments/my_eval_20251213_103000.noo-eval.jsonl
```

---

## Data Format

Test data is a list of dicts (Python) or JSONL file (config):

```python
# Python
data = [
    {"kwargs": {"text": "Great product!"}, "expected": "positive"},
    {"kwargs": {"a": 17, "b": 23}, "expected": 391},
]
```

```jsonl
# data.jsonl
{"kwargs": {"text": "Great product!"}, "expected": "positive"}
{"kwargs": {"a": 17, "b": 23}, "expected": 391}
```

The `kwargs` match the agent method signature:
```python
class SentimentAgent:
    async def classify_single(self, text: str) -> str:  # text comes from kwargs
        ...
```

---

## Scorers

### Built-in Scorers

**ExactMatchScorer** - Case-insensitive string comparison:
```python
from eval_pipeline import ExactMatchScorer
scorers = [ExactMatchScorer()]
```

**LLMJudgeScorer** - LLM evaluates output against a rubric:
```python
from eval_pipeline import LLMJudgeScorer
scorers = [LLMJudgeScorer(rubric="Is this a good answer?", model_spec=my_model_spec)]
```

### Custom Scorers

```python
from eval_pipeline import ScoringContext, ScoreResult

class MyScorer:
    def score(self, ctx: ScoringContext) -> ScoreResult:
        # ctx.input, ctx.expected, ctx.actual, ctx.code, ctx.error
        is_correct = ctx.actual == ctx.expected
        return ScoreResult(
            score=1.0 if is_correct else 0.0,
            reasoning="Match" if is_correct else "No match",
        )

# Use it
evaluator.add_test(..., scorers=[MyScorer()])
```

### Weighted Scoring

Multiple scorers combine via weighted average:
```python
from eval_pipeline import ScorerConfig

scorers = [
    ScorerConfig(name="exact", weight=0.5, scorer=ExactMatchScorer()),
    ScorerConfig(name="quality", weight=0.5, scorer=MyQualityScorer()),
]
```

Pass threshold: weighted score >= 0.5

---

## Config File Reference

```yaml
name: my_eval                    # Experiment name
description: "What this tests"   # Optional description

# All models defined here, referenced by ID
models:
  model-id:                      # Your chosen ID
    model_name: provider/model   # LiteLLM model string
    endpoint: https://...        # API endpoint
    api_key_env: ENV_VAR_NAME    # Environment variable with API key
    max_tokens: 4096             # Max output tokens

# Which models to use for agent evaluation
agent_models:
  - model-id

output_dir: experiments/results  # Output directory

test_suite:
  - name: test_name              # Test identifier
    description: "..."           # Optional
    agent:
      module: path.to.module     # Python module path
      class: AgentClassName      # Agent class name
    method: method_name          # Method to call
    data_file: path/to/data.jsonl
    limit: 10                    # Optional: limit samples
    scorers:
      - name: scorer_name
        class: ExactMatchScorer  # or LLMJudgeScorer
        weight: 1.0
        # For LLMJudgeScorer:
        model: model-id          # Reference to models section
        rubric: |
          Your evaluation criteria...
```

---

## Output Format

Results are written to `.noo-eval.jsonl`:

```text
experiments/my_eval_20251213_103000/
├── my_eval_20251213_103000.noo-eval.jsonl   # Results
└── traces/
    └── *.jsonl                              # Per-sample traces
```

Each result includes:
- `test_id`: Unique ID with model and run suffix
- `passed`: Boolean pass/fail
- `output`: Agent output
- `expected`: Expected output
- `scores`: Per-scorer results
- `trace_file`: Path to detailed trace

---

## Trace export: OTLP vs disk

The pipeline can send traces to an **OTLP API** (e.g. the nooa viewer) or write them to **disk** as `.jsonl` files. The choice is controlled by the **`OTLP_ENDPOINT`** environment variable.

| `OTLP_ENDPOINT`   | Behavior |
|-------------------|----------|
| **Set** (e.g. `http://localhost:5001/v1/traces`) | Traces are exported via OTLP to the given endpoint. No local trace files are created under `output_dir` (the pipeline may fetch trace data temporarily for scoring). Each run uses a unique experiment name so runs appear separately in the viewer. |
| **Unset**         | Traces are written to disk under `output_dir/<experiment>_<timestamp>/traces/*.jsonl`. Scoring reads these files directly. |

Pass `--trace-files` to write local `.jsonl` traces **even when a viewer is
active**, e.g. to cross-validate viewer traces against disk. `--no-files`
suppresses all file output regardless of `--trace-files`.

**Example: send traces to the viewer API**

```bash
export OTLP_ENDPOINT=http://localhost:5001/v1/traces
python -m eval_pipeline --config config.yaml --runs 3
```

Start the viewer (e.g. `nooa start-dev` or the combined viewer on port 5001) before running so the endpoint is available. Experiments will appear in the eval viewer under distinct names (e.g. `capability_20260309_080455_207319_r3_p40`).

**Example: traces to disk only (default)**

```bash
# Do not set OTLP_ENDPOINT
python -m eval_pipeline --config config.yaml --runs 3
```

Output directory will contain both the `.noo-eval.jsonl` results and a `traces/` folder with one `.jsonl` trace per sample.

---

## Self-Consistency Scoring

Run each test multiple times to measure reliability:

```python
results = await evaluator.run(runs=3)
```

Each run gets a unique `run_id` (1, 2, 3). Results can be analyzed for:
- Output consistency across runs
- Variance in responses
- Reliability metrics

---

## Architecture

```text
Evaluator
    │
    ├── add_test() ──→ TestDefinition
    │
    └── run() ──→ for each (test, model, run):
                      │
                      └── Sample ──→ Execute ──→ Score ──→ Write
                                        │          │         │
                                        ▼          ▼         ▼
                                    Run agent   Apply     Append to
                                    method     scorers   .noo-eval.jsonl
```

Each sample runs independently with its own trace file, enabling:
- Parallel execution
- Incremental output
- Re-scoring without re-running

---

## Tips

1. **Start with Python API** for rapid iteration, switch to YAML for production
2. **Use `limit=5`** during development to speed up iteration
3. **Check traces** in `.jsonl` trace files to debug agent behavior
4. **Use quiet mode** (`-q`) for scripting and CI
5. **Multiple runs** (`runs=3`) help identify flaky tests
