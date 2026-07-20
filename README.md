# NeMo OO Agents (NOOA) Framework

**What if your Python methods could think?** With the NeMo Object-Oriented Agents (NOOA) framework, they can. Write AI agents using familiar Python OOP—no new paradigms to learn. Define method signatures with type hints and docstrings, and your methods automatically become intelligent agents that can reason, execute code, and even create new methods to decompose complex tasks.

## Why the NOOA Framework?

- **Familiar Python OOP**: Classes for scope, methods for capabilities, inheritance for composition
- **Zero Boilerplate**: Define `async def analyze_data(self, text: str) -> str: ...` and it just works
- **Type-Safe**: All returns are validated, including full Pydantic integration for structured outputs
- **Pluggable Strategies**: Optimize per-method with `@strategy` decorator - fast PredictStrategy or iterative CodeAct
- **Self-Extending**: Your methods can think about complex problems and create new methods dynamically to decompose tasks
- **Async Native**: First-class async/await support throughout
- **Observable Context**: Context blocks show transparently where information comes from—the LLM can even manage its own context
- **Rich Introspection**: `doc()` gives LLMs a rich interface to explore objects, enabling progressive disclosure
- **Deterministic + LLM in One Class**: Seamlessly interleave ordinary Python methods and LLM-powered methods — the boundary is a single `...`
- **Context-Efficient**: Automatic handling of large parameters with smart truncation and `pprint()`

## Installation

### Just want the OO CLI/TUI?

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA-NeMo/labs-OO-Agents/main/scripts/install.sh | sh
exec $SHELL          # pick up the ~/.local/bin PATH change
nooa tui          # launch the interactive agent REPL
```

The script installs `uv`, a managed Python, and the three lockstep packages (CLI + core + the `nemo-oo-agents-nvidia` aliases), then prompts for your `NVIDIA_INFERENCE_API_KEY` and saves it to `~/.config/nooa/secrets.yaml`. Upgrade later with `uv tool upgrade nemo-labs-oo-agents-cli`.

### Use as a library

Add to a Python project (typical workflow when *building* agents, not just running the TUI):

```bash
# Create a new project (if needed)
uv init my-agent-project
cd my-agent-project

# Core framework
uv add "nemo-labs-oo-agents @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main"

# Optional: CLI + TUI (`nooa` command, web terminal, agent REPL)
uv add "nemo-labs-oo-agents-cli @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main#subdirectory=packages/nooa-cli"

# Optional: long-term memory subsystem (MemoryManager, see Long-Term Memory below)
uv add "nemo-labs-oo-agents-memory @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main#subdirectory=packages/nooa-memory"

# Optional: benchmark agent + Harbor runner (BenchAgent, `nemo-harbor`)
uv add "nemo-labs-oo-agents-bench @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main#subdirectory=packages/nooa-bench"
```

NOOA ships as four lockstep packages from this repo:

- **`nemo-labs-oo-agents`** — the core framework. Includes the agent runtime, context blocks, the trace viewer, and the unified LLM client. Optional extras: `[tracing]` (OpenTelemetry exporters), `[viewer]` (FastAPI trace viewer), `[mcp]`, `[nemo-relay]` (NeMo Flow guardrails/intercepts/ATIF export).
- **`nemo-labs-oo-agents-cli`** (`packages/nooa-cli`) — the `nooa` command and agent TUI. Optional `[datascience]` extra pre-loads numpy/pandas/plotly/scipy/sklearn into the LLM REPL execution namespace; `[web]` adds the `nooa term` web frontend.
- **`nemo-labs-oo-agents-memory`** (`packages/nooa-memory`) — opt-in long-term memory subsystem: `MemoryManager.install(agent, config=MemoryConfig(enabled=True))` attaches deliberate recall tools, spontaneous per-turn recall, and offline reflection, all backed by a single human-inspectable SQLite file.
- **`nemo-labs-oo-agents-bench`** (`packages/nooa-bench`) — the tech report's benchmark-agnostic `BenchAgent` (SWE-bench Verified, Terminal-Bench 2.0), the `nemo-harbor` container runner, and the trace analyzer behind the report's per-task token statistics.

NVIDIA-gateway model aliases (`nemo-oo-agents-nvidia`, registered via the `nooa.bundled_configs` entry-point group) are installed by the installer script; without them you get an OSS-only model registry.

### API Keys

Keys are read from `~/.config/nooa/secrets.yaml` unless already exported in your shell (the export wins). The installer writes this file; you can also edit it by hand:

```yaml
# ~/.config/nooa/secrets.yaml   (chmod 600; gitignore the project-local one)
env:
  NVIDIA_API_KEY: your-build.nvidia.com-key-here
  # OPENAI_API_KEY: sk-...
  # ANTHROPIC_API_KEY: sk-ant-...
```

For **library use** (not the CLI), drop a `.env` in your project directory instead — it's loaded by scripts and the viewer:

```bash
echo 'NVIDIA_API_KEY=your-build.nvidia.com-key-here' > .env
```

Run `nooa config show` to see which `secrets.yaml` / `settings.yaml` / `llm_config.yaml` layers are loading (secret values are redacted — only key names are shown).

- **Model providers**: The quickstart examples pick whichever credential you have set — no configuration needed:
  - **NVIDIA build.nvidia.com** (NIM, public): get a key at [build.nvidia.com](https://build.nvidia.com) → set `NVIDIA_API_KEY`. The examples default to `nvidia/nemotron-3-super-120b-a12b`; any NIM model works via the litellm `nvidia_nim/...` prefix (served at `integrate.api.nvidia.com`).
  - **OpenAI**: set `OPENAI_API_KEY`.
  - **NVIDIA internal inference gateway** (NVIDIA employees): set `NVIDIA_INFERENCE_API_KEY` to route through `inference-api.nvidia.com`.
  - **Local models** (Ollama, vLLM, …): run a model on your own machine — no API key needed. See [local deployment](#step-1-your-first-generation-method) in Quick Start Step 1.
  - **Any other provider**: pass any litellm-supported model name to `get_llm_client()` with that provider's key (see Quick Start Step 1).

### Development Setup (Advanced Use)

<details>
<summary><strong>Click here to see instructions for the development setup</strong></summary>

If you want to contribute to the NOOA framework or change code of the library, use the following steps.

```bash
git clone https://github.com/NVIDIA-NeMo/labs-OO-Agents.git
cd nemo_oo_agents/
./setup.sh  # Sets up venv, installs dependencies, installs pre-commit hooks
# Then create .env with your API key (see API Keys above)
```

</details>



## Quick Start

The NOOA framework's key strength is that you can start with zero boilerplate and progressively add structure only when you need it.

> **Note**: The examples below use `from nooa.util.quickstart import *` which provides common imports (`Agent`, `llm`, `BaseModel`, `strategy`, `autorun`, etc.) for brevity. Each example is copy-paste runnable.

### Step 1: Your First Generation Method

Methods with `...` bodies are called **generation methods** (the *agentic methods* of the NOOA tech report) - they're implemented by an agentic strategy using LLMs at runtime. The method signature defines the contract (inputs/outputs), and the **docstring provides instructions** to guide the LLM:

```python
# The quickstart import provides `llm`, configured from whichever public key
# you've set — NVIDIA_API_KEY (build.nvidia.com) or OPENAI_API_KEY (see API
# Keys above). To use another provider: llm = get_llm_client("gpt-4o-mini")
from nooa.util.quickstart import *


class FeedbackAgent(Agent, llm=llm):
    """You are an agent specializing in analyzing customer feedback."""

    async def analyze_feedback(self, text: str) -> str:
        """Analyze customer feedback for sentiment and key topics in one sentence."""
        ...  # Generation method - LLM implements based on docstring


@autorun
async def main():
    agent = FeedbackAgent()
    result = await agent.analyze_feedback("Great product, but shipping was slow")
    print(result)
```
`get_llm_client()` (from `nooa.unifiedllm`) is a thin wrapper on top of [litellm](https://docs.litellm.ai/), so any litellm-supported model name works directly — the NVIDIA gateway is the default, not a requirement:

```python
llm = get_llm_client("gpt-4o-mini")                # OpenAI (needs OPENAI_API_KEY)
llm = get_llm_client("claude-sonnet-4-5-20250514") # Anthropic (needs ANTHROPIC_API_KEY)
llm = get_llm_client("gemini/gemini-2.5-flash")    # Google (needs GEMINI_API_KEY)
```

For **local deployment** against an OpenAI-compatible server, pass the model and
endpoint directly — no key, no YAML alias. A simple convention is one environment
variable for the model and one for the URL:

```python
import os

llm = get_llm_client(
    os.environ["NOOA_MODEL"],
    api_base=os.environ["NOOA_API_BASE"],
)
```

- **[Ollama](https://ollama.com)** — pull a model, then point at the local server
  (the `ollama_chat/` prefix uses litellm's chat API):

  ```bash
  ollama run qwen3:1.7b
  export NOOA_MODEL=ollama_chat/qwen3:1.7b
  export NOOA_API_BASE=http://localhost:11434
  ```

- **[vLLM](https://docs.vllm.ai)** — serve a model with `vllm serve`, then use the
  `hosted_vllm/` prefix:

  ```bash
  export NOOA_MODEL=hosted_vllm/Qwen/Qwen3-1.7B
  export NOOA_API_BASE=http://127.0.0.1:8000/v1
  ```

If you installed `nemo-oo-agents-nvidia` (the installer script includes it), an extra set of NVIDIA-gateway aliases (claude-*, nemotron-*, qwen-*, gemini-*, gpt-*, llama-*) is registered automatically via the `nooa.bundled_configs` entry-point group:

```python
llm = get_llm_client("claude-haiku")          # NVIDIA-gateway Claude Haiku
llm = get_llm_client("nemotron3-nano-30b")    # NVIDIA Nemotron Nano
```

These bundled aliases route through NVIDIA's internal inference gateway and require `NVIDIA_INFERENCE_API_KEY` (NVIDIA employees). External users don't install `nemo-oo-agents-nvidia` and instead use public providers directly — `NVIDIA_API_KEY` for [build.nvidia.com](https://build.nvidia.com) NIM (`nvidia_nim/...`), `OPENAI_API_KEY`, etc. To customize the registry, run `nooa config eject` (writes to `~/.config/nooa/llm_config.yaml`), drop an `llm_config.yaml` in your project's `.nooa/` dir, or point `NEMO_OO_LLM_CONFIG` at one or more YAML files. Run `nooa config show` to inspect which files are loading.

See [`src/nooa/unifiedllm/registry.py`](src/nooa/unifiedllm/registry.py) for the YAML schema, or `CompletionClient()` directly for full control.

> **Key insight**: In NOOA, your method name, parameters, and docstring ARE the prompt. Try renaming `analyze_feedback` to `analyze_feedback_briefly` or `give_detailed_feedback_analysis`—the output changes accordingly, without modifying any other code. This is the fundamental paradigm shift: code structure drives LLM behavior.

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/01_first_generation_method.py
```
</details>

### Step 2: Use Structured Output to Enforce Method Contracts with Auto-Retry

Use any Pydantic model as a return type. NOOA automatically validates outputs and retries on errors. The LLM receives validation messages and corrects its response, ensuring you always get type-safe, valid data. This makes integrating LLM outputs with deterministic code robust and reliable:

```python
from typing import Literal

from nooa.util.quickstart import *


class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    topics: list[str]
    urgency: Literal["low", "medium", "high"]
    summary: str
    confidence: float = Field(ge=0, le=1)  # Pydantic constraints enforced!


class FeedbackAgent(Agent, llm=llm):
    """Agent for analyzing customer feedback with structured output."""

    async def analyze_feedback(self, text: str) -> FeedbackAnalysis:
        """Analyze customer feedback comprehensively."""
        ...


@autorun
async def main():
    agent = FeedbackAgent()
    result = await agent.analyze_feedback("Broken feature, needs immediate fix!")
    print(result)  # Guaranteed valid FeedbackAnalysis instance
```

Any Pydantic features work: `Field` constraints, validators, nested models, optional fields, and more. In addition, NOOA also supports `dataclasses`, `TypedDict`s, and validates basic types like `str`, `int`, `bool`, plus container types like `dict` and `list`.

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/02_structured_outputs.py
```
</details>

### Step 3: Your Methods Are Your Tools

In NOOA, you don't need a separate "tool" abstraction—**your regular Python methods ARE the tools**. The LLM can call any method on `self`, enabling seamless interleaving of deterministic code and LLM reasoning. No decorators, no registration, no schema definitions:

```python
from typing import TypedDict

from nooa.util.quickstart import *


class Result(TypedDict):
    can_fulfill: bool
    total_cost: float
    unavailable_items: list[str]


class InventoryAgent(Agent, llm=llm):
    """You are an agent that checks inventory using deterministic helper methods."""

    def __init__(self):
        super().__init__()
        self.inventory = {
            "apple": {"stock": 50, "price": 0.75},
            "banana": {"stock": 30, "price": 0.50},
            "orange": {"stock": 0, "price": 0.80},  # Out of stock
        }

    # Deterministic Python - automatically available as "tools" for the LLM
    def get_stock(self, item: str) -> int:
        """Get current stock for an item."""
        return self.inventory.get(item, {}).get("stock", 0)

    def get_price(self, item: str) -> float:
        """Get price for an item."""
        return self.inventory.get(item, {}).get("price", 0.0)

    # Generation method - LLM implements this, calling the helpers above as needed
    async def can_fulfill_order(self, items: list[str], budget: float) -> Result:
        """Check if order can be fulfilled within budget."""
        ...


@autorun
async def main():
    agent = InventoryAgent()
    result = await agent.can_fulfill_order(["apple", "banana", "orange"], budget=5.0)
    print(f"Can fulfill: {result['can_fulfill']}")
    print(f"Total cost: {result['total_cost']}")
    print(f"Unavailable items: {result['unavailable_items']}")
```

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/03_codeact_tools.py
```
</details>

### Step 4: Choose How Your Methods Think

Use `@strategy` to control reasoning style (agentic strategy) per-method. You can also add external tools (APIs, databases, MCP servers) as class attributes—they become callable just like your methods:

```python
from typing import Annotated

from nooa.config import CodeActConfig
from nooa.tools import ShellTools
from nooa.util.quickstart import *


class AnalysisAgent(Agent, llm=llm):
    """Agent demonstrating different strategy options."""

    shell = ShellTools()  # External tool - LLM can call this too

    @strategy(PredictStrategy())
    async def classify_sentiment(self, text: str) -> str:
        """Classify as positive, negative, or neutral."""
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=10)))
    async def perform_task(
        self,
        request: str,
    ) -> Annotated[str, "Your answer"]:
        """Perform the task requested by the user and provide a friendly response."""
        ...


@autorun
async def main():
    agent = AnalysisAgent()

    # PredictStrategy (fast, single-shot)
    sentiment = await agent.classify_sentiment("I love this product! Best purchase ever!")
    print("Sentence: I love this product! Best purchase ever!")
    print(f"Sentiment: {sentiment}\n")

    # CodeActStrategy (can execute Python code, call methods, iterate)
    requests = [
        "What is the current working directory?",
        "List the files in the current directory.",
    ]
    for request in requests:
        response = await agent.perform_task(request=request)
        print(f"Request: {request}")
        print(f"Response: {response}\n")
```


The NOOA framework supports two strategies (with more planned in the future). Advanced users can implement their own strategies.

| Strategy | Best For | Description |
|----------|----------|-------------|
| `CodeActStrategy` | Complex tasks (default) | Executes Python code iteratively, can call methods on `self` |
| `PredictStrategy` | Classification, extraction | Fast single-shot structured output, no code execution |

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/04_strategies.py
```
</details>

### Step 5: Explore with Progressive Disclosure

LLMs can use `doc()` to explore unknown objects. This is powerful when working with factories or APIs that return `Any`:

```python
from typing import Any

from nooa.util.quickstart import *

_WAREHOUSE = {
    "ART-001": Artwork("Starry Night Print", "Van Gogh Studio", appraised_value=15000.0),
    "STK-001": StockHolding("NVDA", shares=100, price_per_share=875.50),
    "JWL-001": Jewelry("Diamond Ring", carats=2.5, rate_per_carat=8000.0),
    "COL-001": Collectible("Vintage Baseball Card", base_value=5000.0, condition="excellent"),
    "ART-002": Artwork("Modern Abstract", "Local Artist", appraised_value=2500.0),
    "STK-002": StockHolding("AAPL", shares=50, price_per_share=185.25),
}


def get_item(item_id: str) -> Any:
    """Retrieve an item from the warehouse by ID."""
    return _WAREHOUSE.get(item_id)


class WarehouseAppraiser(Agent, llm=llm):
    """Agent that appraises items without knowing their types ahead of time."""

    get_item = staticmethod(get_item)

    async def appraise_item(self, item_id: str) -> float:
        """Get the monetary value of an item."""
        ...


@autorun
async def main():
    appraiser = WarehouseAppraiser()
    test_items = ["ART-001", "STK-001", "JWL-001", "COL-001"]

    for item_id in test_items:
        value = await appraiser.appraise_item(item_id)
        item = get_item(item_id)
        item_type = type(item).__name__
        print(f"  {item_id} ({item_type}): ${value:,.2f}")
```

The LLM must use `doc(item)` to discover how to extract value from each type:
- `Artwork`: `item.get_appraisal()["value"]`
- `StockHolding`: `item.get_total_value()`
- `Jewelry`: `item.compute_value()`
- `Collectible`: `item.estimate_value()`

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/05_progressive_disclosure.py
```
</details>

### Step 6: Tracing

Tracing is automatic. Start the development server and all agent method calls are traced — orchestrators, LLM methods, and private helpers — with parent-child relationships preserved:

```bash
nooa start-dev   # start trace viewer on http://localhost:5001
```

```python
from nooa import hidden
from nooa.util.quickstart import *


class MathAgent(Agent, llm=llm):
    """Agent that performs calculations with full tracing."""

    async def run(self, expression: str) -> str:
        """Orchestrator: evaluate the expression, then explain it."""
        value = await self.calculate(expression)
        formatted = await self._format(value)
        explanation = await self.explain(expression, formatted)
        return explanation

    async def calculate(self, expression: str) -> float:
        """Evaluate the mathematical expression and return the numeric result."""
        ...

    async def explain(self, expression: str, result: str) -> str:
        """Explain in one sentence why the expression evaluates to the result."""
        ...

    @hidden
    async def _format(self, value: float) -> str:
        """Private helper — formats the result for display."""
        return f"{value:g}"


@autorun
async def main():
    agent = MathAgent()
    result = await agent.run("(10 + 5) * 2")
    print(f"Result: {result}")
```

Traces capture every LLM call, code execution, and tool invocation, with spans nested by call hierarchy. If the viewer is not running, tracing is silently disabled. Set `OTLP_ENDPOINT` to send traces to a viewer on a different host or port. The viewer ships in the core package ([`src/nooa/viewer/`](src/nooa/viewer/), enabled via the `[viewer]` extra) — `nooa start-dev` runs it locally.

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/06_tracing.py
```
</details>

### Step 7: Dynamic Prompts with Templating

Use `{self.attribute}` in docstrings to inject specific runtime values in your prompts:

```python
from nooa.util.quickstart import *


class TranslatorAgent(Agent, llm=llm):
    """Agent that translates text with configurable behavior."""

    def __init__(self, target_language: str = "Spanish", **kwargs):
        super().__init__(**kwargs)
        self.target_language = target_language
        self.translation_count = 0

    async def translate(self, text: str) -> str:
        """Translate the text to {self.target_language}.

        Keep the translation natural and idiomatic.
        """
        ...

    async def translate_formal(self, text: str) -> str:
        """Translate the text to {self.target_language} using formal register.

        Use polite/formal forms (e.g., 'usted' in Spanish, 'Sie' in German).
        """
        ...


@autorun
async def main():
    # The {self.target_language} in docstrings is expanded at runtime
    spanish = TranslatorAgent(target_language="Spanish")
    german = TranslatorAgent(target_language="German")

    text = "Hello, how are you today?"

    print(f"Original: {text}\n")

    result_es = await spanish.translate(text)
    print(f"Spanish: {result_es}")

    result_de = await german.translate(text)
    print(f"German: {result_de}")

    result_formal = await spanish.translate_formal(text)
    print(f"Spanish (formal): {result_formal}")
```


Template variables work with instance state and computed expressions: `{self.attr}`, `{len(self._notes)}`. **Don't re-inject method parameters** (e.g. `{text}`): arguments are already rendered to the LLM automatically, with size-aware truncation — repeating them in the docstring duplicates them untruncated and mixes data into the instruction text. Reserve templating for what the signature cannot show.

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/07_dynamic_prompts.py
```
</details>

### Step 8: Context Blocks

Context blocks let you pin information directly into the LLM's system prompt—so it's always visible without re-passing it in every method call. Static blocks hold a fixed value; dynamic blocks re-evaluate an expression each turn so the LLM always sees live state:

```python
from nooa.util.quickstart import *
from nooa.agentdoc import spec


class NoteTakingAgent(Agent, llm=llm):
    """Agent that stores notes and answers questions about them."""

    def __init__(self):
        super().__init__()
        self._notes: list[str] = []
        spec(self, "context", hidden=False)  # Expose context management to LLM

    def add_note(self, text: str) -> None:
        """Add a note to the collection."""
        self._notes.append(text)

    def render_notes(self) -> str:
        """Render all stored notes as a formatted list."""
        return "\n".join(f"- {n}" for n in self._notes) or "No notes yet."

    async def record(self, note: str) -> str:
        """Store this note using add_note and confirm what was saved."""
        ...

    async def answer(self, question: str) -> str:
        """Answer the question using the notes visible in your context."""
        ...


@autorun
async def main():
    agent = NoteTakingAgent()

    # Dynamic block: expression re-evaluated every turn — LLM always sees latest notes
    agent.context.set_dynamic("notes", "self.render_notes()")

    notes = [
        "Deploy uses blue-green strategy with 5-minute health checks.",
        "Database migrations run before traffic shifts.",
        "Rollback is automatic if error rate exceeds 1% for 2 minutes.",
    ]
    for note in notes:
        await agent.record(note)

    # LLM sees all three notes in its context — no need to pass them explicitly
    answer = await agent.answer("What triggers an automatic rollback?")
    print(f"Answer: {answer}")

    # Static block: pin a value once — useful for specs, plans, decisions
    agent.context["policy"] = "Always prefer rollback over forward-fix during incidents."
    answer2 = await agent.answer("Should we try to fix forward or roll back?")
    print(f"Answer: {answer2}")
```

Both block types appear in the LLM's system prompt as labelled sections. The LLM can also manage blocks itself in CodeAct—adding, updating, or removing them as its understanding evolves.

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/08_context_blocks.py
```
</details>

### Step 9: Automatic History Summarization

Each exchange in a multi-turn conversation adds events to the agent's history. Left unchecked, this eventually fills the model's context window. `TokenBudgetSummarizer` compresses older turns automatically when a budget threshold is crossed—keeping history bounded so the conversation can run indefinitely.

```python
from nooa.agents import TokenBudgetSummarizer
from nooa.config import TokenBudgetConfig
from nooa.util.quickstart import *
from nooa.agentdoc import spec


class InterviewAgent(Agent, llm=llm):
    """A technical interviewer conducting a multi-turn conversation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "events", hidden=False)  # Expose event querying

    async def ask(self, candidate_answer: str) -> str:
        """Continue the technical interview based on the candidate's latest answer.

        Ask a relevant follow-up or move to a new topic as appropriate.
        Keep track of what has already been covered.
        """
        ...

    async def evaluate(self) -> str:
        """Based on the full interview so far, provide a brief candidate evaluation."""
        ...


ANSWERS = [
    "I have 3 years of Python experience, mostly building REST APIs.",
    "I use FastAPI for most projects — I like its async support and automatic OpenAPI docs.",
    "For databases, I usually go with PostgreSQL and SQLAlchemy as the ORM.",
    "I handle errors with try/except blocks and return proper HTTP status codes.",
    "For testing I write unit tests with pytest and mock out external dependencies.",
    "I've used Redis for caching and Celery for background task queues.",
    "I deploy on AWS — mostly ECS for long-running services and Lambda for event-driven jobs.",
]


@autorun
async def main():
    agent = InterviewAgent()
    TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=1000))

    for i, answer in enumerate(ANSWERS, 1):
        await agent.ask(answer)
        n = len(agent.events.query())
        print(f"  {i:<6}  {n:>14}")

    evaluation = await agent.evaluate()
    print(f"\nEvaluation: {evaluation}")
```

For agents that process discrete batches rather than open-ended conversations, `MethodSummarizer` compresses each completed method call's history instead:

```python
from nooa.agents import MethodSummarizer

MethodSummarizer.install(agent)  # Compress after each method call completes
```

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/09_summarization.py
```
</details>

## Documentation

Finished the Quick Start? The deeper material lives in the repo:

- **Guides** ([`docs/guides/`](docs/guides/)) — [prompt mechanics](docs/guides/prompt-mechanics.md), [writing generation methods](docs/guides/writing-generation-methods.md), [strategies](docs/guides/strategies.md), [structured output](docs/guides/structured-output.md), [context blocks](docs/guides/context-blocks.md), [truncation](docs/guides/truncation.md), [single vs. multi-agent](docs/guides/single-vs-multi-agent.md), and [config migration](docs/guides/config-migration.md).
- **Runnable examples** ([`examples/quickstart/`](examples/quickstart/)) — fifteen numbered scripts: `01`–`11` mirror the steps above and the Skills/MCP sections, then [`12_memory.py`](examples/quickstart/12_memory.py), [`13_multimodal.py`](examples/quickstart/13_multimodal.py), [`14_atif_trajectory.py`](examples/quickstart/14_atif_trajectory.py), and [`15_nemo_relay.py`](examples/quickstart/15_nemo_relay.py) go beyond it.
- **More examples** — [`examples/advanced/`](examples/advanced/) (memory internals, CodeAct event flow, prefill, OTLP/Langfuse/Phoenix tracing, swappable execution engines), [`examples/benchmarks/`](examples/benchmarks/) (minimal Harbor-compatible agent), and [`examples/arc_agi_3/`](examples/arc_agi_3/) (the ARC-AGI-3 world-model agent).
- **Reference** — [REFERENCE.md](REFERENCE.md): paths, commands, environment variables, and a map of all of the above.

## More Features

The features below follow the same Agent pattern — class attributes, method signatures, docstrings — so there's nothing new to learn.

### Long-Term Memory

Everything above is method- or session-scoped. The opt-in memory subsystem (`nemo-labs-oo-agents-memory`, see Installation) gives an agent state that persists across sessions — and the **agent curates its own store**: model-callable tools (`remember`, `recall`, `search`, `update_memory`, `forget`, …) let it write and consult memories deliberately, while a per-turn hook injects associated memories spontaneously. Offline reflection consolidates the store (merge → abstract → prune), and everything lives in a single human-inspectable SQLite file.

`MemoryManager.install(agent, config=MemoryConfig(enabled=True))` attaches it to an unmodified agent (`enabled` defaults to `False`, so pass the config — install is inert without it); uninstalling leaves no trace:

```python
from nooa import Agent
from nooa_memory import MemoryConfig, MemoryManager, MemoryToolsMixin


class MyAgent(MemoryToolsMixin, Agent, llm=llm):
    async def work(self, task: str) -> str:
        """Do the task. Use self.recall(...) to consult memory."""
        ...


agent = MyAgent()
MemoryManager.install(agent, config=MemoryConfig(enabled=True))
agent.remember("Deploy with `make ship`.", type="skill", importance="HIGH")
```

The store defaults to `.nooa/memory/memory.sqlite`. See [`packages/nooa-memory`](packages/nooa-memory/) for the full design.

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/12_memory.py
```
</details>

### Skills

Skills inject curated context (guidelines, examples, domain knowledge) directly into your agent. Skills are included in the core package — no extra install needed.

Attach a skill as an instance attribute—every instance automatically gets that context:

```python
from pathlib import Path

from nooa import TextSkill
from nooa.util.quickstart import *

ASSETS = Path("path/to/skills")  # directory containing skill folders


class FrontendAgent(Agent, llm=llm):
    """Agent with a single file-based skill."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frontend_design = TextSkill(path=ASSETS / "frontend-design")

    async def respond(self, prompt: str) -> str:
        """Respond to a user message."""
        ...


@autorun
async def main():
    agent = FrontendAgent()
    result = await agent.respond("What are the best practices for responsive layout?")
    print(result)
```

You can also discover and activate all skills in a directory with `SkillRegistry` (`from nooa.skill_registry import SkillRegistry`): construct `self.skills = SkillRegistry(self)` in `__init__`, then `self.skills.discover_skills_dirs([skills_dir])` and `self.skills.load(["cmd.*"])` / `self.skills.activate(["cmd.*"])`. Skills are instance attributes — every instance of the agent automatically has the skill context available.

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/10_skills.py
```
</details>

### MCP Tools

MCP (Model Context Protocol) tools let your agent call external services through a standard interface. The TUI (`nemo-labs-oo-agents-cli`) ships MCP support out of the box — it's a hard dependency, so `self.mcp` and the `/mcp` command are always available. In the **core** `nemo-labs-oo-agents` library MCP stays an optional extra (for glibc<2.28 hosts): install with `uv sync --extra mcp` (or `uv add 'nemo-labs-oo-agents[mcp]'`).

For the TUI, declare MCP servers in the same project config file used for
models, skills, and other TUI settings: `.nooa/config.toml`. Use
environment variables for secrets; string values are expanded when the server is
connected.

```toml
[tui]
mcp_auto_connect = ["maas-confluence-stg"]

[tui.mcp_servers.maas-confluence-stg]
url = "https://your-mcp-server.example.com/mcp"
transport = "streamable-http"

[tui.mcp_servers.maas-confluence-stg.headers]
Authorization = "Bearer ${MAAS_API_KEY}"
```

At startup the TUI attaches auto-connected servers to the agent using a valid
Python attribute name, so `maas-confluence-stg` becomes
`self.maas_confluence_stg`. You can also connect manually with
`/mcp connect maas-confluence-stg` and inspect configured servers with
`/mcp list`.

A VS Code / Claude-style `.mcp.json` file is still supported via `mcp_file` or
`--mcp-file`; inline `mcp_servers` in `config.toml` override file entries with
the same name.

Inside the TUI, MCP servers are managed through `self.mcp`, an `MCPRegistry`
skill that mirrors `self.skills`:

- `self.mcp.discovered()` — configured server names (from `config.toml` /
  `.mcp.json`).
- `await self.mcp.connect(["name"])` — open a server (and activate it),
  attaching it as `self.<name>`.
- `self.mcp.activate([...])` / `self.mcp.deactivate([...])` — control whether a
  connected server's tools are listed for the agent; a deactivated server keeps
  its authenticated client alive.
- An `<mcp>` context block lists every server each turn under **Active**,
  **Connected but inactive**, or **Configured**.

The same actions are available as a slash command — `/mcp list`,
`/mcp connect <server>`, `/mcp disconnect <server>` — whose output is shown to
you without spending an agent turn.

Outside the TUI (library use), there's no `self.mcp` — call the stateless
`MCPManager` factory directly. Reference the server by name in your agent:

```python
from nooa.mcp import MCPManager

from nooa.util.quickstart import *


class ConfluenceAgent(Agent, llm=llm):
    """Agent with MCP tool access."""

    confluence_tool = MCPManager.create_from_server("maas-confluence-stg")

    async def respond(self, prompt: str) -> str:
        """Respond to a user message using the Confluence MCP tool."""
        ...


@autorun
async def main():
    agent = ConfluenceAgent()
    result = await agent.respond("What are the best practices for claude code?")
    print(result)
```

Alternatively, pass the connection details inline to skip the `.mcp.json` file:

```python
confluence_tool = MCPManager.create_from_server(
    "maas-confluence-stg",
    url="https://your-mcp-server.example.com/mcp",
    transport="streamable-http",
    headers={},
)
```

The MCP tool appears alongside your regular methods — the LLM calls it the same way it calls any other `self` method.

<details>
<summary><strong>Show cloned repo command</strong></summary>

```bash
uv run python examples/quickstart/11_mcp.py
```
</details>

### Sandbox

For isolated, ephemeral execution environments, pair NOOA with [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell) — run your agent script inside an OpenShell sandbox and it gets credential brokering, network policy, and filesystem isolation without code changes. See the OpenShell documentation for setup.

## Advanced Features

Beyond the quickstart, the NOOA framework offers these advanced capabilities:

### Self-Extending Agents

Within CodeAct, the LLM can define new helper methods dynamically:

```python
class DataAgent(Agent, llm=llm):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=20)))
    async def process_dataset(self, data: list[dict]) -> dict:
        """Process dataset. Create helper methods as needed."""
        ...

# LLM can define new methods at runtime:
#     @strategy(PredictStrategy())
#     async def extract_features(self, item: dict) -> dict:
#         """Extract features from a single item."""
#         ...
#     features = [await self.extract_features(item) for item in data]
```

### LLM Cascading Resolution

Configure LLMs at any granularity—from class-wide defaults to per-method overrides—enabling cost optimization (cheap models for simple tasks, powerful models for complex ones), A/B testing, and gradual model rollouts, all without changing your agent's interfaces or business logic:

```python
class MyAgent(Agent, llm=default_llm):             # 1. Class-level default
    @strategy(CodeActStrategy(), llm=special_llm)  # 2. Method-level override
    async def complex_task(self) -> str:
        sub = MySubAgent()                         # 3. Subagent without llm= inherits the
        ...                                        #    calling parent's LLM (any explicit
                                                   #    llm= on the subagent overrides it)

agent = MyAgent(llm=different_llm)                 # 4. Instance-level override
```

Child agents inherit their parent's LLM by default; give a subagent its own `llm=` (class or instance level) to override. Inheritance is resolved at construction from the calling agent, so construct no-`llm` subagents inside a parent agent method — not in `__init__` or at module level.

### Event-Driven History

Subscribe to agent events or query past events:

```python
agent.event_manager.on("message", lambda e: print(f"Message: {e.content}"))
recent = agent.events.query()
```

---

*Start with a simple agent and progressively add complexity only when you need it.*
