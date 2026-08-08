# NVIDIA OO Agents — Examples

A progressive tour of the framework. Each step is a standalone, copy-paste-runnable file under [`quickstart/`](quickstart/).


**Contents**

| # | Topic | File |
|---|---|---|
| 1 | Your first generation method | [`01_first_generation_method.py`](quickstart/01_first_generation_method.py) |
| 2 | Structured outputs (Pydantic) | [`02_structured_outputs.py`](quickstart/02_structured_outputs.py) |
| 3 | Methods as tools (SW1 / SW3 interleaving) | [`03_codeact_tools.py`](quickstart/03_codeact_tools.py) |
| 4 | Choosing a strategy | [`04_strategies.py`](quickstart/04_strategies.py) |
| 5 | Progressive disclosure with `doc()` | [`05_progressive_disclosure.py`](quickstart/05_progressive_disclosure.py) |
| 6 | Tracing | [`06_tracing.py`](quickstart/06_tracing.py) |
| 7 | Dynamic prompts | [`07_dynamic_prompts.py`](quickstart/07_dynamic_prompts.py) |
| 8 | Context blocks | [`08_context_blocks.py`](quickstart/08_context_blocks.py) |
| 9 | Automatic history summarization | [`09_summarization.py`](quickstart/09_summarization.py) |
| 10 | Skills | [`10_skills.py`](quickstart/10_skills.py) |
| 11 | MCP tools | [`11_mcp.py`](quickstart/11_mcp.py) |
| — | Sandbox, memory, multimodal, NeMo Relay | see the [Advanced](#advanced-topics) section |

---

## Step 1: Your first generation method

Methods with `...` bodies are **generation methods** — implemented at runtime by an LLM-driven strategy. The signature defines the contract; the docstring guides the LLM.

```python
from nooa.util.quickstart import *


class FeedbackAgent(Agent, llm=llm):
    """You are an agent specializing in analyzing customer feedback."""

    async def analyze_feedback(self, text: str) -> str:
        """Analyze customer feedback for sentiment and key topics in one sentence."""
        ...


@autorun
async def main():
    agent = FeedbackAgent()
    result = await agent.analyze_feedback("Great product, but shipping was slow")
    print(result)
```

The quickstart import provides `llm`. To choose a model explicitly, import `get_llm_client()` from `nooa.unifiedllm`; it wraps [litellm](https://docs.litellm.ai/), so any litellm model name works:

```python
from nooa.unifiedllm import get_llm_client

llm = get_llm_client("gpt-4o-mini")                 # OpenAI (needs OPENAI_API_KEY)
llm = get_llm_client("claude-sonnet-4-5-20250514")  # Anthropic (needs ANTHROPIC_API_KEY)
llm = get_llm_client("gemini/gemini-2.5-flash")     # Google (needs GEMINI_API_KEY)
llm = get_llm_client("ollama_chat/qwen3:1.7b",      # Ollama (no key)
                     api_base="http://localhost:11434")
llm = get_llm_client("hosted_vllm/Qwen/Qwen3-1.7B", # vLLM (no key)
                     api_base="http://localhost:8000/v1")
```

Provider packages can register bundled model aliases automatically through the `nooa.bundled_configs` entry-point group. To customize the registry, run `nooa config eject`, drop an `llm_config.yaml` in your project's `.nooa/`, or point `NEMO_OO_LLM_CONFIG` at one or more YAML files. See [`src/nooa/unifiedllm/registry.py`](../src/nooa/unifiedllm/registry.py) for the YAML schema.

> **Key insight.** In NVIDIA OO Agents, your method name, parameters, and docstring ARE the prompt. Rename `analyze_feedback` to `analyze_feedback_briefly` or `give_detailed_feedback_analysis` — the output changes accordingly, without touching any other code.

```bash
uv run python examples/quickstart/01_first_generation_method.py
```

## Step 2: Structured outputs

Use any Pydantic model as the return type. NVIDIA OO Agents validates outputs and auto-retries on error; the LLM sees the validation message and corrects itself.

```python
from typing import Literal
from nooa.util.quickstart import *


class FeedbackAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    topics: list[str]
    urgency: Literal["low", "medium", "high"]
    summary: str
    confidence: float = Field(ge=0, le=1)          # Pydantic constraints enforced


class FeedbackAgent(Agent, llm=llm):
    """Agent for analyzing customer feedback with structured output."""

    async def analyze_feedback(self, text: str) -> FeedbackAnalysis:
        """Analyze customer feedback comprehensively."""
        ...


@autorun
async def main():
    agent = FeedbackAgent()
    result = await agent.analyze_feedback("Broken feature, needs immediate fix!")
    print(result)                                  # guaranteed valid FeedbackAnalysis
```

Any Pydantic feature works — `Field` constraints, validators, nested models, optionals. The framework also validates `dataclass`, `TypedDict`, primitives (`str`, `int`, `bool`), and containers (`dict`, `list`).

> **Key insight.** The return type is the contract. There is no boundary between "the model produced text" and "your code has a valid object" — validation and retry happen inside the harness, so callers only ever see well-typed data.

```bash
uv run python examples/quickstart/02_structured_outputs.py
```

## Step 3: Your methods are your tools (SW1 / SW3 interleaving)

There's no separate "tool" abstraction — regular Python methods on `self` ARE the tools. No decorators, no registration, no schemas.

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
            "apple":  {"stock": 50, "price": 0.75},
            "banana": {"stock": 30, "price": 0.50},
            "orange": {"stock":  0, "price": 0.80},
        }

    # SW1: deterministic Python — automatically visible to the LLM
    def get_stock(self, item: str) -> int:
        """Get current stock for an item."""
        return self.inventory.get(item, {}).get("stock", 0)

    def get_price(self, item: str) -> float:
        """Get price for an item."""
        return self.inventory.get(item, {}).get("price", 0.0)

    # SW3: generation method — LLM writes Python that calls the helpers above
    async def can_fulfill_order(self, items: list[str], budget: float) -> Result:
        """Check if order can be fulfilled within budget."""
        ...


@autorun
async def main():
    agent = InventoryAgent()
    result = await agent.can_fulfill_order(["apple", "banana", "orange"], budget=5.0)
    print(result)
```

> **Key insight.** There is no separate tool abstraction. Adding a tool = writing a Python method. Deleting a tool = deleting a method. Nothing to register, no schema to keep in sync with the implementation.

```bash
uv run python examples/quickstart/03_codeact_tools.py
```

## Step 4: Choose how your methods think

Use `@strategy` to control the reasoning style per method. External tools (APIs, DBs, MCP servers) as class attributes become callable just like your methods.

```python
from typing import Annotated
from nooa.config import CodeActConfig
from nooa.tools import ShellTools
from nooa.util.quickstart import *


class AnalysisAgent(Agent, llm=llm):
    """Agent demonstrating different strategy options."""

    shell = ShellTools()                          # external tool, LLM-callable

    @strategy(PredictStrategy())
    async def classify_sentiment(self, text: str) -> str:
        """Classify as positive, negative, or neutral."""
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=10)))
    async def perform_task(self, request: str) -> Annotated[str, "Your answer"]:
        """Perform the task requested by the user and provide a friendly response."""
        ...
```

| Strategy | Best for | Description |
|---|---|---|
| `CodeActStrategy` *(default)* | Complex tasks, tool use | Iterative Python REPL; can call methods on `self` |
| `PredictStrategy` | Classification, extraction | Fast single-shot structured output, no code execution |

Advanced users can implement their own strategies.

> **Key insight.** The strategy is an execution detail, not part of the method's contract. Swap `@strategy(PredictStrategy())` for `@strategy(CodeActStrategy())` and no caller changes — you're optimizing for cost or capability, not rewriting an interface.

```bash
uv run python examples/quickstart/04_strategies.py
```

## Step 5: Progressive disclosure

The LLM can use `doc(obj)` to explore unknown objects — powerful when working with factories or APIs that return `Any`.

```python
from typing import Any
from nooa.util.quickstart import *

_WAREHOUSE = {
    "ART-001": Artwork("Starry Night Print", "Van Gogh Studio", appraised_value=15000.0),
    "STK-001": StockHolding("NVDA", shares=100, price_per_share=875.50),
    "JWL-001": Jewelry("Diamond Ring", carats=2.5, rate_per_carat=8000.0),
    "COL-001": Collectible("Vintage Baseball Card", base_value=5000.0, condition="excellent"),
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
```

The LLM uses `doc(item)` at runtime to discover the right accessor for each type — `item.get_appraisal()["value"]` for `Artwork`, `item.get_total_value()` for `StockHolding`, and so on.

> **Key insight.** You don't have to describe every type in the system prompt. `doc()` lets the model discover the shape of an object at the moment it needs to — so an agent can operate on data it has never seen, and your prompt stays bounded even as the domain grows.

```bash
uv run python examples/quickstart/05_progressive_disclosure.py
```

## Step 6: Tracing

Tracing is automatic. All agent method calls — orchestrators, LLM methods, private helpers — are traced with parent-child relationships preserved.

```bash
nooa start-dev             # trace viewer on http://localhost:5001
```

```python
from nooa import hidden
from nooa.util.quickstart import *


class MathAgent(Agent, llm=llm):
    """Agent that performs calculations with full tracing."""

    async def run(self, expression: str) -> str:
        """Orchestrator: evaluate, then explain."""
        value = await self.calculate(expression)
        formatted = await self._format(value)
        return await self.explain(expression, formatted)

    async def calculate(self, expression: str) -> float:
        """Evaluate the mathematical expression and return the numeric result."""
        ...

    async def explain(self, expression: str, result: str) -> str:
        """Explain in one sentence why {expression} equals {result}."""
        ...

    @hidden
    async def _format(self, value: float) -> str:
        """Private helper — formats the result for display."""
        return f"{value:g}"
```

If the viewer is not running, tracing is silently disabled. Set `OTLP_ENDPOINT` to send traces elsewhere.

> **Key insight.** Traces follow the call graph, not the transcript. Because agents are Python objects, every method — LLM-driven or deterministic, public or private — becomes a nested span. You debug agents the same way you debug programs.

```bash
uv run python examples/quickstart/06_tracing.py
```

## Step 7: Dynamic prompts with templating

Use `{self.attribute}` (or any Python expression) in docstrings to inject runtime values.

```python
from nooa.util.quickstart import *


class TranslatorAgent(Agent, llm=llm):
    """Agent that translates text with configurable behavior."""

    def __init__(self, target_language: str = "Spanish", **kwargs):
        super().__init__(**kwargs)
        self.target_language = target_language

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
```

Template variables work with any Python expression: `{self.attr}`, `{len(items)}`, `{param.upper()}`.

> **Key insight.** A docstring is an f-string. Runtime values from `self` (or any expression in scope) interpolate at call time, so one method definition serves many configured behaviors — no templating engine, no prompt-management layer.

```bash
uv run python examples/quickstart/07_dynamic_prompts.py
```

## Step 8: Context blocks

Context blocks pin information into the LLM's system prompt so you don't have to pass it into every method call. **Static** blocks hold a fixed value; **dynamic** blocks re-evaluate a Python expression each turn.

```python
from nooa import Context
from nooa.agentdoc import spec
from nooa.util.quickstart import *


class NoteTakingAgent(Agent, llm=llm):
    """Agent that stores notes and answers questions about them."""

    def __init__(self):
        super().__init__()
        self._notes: list[str] = []
        spec(self, "context", hidden=False)        # expose context management to LLM

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
    agent.context["notes"] = Context(expr="self.render_notes()")   # re-evaluated every turn

    for note in [
        "Deploy uses blue-green strategy with 5-minute health checks.",
        "Database migrations run before traffic shifts.",
        "Rollback is automatic if error rate exceeds 1% for 2 minutes.",
    ]:
        await agent.record(note)

    print(await agent.answer("What triggers an automatic rollback?"))

    agent.context["policy"] = "Always prefer rollback over forward-fix during incidents."
    print(await agent.answer("Should we try to fix forward or roll back?"))
```

Both block types appear as labelled sections in the LLM's system prompt. In CodeAct the LLM can add, update, or remove blocks itself as its understanding evolves.

> **Key insight.** Context is a first-class API, not a hidden implementation detail. Both the developer and the model manipulate the same blocks through the same `self.context` interface — so agents can engineer their own context as their understanding evolves.

```bash
uv run python examples/quickstart/08_context_blocks.py
```

## Step 9: Automatic history summarization

Every method call adds events to the agent's history. `TokenBudgetSummarizer` compresses older turns when a budget threshold is crossed — history stays bounded so conversations can run indefinitely.

```python
from nooa.agentdoc import spec
from nooa.agents import TokenBudgetSummarizer
from nooa.config import TokenBudgetConfig
from nooa.util.quickstart import *


class InterviewAgent(Agent, llm=llm):
    """A technical interviewer conducting a multi-turn conversation."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        spec(self, "events", hidden=False)

    async def ask(self, candidate_answer: str) -> str:
        """Continue the technical interview based on the candidate's latest answer.

        Ask a relevant follow-up or move to a new topic. Track what has been covered.
        """
        ...

    async def evaluate(self) -> str:
        """Based on the full interview so far, provide a brief candidate evaluation."""
        ...


@autorun
async def main():
    agent = InterviewAgent()
    TokenBudgetSummarizer.install(agent, config=TokenBudgetConfig(max_tokens=1000))
    ...
```

For agents that process discrete batches, `MethodSummarizer` compresses each completed method call's history instead:

```python
from nooa.agents import MethodSummarizer
MethodSummarizer.install(agent)
```

> **Key insight.** Long-running agents don't need unbounded history — they need bounded, compressed history. Install a summarizer once and the conversation can run indefinitely without ever hitting the context window.

```bash
uv run python examples/quickstart/09_summarization.py
```

---

## More features

The features below follow the same Agent pattern — class attributes, method signatures, docstrings — nothing new to learn.

### Skills

Skills inject curated context (guidelines, examples, domain knowledge) into your agent. Attach a skill as an instance attribute and every instance gets that context.

```python
from pathlib import Path
from nooa import TextSkill
from nooa.util.quickstart import *

ASSETS = Path("path/to/skills")


class FrontendAgent(Agent, llm=llm):
    """Agent with a single file-based skill."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frontend_design = TextSkill(path=ASSETS / "frontend-design")

    async def respond(self, prompt: str) -> str:
        """Respond to a user message."""
        ...
```

Load a whole directory by attaching each `SKILL.md` subdirectory as a `TextSkill` attribute:

```python
for entry in sorted(ASSETS.iterdir()):
    if entry.is_dir() and (entry / "SKILL.md").exists():
        setattr(self, entry.name.replace("-", "_"), TextSkill(path=entry))
```

> **Key insight.** A skill is just a class attribute. Because visibility follows Python rules, anything you attach to `self` is discoverable to the model — no separate registration step, no split between "code" and "knowledge".

```bash
uv run python examples/quickstart/10_skills.py
```

### MCP tools

MCP (Model Context Protocol) tools let your agent call external services through a standard interface. Install with `uv add 'nooa[mcp]'` and call the stateless `MCPManager` factory:

```python
from nooa.mcp import MCPManager
from nooa.util.quickstart import *


class ConfluenceAgent(Agent, llm=llm):
    """Agent with MCP tool access."""

    confluence_tool = MCPManager.create_from_server("maas-confluence-stg")

    async def respond(self, prompt: str) -> str:
        """Respond to a user message using the Confluence MCP tool."""
        ...
```

Or inline the connection details:

```python
confluence_tool = MCPManager.create_from_server(
    "maas-confluence-stg",
    url="https://your-mcp-server.example.com/mcp",
    transport="streamable-http",
    headers={},
)
```

Environment placeholders in `.mcp.json` and inline/TUI `servers` configurations
are always treated literally so repository data cannot copy host secrets
into MCP URLs, headers, arguments, or child-process environments. Resolve secrets
only in trusted caller code and pass the resulting values explicitly:

```python
import os

trusted_tool = MCPManager.create_from_server(
    "trusted",
    url="https://trusted.example/mcp",
    transport="streamable-http",
    headers={"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"},
)
```

> **Key insight.** MCP servers surface as regular `self.<name>` attributes. The model calls them the same way it calls a local method — external services stop being second-class relative to your own code.

```bash
uv run python examples/quickstart/11_mcp.py
```

### Sandbox

Run agents in isolated, ephemeral compute environments. Install the `sandbox` extra:

```bash
uv add 'nooa[sandbox]'
openshell gateway start           # see https://github.com/NVIDIA/OpenShell
```

Use OpenShell directly for launch, port forwarding, long-running tasks, or connecting to existing sandboxes.

> **Key insight.** The sandbox is a wrapper, not a code change. The same agent script runs locally or isolated with identical semantics — isolation is a deploy-time decision, not a rewrite.

---

## Advanced topics

### Self-extending agents

Inside CodeAct, the LLM can define new helper methods at runtime and fan them out:

```python
class DataAgent(Agent, llm=llm):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=20)))
    async def process_dataset(self, data: list[dict]) -> dict:
        """Process dataset. Create helper methods as needed."""
        ...

# The model can, mid-run, define:
#     @strategy(PredictStrategy())
#     async def extract_features(self, item: dict) -> dict:
#         """Extract features from a single item."""
#         ...
#     features = await asyncio.gather(*(self.extract_features(x) for x in data))
```

### LLM cascading resolution

Configure LLMs at any granularity — class default, method override, instance override — for cost/latency optimization, A/B testing, or gradual rollouts.

```python
class MyAgent(Agent, llm=default_llm):              # 1. class default
    sub_agent = MySubAgent()                        # 2. inherits from outer class
    @strategy(CodeActStrategy(), llm=special_llm)   # 3. method override
    async def complex_task(self) -> str:
        ...

agent = MyAgent(llm=different_llm)                  # 4. instance override
await agent.complex_task(llm=one_off_llm)           # 5. call override
```

A method override can also be a callable taking the agent and returning a
client. It is resolved on each call, so the per-method model can be chosen at
agent-construction time (or vary with instance state) instead of being fixed
when the module is imported:

```python
class MyAgent(Agent, llm=default_llm):
    @strategy(CodeActStrategy(), llm=lambda self: self.big_model)
    async def complex_task(self) -> str:
        ...

alice = MyAgent(); alice.big_model = strong_llm     # same method, different
bob = MyAgent();   bob.big_model = cheap_llm        # model per instance
```

### Event-driven history

Subscribe to agent events or query past events:

```python
agent.event_manager.on("message", lambda e: print(f"Message: {e.content}"))
recent = agent.events.query()
```

### Additional examples

Beyond the numbered quickstart, [`advanced/`](advanced/) contains focused demos of specific mechanics:

- [`codeact_event_sequence.py`](advanced/codeact_event_sequence.py) — inspect the raw event stream during a CodeAct run
- [`memory.py`](advanced/memory.py) — persistent memory patterns
- [`prefill.py`](advanced/prefill.py) — pre-populate agent state before running
- [`swappable_execution_engines.py`](advanced/swappable_execution_engines.py) — replace the default Python execution engine
- [`tracing_langfuse.py`](advanced/tracing_langfuse.py), [`tracing_otlp.py`](advanced/tracing_otlp.py), [`tracing_phoenix.py`](advanced/tracing_phoenix.py) — export traces to third-party backends

And [`quickstart/`](quickstart/) also contains a few beyond-numbered examples: `12_memory.py`, `13_multimodal.py`, `14_atif_trajectory.py`, `15_nemo_relay.py`.

### Community examples

- [`gspc_provision_eval/`](gspc_provision_eval/) — deterministic, provision-anchored AI-compliance evaluator with signed verdict records and an LLM-narrated explanation layer (Python decides, the LLM narrates). Contributed by [CSOAI](https://csoai.org) (Council for the Safety of Artificial Intelligence) for the Open Secure AI Alliance's evaluations-and-benchmarks lane.
