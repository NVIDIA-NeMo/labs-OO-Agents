---
name: nooa-agent-authoring
description: Author agents with the NVIDIA OO Agents (nooa) framework. Use when writing or modifying an Agent subclass, agentic methods (ellipsis bodies), docstring prompts, structured output contracts, strategy selection (CodeAct/Predict), visibility control, orchestrators, or subagent composition.
compatibility: Python >= 3.12, uv, nooa package (CLI: nooa)
---

# Authoring NVIDIA OO Agents (nooa)

An agent is a Python class; its methods are its capabilities. Three rules drive everything:

- **`...` body = agentic method.** An async method whose body ends in `...` is implemented by an LLM at call time. A real body = plain Python.
- **Docstring = prompt.** The method docstring (plus name, signature, and class docstring) is the instruction the LLM receives.
- **Return type = contract.** The framework validates the LLM's output against the annotation and retries on failure.

The **class docstring** is part of the static system-prompt prefix — it appears in every LLM call on that agent. Keep it to a sentence or two describing the agent's role and constraints.

```python
import asyncio
from typing import Annotated
from nooa import Agent
from nooa.unifiedllm import get_llm_client

llm = get_llm_client("gpt-4o-mini")    # any litellm model name, or a registry alias

class Summarizer(Agent, llm=llm):
    """Produce concise, faithful summaries."""          # class docstring → static prefix

    async def summarize(
        self,
        text: Annotated[str, "The document to summarize"],
    ) -> Annotated[str, "Three bullet points"]:
        """Summarize the text in three bullet points."""
        ...                                              # LLM implements this (agentic method)

async def main():
    result = await Summarizer().summarize("...")
    print(result)

asyncio.run(main())
```

Use `Annotated[type, "description"]` on parameters and return types to give the LLM richer context about what each value means.

## LLM clients and cascading

```python
from nooa.unifiedllm import get_llm_client, CompletionClient, RetryConfig

llm = get_llm_client("gpt-4o-mini")                        # any litellm name passes through (needs provider key)
llm = get_llm_client("qwen3.5-397b")                       # registry alias (public NIM via NVIDIA_API_KEY from build.nvidia.com)
llm = get_llm_client("gpt-4o-mini", retry_config=RetryConfig(max_retries=5))  # retries default ON (max_retries=3)
llm = CompletionClient(model="m", base_url="https://.../v1", api_key="...")  # any OpenAI-compatible endpoint
```

**Registry:** models are defined in YAML configs loaded from `~/.config/nooa/models.yaml` (user) and the package's built-in `model_registry.yaml`. Each entry maps an alias to `{model, base_url, api_key_env}`. Inspect with `nooa config show` (shows all config layers and resolved values) or programmatically via `from nooa.unifiedllm.registry import reload_registry; configs = reload_registry()`.

Keys come from `.env` (library use) or `~/.config/nooa/secrets.yaml`.

**Resolution cascade** for which LLM a method uses — first match wins:

1. `@strategy(..., llm=special_llm)` — method override
2. `agent = MyAgent(llm=other)` — instance override
3. `class MyAgent(Agent, llm=default)` — class default
4. **Parent inheritance** — a subagent with no `llm=` of its own inherits from the agent that calls it

Child agents inherit the parent's LLM by default. Any explicit `llm=` on the child overrides it — that's how you run a cheap model for one phase:

```python
class Reviewer(Agent):                # no llm= → inherits the calling parent's LLM
    async def review(self, draft: str) -> Feedback: ...

class Writer(Agent, llm=big_llm):
    async def run(self, topic: str) -> str:
        reviewer = Reviewer()                  # runs on big_llm (inherited from Writer)
        triage = Reviewer(llm=cheap_llm)       # explicit override beats inheritance
        ...
```

**Timing constraint:** resolution happens eagerly at construction via a runtime context variable — so a no-`llm` subagent must be constructed *inside a parent's agentic method body* (an orchestrator or LLM-generated code). Constructing one at module level or in `__init__` (before the parent's LLM context is active) raises `ValueError: No LLM available`.

To construct subagents in `__init__`, pass the LLM explicitly:

```python
class Orchestrator(Agent, llm=big_llm):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self._llm is resolved after super().__init__()
        self.helper = HelperAgent(llm=self._llm)
```

## Docstrings are prompts

**Arguments are rendered to the LLM automatically — avoid writing `{param}` in docstrings.** Each strategy renders parameters with protections: CodeAct passes them as live variables and `pprint()`s them under truncation config; Predict serializes them with size caps. Interpolating `{text}` into the docstring is usually redundant and bypasses those protections (no truncation, untrusted data in instruction channel).

```python
# AVOID — argument re-injected into the prompt (redundant, unprotected)
async def summarize(self, text: str) -> str:
    """Summarize the following text: {text}"""
    ...

# PREFERRED — the LLM already sees `text`; the docstring only instructs
async def summarize(self, text: str) -> str:
    """Summarize the text in three bullet points."""
    ...
```

There are cases where `{param}` is appropriate — e.g. when prefill is disabled and you need custom rendering, or when you want to embed a short value directly in the instruction. For advanced prefill and truncation control, see `nooa-codeact-advanced`.

Template expansion is primarily for values the signature *can't* show: `{self.attr}` instance state and computed expressions like `{len(items)}`. Escape literal braces as `{{ }}`.

**Glossary:**
- **Static prefix** — the class docstring and context blocks prepended to every LLM call on the agent.
- **Agentic method** — an async method with `...` body, implemented by an LLM at call time. Has an internal lock (only one concurrent execution per agent instance).
- **Generation method** — synonym for agentic method.

Additional prompt mechanics:
- The LLM sees `doc(self)` — auto-generated API docs of all visible methods/fields — so helper methods are discoverable without listing them (though naming them in the docstring still helps).
- Method names are part of the prompt: `order_ingredients` beats `process`.

## Structured output

Return types can be Pydantic models, dataclasses, TypedDicts, `Literal`, basic and container types. Validation failures are fed back to the LLM, which retries:

```python
class Analysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
    topics: list[str]

async def analyze(self, text: str) -> Analysis:
    """Analyze the text for sentiment and topics."""
    ...
```

- Prefer a typed `BaseModel` over raw `dict`/`list` — it gives the LLM a schema to target.
- For an annotated free-form string: `-> Annotated[str, "Your answer"]`.
- When the LLM generates code, validate it in the model: a `@field_validator` that calls `ast.parse()` turns syntax errors into automatic retries.
- Types used in signatures must be defined or imported at **module level** so they exist in the CodeAct execution namespace.

## Strategies

| Strategy | When | How inputs are passed | Behavior |
|---|---|---|---|
| `CodeActStrategy` (default) | anything needing code execution, tool calls, or iteration | Parameters are live Python objects in the REPL; the LLM sees source code + stdout/stderr from each execution | REPL loop: `execute_python(code)` + `return_result(value)` tools; `reasoning(text)` builtin for chain-of-thought |
| `PredictStrategy` | single-shot classification/extraction returning a typed value | Parameters serialized as text with size caps | One LLM call, output validated against the return type |

```python
from nooa import strategy
from nooa.strategies import CodeActStrategy, PredictStrategy
from nooa.config import CodeActConfig, PredictConfig

@strategy(PredictStrategy())                                   # fast single-shot
async def classify(self, text: str) -> Intent: ...

@strategy(CodeActStrategy(config=CodeActConfig(max_iterations=10)))
async def implement(self, task: str) -> str: ...
```

**Constructors take `config=` only.** `PredictStrategy(max_retries=3)` and `CodeActStrategy(max_iterations=10)` are errors — wrap options in `PredictConfig(...)`/`CodeActConfig(...)`. Useful `CodeActConfig` fields: `max_iterations`, `max_retries`, `cell_timeout`, `max_tokens`, `temperature`, `max_consecutive_text_only`, `restrictions`.

`max_iterations` is a safety net, not the main tuning dial — decompose the task instead of raising the cap. For prefill control, truncation tuning, code restrictions, and the full config surface, see `nooa-codeact-advanced`.

**Reserved parameter:** naming an agentic method parameter `reasoning` raises `ValueError` at class creation (chain-of-thought is provided via the injected `reasoning()` builtin instead).

**Prefill:** code between the docstring and the `...` runs first and is shown to the LLM as a starting point. The LLM sees the executed code and its output:

```python
async def order(self, recipe: dict) -> OrderResult:
    """Order ingredients for the recipe."""
    stock = {name: self.check_stock(name) for name in recipe}
    ...   # LLM continues with `stock` already computed
```

## Visibility

Everything public is visible to the LLM by default (in `doc(self)` and the CodeAct namespace); hide explicitly. `_private` names are hidden by default.

```python
from typing import Annotated
from nooa import Agent, hidden
from nooa.agentdoc import spec        # NOT `from agentdoc import ...` — that package doesn't exist

with hidden:
    import secrets                              # keep out of the LLM's execution namespace

class SearchAgent(Agent, llm=llm):
    api_key: Annotated[str, hidden] = ""        # hidden field

    @hidden                                     # hidden method (still callable from Python)
    def rebuild_index(self) -> None: ...

    @spec(hidden=False)                         # opt a _private method back in
    def _shown_helper(self) -> str: ...
```

- **Hide the entry-point agentic method** (`@hidden` on `run`/`respond`) when the LLM might recursively call itself through `doc(self)`.
- Never hide a method the LLM must call as a tool.
- Module-level imports are the LLM's execution namespace: if generated code needs `json.loads`, `import json  # noqa: F401` at the top of the agent file.
- `self.context` / `self.events` are always present but hidden; expose with `spec(self, "context", hidden=False)` in `__init__` (see `nooa-context-and-state`).
- Making the visible surface render *well* — field descriptions, referenced-type expansion, `pformat` caps — is its own craft: see `nooa-agentdoc`.

## Orchestration and decomposition

- **Orchestrators are pure Python.** The entrypoint that sequences phases has a real body and calls agentic methods; the LLM cannot skip steps you encode in Python. Enforce verification gates in the orchestrator, not the prompt.
- **One method = one LLM task.** If a method classifies AND greps AND summarizes, split it.
- **Helpers beat prompts.** Deterministic logic (exact matching, parsing, formatting) goes in regular methods the LLM calls as tools; LLM generation is for fuzzy interpretation. Define helpers as class methods, not lambdas assigned to `self`.
- **Subagents** for context isolation, per-phase LLMs, or reusable sub-tasks: store an instance on `self`, `await` its methods. Subagents share nothing (no context blocks, no history) — pass data explicitly. Define them without `llm=` to inherit the parent's.
- A class with no `...` methods doesn't need to subclass `Agent` at all.
- **Concurrency:** each agentic method holds an internal lock on its agent instance — `asyncio.gather` on one instance's agentic methods runs them sequentially. For true parallelism, use one instance per concurrent task.
- Keep interesting logic *inside* agent methods: preprocessing done in `main()` is invisible in traces and can't be adapted to by the agent.

## Debugging while authoring

```python
from nooa import print_prompt, build_prompt_data, enable_logging
from nooa.unifiedllm import FakeLLMClient

agent = MyAgent(llm=FakeLLMClient())            # no network
await print_prompt(agent.my_method, sample_arg)  # exact system + task prompt the LLM would see
data = await build_prompt_data(agent.my_method, sample_arg)  # structured PromptData

enable_logging(level="DEBUG")                    # nooa.* logger hierarchy
# kill -USR2 <pid>                               # dump traceback + all registered cells to debug_dump_<pid>.txt
```

Most bugs are visible in the rendered prompt. For runtime behavior, capture traces and inspect them — see `nooa-capturing-traces` and `nooa-trace-explorer`.

## Where to look

- Examples: `examples/README.md` and `examples/quickstart/`.
- Runnable examples: `examples/quickstart/01`–`15`.
- Related skills: `nooa-codeact-advanced`, `nooa-context-and-state`, `nooa-tools-and-skills`, `nooa-agentdoc`, `nooa-channels`, `nooa-capturing-traces`, `nooa-trace-viewer`, `nooa-trace-explorer`.
