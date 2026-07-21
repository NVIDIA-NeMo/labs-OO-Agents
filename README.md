<div align="center">

<br />

# NeMo Object Oriented Agents

[![nemo-labs | NVIDIA](https://img.shields.io/badge/nemo--labs-NVIDIA-76B900)](https://www.nvidia.com/)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](PAPER_URL)
[![Blog](https://img.shields.io/badge/blog-coming%20soon-lightgrey)](BLOG_URL)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**[Quick Start](#quick-start)** &nbsp;·&nbsp; **[Examples](examples/README.md)** &nbsp;·&nbsp; **[Paper](PAPER_URL)**

<!-- TODO: add hero image at assets/hero.png -->
<!-- <img src="assets/hero.png" alt="NeMo OO Agents at a glance" width="720"> -->

<br />

</div>


NeMo OO Agents (NOOA) is a model-agnostic Python framework for building reliable AI agents. Traditional agent frameworks scatter your code across prompt templates, tool schemas, callback handlers, and workflow graphs. NOOA collapses all of that into a single Python class:

- **Agents are Python classes.** Fields are state, methods are capabilities, docstrings are prompts, type annotations are contracts.
- **`...` bodies are LLM-driven.** A method with `...` becomes an agentic loop; a real body stays deterministic Python. The boundary is one character wide.
- **Code as action.** The model acts by writing Python in a Jupyter-style REPL with access to `self`, imports, and helpers — no bespoke tool schemas.
- **Pythonic and agent-ready.** Typed I/O with auto-retry, live-object arguments passed by reference, and model-callable context and event APIs — designed for agents from the ground up.

The result: agents you can test, trace, refactor, and version — **just like the rest of your software**.

```python
from nooa import Agent, strategy, PredictStrategy

class SupportAgent(Agent):
    """You are a support agent for a customer service system."""

    order_db: OrderDB                                        # model-visible state

    def is_refund_eligible(self, order: Order) -> bool:      # deterministic tool
        """Return whether an order is eligible for a refund."""
        return order.delivered and order.days_since_delivery <= 30

    @strategy(PredictStrategy())
    async def classify(self, message: str) -> TicketKind:    # single-shot LLM call
        """Classify the customer message into the best ticket kind."""
        ...

    async def triage(                                        # CodeAct loop (default)
        self, message: str, photo: Image | None, order: Order | None
    ) -> Ticket:
        """Triage a customer message and create a support ticket."""
        ...
```

Read the paper for the design principles and evaluation results: [Nemo OO Agents: Native Python Object-Oriented Agents](PAPER_URL).

## Installation

Install the core distribution into a Python project with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv init my-agent-project
cd my-agent-project
uv add nemo-labs-oo-agents
```

If you want the CLI without a source checkout, install the separate CLI package:

```bash
uv add nemo-labs-oo-agents-cli
```

## Quick Start

### Choose a model

Pick any [LiteLLM-supported](https://docs.litellm.ai/) model — hosted or local:

```python
from nooa.unifiedllm.registry import get_llm_client

llm = get_llm_client("claude-haiku-4-5")                                            # Anthropic (after `export ANTHROPIC_API_KEY=...`)
llm = get_llm_client("gpt-5-mini")                                                  # OpenAI    (after `export OPENAI_API_KEY=...`)
llm = get_llm_client("ollama_chat/qwen3:1.7b", api_base="http://localhost:11434")   # Ollama    (no key)
llm = get_llm_client("hosted_vllm/Qwen/Qwen3-1.7B", api_base="http://localhost:8000/v1")  # vLLM (no key)
```

### Your first agent

***Agents are Python objects***. Methods with `...` bodies are **generation methods** — implemented at runtime by an LLM-driven strategy. The signature defines the contract; the docstring is the prompt.

```python
from nooa import Agent


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

Run the same code from your own project with `python`. You can run the checked-in example:

```bash
uv run python examples/quickstart/01_first_generation_method.py
```

Rename `analyze_feedback` to `analyze_feedback_briefly` and the output changes — your method name, parameters, and docstring *are* the prompt.

Ready for more? See [**examples/**](examples/README.md) for the full progressive tutorial — structured output, tools, strategies, tracing, context blocks, MCP, and more.

### See what your agent is doing

Every LLM call, code execution, and method invocation is traced by default — orchestrators, generation methods, and helpers, with parent-child spans preserved. If you installed the CLI and viewer dependencies, start the trace viewer and open the run in your browser:

```bash
uv run nooa start-dev        # trace viewer on http://localhost:5001
```

If the viewer isn't running, tracing is silently disabled — no configuration needed either way.

## Learn more

- **[examples/README.md](examples/README.md)** — the full progressive tutorial: structured output, tools via `self`, strategies, progressive disclosure with `doc()`, tracing, dynamic prompts, context blocks, summarization, skills, MCP, sandbox, and more.
- **[Paper](PAPER_URL)** — design principles, harness details, capability tests, and SWE-bench Verified / Terminal-Bench 2.0 results.
- **[Blog post](BLOG_URL)** — WIP.
- **[CLAUDE.md](CLAUDE.md)** — conventions used inside this repo (helpful when reading the source).

## Contributing

For a local editable install, clone the repo and sync the development environment with `uv`:

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/interactive-agents/nemo_oo_agents.git
cd nemo_oo_agents
uv sync --group dev
```

This installs the core framework, workspace packages, development tools, the `nooa` CLI, and the trace viewer runtime in the repo's `.venv`. Run CLI commands through `uv`:

```bash
uv run nooa --help
uv run nooa start-dev       # trace viewer on http://localhost:5001
```

Enable pre-commit hooks and run the test/lint suite:

```bash
uv run pre-commit install
uv run pytest                # run tests
uv run ruff check            # lint
uv run pyright               # type check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Citation

If you use NeMo OO Agents in your research, please cite:

```bibtex
@techreport{nemo_oo_agents_2026,
  title  = {Nemo OO Agents: Native Python Object-Oriented Agents},
  author = {Furgale, Paul and Klingler, Severin and Nolan, James and Staats, Matt and
            Di Lorenzo, Gaia and Martinez Abad, Elisa and Schueler, Christian and
            Dinu, Razvan and Devoto, Alessio and Berard, Pascal and Silveira Cabral, Ricardo},
  year   = {2026},
}
```

## License

Apache 2.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
