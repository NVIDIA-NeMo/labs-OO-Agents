<div align="center">

<br />

# NeMo Object Oriented Agents

[![NVIDIA](https://img.shields.io/badge/made%20by-NVIDIA-76B900)](https://www.nvidia.com/)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](PAPER_URL)
[![Blog](https://img.shields.io/badge/blog-coming%20soon-lightgrey)](BLOG_URL)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

**[Quick Start](#quick-start)** &nbsp;·&nbsp; **[Examples](examples/README.md)** &nbsp;·&nbsp; **[Paper](PAPER_URL)**

<!-- TODO: add hero image at assets/hero.png -->
<!-- <img src="assets/hero.png" alt="NeMo OO Agents at a glance" width="720"> -->

<br />

</div>


NeMo OO Agents (NOOA) is a model-agnostic Python framework for building reliable AI agents. Traditional agent frameworks scatter your code across prompt templates, tool schemas, callback handlers, and workflow graphs. NOOA collapses all of that into a single Python class:

- **Classes are agents.** Fields are state, methods are capabilities, docstrings are prompts, type annotations are contracts.
- **`...` bodies are LLM-driven.** A method whose body is `...` becomes an agentic loop; a real body stays deterministic Python. The boundary is one character wide and visible in the source.
- **Live objects, not serialized prompts.** Arguments are passed by reference. A method can accept a million-row dataframe or a live database handle — the model sees a bounded preview and operates on the real object.
- **Code as action.** By default, the model acts by writing Python in a Jupyter-style REPL that has access to `self`, imports, and helpers — no bespoke tool schemas.
- **Typed I/O with auto-retry.** Return values (Pydantic, dataclass, TypedDict, primitives) are validated on the way out; the harness reprompts on failure.
- **Explicit, model-callable context.** Static and dynamic context blocks, plus a typed event history, are Pythonic APIs available to both the developer and the model.

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

The distribution package is `nemo-labs-oo-agents`; the Python import package is `nooa`.

Then set your API key (any [LiteLLM-supported](https://docs.litellm.ai/) model works):

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env       # or OPENAI_API_KEY, GEMINI_API_KEY, ...
```

That's it — jump to the [Quick Start](#quick-start).

### Advanced Installation

For a local editable installation, clone the repo and sync the development environment with `uv`:

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/interactive-agents/nemo_oo_agents.git
cd nemo_oo_agents
uv sync --group dev
```

This installs the core framework, workspace packages, development tools, the `nooa` CLI,
and the trace viewer runtime in the repo's `.venv`. Run CLI commands through `uv`:

```bash
uv run nooa --help
uv run nooa start-dev       # trace viewer on http://localhost:5001
```

If you want the CLI without a source checkout, install the separate CLI package:

```bash
uv add nemo-labs-oo-agents-cli
```

<details>
<summary><strong>Contributor details</strong> — tests, lint, optional integrations</summary>

**Contributor setup** — enable pre-commit hooks and run the test/lint suite:

```bash
uv run pre-commit install
uv run pytest                # run tests
uv run ruff check            # lint
uv run pyright               # type check
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

</details>

## Quick Start

***Agents are Python objects***. Methods with `...` bodies are **generation methods** — implemented at runtime by an LLM-driven strategy. The signature defines the contract; the docstring is the prompt.

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

Run the same code from your own project with `python`. You can run the checked-in example:

```bash
uv run python examples/quickstart/01_first_generation_method.py
```

Rename `analyze_feedback` to `analyze_feedback_briefly` and the output changes — your method name, parameters, and docstring *are* the prompt.

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
