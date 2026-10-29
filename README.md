<div align="center">

<br />

<!-- Absolute URLs, not repo-relative paths: this README is also the PyPI
     long_description, and PyPI renders it standalone with no assets/
     directory alongside it, so relative paths 404 there. -->
<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="https://raw.githubusercontent.com/NVIDIA-NeMo/labs-OO-Agents/main/assets/nvidia-labs-object-oriented-agents-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="https://raw.githubusercontent.com/NVIDIA-NeMo/labs-OO-Agents/main/assets/nvidia-labs-object-oriented-agents-light.svg"
  >
  <img
    alt="NVIDIA-labs Object Oriented Agents"
    src="https://raw.githubusercontent.com/NVIDIA-NeMo/labs-OO-Agents/main/assets/nvidia-labs-object-oriented-agents-light.svg"
    width="820"
  >
</picture>

<p align="center"><b>A Pythonic way to build AI agents.</b></p>

[![NVIDIA](https://img.shields.io/badge/NVIDIA-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Paper](https://img.shields.io/badge/paper-arXiv-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.20709)
[![Blog](https://img.shields.io/badge/blog-NVIDIA-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/LICENSE)

**[Docs](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/README.md)** &nbsp;·&nbsp; **[Quick Start](#quick-start)** &nbsp;·&nbsp; **[Notebook Tutorials](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/notebook_tutorials/README.md)** &nbsp;·&nbsp; **[Examples](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/examples/README.md)** &nbsp;·&nbsp; **[Paper](https://arxiv.org/abs/2607.20709)** &nbsp;·&nbsp; **[Blog](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/)**

<br />

</div>


NVIDIA-labs Object Oriented Agents (NOOA) is a model-agnostic Python framework designed to support reliable AI agent development. Many agent frameworks represent prompts, tools, callbacks, and workflows as separate abstractions. NOOA offers an alternative object-oriented interface that brings these concepts together in a Python class. NOOA lets developers express an agent’s state, capabilities, prompts, and typed interfaces through a single Python class:

```python
from nooa import Agent

# The agent is a Python object.
class SupportAgent(Agent):
    """You are a support agent."""

    # State lives on the object. Fields are typed.
    order_db: OrderDB

    # Ordinary method. Just Python.
    def is_refund_eligible(self, order: Order) -> bool:
        return order.delivered and order.days_since_delivery <= 30

    # Agentic method: the runtime hands this to an LLM.
    async def triage(self, message: str, order: Order) -> Ticket:
        """Create a typed support ticket."""
        ...
```

**What's happening here:**

- **Agents are Python objects.** Fields are state, methods are capabilities, docstrings are prompts, type annotations are contracts.
- **`...` bodies are LLM-driven.** A method with `...` becomes an agentic loop; a real body stays deterministic Python.
- **Generation commits one result.** Deterministic generators are supported, but generated streams need a separate stream contract and are rejected clearly.
- **Code as action.** The model acts by writing Python in a Jupyter-style REPL with access to `self`, imports, and helpers — Python methods and type annotations supply the callable interfaces, reducing the need to write separate tool-schema definitions.
- **Pythonic and agent-ready.** Typed I/O with auto-retry, live-object arguments passed by reference, and model-callable context and event APIs — designed around agent-oriented Python workflows.

This design supports familiar Python testing, tracing, refactoring, and version-control workflows — **just like the rest of your software**. Read [the paper](https://arxiv.org/abs/2607.20709) for the design principles and evaluation results.

Want to see how the pieces compose? Take the [**10-minute tour**](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/tour.md), from one thinking method through tools, typed contracts, deterministic orchestration, and object composition.

## Installation

Add the **core** framework to a new (or existing) Python project with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv init my-agent-project
cd my-agent-project

uv add nooa
```

Or with pip: `pip install nooa`.

> **Windows:** use [WSL 2](https://learn.microsoft.com/windows/wsl/install) for the full product. On native Windows only the pure-Python core runs: agents, PredictStrategy, in-process CodeAct, and in-memory storage. Features that need Linux or macOS (shell tools, the CodeAct sandbox, file-backed SQLite storage, `monitor()`, and the Atom agent) stop with a `PosixOnlyFeatureError` that points to WSL 2.

<details>
<summary><b>Optional sub-packages</b> — CLI, ACP, memory, benchmarks, evaluation pipeline</summary>

<br />

The CLI, ACP, memory, and benchmark packages are separate distributions. Install
them by name, or pull them in as extras of the core package:

```bash
uv add nooa-cli                 # or: uv add "nooa[cli]"
uv add nooa-acp                 # or: uv add "nooa[acp]"
uv add nooa-memory              # or: uv add "nooa[memory]"
uv add nooa-bench               # or: uv add "nooa[bench]"

uv add "nooa[cli,memory]"       # several at once
```

| Package | Extra | What it adds |
|---|---|---|
| `nooa-cli` | `nooa[cli]` | the `nooa` command, trace-viewer command, eval runner |
| `nooa-acp` | `nooa[acp]` | coding agent for Agent Client Protocol hosts such as Zed — [setup](packages/nooa-acp/README.md) |
| `nooa-memory` | `nooa[memory]` | long-term memory subsystem (`MemoryManager`) |
| `nooa-bench` | `nooa[bench]` | `BenchAgent` and the Harbor benchmark runner |

`eval_pipeline` is not published to PyPI — install it from the repo:

```bash
uv add "eval_pipeline @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main#subdirectory=util/eval_pipeline"
```

</details>

<details>
<summary><b>Installing from source</b> — track <code>main</code> or pin a tag</summary>

<br />

```bash
# latest development state
uv add "nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@main"

# pinned to a release tag
uv add "nooa @ git+https://github.com/NVIDIA-NeMo/labs-OO-Agents.git@v0.0.7"
```

</details>

## Quick Start

### ⚠️ Before Starting: safety note
NOOA is **research software**, and agents can be configured to execute LLM-generated code. We welcome contributions and fixes, but expect rough edges. LLM-generated code may take dangerous or unwanted actions, including sending private data to uncontrolled locations, deleting files, or modifying its environments.  Ensure you run NOOA agents in a sandboxed environment isolated from your primary filesystem, such as [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell).


NOOA validates generated code (AST checks) and applies module deny-lists before execution. **These are defense-in-depth guardrails, not a containment boundary.** They exist to keep generated code from freezing the event loop and to catch common mistakes early — not to stop code that is actively trying to escape. A static checker over Python cannot provide that guarantee: `open()` gives arbitrary file access, `importlib` can load modules straight from a path, and reflection reaches the rest. **The containment boundary is OS-level isolation** — always run agents that execute generated code inside a sandbox such as a container, VM, or [NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell). Do not rely on the in-process validators alone.


### 1. Connect a model

Install the CLI and run the guided setup:

```bash
uv add "nooa[cli]"
uv run nooa connect
```

Choose your provider or endpoint, API key, exact model, and an alias such as
`my-model`. Connect checks the configured route and saves the alias in NOOA's
user configuration. **The checks send real requests and can cost money**; the
wizard asks before running them and before saving. Keep keys in environment
variables or NOOA's secrets file, not in model configuration or committed code.

Use the saved alias in Python:

```python
from nooa.unifiedllm import get_llm_client

llm = get_llm_client("my-model")
```

See [Connect setup](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/model-connect.md)
for providers, editing, and diagnostic stages. For setup **without inference
calls**, use [offline model configuration](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/model-configuration.md).
`--no-probe` alone may still fetch discovery/catalogue metadata; the offline
guide explains how to avoid that too. Agent-driven stages do not ask for
spending approval; authorize them before use.

You can also construct clients directly using
[LiteLLM-supported model names](https://docs.litellm.ai/), with your endpoint
and credentials. See [manual configuration](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/model-configuration.md).

### 2. Your first agent

***Agents are Python objects***. Methods with `...` bodies are **generation methods** — implemented at runtime by an LLM-driven strategy. The signature defines the contract; the docstring is the prompt.

```python
import asyncio

from nooa import Agent


class FeedbackAgent(Agent, llm=llm):
    """You are an agent specializing in analyzing customer feedback."""

    async def analyze_feedback(self, text: str) -> str:
        """Analyze customer feedback for sentiment and key topics in one sentence."""
        ...


async def main():
    try:
        agent = FeedbackAgent()
        result = await agent.analyze_feedback("Great product, but shipping was slow")
        print(result)
    finally:
        await llm.aclose()


asyncio.run(main())
```

Save the model-selection and agent snippets together as `first_agent.py`, then
run `uv run python first_agent.py` in your project with approved inference
allowance. After your final call, `await llm.aclose()` releases client resources.

The [checked-in first example](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/examples/quickstart/01_first_generation_method.py)
is a separate source-checkout example: it chooses its own client from environment
credentials and does not automatically use your `my-model` alias.

Rename `analyze_feedback` to `analyze_feedback_briefly` and the output changes — your method name, parameters, and docstring *are* the prompt.

Prefer a guided notebook path? Start with the [**notebook tutorials**](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/notebook_tutorials/README.md), which walk through the same ideas in Colab-friendly steps, with more notebooks planned.

Ready to run something specific? Use the [**examples catalog**](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/examples/README.md) to find quickstarts for structured output, tools, strategies, tracing, context blocks, MCP, and more.

### 3. See what your agent is doing

Every LLM call, code execution, and method invocation is traced by default — orchestrators, generation methods, and helpers, with parent-child spans preserved. If you installed the CLI and viewer dependencies, start the trace viewer and open the run in your browser:

```bash
uv add "nooa[cli,viewer]"    # includes viewer runtime dependencies
uv run nooa start-dev        # trace viewer on http://localhost:5001
```

If the viewer isn't running, tracing is silently disabled — no configuration needed either way.

### 4. Give your agent media inputs

Pass `Image`, `Audio`, `Video`, or PDF `File` objects as method arguments.
CodeAct can also use `show()` to see media created while it runs.

NOOA assumes models are text-only until configured. Set `media_inputs`, such
as `[image, pdf]`, in the model entry. Connect lets you choose formats manually
or test them with generated samples. If you enable a format the provider does
not accept, requests may fail.

```python
from nooa import Agent, PredictStrategy, strategy
from nooa.media import Image
from nooa.unifiedllm import get_llm_client

# This alias must allow image input and use an endpoint that accepts it.
vision = get_llm_client("vision")

class PhotoReader(Agent, llm=vision):
    @strategy(PredictStrategy())
    async def describe(self, image: Image) -> str:
        """Describe the image briefly. Do not invent details."""
        ...

# In your async application, with permission to send the photo:
# result = await PhotoReader().describe(Image.from_file("photo.png"))
# await vision.aclose()
```

Load media with `from_file`, `from_bytes(..., media_type=...)`, or `from_url`.
URL support depends on the provider. `File` supports PDFs. Formats left out of
`media_inputs` are replaced with text notices when sending a request.

See the [media API guide](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/media.md)
for a complete example, CodeAct `show()`, audio/video/PDF inputs, and limits.

## Learn more

- **[Documentation](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/README.md)** — human-oriented reading paths, core concepts, architecture, and safety guidance.
- **[Framework tour](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/tour.md)** — a concise conceptual showcase of NOOA's core ideas and Python-first design.
- **[Decision models](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/docs/concepts/decisions.md)** — route classification, yes/no, and scoring methods to a calibrated decision model such as `typesafe/jev-1.13` with `DecideStrategy`; agents that only make decisions need no chat LLM. Runnable example: [`16_decisions.py`](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/examples/quickstart/16_decisions.py).
- **[Notebook tutorials](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/notebook_tutorials/README.md)** — the primary hands-on path for your first agent, strategy selection, CodeAct's live-object workflow, and composing subagents. More notebooks are planned.
- **[Examples catalog](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/examples/README.md)** — runnable quickstarts, advanced mechanics, and complete benchmark systems, indexed by capability and setup requirements.
- **[Paper](https://arxiv.org/abs/2607.20709)** — design principles, harness details, capability tests, and SWE-bench Verified / Terminal-Bench 2.0 results.
- **[Blog post](https://developer.nvidia.com/blog/six-agent-harness-capabilities-for-higher-model-performance/)** — Six Agent Harness Capabilities for Higher Model Performance.
- **[AGENTS.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/AGENTS.md)** — conventions used inside this repo (helpful when reading the source).

## Contributing

For a local editable install, clone the repo and sync the development environment with `uv`:

```bash
git clone https://github.com/NVIDIA-NeMo/labs-OO-Agents.git
cd labs-OO-Agents
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

See [CONTRIBUTING.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/CONTRIBUTING.md) for the full workflow.

## Citation

If you use NVIDIA-labs Object Oriented Agents in your research, please cite:

```bibtex
@techreport{nvidia_oo_agents_2026,
  title  = {NVIDIA-labs OO Agents: Native Python Object-Oriented Agents},
  author = {Furgale, Paul and Klingler, Severin and Nolan, James and Staats, Matt and
            Di Lorenzo, Gaia and Martinez Abad, Elisa and Schueler, Christian and
            Dinu, Razvan and Devoto, Alessio and Berard, Pascal and Kaplun, Gal and Sarafian, Elad and
            Roveri, Riccardo and Derczynski, Leon and Silveira Cabral, Ricardo},
  year   = {2026},
}
```

## License

Apache 2.0. See [LICENSE](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/LICENSE) and [THIRD_PARTY_NOTICES.md](https://github.com/NVIDIA-NeMo/labs-OO-Agents/blob/main/THIRD_PARTY_NOTICES.md).
