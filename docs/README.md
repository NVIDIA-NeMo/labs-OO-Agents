<div align="center">

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="../assets/nvidia-labs-object-oriented-agents-dark.svg"
  >
  <source
    media="(prefers-color-scheme: light)"
    srcset="../assets/nvidia-labs-object-oriented-agents-light.svg"
  >
  <img
    alt="NVIDIA-labs Object Oriented Agents"
    src="../assets/nvidia-labs-object-oriented-agents-light.svg"
    width="820"
  >
</picture>

</div>

# Documentation

NVIDIA-labs Object Oriented Agents (NOOA) is a Python framework in which an
agent is an object and its methods are its capabilities. An asynchronous method
ending in `...` is implemented by an LLM at runtime; a method with a real body
remains ordinary Python.

These pages explain that model for human readers. For detailed instructions
written for coding agents, see the repository's [`skills/`](../skills/README.md)
directory. For code you can run immediately, use the
[`examples/` catalog](../examples/README.md).

## Choose a path

### Understand NOOA in 10 minutes

1. Read the [framework tour](tour.md).
2. If you want the runtime mechanics, skim
   [how a method call runs](architecture.md).

### Learn by building

1. Work through the [notebook tutorials](../notebook_tutorials/README.md) in
   order.
2. Use the [quickstart catalog](../examples/README.md) for compact,
   copy-paste programs.
3. Return here for focused explanations when a design question arises.

### Build a reliable workflow

1. [Orchestration](concepts/orchestration.md)
2. [Tracing](concepts/tracing.md)
3. [Safety](concepts/safety.md)

### Scale beyond one agent

1. [Multi-agent systems](concepts/multi-agent-systems.md)
2. [Architecture](architecture.md)

## Concepts at a glance

| Document | Question it answers |
|---|---|
| [Agents and methods](concepts/agents-and-methods.md) | What is an agent in NOOA, and what does `...` mean? |
| [Strategies](concepts/strategies.md) | When should a method use Predict or CodeAct? |
| [Tools and visibility](concepts/tools-and-visibility.md) | How does generated code discover and call capabilities? |
| [Prompts and context](concepts/prompts-and-context.md) | Where should instructions, inputs, and cross-call information live? |
| [Local models](local-models.md) | How do I run NOOA with Ollama or vLLM? |
| [Orchestration](concepts/orchestration.md) | How do I make a workflow deterministic without turning it into one giant prompt? |
| [Multi-agent systems](concepts/multi-agent-systems.md) | When should I use another agent, and what state does it share? |
| [Tracing](concepts/tracing.md) | How do I inspect the complete Python and LLM call tree? |
| [Safety](concepts/safety.md) | What security boundary does NOOA provide, and what must the deployment provide? |

## A note for users of graph and chain frameworks

NOOA does not require a separate graph, chain, or tool-schema representation of
your program. Python remains the control plane:

- Agentic methods contain the fuzzy work delegated to an LLM.
- Regular methods contain deterministic capabilities.
- Regular Python methods and classes orchestrate ordering, branching, retries,
  concurrency, and verification.
- Type annotations define the boundary between model output and application
  code.

You can still build graphs, routers, supervisors, and worker pools. In NOOA,
they are normally expressed as Python control flow over agent objects rather
than as a second declarative program.

## Documentation conventions

Code blocks in the concept pages are intentionally focused on one idea. Some
use an existing `llm` variable or omit application setup. Follow the linked
quickstart for the complete runnable version.

The concepts describe stable design principles. Exact configuration surfaces,
advanced controls, and framework internals remain in the
[`skills/`](../skills/README.md) reference and source code.

## Other resources

- [Root README](../README.md) — installation, model selection, and the shortest
  quick start.
- [Notebook tutorials](../notebook_tutorials/README.md) — guided,
  notebook-oriented learning.
- [Examples catalog](../examples/README.md) — standalone quickstarts, advanced
  mechanics, and complete systems.
- [Repository conventions](../AGENTS.md) — precise authoring rules used by
  contributors and coding agents.
