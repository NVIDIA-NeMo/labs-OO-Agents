# Notebook Tutorials

Start here if you want to learn NOOA by running code, inspecting prompts, and
seeing how small object-oriented agents come together step by step.

The notebooks cover the fundamentals: agents as Python objects, strategy
selection, CodeAct's live-object workflow, and composing subagents. Run them in
order; more notebooks are coming.

| # | Tutorial | Covers | Colab |
|---:|---|---|---|
| 1 | [Your First Object-Oriented Agent](./01_your_first_agent.ipynb) | Agents as Python objects, generation methods, methods as tools, typed returns, state, and the first trace-viewer hook. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA-NeMo/labs-OO-Agents/blob/main/notebook_tutorials/01_your_first_agent.ipynb) |
| 2 | [Choosing a Strategy](./02_choosing_a_strategy.ipynb) | How to choose between single-shot structured prediction and CodeAct's iterative Python execution. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA-NeMo/labs-OO-Agents/blob/main/notebook_tutorials/02_choosing_a_strategy.ipynb) |
| 3 | [CodeAct REPL and Pass by Reference](./03_codeact_tools_and_live_objects.ipynb) | How CodeAct uses a persistent REPL, passes large live objects by reference, and points toward Recursive Language Models (RLMs) through small helper surfaces. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA-NeMo/labs-OO-Agents/blob/main/notebook_tutorials/03_codeact_tools_and_live_objects.ipynb) |
| 4 | [Composing Subagents](./04_composing_subagents.ipynb) | How to compose subagents with plain Python — sequential, parallel with `asyncio.gather`, and LLM-driven spawning from generated CodeAct code. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA-NeMo/labs-OO-Agents/blob/main/notebook_tutorials/04_composing_subagents.ipynb) |

Looking for compact copy-paste examples? Use `examples/quickstart/`.
For deeper authoring reference material, see the `skills/nooa-*` documents.
