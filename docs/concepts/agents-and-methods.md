# Agents and methods

An agent in NOOA is a Python object. Its methods define what it can do, its
fields hold its state, and ordinary Python code decides how those capabilities
compose.

The one piece of special syntax is an ellipsis at the end of an asynchronous
method:

```python
class SupportAgent(Agent, llm=llm):
    async def answer(self, question: str) -> str:
        """Answer the support question accurately and concisely."""
        ...
```

The ellipsis tells NOOA to implement that method with an LLM at call time. A
method with a real body is regular Python.

## The method is the model boundary

Three familiar Python elements define an agentic call:

- The method name and docstring state the task.
- The parameters are the inputs.
- The return annotation is the output contract.

```python
from typing import Literal

from pydantic import BaseModel, Field

from nooa import strategy
from nooa.strategies import PredictStrategy


class Triage(BaseModel):
    category: Literal["billing", "technical", "general"]
    urgency: Literal["low", "medium", "high"]
    summary: str = Field(description="One-sentence problem summary")


class SupportAgent(Agent, llm=llm):
    @strategy(PredictStrategy())
    async def triage(self, message: str) -> Triage:
        """Classify the support request."""
        ...
```

The caller receives a `Triage`, not an unvalidated JSON string. If model output
does not match the annotation, the strategy can return the validation error to
the model and retry.

## Regular methods remain useful

Deterministic behavior belongs in ordinary methods:

```python
class OrderAgent(Agent, llm=llm):
    def price(self, sku: str) -> float:
        return PRICE_LIST[sku]

    async def explain_quote(self, skus: list[str]) -> str:
        """Explain the quote. Use self.price() for exact prices."""
        ...
```

Generated CodeAct code can call visible methods such as `self.price()`. You do
not need to duplicate the calculation in the prompt or register a second tool
schema.

## Arguments are already shown to the model

Keep a docstring focused on instructions:

```python
async def summarize(self, document: str) -> str:
    """Summarize the document in three bullet points."""
    ...
```

Do not write `"Summarize this: {document}"`. Strategies already render method
arguments with size controls. Interpolating the raw value again is redundant,
bypasses those controls, and places untrusted data in the instruction text.

Template expressions remain useful for trusted configuration the signature
cannot show, such as `"Translate to {self.target_language}."`.

## Design one method around one judgment

An agentic method works best when it has one clear purpose. Split a method that
must classify, investigate, edit, and verify into separate agentic methods and
sequence them in Python.

Use:

- an agentic method for fuzzy interpretation or generation;
- a regular method for exact transformations and checks;
- a Python orchestrator for ordering, branching, retries, and acceptance.

If a class contains no agentic methods, it probably does not need to inherit
from `Agent`.

## If you know chain or graph frameworks

Think of an agentic method as a typed LLM-backed node whose interface is an
ordinary Python signature. The important difference is that the Python object
remains the primary program: its methods can call each other directly, and
control flow does not need a second graph representation.

## Common mistakes

- Returning a raw `dict` when a named Pydantic model would express the contract.
- Asking one method to perform several unrelated LLM tasks.
- Describing exact business rules in prose instead of implementing them in
  Python.
- Repeating raw parameter values in the docstring.
- Defining return or parameter types inside a function; CodeAct-visible types
  should live at module level.

## Continue

- [Strategies](strategies.md)
- [Tools and visibility](tools-and-visibility.md)
- Runnable examples: [first method](../../examples/quickstart/01_first_generation_method.py),
  [structured output](../../examples/quickstart/02_structured_outputs.py), and
  [methods as tools](../../examples/quickstart/03_codeact_tools.py)
