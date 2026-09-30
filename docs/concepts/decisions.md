# Decision models

`DecideStrategy` uses an agent's decision model when one is configured. Method
arguments form the decision state, the return annotation declares one or more
questions, and the model returns probabilities or score distributions. The
agent's chat LLM remains available for its other generation methods.

```python
import os
from enum import StrEnum

from nooa import Agent, DecisionClient, DecideStrategy, strategy
from nooa.unifiedllm.registry import get_llm_client


class Department(StrEnum):
    """Support team responsible for a message.

    Attributes:
        BILLING: Payments, invoicing, and refunds.
        TECHNICAL: Bugs, outages, and integrations.
    """

    BILLING = "billing"
    TECHNICAL = "technical"


llm = get_llm_client("gpt-5-mini")
decision_model = DecisionClient(
    "typesafe/jev-1.13",
    endpoint="https://openrouter.ai/api/alpha/decisions",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


class Router(Agent, llm=llm, decision_model=decision_model):
    @strategy(DecideStrategy())
    async def department(self, message: str) -> Department:
        """Choose the team that should handle the message."""
        ...

    async def draft_reply(self, message: str) -> str:
        """Draft a concise and helpful response."""
        ...
```

## Advanced usage

### Render trusted agent state into the instructions

Method docstrings are expanded for every call before `DecideStrategy` compiles
the question. Use `{self.attribute}` or a computed expression when trusted
instance configuration changes what the method should decide:

```python
class RefundAgent(Agent, llm=llm, decision_model=decision_model):
    region = "EU"

    def render_refund_policy(self) -> str:
        return POLICIES[self.region]

    @strategy(DecideStrategy())
    async def should_refund(self, request: str) -> bool:
        """Apply the {self.region} policy: {self.render_refund_policy()}"""
        ...
```

For an EU agent, the rendered docstring becomes the boolean question's
instructions. In a composite result, where every field has its own
`Instructions`, the rendered method docstring becomes shared guidance instead.
This expansion applies to the method docstring, not to strings inside
`Instructions(...)` annotations.

Method arguments are already present under `state.inputs`; do not repeat
untrusted or potentially large argument values in the docstring.

### Add supporting context and event history

A native decision request does not receive the agent's entire chat prompt.
Opt in to supporting state with the same `ScopedContext` and `EventQuery`
mechanisms used by other strategies:

```python
from nooa import Context, EventQuery
from nooa.context_blocks import ScopedContext


class SupportAgent(Agent, llm=llm, decision_model=decision_model):
    customer_tier = "enterprise"

    def render_refund_policy(self) -> str:
        return "Refunds over $500 require manual review."

    @strategy(
        DecideStrategy(),
        context=ScopedContext(
            context={
                "support_policy": Context(expr="self.render_refund_policy()"),
                "customer_tier": Context(expr="self.customer_tier"),
            },
            events=EventQuery.last_n(5),
        ),
    )
    async def should_escalate(self, message: str) -> bool:
        """Should this customer request be escalated?"""
        ...
```

The resulting request state has this shape:

```python
{
    "inputs": {
        "message": "Please refund my $900 purchase",
    },
    "context": {
        "support_policy": "Refunds over $500 require manual review.",
        "customer_tier": "enterprise",
    },
    "events": [
        {
            "type": "Message",
            "role": "assistant",
            "data": {"content": "The customer requested a refund."},
        },
    ],
}
```

Context expressions are resolved at call time. Decorator context is inherited
by nested calls, and an active `with ScopedContext(...)` can add or override it.
Ordinary persistent context blocks are not copied into native decision state
automatically; reference one explicitly with, for example,
`Context(expr='self.context["policy"]')`.

Events are included only when an effective `EventQuery` is configured. Runtime,
active scoped, decorator, and agent queries use the normal precedence rules.
Selected events retain chronological order. Stored metadata such as
`DecisionRecord` and runtime lifecycle events are never sent back to the model.
All inputs, resolved context values, and event data must be JSON-compatible;
unsupported values fail with the offending parameter, context-block name, or
event index instead of being stringified or truncated silently.

### Ask several questions in one call

Use a Pydantic model to group related outputs. Each field declares its own
instructions, while the method docstring supplies guidance shared by all the
questions:

```python
from typing import Annotated

from pydantic import BaseModel
from nooa import Criteria, Instructions


class Triage(BaseModel):
    urgent: Annotated[
        bool,
        Instructions("Determine whether this requires immediate action."),
        Criteria(
            by_value={
                True: "Delay would cause immediate harm.",
                False: "The request can safely wait.",
            }
        ),
    ]
    department: Annotated[
        Department,
        Instructions("Select the team that should own the request."),
    ]


class SupportAgent(Agent, llm=llm, decision_model=decision_model):
    @strategy(DecideStrategy())
    async def triage(self, message: str) -> Triage:
        """Use only the supplied message and documented routing policy."""
        ...
```

This produces one decision-model request with two named questions rather than
two round trips.

### Choose the model at the right scope

The agent can use its chat LLM for ordinary generation and a separate decision
model for `DecideStrategy` methods. A method-level override takes precedence
when one decision needs a specialized model:

```python
class RiskAgent(Agent, llm=llm, decision_model=general_decision_model):
    @strategy(DecideStrategy(), decision_model=fraud_decision_model)
    async def detect_fraud(self, transaction: Transaction) -> bool:
        """Is this transaction likely fraudulent?"""
        ...
```

Resolution is method decision model, then agent decision model, then chat-LLM
fallback. The fallback supports primitive `bool`, enum, `Literal`, and scored
`float` results. Detailed decision objects and thresholded outputs require a
native decision model because a chat completion does not provide calibrated
probability evidence.

Every decision call, native or fallback, stores a `DecisionRecord` with the
request state, normalized questions, answers, and provenance:

- `decision_source`: `native` or `llm_fallback`.
- `question_digest`: a SHA-256 digest of the normalized question names,
  instructions, criteria, and candidate IDs. Native and fallback calls that ask
  the same question share a digest. For native calls it is computed after
  `decision_call` middleware, so it reflects the request actually sent.
- `requested_model` and `resolved_model`: the configured model and, for native
  calls, the model the endpoint reports it served.
- `fallback_schema_version`: the version of the Predict adapter used by an LLM
  fallback. Fallback answers contain only `value`; they never contain
  probabilities.

The generation trace span carries the same identifiers as
`generation.decision.*` attributes, but not the request state or answers.
Compare native and fallback results only on selected values; calibration and
threshold metrics apply only to `native` records.

The supported primitive outputs are:

- `bool`, optionally with criteria for both outcomes and a `Threshold`.
- An enum or `Literal[...]`, with positional or mapped criteria.
- `float`, with 2–10 ordered score criteria. A bare `float` is invalid.

Use `BooleanDecision`, `ChoiceDecision[E]`, or `ScoreDecision` instead of a
primitive to retain the evidence. `ChoiceDecision.selected` preserves the
backend selection even when a threshold makes `.value` become `None`.

If `decision_model` is omitted, primitive `bool`, enum, `Literal`, and `float`
results fall back to a Predict-style call through `llm`. Composites containing
only those primitive results can also fall back. Detailed decision objects and
any output annotated with `Threshold` require probability evidence, so calling
such a method without a decision model raises `DecisionModelRequiredError`
before making an LLM request.

Standalone strategy functions can configure the same capability directly:

```python
@strategy(DecideStrategy(), decision_model=decision_model)
async def is_urgent(message: str) -> bool:
    """Decide whether the message needs immediate attention."""
    ...
```

When a standalone function is called from an agent and does not set
`decision_model`, it inherits the calling agent's decision model. Without one,
primitive outputs retain the same chat-LLM fallback behavior described above.

Group several outputs in a Pydantic model. Every field then needs its own
`Instructions`; the method docstring becomes shared guidance. The strategy
sends one request and reconstructs the declared model.

`DecisionClient` requires an explicit endpoint and accepts an optional bearer
token, timeout, retry configuration, or caller-owned `httpx.AsyncClient`. The
example uses OpenRouter, but `UnifiedDecisionModel` is the provider-neutral
interface consumed by the runtime. A supplied HTTP client remains owned by the
caller; otherwise call
`await decision_model.aclose()` when done.

Decision models are intentionally distinct from `UnifiedLLM`: a chat client
does not provide calibrated distributions. A configured registry alias with
`client_type: decision` resolves through the same flat model registry:

```yaml
models:
  decisions:
    model_name: typesafe/jev-1.13
    client_type: decision
    api_style: systemone
    endpoint: https://openrouter.ai/api/alpha/decisions
    api_key_env: OPENROUTER_API_KEY
```

Pass the alias at agent or method level:

```python
class SupportAgent(Agent, llm=llm, decision_model="decisions"):
    ...

@strategy(DecideStrategy(), decision_model="decisions")
async def is_urgent(message: str) -> bool:
    """Does this message require immediate attention?"""
    ...
```

`nooa connect` can validate and save this entry with
`--api-style systemone`. It sends one small boolean decision probe and skips
chat-only tool, reasoning, session, and reply-limit checks.
