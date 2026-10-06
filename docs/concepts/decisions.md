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

## Decision-only agents

An agent whose generated methods all use `DecideStrategy` does not need a chat
LLM. Configure only the decision model:

```python
class SupportRouter(Agent, decision_model=decision_model):
    @strategy(DecideStrategy())
    async def department(self, message: str) -> Department:
        """Choose the team that should handle the message."""
        ...


router = SupportRouter()
try:
    print(await router.department("I was charged twice for my invoice."))
finally:
    await decision_model.aclose()
```

Calling a non-decision generation method on such an agent, or reading
`agent.llm`, raises a `RuntimeError` that asks for `llm=...`. Without a decision
model, an agent still requires a chat LLM.

[`examples/quickstart/16_decisions.py`](../../examples/quickstart/16_decisions.py)
is a complete, runnable version with thresholded abstention and batching:

```bash
OPENROUTER_API_KEY=... uv run python examples/quickstart/16_decisions.py
```

## Accept or abstain with a threshold

A `Threshold` turns a decision into an accept-or-abstain rule. Use a detailed
result type to see why a value was accepted or rejected:

```python
from nooa import ChoiceDecision, Threshold


class SupportRouter(Agent, decision_model=decision_model):
    @strategy(DecideStrategy())
    async def confident_department(
        self, message: str
    ) -> Annotated[ChoiceDecision[Department], Threshold(0.8)]:
        """Choose the team that should handle the message."""
        ...


decision = await router.confident_department(message)
if decision.value is None:
    send_to_manual_review(message, suggested=decision.selected)
```

The result fields are:

- `selected`: the option the model ranked highest. It is always set.
- `probabilities`: the probability of every option.
- `value`: `selected` when `probabilities[selected] >= threshold`, otherwise
  `None`.
- `confidence`: a separate confidence value reported by the service. It is
  useful for analysis but is **not** used for the threshold.

So a result can have `probabilities[selected] == 0.54` and `value is None` for
`Threshold(0.8)`, regardless of `confidence`. For booleans, `BooleanDecision`
exposes `probability_true`, and `value` is `probability_true >= threshold`
(default `0.5`). A primitive enum or `Literal` result with a `Threshold` must
include `None` in its annotation, for example
`Annotated[Department | None, Threshold(0.8)]`.

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

A single call can also override the model, exactly like `llm=`:

```python
await agent.detect_fraud(transaction, decision_model=review_decision_model)
```

Resolution is call argument, then method decorator, then agent, then the calling
parent agent. Each level accepts a client or a configured alias; the call
argument and method decorator also accept a callable that receives the agent. A
method parameter named `decision_model` is passed to the method instead of
selecting a model.

A `DecideStrategy` method always uses a decision model. If none resolves, the
call raises `DecisionModelRequiredError` before making any request; it never
switches to the chat LLM.

Every decision call stores a `DecisionRecord` with the request state,
normalized questions, answers, and provenance:

- `decision_source`: currently always `native`.
- `question_digest`: a SHA-256 digest of the normalized question names,
  instructions, criteria, and candidate IDs. It is computed after
  `decision_call` middleware, so it reflects the request actually sent.
- `requested_model` and `resolved_model`: the configured model and the model
  the endpoint reports it served.

The generation trace span carries the same identifiers as
`generation.decision.*` attributes, but not the request state or answers.

The supported primitive outputs are:

- `bool`, optionally with criteria for both outcomes and a `Threshold`.
- An enum or `Literal[...]`, with positional or mapped criteria.
- `float`, with 2–10 ordered score criteria. A bare `float` is invalid.

Use `BooleanDecision`, `ChoiceDecision[E]`, or `ScoreDecision` instead of a
primitive to retain the evidence. `ChoiceDecision.selected` preserves the
backend selection even when a threshold makes `.value` become `None`.

### Inspect the raw API response

For advanced cases, such as reading fields a decision server adds beyond the
standard answers, opt in when you create the strategy:

```python
class SupportAgent(Agent, decision_model=decision_model):
    @strategy(DecideStrategy(include_raw_response=True))
    async def department(self, message: str) -> ChoiceDecision[Department]:
        """Choose the team that should handle the message."""
        ...


decision = await agent.department(message)
decision.raw_response  # read-only mapping of the decision API's response body
```

`raw_response` is set on detailed results (`BooleanDecision`,
`ChoiceDecision`, `ScoreDecision`); primitive results return only the value. In
a composite, every detailed field shares the same response object. The same
body is stored in the call's `DecisionRecord.raw_response`, and never in trace
attributes. NOOA does not interpret it. It is excluded from equality, `repr`,
and `model_dump()`, and it is `None` for responses created by `decision_call`
middleware and for decision clients that do not provide a
raw body.

Standalone strategy functions can configure the same capability directly:

```python
@strategy(DecideStrategy(), decision_model=decision_model)
async def is_urgent(message: str) -> bool:
    """Decide whether the message needs immediate attention."""
    ...
```

When a standalone function is called from an agent and does not set
`decision_model`, it inherits the calling agent's decision model. Without one,
it raises `DecisionModelRequiredError`. A call argument `decision_model=`
overrides the decorator, as for agent methods; standalone functions accept a
client or alias there, but not a callable.

Group several outputs in a Pydantic model. Every field then needs its own
`Instructions`; the method docstring becomes shared guidance. The strategy
sends one request and reconstructs the declared model.

`DecisionClient` requires an explicit endpoint and accepts an optional bearer
token, timeout, retry configuration, or caller-owned `httpx.AsyncClient`. The
example uses OpenRouter, but `UnifiedDecisionModel` is the provider-neutral
interface consumed by the runtime. A supplied HTTP client remains owned by the
caller; otherwise call
`await decision_model.aclose()` when done.

`api_key` is attached only to the HTTP client that `DecisionClient` creates. When
you pass `client=`, configure authentication on that client yourself; `api_key`
is then ignored:

```python
http = httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"}, timeout=60)
decision_model = DecisionClient("typesafe/jev-1.13", endpoint=endpoint, client=http)
```

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
