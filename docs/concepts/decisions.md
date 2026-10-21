# Decision models

`DecideStrategy` uses an agent's decision model when one is configured. Method
arguments form the decision state, the return annotation declares one or more
questions, and the model returns probabilities or score distributions. The
agent's chat LLM remains available for its other generation methods.

```python
import os
from enum import StrEnum
from typing import Annotated

from nooa import Agent, Criteria, DecisionClient, DecideStrategy, strategy
from nooa.unifiedllm.registry import get_llm_client


class Department(StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"


DepartmentChoice = Annotated[
    Department,
    Criteria(
        by_value={
            Department.BILLING: "Payments, invoices, and refunds.",
            Department.TECHNICAL: "Bugs, outages, and integrations.",
        }
    ),
]


llm = get_llm_client("gpt-5-mini")
decision_model = DecisionClient(
    "typesafe/jev-1.13",
    endpoint="https://openrouter.ai/api/alpha/decisions",
    api_key=os.environ["OPENROUTER_API_KEY"],
)


class Router(Agent, llm=llm, decision_model=decision_model):
    @strategy(DecideStrategy())
    async def department(self, message: str) -> DepartmentChoice:
        """Choose the team that should handle the message."""
        ...

    async def draft_reply(self, message: str) -> str:
        """Draft a concise and helpful response."""
        ...
```

The decision model chooses among the enum's members and reads a description,
or criterion, for each one. `DepartmentChoice` declares those criteria once,
next to the enum, so any method can return it.
[Describe enum options](#describe-enum-options) explains the rules.

## Describe enum options

Two annotations shape a choice:

- `Instructions(...)` describes the question: what to decide. For a method
  that returns a single value, the method docstring is the instruction, so you
  usually don't need `Instructions`.
- `Criteria(...)` describes the possible answers: what each option means.

The model is sent each member's name as the option ID, with its criterion as
the description.

### Map criteria to members

Describe an enum with `Criteria(by_value=...)`, keyed by the enum members
themselves, as `DepartmentChoice` does above:

- The keys must be the members (`Department.BILLING`), not their names or
  values.
- The mapping must cover every non-alias member exactly. Adding a member to the
  enum without adding its criterion raises a `ValueError` when the method is
  called, before any request is sent.
- Each description stays attached to its member, so reordering the enum is
  safe.

A criterion can also be a JSON object or array when plain text is not enough:

```python
Department.BILLING: {
    "what": "Payments, invoices, and refunds.",
    "not_for": ["Outages that block payments; choose TECHNICAL."],
},
```

### Reuse the annotation

A named annotation such as `DepartmentChoice` can be reused wherever the
choice appears.

To keep the evidence, wrap `ChoiceDecision` in the same criteria:

```python
from nooa import ChoiceDecision

DEPARTMENT_CRITERIA = Criteria(
    by_value={
        Department.BILLING: "Payments, invoices, and refunds.",
        Department.TECHNICAL: "Bugs, outages, and integrations.",
    }
)

DepartmentChoice = Annotated[Department, DEPARTMENT_CRITERIA]
DepartmentDecision = Annotated[ChoiceDecision[Department], DEPARTMENT_CRITERIA]
```

To abstain below a threshold, make the reused annotation nullable:

```python
@strategy(DecideStrategy())
async def department(
    self, message: str
) -> Annotated[DepartmentChoice | None, Threshold(0.8)]:
    """Choose the team that should handle the message."""
    ...
```

In a composite result, each field adds its own `Instructions` to the reused
choice:

```python
class Triage(BaseModel):
    department: Annotated[
        DepartmentChoice,
        Instructions("Select the team that should own the request."),
    ]
```

The field's `Instructions` say what to decide for that field, and
`DepartmentChoice` says what each department means.

### Enum docstrings and other sources are not read

An enum used as a decision result needs explicit `Criteria`. Without it,
calling the method raises a `TypeError` that asks for
`Criteria(by_value=...)`, before any request is sent. NOOA does not read
criteria from the enum's docstring (including an `Attributes:` section),
annotations on individual members, comments, custom attributes, or member
values. Keeping criteria in the annotation means that editing an enum's
documentation can't change a decision's meaning, and the same enum can have
different criteria in different decisions.

`Literal[...]` choices are different: with no `Criteria`, each literal value is
its own description. To describe them, use
`Criteria(by_value={"low": ..., "high": ...})`, keyed by the literal values, or
positional criteria in the literal's order.

## Decision-only agents

An agent whose generated methods all use `DecideStrategy` does not need a chat
LLM. Configure only the decision model:

```python
class SupportRouter(Agent, decision_model=decision_model):
    @strategy(DecideStrategy())
    async def department(self, message: str) -> DepartmentChoice:
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
    ) -> Annotated[DepartmentDecision, Threshold(0.8)]:
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
`Annotated[DepartmentChoice | None, Threshold(0.8)]`.

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
Resolved context values and visible event data must be JSON-compatible.
Arguments must also be JSON-compatible, except for explicitly permitted,
top-level named `Image` inputs (see [Native image inputs](#native-image-inputs)).
Unsupported JSON-state values fail with the offending parameter, context-block
name, or event field/index instead of being stringified or truncated silently.
This naming guarantee does not cover every native image preflight error. Native
media in context or event data is not collected; hidden `repr=False` event
attachment slots remain excluded.

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
        DepartmentChoice,
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
switches to the chat LLM on its own. To use a chat model, configure one
explicitly as described below.

### Use a chat model as a decision model

`DecisionModel.from_llm` wraps a chat client, or a configured chat alias, as a
decision model:

```python
from nooa import DecisionModel

chat_decisions = DecisionModel.from_llm(llm)


class Router(Agent, decision_model=chat_decisions):
    @strategy(DecideStrategy())
    async def department(self, message: str) -> DepartmentChoice:
        """Choose the team that should handle the message."""
        ...


# Use a native decision model for one call instead.
await router.department(message, decision_model=decision_model)
```

The chat model receives the same state and questions a decision model would,
and answers with structured output: one option, a true/false value, or a score
level per question. It provides no probabilities, so it supports only primitive
results (`bool`, enums, `Literal`, scored `float`) and composites of them. A
method that returns `BooleanDecision`, `ChoiceDecision`, or `ScoreDecision`, or
uses `Threshold`, raises `DecisionModelRequiredError` before any request.
Scores are whole levels. A reply that does not match the schema is retried
once, then fails with `InvalidDecisionResponseError`.

Custom decision models can subclass `DecisionModel` and set
`provides_probabilities = False` to get the same checks. Clients that implement
only `UnifiedDecisionModel` are treated as providing probabilities.

Every decision call stores a `DecisionRecord` with the request state,
normalized questions, answers, and provenance:

- `decision_source`: `native` for decision models, or `llm` for a model
  created with `DecisionModel.from_llm`. Compare `llm` records with native
  ones only on selected values; their recorded distributions are one-hot.
- `question_digest`: a SHA-256 digest of the normalized question names,
  instructions, criteria, and candidate IDs. It is computed after
  `decision_call` middleware, so it reflects the request actually sent.
- `requested_model` and `resolved_model`: the configured model and the model
  the endpoint reports it served.

The generation trace span carries the same identifiers as
`generation.decision.*` attributes, but not the request state or answers.

The supported primitive outputs are:

- `bool`, optionally with criteria for both outcomes and a `Threshold`.
- An enum with mapped criteria (see [Describe enum options](#describe-enum-options)),
  or a `Literal[...]`.
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
    async def department(self, message: str) -> DepartmentDecision:
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
`--api-style systemone`. Minimal setup sends one small typed boolean routing probe and skips
chat-only tool, reasoning, session, and reply-limit checks.

### Use OpenAI's Decisions API

`DecisionClient` speaks two wire formats. The default, `api_style="systemone"`,
is the System One API used by TypeSafe's Jev. `api_style="openai"` speaks
OpenAI's `/v1/decisions`:

```python
decision_model = DecisionClient(
    "gpt-6-luna",
    endpoint="https://api.openai.com/v1/decisions",
    api_key=os.environ["OPENAI_API_KEY"],
    api_style="openai",
)
```

Or in the registry:

```yaml
models:
  openai-decisions:
    model_name: gpt-6-luna
    client_type: decision
    api_style: openai
    endpoint: https://api.openai.com/v1/decisions
    api_key_env: OPENAI_API_KEY
```

Declarations, detailed results, thresholds, raw responses, and
`DecisionRecord` provenance work the same way with either format. The client
adapts the request to OpenAI's shapes:

- With no attachments, the decision state is sent as JSON text in `input`,
  preserving the original text-only contract. With permitted images, `input`
  is a user-message list with native image blocks (see below).
- Questions are sent as a named array, in order. Boolean outputs become
  `predicate` questions; their true/false criteria are appended to the
  instructions, because predicates have no criteria field.
- Choice criteria become `choices` and score criteria become `levels`, labelled
  `"0"`, `"1"`, and so on. Structured criteria and instructions are sent as JSON
  text, because OpenAI accepts only text.
- Score legends are rebuilt from the declared criteria. `response_id` comes
  from the `x-request-id` response header.

OpenAI can refuse an individual question. The client then raises
`DecisionRefusedError`, whose `questions` names the refused outputs and whose
`raw` holds the complete response, including the answers to other questions.
The call's `DecisionRecord` is stored as a failure.

`nooa connect` validates and saves these entries with `--api-style openai`,
sending one predicate routing probe for minimal setup, plus fresh typed smoke cases for full setup.

This uses OpenAI's native decision endpoint. `DecisionModel.from_llm(...)` with
an OpenAI chat model is different: it uses the chat API and provides no
probabilities.


### Native image inputs

Image permission is **explicit and local to the decision route**. It is not
inferred from the model name, endpoint hostname, chat LLM, or a catalogue.
The default `media_inputs=()` is text-only: passing an `Image` raises before
HTTP. Only `media_inputs=["image"]` is supported for native decisions; audio,
video and PDF/file decision contracts are unverified and rejected.

For an OpenAI/Hub Decisions endpoint implementing the documented OpenAI
Decisions input contract:

```python
from nooa.media import Image

vision_decisions = DecisionClient(
    "YOUR_DECISION_MODEL",
    endpoint="https://YOUR_VERIFIED_HOST/v1/decisions",
    api_style="openai",
    media_inputs=["image"],
    api_key=os.environ["DECISION_API_KEY"],
)

class Inspector(Agent, decision_model=vision_decisions):
    @strategy(DecideStrategy())
    async def damaged(self, photo: Image, claim: str) -> bool:
        """Does the photo show the claimed damage?"""
        ...

await Inspector().damaged(Image.from_file("return.png"), "chipped rim")
```

Binding uses the same `CurrentCall.bound_parameters()` source as Predict.
Only direct named `Image` arguments are collected. Lists/dicts/models holding
native Media are rejected, including before custom Pydantic serializers can
turn them into text. No PIL conversion, recursive promotion, arbitrary/private
slot traversal, question-media interpolation, or context/event media collection is
performed. No attachments are harvested from hidden slots. Media already
deliberately converted to plain strings/dicts is
ordinary JSON data, **not** an attachment or proof of visual inspection.

The normalized `DecisionRequest(state, questions, attachments=())` remains
compatible with existing two-argument construction. `DecisionAttachment`
(importable from `nooa.decisions`) retains the encoded image URL/bytes, MIME,
optional vendor `detail`, alias, and all associated input names. JSON state
contains token strings instead of image payloads:

```python
{"inputs": {"photo": "<decision-image-0>", "claim": "chipped rim"}}
```

Passing the same Image object to several named inputs sends one attachment
with several associations; distinct objects are not content-hash deduplicated.
For direct or middleware-rewritten requests, each declared input name must map
to its exact alias in `state.inputs`; unrelated substrings, string/list states,
and undeclared named uses of an attachment alias are rejected.

The OpenAI wire body uses `input=[{"role": "user", "content": [...]}]`:
first `{"type": "input_text", "text": JSON_STATE}`, then an alias text block
and `{"type": "input_image", "image_url": FLAT_STRING}` for each image.
`detail` may be `low`, `high`, `auto`, or `original`; other vendor metadata is
rejected instead of overriding trusted fields. PNG/JPEG/WebP/GIF MIME values
are allowed by this bounded normalization (actual model support may be
narrower). Image URLs may be locally encoded base64 data URLs or
credential-free HTTP(S) URLs. NOOA never downloads remote images; the OpenAI
provider may fetch a permitted URL. Supply public URLs only.

#### Direct TypeSafe experimental multipart transport

System One JSON proxies keep the old JSON request shape. Even with image
permission enabled, attachments on that transport raise a clear **unverified
transport** error; image data is never disguised as JSON text. Direct TypeSafe
multipart is selected **only by explicit configuration**, not by hostname:

```yaml
models:
  direct-typesafe-vision:
    client_type: decision
    model_name: jev-preview
    api_style: systemone
    endpoint: https://YOUR_VERIFIED_TYPESAFE_HOST/YOUR_SYSTEM_ONE_PATH
    api_key_env: TYPESAFE_API_KEY
    media_inputs: [image]
    decision_image_transport: typesafe_multipart
    typesafe_danger_zone_images: true
```

These same three settings are accepted by `DecisionClient(...)`. Both image
permission and the experimental boolean opt-in are required for an image
request. `decision_image_transport` defaults to `json`; `typesafe_multipart`
is incompatible with `api_style="openai"`.

The multipart body's **first** part is named `request`, with Content-Type
`application/json`, containing `model`, unchanged `state` and `questions`, and
`assets={ALIAS: {"type": "image", "part": "asset-0"}, ...}`. Subsequent binary
parts are named and have filenames `asset-0`, `asset-1`, etc., with PNG/JPEG
Content-Type and their original bytes. State references the aliases as plain
strings. Requests with images include `TypeSafe-Danger-Zone-Images: true`.
Text-only calls still send ordinary JSON and do not add this header.

Multipart accepts **local base64 data URL bytes only**, with matching PNG/JPEG
MIME and file signatures; remote URLs are rejected without fetching (no SSRF
fetch path). There is no verified TypeSafe `detail` hint contract, so a supplied
hint is rejected rather than lost. Signature checks are not full image decoding.

Preflight is bounded to 16 images, 20 MiB decoded bytes per image, 50 MiB total
local image bytes, and 8,192 characters per HTTP(S) URL. Malformed base64,
conflicting/unsupported MIME, empty/oversized images, bad URIs, unsupported
media, permission denial and incompatible transports fail **before retries or
HTTP sends**. The limits apply to local bytes, not provider-fetched remote
content. Retries replay repeatable bytes, and caller-owned HTTP clients retain
their ownership and authentication. Multipart overrides an inherited JSON
Content-Type for that request only.

Registry endpoint/model/API-style/caller-client replacement clears inherited
image permissions, encoding and experimental opt-in; redeclare them explicitly
for the replacement route. `DecisionModel.from_llm(...)` remains JSON-only and
explicitly rejects decision attachments; chat media permission does not change
this, and text-only calls on multimodal chat routes remain supported.

`nooa connect` supports native decision onboarding and image checks; see
[Check native decisions with Connect](#check-native-decisions-with-connect).
Offline tests verify the wire contract, not live endpoint acceptance or visual accuracy. Decision
records currently retain state aliases, not image bytes/URLs or an attachment
content digest; applications needing reproducible visual audits must retain
assets separately.

Contract references:
[TypeSafe SDK v0.7.0 assets](https://raw.githubusercontent.com/typesafe-ai/typesafe-sdk-js/v0.7.0/src/assets.ts),
[TypeSafe SDK image opt-in](https://github.com/typesafe-ai/typesafe-sdk-js/blob/v0.7.0/README.md#images),
[OpenAI SDK v3.35.1 decision input image](https://raw.githubusercontent.com/openai/openai-python/v3.35.1/src/openai/types/decision_input_image_param.py).

### Check native decisions with Connect

Use `nooa connect MODEL --stage decision --endpoint FULL_URL --api-style openai`
(or `systemone`) to run fresh bounded fixtures through an actual Agent
`DecideStrategy`. One composite result checks boolean, literal choice and
ordinal score reconstruction; positive and negative boolean fixtures check
simple semantics. Acceptance is not the same as answering correctly. Schema
validation includes finite normalized distributions and weighted score
consistency. There are no retries, and each request has a deadline and shares
the approved reservation budget.

`--stage media --test-media-inputs image` runs that baseline before inspecting a
generated red PNG through a named `Image` parameter. Passing requires actual
wire attachment evidence, typed access, and the correct known color. OpenAI
Decisions uses `input_image`. Direct TypeSafe requires explicit
`--decision-image-transport typesafe_multipart --typesafe-danger-zone-images`;
System One JSON proxy image testing is rejected as unverified. Native decision
audio, video and PDF transports are not implemented.

`--media-inputs image` is a manual permission declaration, not a test. Connect
saves permission, image transport and explicit TypeSafe opt-in in runtime YAML,
separately from sanitized case outcomes and usage. It never saves encoded
images, raw responses or test questions there. See [Connect](../model-connect.md).
