# LLM Transaction Telemetry and TUI Usage Design

**Status:** Proposed for design review
**Parent:** `docs/design/llm-reasoning-observability-program.md`

## 1. Problem

NOOA currently records some provider usage, but there is no single transaction
contract. Usage is normalized repeatedly and incompletely:

- clients return loose `dict[str, int]` values despite nested details and float
  cost;
- the actor probes provider aliases and emits flat, zero-defaulted
  `LLMComplete` fields;
- the LiteLLM journal has a separate, narrower extractor;
- ATIF/headless/task accumulators each aggregate a different subset;
- trace and TUI surfaces do not expose cache writes, cache outcome, or a complete
  per-call view.

This makes “show me cache hit rate and tokens for every transaction” impossible
to answer reliably.

## 2. Goals

1. Record one canonical, immutable observation for every logical LLM call.
2. Preserve optionality: absent provider data remains absent.
3. Distinguish logical calls from transport attempts and retries.
4. Normalize input/output/reasoning/cache-read/cache-write/cost once.
5. Store enough provider identity and API-style context to interpret the values.
6. Feed events, traces, journal, ATIF/headless, and TUI from the same contract.
7. Present last-call and session totals without confusing cache subsets with
   additive token usage.
8. Keep provider-reported usage passive: no prompt sizing, summarization,
   eviction, or retry decisions derive from it.

## 3. Definitions

### 3.1 Logical call

A logical call is one immutable request payload plus its provider-independent
intent. Retries of byte-equivalent input are attempts of the same logical call.
If NOOA rebuilds the prompt (for example after context recovery), that is a new
logical call because the input changed.

### 3.2 Canonical token fields

- `input_tokens`: all provider-reported input tokens for the call, including
  cached input where the provider reports inclusive totals.
- `output_tokens`: all generated output tokens, including reasoning where the
  provider reports reasoning as a subset.
- `reasoning_tokens`: subset/category of output tokens, not additive.
- `cache_read_input_tokens`: subset/category of input served from a cache.
- `cache_write_input_tokens`: input tokens newly written to a cache where the
  provider reports this separately.
- `total_tokens`: provider total when supplied; otherwise a clearly marked
  derived `input + output` value only when both are known.

### 3.3 Cache outcome

```python
class CacheStatus(StrEnum):
    HIT = "hit"                 # all eligible/reported input served from cache
    PARTIAL_HIT = "partial_hit" # positive cache read, less than total input
    MISS = "miss"               # provider explicitly reports an eligible lookup with no read
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"         # default when zero/absence is ambiguous
```

`cached_tokens == 0` alone is not proof of a miss. Adapters decide whether the
provider performed/reported a cache lookup. `HIT` is only derivable when the
provider reports cache-eligible input tokens or an explicit outcome. Otherwise a
positive cache read is `PARTIAL_HIT` and zero/absence remains `UNKNOWN`.

### 3.4 Cache rates

The universally comparable metric is **cached-input share**, not cache hit rate:

```text
cached_input_share = sum(cache_read_input_tokens) / sum(input_tokens)
```

This includes ineligible/current-turn input in the denominator and therefore
must never be labeled “cache hit rate.” If a provider reports eligible input,
an additional hit rate may be computed:

```text
eligible_hit_rate = sum(cache_read_input_tokens) / sum(cache_eligible_input_tokens)
```

only over calls where both values are known and semantically compatible.
Additionally report coverage:

```text
coverage = calls_with_cache_observation / successful_calls
```

A “70% hit rate at 20% coverage” must not be displayed as simply “70%.”

## 4. Canonical schemas

```python
class UsageSource(StrEnum):
    PROVIDER = "provider"
    GATEWAY_HEADER = "gateway_header"
    DERIVED = "derived"

class UsageObservation(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    reasoning_tokens: int | None = None
    cache_read_input_tokens: int | None = None
    cache_write_input_tokens: int | None = None
    cache_eligible_input_tokens: int | None = None
    cost_usd: Decimal | None = None
    cache_status: CacheStatus = CacheStatus.UNKNOWN
    field_sources: dict[str, UsageSource] = Field(default_factory=dict)
    raw: dict[str, JsonValue] | None = None

class LLMAttemptRecord(BaseModel):
    attempt: int
    started_at: float
    ended_at: float
    first_token_at: float | None = None
    status: Literal["success", "error", "cancelled"]
    status_code: int | None = None
    error_type: str | None = None
    retryable: bool | None = None
    # Optional provider/gateway observation for this attempt. A failed attempt
    # can still be billed; this is distinct from successful response usage.
    usage: UsageObservation | None = None

class LLMCallRecord(BaseModel):
    call_id: str
    generation_id: str | None
    provider: ProviderIdentity
    request_fingerprint: str
    started_at: float
    ended_at: float
    first_token_at: float | None
    status: Literal["success", "error", "cancelled"]
    finish_reason: str | None
    attempts: tuple[LLMAttemptRecord, ...]
    # Usage of the successful terminal response, when any.
    usage: UsageObservation | None
```

`Decimal` prevents float accumulation surprises. JSON serialization uses a
stable string/number convention decided by the wire schema.

## 5. Provider-shape normalization

One normalizer lives at the transport boundary. It accepts provider/SDK usage
and authoritative gateway headers and returns `UsageObservation`.

Fixtures must cover:

| Source | Input | Output | Cache read | Cache write | Reasoning |
|---|---|---|---|---|---|
| OpenAI Chat | `prompt_tokens` | `completion_tokens` | `prompt_tokens_details.cached_tokens` | provider-dependent | `completion_tokens_details.reasoning_tokens` |
| OpenAI Responses | `input_tokens` | `output_tokens` | `input_tokens_details.cached_tokens` | `input_tokens_details.cache_write_tokens` where supported | `output_tokens_details.reasoning_tokens` |
| Anthropic/LiteLLM | `prompt_tokens` / input | completion/output | `cache_read_input_tokens` | `cache_creation_input_tokens` | provider-dependent |
| NVIDIA gateway | response usage | response usage | response details | response/header if authoritative | response details |
| Unknown dict/object | recognized aliases only | recognized aliases only | optional | optional | optional |

Rules:

- tolerate dicts and attribute/Pydantic objects at the adapter edge only;
- preserve legitimate zero values and distinguish them from absent values;
- reject or ignore booleans, negatives, NaN/Inf, and non-numeric strings;
- retain bounded raw usage for diagnostics after secret/size scrubbing;
- endpoint-reported cost/header values may supplement usage; provenance is
  recorded **per field**, not once per observation;
- precedence is provider response field > authoritative gateway header >
  derived value, unless an endpoint profile explicitly overrides it;
- conflicting non-derived values retain the selected value, all source labels,
  and a bounded diagnostic rather than silently summing them;
- no consumer performs alias probing after normalization.

## 6. Lifecycle and double-counting

The lifecycle is above the transport adapter:

```text
before_llm_call(call snapshot)
  before_llm_attempt(attempt 1)
  after_llm_attempt(error retryable)
  before_llm_attempt(attempt 2)
  after_llm_attempt(success + usage)
after_llm_call(success + exactly one UsageObservation)
```

Invariants:

- exactly one terminal `LLMCallCompleted` per logical call;
- attempt failures/latencies remain observable;
- terminal successful usage enters **successful-response totals** exactly once;
- provider/gateway-reported attempt usage (including failed, timed-out, or
  cancelled attempts) enters separate **observed billed-attempt totals** exactly
  once; never merge the two into a single cost without a label;
- cancellation emits one terminal cancelled call;
- streams accumulate usage and end once; abandoned streams do not fabricate
  success;
- prompt rebuild/context recovery creates a new call ID and fingerprint.

## 7. Event and persistence strategy

Replace or evolve flat `LLMComplete` carefully:

1. Introduce the canonical schema and a new optional nested `usage`/`provider`
   representation.
2. Keep legacy flat accessors (`prompt_tokens`, `completion_tokens`,
   `cached_tokens`, `reasoning_tokens`, `cost_usd`) during one compatibility
   release.
3. New events serialize as JSON in the existing event store; optional fields
   need no SQLite migration.
4. Version journal token JSON (`{"version": 2, ...}`) and dual-read legacy keys.
5. Do not add queryable SQLite columns until a real query requirement exists;
   journal JSON is sufficient for reconstruction and export.

Recommended event name remains `LLMComplete` for compatibility, but its public
payload should carry:

- provider identity;
- logical `call_id` and generation ID;
- terminal status/timing/attempt count;
- optional `UsageObservation`;
- tool calls and bounded reasoning observability.

## 8. Aggregation

Add a session-local `UsageAccumulator` subscribed to terminal call events. It
keeps immutable call records plus a compact aggregate:

```python
class UsageTotals(BaseModel):
    successful_calls: int
    failed_calls: int
    calls_with_usage: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_input_tokens: int
    cache_write_input_tokens: int
    cost_usd: Decimal | None
    cache_observation_calls: int
    billed_attempt_input_tokens: int
    billed_attempt_output_tokens: int
    billed_attempt_cost_usd: Decimal | None
    attempts_with_usage: int
```

Successful-response totals and billed-attempt totals are separate views. The
latter may include work that produced no successful response and is explicitly
labeled “observed billed attempts”; provider omission means it is still a lower
bound. Totals sum each field independently. `reasoning_tokens` and cache fields are
never added to input/output. If cost coverage is partial, show the known total
plus coverage; never imply it is complete.

## 9. TUI experience

### 9.1 `/usage`

A new read-only command:

```text
/usage                # session totals + last call
/usage last           # last completed call detail
/usage session        # aggregate
/usage reset          # reset display accumulator only; persisted events remain
/usage json           # JSON for diagnostics
```

Example:

```text
Last call  gpt-5.6-sol · Responses · 2.8s · 2 attempts
Input      42,310  (cache read 31,232 · 73.8%)
Output      1,248  (reasoning 812)
Cache       partial hit · write 4,096
Cost        $0.0312 (provider reported)

Session     17 calls · input 411k · output 22k · cached-input 62.4%
Billed      19 attempts · $0.0341 observed
Coverage    usage 17/17 · cache 14/17 · attempt cost 12/19
```

### 9.2 Toolbar

Add a `usage` provider to the existing toolbar registry. Compact session label
using monochrome arrows (DECIDED, D-05):

```text
↑411k ↓22k ↻62% (14/17)
```

- `↑` input tokens (provider-reported, session total)
- `↓` output tokens (provider-reported, session total)
- `↻` cached-input share (62% of observed input served from cache), with
  coverage `(14/17)` = calls with a cache observation / successful calls

**Cache symbol (D-05).** `↻` (U+21BB, clockwise open-circle arrow) is the
recommended glyph: it reads as "served again / reused" — exactly what a cache
hit is — is monochrome, renders in essentially every terminal font, and is
visually distinct from `↑`/`↓`. ASCII fallback when the terminal/locale profile
rejects non-ASCII: `c` (label reads `↑411k ↓22k c62% (14/17)`). Alternatives
considered: `◆` (ubiquitous but semantically blank), `⟳` (same meaning, slightly
weaker font coverage), `@`/`#` (safe but unreadable as cache).

**Cache-segment hiding.** The cache segment is emitted only when (a) the
current endpoint's capability profile reports cache support and (b) at least
one session call produced a cache observation. Endpoints without caching (some
inference-API models) show simply:

```text
↑411k ↓22k
```

— the segment is absent, not `↻—` or `↻0%`. Capability is decided once per
model switch, not per call.

The toolbar shows session successful-response aggregate; `/usage last` gives
transaction detail and `/usage session` distinguishes successful response usage
from observed billed-attempt usage. Context window usage (`ctx 42%`) remains a
separate concept and toolbar item.

### 9.3 Event explorer and viewer

- Fix `LLMComplete` preferred fields to match the actual/canonical schema.
- Trace Explorer reads canonical nested usage plus legacy attributes.
- React LLM card shows input/output/reasoning/cache read/cache write/cost,
  attempt count, latency, and field provenance.
- All UI values say “provider reported” where appropriate.

## 10. Redaction and privacy

- Token counts and aggregate timings are safe observability metadata.
- Raw usage may contain provider-defined extras; scrub and size-cap it.
- **Product decision (overrides the earlier draft): reasoning is exported by
  default.** OTLP spans, the journal, trace downloads, bug reports, and normal
  Event Explorer previews include reasoning content because tracing must
  retain everything that was sent to the model. `export_reasoning=false`
  suppresses reasoning bodies for compliance-sensitive deployments.
- Even with export on, keep reasoning out of *accidental* channels: exception
  tracebacks, debug logs, and `repr()` output. Those are not tracing surfaces.
- Report whether metrics cover calls whose reasoning body was redacted; never
  require reasoning content to compute usage.

## 11. Acceptance criteria

1. Fixtures normalize all listed provider shapes from dict and object forms.
2. Missing fields remain `None`; explicit zero remains zero.
3. Cache status never calls an unobserved zero a miss; cached-input share is
   not labeled hit rate, and eligible hit rate is shown only with an eligible
   denominator or explicit provider status.
4. One successful logical call contributes successful response usage exactly
   once despite retries; provider-observed failed/timeout/cancelled attempt
   usage is retained separately and never double-counted.
5. Cache writes are distinct from cache reads and input total.
6. Mixed-source fixtures prove field-level provenance and precedence for
   provider tokens plus gateway-header cost.
7. Event, journal, ATIF/headless, trace, and TUI consume the canonical schema.
8. `/usage` and toolbar pass deterministic rendering tests with full, partial,
   and zero coverage.
9. Old events/traces/journal rows continue to render.
10. No usage field is read by context control/summarization code.
11. Export tests prove reasoning bodies appear in journal/OTLP/trace-download/
    Event Explorer by default, and are suppressed when `export_reasoning=false`.
