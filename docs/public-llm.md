# Public LLM loading (incremental migration)

`load_llm` returns the provider-neutral `LLM` shell. `get_llm_client` and
`client_from_config` continue returning legacy clients for runtime and Connect;
this change does not switch the runtime's global default.

```python
from nooa.unifiedllm import LLMRequest, load_llm

async with load_llm("my-alias", temperature=0.2, reasoning_level="low",
                    max_output_tokens=512, timeout=30) as llm:
    response = await llm.acall(LLMRequest(
        messages=[{"role": "user", "content": "Explain leases briefly."}],
        tools=[], tool_choice="auto", output_type=None,
    ))
```

The request has **exactly four fields**: `messages`, `tools`, `tool_choice`,
`output_type`. Unknown fields are rejected. Tool choice is a neutral string:
`auto`, `required`, `none`, or the name of one supplied tool. Native mappings,
reasoning selection, sampling, timeout and reply caps are not request fields.

Settings belong at load time. Omission inherits registry defaults; explicit
`None` clears sampling/cap or disables a managed reasoning selection while
retaining raw native defaults. `reasoning_default` remains descriptive metadata.
Settings are frozen in each view. Pool-sharing scalar sampling views cover
`temperature`, `top_p`, `seed`, `frequency_penalty` and `presence_penalty`;
other native defaults conservatively remain in the route key. Reply caps under
`extra_body` share with top-level cap views, without discarding unrelated native
defaults. Registry defaults retain the existing resolver's recognized fields.
Levels are route-local declarations, not
inferred from model names. Loading validates supported selection and positive
reply caps against known facts without provider requests. Native configuration
remains an adapter/registry concern, not a public settings model or LLM config.
Public built-in SDK attempts have zero retries; the shell owns retry policy.
Legacy clients retain their existing per-call knobs and explicit retry overrides.
A legacy facade over a strict custom adapter rejects settings it cannot express,
rather than silently losing them.

## Pools and lifetime

Repeated same-route loads with different sampling, caps, selected levels or wire
timeouts share the live built-in connection pool, cached capabilities and numeric
calibration. Loads take one registry-entry snapshot; a concurrent reload cannot
validate one entry and construct another. Routing and alias precedence still use
the legacy resolver. Only the existing LiteLLM Chat and Responses adapters are
supported; decision aliases use the decision loader instead.

Each loaded view and in-flight call leases its pool. `close()` / `aclose()` and
sync / async context managers release the view idempotently. Closing one view
cannot close another live view's pool. The final lease closes the owned HTTP
pool once. Caller-supplied SDK clients are borrowed and are **not** closed or
mutated. Directly supplied adapters are also borrowed. In an event loop use
`await llm.aclose()` to await final async disposal; synchronous `close()` schedules
it. Use explicit disposal: abandoning a view is not a promise of awaited cleanup.

The pool cache is weak and its keys are process-private keyed digests, never
persisted credentials. Cross-load sharing requires an explicit normalized HTTP(S)
endpoint plus an explicit resolved credential or weakly identifiable supplied SDK.
Settings-only changes share; route/native defaults and cached registry facts
participate in the key. Environment-dependent routes, ambiguous endpoints,
unsupported opaque carriers, cyclic/oversized configurations and custom transport
configurations are isolated instead of guessed equivalent. Reloaded declarations
can therefore create a separate pool. This is a deliberately conservative cache,
not a universal provider/client cache.

## Estimated context fit

An owned built-in public attempt prepares its wire projection once, including
media, tools, cache markers, native replay and Responses instructions. The adapter
reports only raw numeric input estimate and estimate quality to the shell. The
shell uses the **same route's EMA**, conservatively never below the raw estimate.
For known context window `W`, a positive selected reply cap `R`, and a full
projection estimate `I`, the wire cap is at most
`min(R, known_output_ceiling_if_any, W - I)`. No estimated reply room raises a
typed `context_window` failure before transmission. Only the prepared cap aliases
are changed; frozen settings, history, pool and projection are not rebuilt.

Unknown windows, absent/None caps, unavailable estimates and text-only fallback
estimates do not establish a context-fit guarantee. Absent caps preserve adapter
behavior; no reply limit is invented from a server default. Even a full projection
is **not proof** of actual provider tokenization. Delivered positive input
estimate/usage pairs update route EMA once, including replies that later fail to
parse; observation buffering is independent of calibration ownership. Native
provider payloads and tokenization stay on the adapter side.

This is a coherent loader-first slice, not full design-v27 migration: the existing
portable mapping-message/media surface, legacy compatibility paths, logical
accounting and provider error normalization remain incremental work.
