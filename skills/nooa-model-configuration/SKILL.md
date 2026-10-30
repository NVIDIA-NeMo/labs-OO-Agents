---
name: nooa-model-configuration
description: Understand NOOA model registry precedence and saved settings, client selection, reply caps, reasoning levels, caching and replay semantics. For model onboarding, checks and writes use the canonical nooa-connect skill.
---

# NOOA registry settings and semantics

For onboarding, network checks, credentials, budgets, diagnostic handoffs and
configuration writes, follow [nooa-connect](../nooa-connect/SKILL.md), the
canonical permission-first headless guide. Do not configure a model merely
because you are authoring an agent or have a key available. Use existing aliases
where appropriate; obtain scoped permission before changes or checks.

This skill explains saved registry settings, precedence and client selection,
not an alternative Connect workflow. See
[Model configuration](../../docs/model-configuration.md) for file discovery and
[Connect reference](../../docs/model-connect.md) for guided human setup/library APIs.
Copied skills need sibling skills and the source docs matching the installed
release/commit; use absolute installation references from a diagnostic handoff.

## Locate the effective configuration

- Inspect `nooa.llm_config.llm_config_chain()` and the effective alias source.
  An explicit output file is not necessarily in that chain, and a higher-priority
  entry may shadow it. `reload_registry(Path(...))` loads a specific file; local
  loading does not prove endpoint support or credential validity.
- Preserve unrelated aliases/comments and review the selected alias's custom
  fields: stage save replaces that alias as a whole, not a merge.
- Registry entries contain key variable names, never literal credentials.
  Environment variables override layered `secrets.yaml`; credential persistence
  requires separate consent. Follow the canonical skill for headless key use.

**Windows:** run Connect inside WSL 2. Native Windows refuses `nooa` unless
`NOOA_SKIP_PLATFORM_CHECK=1` is set; with it, `nooa connect` works and is tested.

## Keep settings truthful

- Use the endpoint's exact model ID and supported interface, not guesses from
  its name. Chat uses `client_type: completion`; Responses uses
  `client_type: responses`; Anthropic Messages uses the completion client and
  its Anthropic route prefix. Connect normalizes provider prefixes and base URLs.
- New Connect entries carry `transport: direct`. Runtimes with direct SDK support
  use it to bypass LiteLLM; older runtimes ignore it. Preserve explicit transport
  choices when editing. Check the installed runtime before describing probes as
  direct-transport validation: the saved field alone is not evidence.
- Explicit user choices win; endpoint-reported limits take precedence over
  public catalogue suggestions. Do not add input/output limits to invent a total
  context window. Record unknown context capacity as unverified.
- The saved reply cap is `max_tokens` for all three interfaces; Responses maps
  it to `max_output_tokens` on the wire. Catalogue output ceilings are metadata,
  not recommendations. Keep the selected cap below the known context window.
  Configured checks send this cap, including a selected level’s override; only
  initial interface discovery uses a smaller cap. Insufficient token budget
  skips checks rather than lowering their caps. If no check sent the saved cap, report it
  as unverified. Thinking and the final answer can share the cap; a level-specific
  thinking budget needs answer headroom.
- Connect defaults Responses entries to `store: false` and
  `include: [reasoning.encrypted_content]`. Preserve explicit choices.
  `include: []` is the persisted opt-out for a route that rejects the field;
  unrelated errors are not evidence of rejection. Do not hide these fields or
  reply caps inside `extra_body`.
- Caching defaults to `auto`: use the default cached renderer's stable boundary,
  not a test-only cache override. Other Chat routes may cache implicitly;
  Responses and recognized Anthropic routes have explicit marker mappings.
  `cache_breakpoint: null` opts out of NOOA markers, not provider implicit caching.

## Media declarations and diagnostic evidence

Each model's `media_inputs` declares accepted `image`, `audio`, `video`, `pdf`
formats; omitted/null/empty means text-only. `File` supports PDFs with
`application/pdf`. Use [nooa-media](../nooa-media/SKILL.md) for typed examples and
[nooa-connect](../nooa-connect/SKILL.md) for authorized checks before changing
these settings. Declaration alone is not verification.

Distinguish request acceptance, configured wire settings, observed reasoning,
cache reuse and reasoning replay. A correct answer alone does not prove a level
was honored; no visible reasoning is unknown, not necessarily unsupported.
A cache miss does not erase a previous continuation hit. Preserve explicit
retention/transport choices and never publish raw reasoning/signatures/state.

## Configure reasoning levels

Declare choices for the **configured route**, not a model family inferred from
its name. Each `reasoning_levels` label maps to exact request parameters;
selection replaces whole top-level values, with no nested merge. Repeat any
nested settings that must survive a level change.

For a Responses route that accepts this request shape, adapt this
`llm_config.yaml` example (replace the placeholder model and configure its key):

```yaml
models:
  my-route:
    model_name: openai/your-model
    client_type: responses
    max_tokens: 32768
    store: false
    include: [reasoning.encrypted_content]
    reasoning: {effort: medium, summary: auto}
    reasoning_default: medium
    reasoning_levels:
      low: {reasoning: {effort: low, summary: auto}}
      medium: {reasoning: {effort: medium, summary: auto}}
      high: {reasoning: {effort: high, summary: auto}}
```

The following example makes an inference request.

```python
from pathlib import Path
from nooa.unifiedllm import get_llm_client
from nooa.unifiedllm.registry import reload_registry

reload_registry(Path("llm_config.yaml"))
llm = get_llm_client("my-route", reasoning_level="high")  # persistent selection
levels = llm.reasoning_levels
default = llm.reasoning_default
# Pass llm to an Agent, or make a direct UnifiedLLM call:
# For a direct UnifiedLLM call, override only this request:
try:
    reply = await llm.acall(messages, reasoning_level="low")
finally:
    await llm.aclose()
```

- Omit `reasoning_levels` (or use null) for unknown support; `{}` declares
  selection unsupported. The public property returns `None`, `()`, or a tuple
  of valid labels respectively. Invalid selections raise, naming the choices.
- `reasoning_default` is metadata, not a selection: if supplied it must name
  a declared level and describe the base configuration. `reasoning_level=None`
  on a call bypasses a persistent selection and uses those base parameters.
- Declare levels/defaults on the registry entry or client constructor, never
  per-call or in `extra_body`. Do not combine a selected level with a per-call
  raw setting of the same key (including inside `extra_body`); that raises.
  Level blocks cannot change routing, credentials, messages or framework controls.
  Construct a new client when changing routes rather than reusing its level map.
- Use provider documentation or catalogues as candidates, then check the actual
  route's outgoing HTTP fields: SDKs and gateways may reject or silently drop
  settings. HTTP acceptance alone does not prove a level was honored. Leave
  unverified choices unknown rather than inventing mappings.
- This controls effort, not retention of stored reasoning. Changing effort may
  invalidate the cached prefix; labels are not comparable across providers.

See [reasoning-level configuration](../../docs/reasoning-levels.md) for reserved
keys, transport caveats, and bounded validation commands. The
`examples/reasoning_levels/llm_config.yaml` entries require explicit loading;
they are not automatically available registry aliases.
