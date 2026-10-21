---
name: nooa-model-configuration
description: Understand and troubleshoot NOOA registry settings, aliases, credentials, reply budgets, reasoning levels, caching and reasoning replay. For permission and running Connect headlessly to onboard or edit models, use the nooa-connect skill.
---

# Configure NOOA models

Produce a usable registry entry for the user's actual route, and state what was
verified. Keep model configuration separate from agent prompts and strategies.

## Locate the configuration before changing it

- For a diagnostic handoff, inspect its running package/version, target file,
  effective alias source, registry chain, credential source and remaining budget.
  A suggested rerun is not permission for additional paid calls.
- Use `nooa.llm_config.llm_config_chain()` to inspect active sources and
  `nooa.unifiedllm.registry.reload_registry(Path(...))` to load a specific file.
  An explicit output file is not necessarily in the runtime's discovery chain;
  a higher-priority alias can shadow the saved entry.
- Confirm the target alias/file and preserve other aliases and unknown fields.
  Prefer Connect's edit/write path over rebuilding a registry from selected entries.
- Keep keys in environment variables or NOOA's layered `secrets.yaml`, never in
  registry entries, shell arguments, reports or commits. `--prompt-key` provides
  masked input. Persisting a pasted key requires separate consent; check whether
  a project secrets file shadows the user file.

## Run Connect through the headless skill

Before running Connect to configure a model, get the user's permission for the
alias/file, checks and writes unless the request already authorizes them.
Agents use stage mode, not the interactive wizard.

Load [nooa-connect](../nooa-connect/SKILL.md) for the complete headless workflow:
model/interface discovery, offline proposals, bounded checks, JSON results,
media and decision stages, saving/replacing aliases, and diagnostic handoffs.
Keep that skill as the single source for Connect commands; do not duplicate its
workflow here. Install it alongside this skill when copying skill bundles.

This skill covers registry semantics and manual configuration. See
[manual configuration](../../docs/model-configuration.md) for the reference.
For library automation, `nooa.unifiedllm.connect` exposes proposal/check/write
operations but never grants user permission; use the same scope and evidence
rules as the headless skill.

## Keep settings truthful

- Use the endpoint's exact model ID and supported interface, not guesses from
  its name. Chat uses `client_type: completion`; Responses uses
  `client_type: responses`; Anthropic Messages uses the completion client and
  its Anthropic route prefix. Native decisions use `client_type: decision`, the
  exact model ID, and a full POST `endpoint` with `api_style: systemone` or
  `openai` (not Chat). Connect normalizes provider prefixes and base URLs.
- New Connect entries carry `transport: direct`. Runtimes with direct SDK support
  use it to bypass LiteLLM; older runtimes ignore it. Preserve explicit transport
  choices when editing. Check the installed runtime before describing probes as
  direct-transport validation: the saved field alone is not evidence.
- Explicit user choices win; endpoint-reported limits take precedence over
  public catalogue suggestions. Do not add input/output limits to invent a total
  context window. Record unknown context capacity as unverified.
- The saved reply cap is `max_tokens` for the three chat-family interfaces; Responses maps
  it to `max_output_tokens` on the wire. Catalogue output ceilings are metadata,
  not recommendations. Keep the selected cap below the known context window.
  Configured checks send this cap, including a selected level’s override; only
  initial interface discovery uses a smaller cap. Insufficient approved budget
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

## Configure media

Set each model's `media_inputs` to the formats its provider accepts: `image`,
`audio`, `video`, `pdf`. `File` supports PDFs with MIME type `application/pdf`.
Use [nooa-media](../nooa-media/SKILL.md) for examples.

Headless configuration, manual declarations and generated checks follow the
permission and evidence rules in [nooa-connect](../nooa-connect/SKILL.md).
An empty media list means text-only; omitted formats become text notices.

Keep the existing formats when editing the same model. Check the provider's
supported formats when configuring a different model or endpoint.

## Native decision settings

Use `client_type: decision` and the exact model ID with a full decision endpoint.
Chat reply caps and reasoning levels do not apply. Only named image inputs have
implemented native media transports: OpenAI/Hub Decisions image parts, or direct
TypeSafe multipart with explicit experimental opt-in. Route changes clear
inherited image permissions. See [nooa-connect](../nooa-connect/SKILL.md#media-and-native-decisions)
for headless checks, permission/transport options, evidence and save behavior.

## Interpret failures and evidence separately

Model listing may be public: it does not validate inference credentials.
A timeout or server error does not prove an unsupported API. A truncated reply
is inconclusive. To test a larger limit, explicitly change the configuration,
rebuild the plan, and rerun within the approved allowance and known ceiling.
Connect does not silently retry with a different cap than the saved configuration.

For reasoning, distinguish request acceptance, settings reaching the wire,
reasoning observed in the reply, and state replayed in the next request. An
accepted effort field may be ignored; a correct puzzle answer alone does not
prove reasoning support. A route that exposes no reasoning is unconfirmed, not
necessarily non-reasoning. Session checks must retain both settings and available
state. Never print reasoning text, signatures, encrypted state or raw error bodies.

For caching, compare a stable prefix and continuation cache-read usage. A later
miss does not erase a successful continuation hit; implicit provider cache misses
are warnings, not proof that an otherwise working entry is broken. Report missing
markers separately from markers sent with no reported reuse.

After a fix, rerun only affected checks within the existing allowance. Hand off
the alias, target/effective file, change, sanitized outcomes and unverified
capabilities. Do not overwrite unrelated entries or raise budgets implicitly.

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

The following call makes inference requests; run it only with approved allowance.

```python
from pathlib import Path
from nooa.unifiedllm import get_llm_client
from nooa.unifiedllm.registry import reload_registry

reload_registry(Path("llm_config.yaml"))
llm = get_llm_client("my-route", reasoning_level="high")  # persistent selection
levels = llm.reasoning_levels
default = llm.reasoning_default
# Pass llm to an Agent, or make an approved direct UnifiedLLM call:
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
