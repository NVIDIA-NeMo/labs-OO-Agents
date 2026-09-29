---
name: nooa-model-configuration
description: Configure and troubleshoot NOOA model registry entries with Connect or the library, including endpoints, credentials, reply budgets, reasoning levels, cache reuse and reasoning replay. Use for model onboarding, editing aliases, or Connect diagnostic handoffs, not Agent class authoring.
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

Read [Connect](../../docs/model-connect.md) for CLI stages, editing, budgets and
library contracts, or [manual configuration](../../docs/model-configuration.md)
for offline setup. These paths are relative to the source checkout. If the skill
was copied elsewhere, use the absolute installation references in the handoff;
a wheel without docs needs the matching source version/commit, not arbitrary main.

## Choose the least expensive relevant workflow

**Offline:** build an unsaved entry with `connect.plan` or `--stage plan`.
Validate the resulting YAML with `reload_registry(Path(...))`; that proves only
local loading, not credentials or endpoint support. Unknown capabilities stay
unknown. `--no-probe` prevents generation but may still fetch discovery/catalogue
metadata; use an explicit model/interface/endpoint plus `--no-catalogue` when
using the wizard without any network access.

**Interactive:** `uv run nooa connect` offers the wizard; `--edit-model ALIAS`
edits an existing entry, and `--edit-model` alone opens the registry selector.
Saving/replacing an alias and storing credentials are separate decisions from
permission to make model calls.

**Agent CLI:** stage mode produces JSON and never saves implicitly.
`plan` and `save` are offline; `discover` and `catalogue` fetch metadata;
`interfaces`, `routing`, `tools`, `reasoning`, `session`, and `all` can
incur inference charges. Stage mode does not ask for spending approval: obtain
the user's allowance before invoking a paid stage, set explicit token caps and
budget, inspect the planned timeouts, and stop on exhaustion. Failed calls consume
allowance too. Do not run all
stages to investigate a single known failure.

```sh
# Offline proposal, not a saved alias or a verified connection:
uv run nooa connect your-model --stage plan --as work \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --max-tokens 32768 > model-plan.json

# Only when the user has approved this write:
uv run nooa connect --stage save --input model-plan.json --output llm_config.yaml
```

Use `--yes` only when replacement is authorized; it is not a substitute for
spending approval. Read `checks`, `warnings` and `run_context`, not just exit
status: 0 means that stage met its criterion, 1 means failure or missing evidence,
and 2 means invalid options. Saving an untested plan is not successful validation.

**Library:** import `from nooa.unifiedllm import connect`. Use `plan` to propose,
`check_stage` or `run_steps` for approved checks, and `write` for explicit
persistence. The library never prompts or grants consent. Run through the same
registry client and formatter as the application; do not substitute raw SDK
probes and call that validation of a NOOA entry.

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
  Read [stable-prefix caching](../../docs/stable-prefix-caching.md) for details.

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
