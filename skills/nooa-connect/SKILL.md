---
name: nooa-connect
description: Run NOOA Connect headlessly as an agent to discover interfaces, propose model settings, check chat or native decision routes and media, and save registry aliases with the user's permission. Canonical guide for Connect stages, JSON reports, bounded checks, credentials and configuration writes.
---

# Run Connect headlessly as an agent

Use this skill whenever you need to run `nooa connect` to configure or
troubleshoot a model. Other NOOA skills refer here for the complete headless
workflow. Agents use **`--stage NAME`**, not the interactive wizard, piped answers
or `--yes` to drive guided setup.

## Get permission before configuring the model

Ask the user before running Connect to configure a model unless the current
request already authorizes that work. Agree on:

- The endpoint, model, alias and target registry file.
- Which network lookups and model checks you may run, and the check budget.
- Whether you may create an alias or replace an existing one.
- Whether media samples and experimental transport options are allowed.

For example: “May I configure alias `work` in this project's registry, run one
connection check with a 4,096-token check budget, and save it if it passes?”

Permission to author an agent, an available key, a suggested diagnostic command,
or permission for one check is not permission to configure or test other models.
Once the scope is approved, do not ask again for every command within it. Ask
again before expanding the checks, raising the allowance, changing endpoints,
replacing an unapproved alias, or saving credentials.

`--stage` never asks for this permission. `--yes` on **stage save** only permits
replacement of an existing alias; it is not authorization from the user.

## Inspect the installed CLI and active configuration

Run from the user's project with its installed version:

```sh
uv run nooa connect --help
```

Check available stages/options before using examples from another release. Find
the effective registry sources and target before changing anything:

```python
from nooa.llm_config import llm_config_chain

print(llm_config_chain())
```

Use existing aliases where possible. Stage checks do not load an alias as an
editable connection: supply the actual endpoint, model ID and settings yourself.
Keep unrelated aliases, comments and fields. For settings and precedence details,
see [nooa-model-configuration](../nooa-model-configuration/SKILL.md).
For released media constructors, CodeAct `show()` and format/size limits, see
[nooa-media](../nooa-media/SKILL.md) and the [media guide](../../docs/media.md).

Use **key variable names**, never key values, with `--api-key-env`. Let the user
configure environment variables or layered `secrets.yaml`; never put secrets in
shell arguments, registry files, logs or commits. Avoid `--prompt-key` in a
headless agent: it can prompt for masked input. Stage save does not store a key.

## Stage contract

`--stage NAME` runs without wizard prompts and emits one JSON report to stdout.
Capture stderr separately if needed. A stage report has `version`, `stage`, `ok`,
`data`, `error`, `checks`, `warnings`, `run_context` and `diagnostic_prompt`.

| Stage | Operation |
|---|---|
| `discover` | Fetch the endpoint's advertised model list. No model test. |
| `model-info` | GET-only selected server facts; optional validated LiteLLM group enrichment. No model test or writes. |
| `catalogue` | Fetch public model metadata; useful for suggestions, not route verification. |
| `plan` | Build an unsaved proposal locally. No network. |
| `interfaces` | Try request interfaces with bounded routing requests; optionally compare chat reasoning. |
| `routing` | Chat-family: configured typed text baseline. Native decisions: typed routing acceptance. |
| `tools` | Chat-family tool-call check; the requested tool is not executed. |
| `reasoning` | Chat-family checks for settings declared in `--levels-file`. |
| `session` | Chat-family continuation/cache/reasoning replay checks. |
| `decision` | Native decisions: fresh routing and three typed decision smoke requests. |
| `media` | Baseline followed by the selected media samples. Native decisions include their typed smoke baseline. |
| `all` | All applicable checks: chat tools/reasoning/conversation, or native decision smoke cases, plus selected media. |
| `save` | Write one alias from a plan/result JSON file. No model requests. |

### Selected server facts (explicit metadata permission)

For a permitted metadata lookup, use:

```sh
uv run nooa connect EXACT_MODEL --stage model-info \
  --endpoint https://approved-gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --server-metadata litellm
```

`--server-metadata listing` (default) uses known listing headers/shapes for native
OpenAI/Anthropic and custom servers. Native OpenAI identity-only records leave
limits unknown. Native Anthropic reads exact `max_input_tokens`, `max_tokens`,
`capabilities.image_input.supported` and `capabilities.pdf_input.supported` fields.
The exact OpenRouter API base uses the public catalogue without an inference key.
`litellm` additionally permits a public OpenAPI product/GET-contract check on the
configured root (or a root obtained by removing its trailing `/v1`, retaining a
deployment prefix), then **one** selected `/model_group/info?model_group=EXACT_MODEL`
GET with the same approved work key. No hostname-only detection, new origin,
admin endpoints, master key, health call, or inference is used.

`model-info` returns `model_id`, effective `model`, source/field `records`, concise
`notices`, GET `requests`, and optional exact OpenRouter `catalogue_model` facts.
Null/denied/unavailable metadata is a normal result: retain known facts/manual
settings and continue to an independently authorized check. Group maxima cannot
raise tighter listing limits, false flags do not revoke permissions, and
`mode: evaluation` does not choose a decision URL/interface. No false flag means
"tested unsupported". Source facts remain separate from runtime configuration.
Lookups are bounded (30-second discovery and 30-second enrichment components,
5 MiB, 5,000 records), have no redirects/retries, and do not store keys.

`--discovery-file` reuses an approved fetched listing without a duplicate GET;
`--models-url` and `--listing-style` stay separate from inference routing.
Different origins require `--discovery-api-key-env NAME` (or `-` for anonymous
listing); no automatic credential forwarding. Keep invocation-local records
within their original connection/key visibility scope. The public helper is
`await connect.server_details(endpoint, model_id, listing=..., litellm=True)`;
`metadata=False` is its zero-GET bypass.

For known connections use `--no-metadata`: it blocks all metadata GETs in wizard
or stage execution; metadata stages and metadata files/URLs/mappings then reject
as contradictory options. Routing/interfaces still send only their explicitly
authorized tests. `--no-probe` remains a wizard inference bypass and still permits
metadata; `--no-catalogue` retains its historical exact-MODEL metadata bypass.
The explicit model-info stage requires metadata to be enabled. **Plan remains
offline**, and `all` never adds metadata GETs. OpenRouter fallback mappings must
be exact/user-selected, never guessed route profiles; do not import a different
server/model's reasoning defaults, reply allocation or price.

Only `save` writes the model registry. Redirecting reports to files is still a
local write: choose an approved workspace. Exit **0** means that stage met its
criterion, **1** means failure or missing evidence, and **2** means invalid
options. Inspect the report, not just the exit code. CLI parser errors can appear
on stderr without a JSON report.

Guided humans see provider/connection credentials **first**, then the model
picker/exact ID, metadata, and **Request interface** for that selected model—not
an interface menu immediately after provider selection. Explicit `--api-style`,
`--yes`, and edit flows retain their nonprompted/current interface behavior. No
inference occurs before **Test now** (or explicit `--yes` approval).

Wizard presets know System One routes for Hub
(`https://inference-api.nvidia.com/v1/systemone`) and OpenRouter
(`https://openrouter.ai/api/alpha/decisions`). OpenRouter Discover offers all
accepted chat/Responses/decision formats for the same exact ID rather than
classifying a router by name. Decision probes have no chat limits/reasoning/media
parameters and share the existing capped budget without retries. Explicit full
endpoint/mapped route overrides remain authoritative. Native System One is
labelled **System One Decisions**. This is a wizard convenience, not automatic
headless routing: stage interfaces still requires an explicit
`--systemone-endpoint` for a decision comparison. Custom decision paths are never
guessed. Key reuse across origins also requires explicit `--share-interface-key`
when a post-model known-interface choice changes the server.

Wizard options such as `--provider`, `--edit-model`,
`--levels`, `--no-probe`, a nondefault `--probe`, and `--show-config` are not
accepted in stage mode. Use explicit connection options and `--levels-file`.
`--catalogue-model` is accepted in plan/check stages only with `--catalogue-file`: it is an explicit mapping to an OpenRouter metadata ID, not a change to the inference model ID.
`--reasoning-template` and `--reasoning-level` apply only to `interfaces`.
`--input` and `--output` belong to `save`; redirect other stage reports yourself.

`--input` is for **save only**. Check stages construct a new proposal from their
arguments; they do not consume a previous plan or accumulate earlier reports.
Repeat the same settings when checking a plan. Retain each report separately and
save the reviewed result containing the settings you intend to use.

## Minimal configure → check → save workflow

Replace placeholders with the approved connection. This example is for a
Chat-compatible route and an alias that does not already exist.

### 1. Propose settings without calling the model

```sh
uv run nooa connect your-model --stage plan --as work \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --context-window 16384 \
  --max-tokens 1024 --budget-tokens 4096 > model-plan.json
```

Review `data.entry`, planned `data.probes`, estimates and timeouts. These are
proposals, not evidence of support. Headless plan does not automatically fetch
catalogue suggestions. Supply known limits, a saved discovery file, or an explicitly selected `--catalogue-file` as below; do not invent the context window.

### 2. Run only the approved check with the same settings

```sh
uv run nooa connect your-model --stage routing --as work \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --context-window 16384 \
  --max-tokens 1024 --budget-tokens 4096 > model-result.json
```

Check `ok`, `checks`, `warnings` and the actual entry before saving. For chat,
routing requires a complete validated integer `323` for `17 * 19`, with the
configured settings observed. Native decision routing only checks a valid
accepted decision response; use `decision` for typed correctness smoke cases.

Do not automatically chain save after a failed or uninspected check. **Stage
save can write an untested plan or failed report with a warning.** If the user
wants manual unverified configuration, make that choice explicit and save the
plan instead of describing it as tested.

### 3. Save the reviewed entry only when authorized

```sh
uv run nooa connect --stage save --input model-result.json \
  --output .nooa/llm_config.yaml > model-save.json
```

Use `--working-dir /absolute/project` instead of `--output` to target that
project's `.nooa/llm_config.yaml`. The directory must exist; the two options are
mutually exclusive. Other stages cannot use `--working-dir`.

If the alias already exists, inspect the existing entry and obtain replacement
permission, then add `--yes` to **save**. Save preserves other entries and comments, but **replaces the selected alias**
as a whole. Inspect and carry forward its approved custom fields; a fresh plan
does not merge them automatically.
Do not change media/transport/reply settings at save time and keep claiming the
old checks verified them. Save removes transient probe provenance from YAML;
keep result JSON as the evidence artifact.

### 4. Confirm what the application will actually load

Inspect `model-save.json` for `data.path` and `data.shadowed_by`. A successful
write may be shadowed by a higher-priority registry. Load an explicit file when
appropriate:

```python
from pathlib import Path
from nooa.unifiedllm.registry import reload_registry
from nooa.unifiedllm import get_llm_client

reload_registry(Path(".nooa/llm_config.yaml"))
llm = get_llm_client("work")
# Pass llm to the application; construction alone is not a model test.
await llm.aclose()
```

Use `get_decision_model("work")` for a native decision alias instead of
`get_llm_client`. Report the effective alias/file and which checks actually ran.

## Discover models and interfaces separately

List models first if the model ID is unknown, or use the exact ID supplied by the
user. A shared Hub list may contain both LLMs and decision models.

```sh
uv run nooa connect --stage discover \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY > models.json
```

Reuse this metadata with `--discovery-file models.json` on plan/check stages to
avoid another listing request. Repeat `--api-key-env MODEL_KEY`; the discovery
file carries metadata, not credentials. Advertised interfaces/modalities are
suggestions; they do not enable media or verify inference access.

For a decision route, supply its full POST endpoint and a separate full catalogue
GET URL with `--models-url`. Match reused discovery files against that listing
URL. `--listing-style chat|responses|anthropic` controls catalogue authentication
and pagination, not inference. Same-origin listing can use the inference key;
for another origin specify `--discovery-api-key-env CATALOGUE_KEY`, or `-` for no
authentication. Never forward a credential to another origin without permission.

Known interface: supply `--api-style`; `interfaces` checks that style alone.
Unknown interface: omit it to compare Chat, Responses and Anthropic. Add System
One with an explicit `--systemone-endpoint` (including when using the Hub):

```sh
uv run nooa connect your-model --stage interfaces --as work \
  --endpoint https://gateway.example/v1 --api-key-env MODEL_KEY \
  --systemone-endpoint https://gateway.example/v1/systemone \
  --output-tokens 200 --budget-tokens 4096 > interfaces.json
```

No custom System One URL is guessed. Results retain per-style entries, accepted
interfaces/client types and usage; **the combined interfaces report is not a
saveable single-entry result**. Select an accepted interface and run plan/routing
with its actual endpoint. If several work, choose for the intended task or ask
the user rather than preferring chat because it returned reasoning.

`--share-interface-key` explicitly permits cross-origin interface checks to reuse
the inference key. Use it only after the user approves those destinations.
`openai` means **OpenAI Decisions**, not Chat; chat-family styles are `chat`,
`responses`, and `anthropic`.

## Explicit structured-output profile

Use `--structured-output-mode anthropic` only with an explicitly selected
Completion interface (`chat` or `anthropic`) and authorization for the configured
checks. It declares native `output_config.format: {type: json_schema, schema: ...}`;
it does not prove gateway acceptance or change transport. `none` (Python `None`,
YAML `null`) and `auto` retain the existing carrier, with no inferred native support
or fallback. Omission preserves a saved same-route policy; route/model/interface/
client/credential changes do not inherit it. Connect rejects explicit controls on
Responses/Decision and metadata/interface discovery stages before credential use.
Keep this framework setting on the entry, not in `extra_body`, level patches or
catalogue metadata. Guided review offers Change structured output profile;
plan/check/save stages carry it.

A mode change invalidates all prior probe/session evidence; a save-time override
is not freshly verified. Inspect actual typed Predict baseline/media evidence:
`structured_output_schema_observed`, `structured_output_conflicts_observed`,
`structured_output_carrier`, and `settings_sent`, alongside typed validation,
final-stop/correct-answer facts. Native checks compare the exact original generated
response-model schema on the wire and reject generic/schema-tool surrogates;
ordinary no-output-model tool checks are unchanged. Format insertion preserves
`output_config.effort`. `--no-cache` remains independent, not a carrier fix or proof
of compatibility. The profile adds no extra send, carrier fallback or recovery
UI; existing retry/cap handling retains its explicit selection or opt-out.
See [model-connect](../../docs/model-connect.md#explicit-native-structured-output-profile)
for the official Anthropic SDK 0.109.2 source contract and its limits: gateway
acceptance is unknown until an authorized actual check succeeds.

## Keep checks bounded and settings consistent

Always specify `--budget-tokens` for model checks: the default is `10**15`,
effectively unlimited. Track remaining allowance across commands yourself; each invocation
starts a new budget. Failed attempts consume allowance. Inspect report charges
and skipped cases before deciding whether another call fits.

For chat-family routes, `--max-tokens` / `--reply-tokens` selects the saved cap and
the cap used by configured checks. Responses maps it to `max_output_tokens` on
the wire. `--output-tokens` is the initial interface-discovery cap (default 2,048), not
the saved reply cap. Check estimates include input reservations. Insufficient
budget skips a check; it does not silently lower the saved cap. Increase caps or
budgets only with authorization. Typical defaults and reservations:

- Configured chat reply cap: 32,768 when limits are unknown. Routing/tools
  reserve the cap plus 512 input tokens each; configured checks use 120-second
  timeouts. A 4,096-token budget without a smaller explicit cap may skip them.
- Initial interface routing defaults to 2,048 output tokens plus 512 input tokens,
  with a 30-second timeout. The example explicitly selects `--output-tokens 200`
  for a cheaper check; this is not the default and can truncate responses.
  Optional reasoning comparisons need additional allowance.
- Chat session: three turns, each reserving `8192 + 3 * selected reply cap`,
  with a 120-second timeout per turn. Inspect the plan before approving it.
- Native decisions: four routing/smoke requests reserve 512 each with
  30-second timeouts. Image adds a 4,096-token reservation: the five-request
  image plan needs at least 6,144 estimated tokens, plus reported-usage headroom.

These are local reservations, not measured request lengths or hard spending caps.
Native decisions have no provider-enforced
reply-token field; bound request count, input and time as well as estimates.

For exact chat reasoning settings, prepare `--levels-file levels.yaml` (a mapping
from labels to complete request settings), then run `--stage plan` and
`--stage reasoning` with that same file and a sufficient approved budget.
`--reasoning-template` and `--reasoning-level` are for interface comparisons;
headless reasoning stages require the levels file. Session/all checks make
multiple requests; do not run them to investigate one unrelated failure.

## Known routes: metadata → plan → save, with optional checks

For a known OpenRouter, native Anthropic or native OpenAI chat-family route,
start with the **exact model ID and interface**. Do not run Hub-style interface
comparisons or all checks just to configure it. Metadata lookups are GETs, not
inference requests, but still require network permission. With authorization to
configure and save, no generation calls are required:

```sh
# One public GET, without the native endpoint's key. Save in an approved workspace.
uv run nooa connect openai/example-model --stage catalogue > catalogue.json

# Offline proposal for the exact native ID. The OpenRouter ID is an explicit mapping.
uv run nooa connect example-model --stage plan --as work \
  --endpoint https://api.openai.com/v1 --api-style responses \
  --api-key-env OPENAI_API_KEY --catalogue-file catalogue.json \
  --catalogue-model openai/example-model > model-plan.json

# After reviewing and approving this proposal, write it without model calls.
uv run nooa connect --stage save --input model-plan.json --output llm_config.yaml
```

These model IDs are placeholders, not asserted API profiles. Select a current
catalogue ID yourself. `--catalogue-file` works in plan/check stages, including
decision routing, decision and media checks. There is **no network on plan or
save**, and nothing is silently fuzzy
matched: without `--catalogue-model`, the inference ID must occur exactly once in
the file. The file must be a successful version-1 `catalogue` report with
`data.source: https://openrouter.ai/api/v1/models` and `data.models`, or the
explicit raw form `{"source": "https://openrouter.ai/api/v1/models", "models": [...]}`.
Other sources, failed reports, duplicate matches and mismatched IDs are rejected.
Do not repurpose a gateway's model list as an OpenRouter catalogue file.

For native decision models, mapped OpenRouter details are metadata only: bounded
model descriptions, media advertisements and explicit source/ID provenance.
Exact endpoint media facts take priority. They never initialize chat caps,
context, reasoning, tools, sessions, store/include settings, image permissions or
transport opt-ins, and add no inference checks. `typesafe/jev-router` is not an
implicit match for Hub `nvidia/alexwortega/openjev-4b`. The wizard uses an exact
OpenRouter ID or a match you explicitly select; `--no-catalogue` skips lookup,
and editing and stage `plan` never fetch metadata.

For chat-family models, review initialized context bounds, the published output
**ceiling** in
`data.entry.provenance.catalogue_limits`, the proposed `max_tokens` request cap,
and any reasoning settings. The recommendation is clipped to known limits;
`--max-tokens` explicitly selects a different cap. Pricing, supported parameter
lists and miscellaneous default parameters are not copied into requests.
Check your server's limits if they differ from OpenRouter's top-provider facts.
Models API metadata from the actual endpoint can
be saved with `--stage discover` and supplied through `--discovery-file`; its
reported limits take preference when it belongs to the selected endpoint/model.
Anthropic's exact advertised output `max_tokens` and `max_input_tokens` are
retained. Missing native OpenAI limits remain unknown, not inferred from names.

The proposal's `data.entry.provenance.media_suggestions` reports the selected
model/source, advertised formats, constrained suggestions, checkable formats,
unavailable checks and raw allowlisted evidence. Choose permissions explicitly.
For PDFs, use the specific PDF flag or an approved generated PDF check. Review suggestions with
the user; only an explicitly approved `--media-inputs image,pdf` (or reviewed
entry change) makes a manual, unverified permission. Existing explicit choices,
including text-only, must not be overwritten by a suggestion.

To test instead, get approval for the generated samples and budget, repeat the
same catalogue/discovery/settings options with `--stage media` and explicit
`--test-media-inputs`, inspect the result, and save that reviewed result. Do not
expand to `all` or raise the allowance without authorization. Manual permission
and generated tests are alternative choices, not interchangeable evidence.
Reusing an existing registry file preserves other aliases and comments; replacing
an existing alias still needs explicit permission and stage-save `--yes`.

The wizard's `--no-probe` path also offers metadata and saving without inference
calls. `--no-probe --no-catalogue` with an explicit model/interface remains offline.
Full interface/media/decision checks remain available for unknown shared hubs.

## Media and native decisions

Media tests send generated samples to the selected route. Get permission for
those calls and formats. Use `--test-media-inputs image,audio,video,pdf` to test
selected formats in **plan**, **media** or **all**. Use `--media-inputs` only for
manual declarations saved as not tested; do not combine the two. Only passing tests add
permissions in stage results. Supported formats depend on the actual transport:

- Chat-family image/PDF checks exist. Audio/video testing for Responses and
  Anthropic is unavailable here. Use Chat to test audio/video, or enable those
  inputs without testing; retain the local `not_probed` outcome.
- Native decisions support named **Image** inputs only. OpenAI/Hub Decisions
  (`--api-style openai`) uses native `input_image` blocks.
- Direct TypeSafe System One requires
  `--decision-image-transport typesafe_multipart` and
  `--typesafe-danger-zone-images`; obtain permission for the experimental opt-in.
- System One JSON blocks image requests before sending. Use OpenAI Decisions or
  an approved direct TypeSafe multipart connection, or turn images off. A manual
  permission still cannot send images through System One JSON. Choose multipart
  and experimental opt-in explicitly, never from the hostname or listing.
  Decision audio/video/PDF are unavailable.
- Read listed formats from OpenRouter `architecture.input_modalities` and exact
  Anthropic `capabilities.image_input.supported` / `capabilities.pdf_input.supported`
  flags. With missing metadata, explicitly choose inputs or approve generated
  checks. Empty lists omit the source/advertisement block in the wizard.
- Review effective permissions separately from check evidence. The wizard shows
  Images off, enabled (test passed), enabled (not confirmed by this test) for
  failed-retained inputs, or enabled (not tested) for manual/skipped inputs.
  Preserve the raw outcomes and reasons in headless reports.

```sh
uv run nooa connect decision-model --stage media --as decisions \
  --endpoint https://decision.example/v1/decisions --api-style openai \
  --api-key-env DECISION_KEY --test-media-inputs image \
  --budget-tokens 8192 > decision-media.json
```

Use the same arguments with `--stage plan` first to inspect the five-request
routing/smoke/image plan. A fresh `--stage decision` runs routing plus three
smoke cases (boolean, literal choice, ordinal score); image testing follows the
decision baseline. Native image passing requires the image on the actual request,
a validated typed result and the expected color. Review answers on the user's
actual examples before relying on accuracy or calibration.

For TypeSafe, use its full System One endpoint and the explicit transport/opt-in
flags above. Do not pass chat reply caps or reasoning settings to native decisions.
Changing a route clears inherited media permissions: review and redeclare them
only for the selected connection.

## Diagnose and hand off

Read `checks`, `error`, `warnings`, `run_context` and `diagnostic_prompt` after a
failure. Confirm permission for additional paid calls, configuration or writes
and check the remaining allowance before following a suggested rerun.
Distinguish authentication, timeout, rejection, malformed response, missing usage,
and skipped budget/transport checks. Stop on authentication failures and ask the
user to repair credentials locally; do not try unrelated keys or endpoints.

Request acceptance, schema validity, correctness, reasoning observed and cache
reuse are separate evidence. After saving, inspect the effective registry alias
and review the affected check results before using the model.
Never print raw error bodies, reasoning, encrypted state or credential values.
After a fix, rerun only the affected stage within the remaining allowance.

Hand off the alias, target/effective file, selected model/interface, changed
settings, check outcomes, remaining unknowns and the evidence report paths.
For implementation details, consult [Connect reference](../../docs/model-connect.md).
If this skill is installed separately, keep sibling NOOA skills together and use
the docs matching the installed package version rather than arbitrary `main`.
