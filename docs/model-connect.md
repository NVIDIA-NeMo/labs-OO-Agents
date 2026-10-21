# Configure a model with NOOA Connect

`nooa connect` sets up a model for NOOA. It finds the model on your endpoint,
runs a few small test requests, and saves a named entry (an *alias*) in
`llm_config.yaml`. Your agents then use the model by that alias.

Test requests can cost money. The interactive wizard asks before its first
test request. Agent-driven stages run without a spending prompt; authorize
those calls before invoking them. `--yes` approves wizard checks and saving.
To set up a model without any requests, see [Configure a model without paid
checks](model-configuration.md).

## CLI

Run the guided setup:

```sh
uv run nooa connect
```

Choose a provider (NVIDIA, NVIDIA Inference Hub, OpenAI, Anthropic,
Google, OpenRouter, or a custom endpoint), a known request interface or
**Discover request interfaces**, your key, and the model. Connect first reads
available model metadata using GET requests, not model calls. It fills in a
proposal while retaining your explicit limits, reasoning settings and media
choices, then shows one concise **Proposed configuration** summary for the current
proposal and asks:

**Would you like to test these settings?**

- **Test now** approves the selected checks with the displayed token budget.
  If the request style is unknown, bounded interface discovery runs first;
  metadata alone never proves an interface works. A known style is tested too.
- **Save without testing** skips interface discovery and all model calls. If
  the style is still unknown, choose a provider preset/manual interface before
  saving an explicitly not-tested entry. Saving still needs final confirmation.
- **Change settings** opens the current settings editor, including the reply
  limit chooser for chat models, then returns directly to this decision. It
  shows an updated summary before the next test/save decision. It does not
  fetch metadata again or discard explicit media declarations or custom
  reasoning-level request blocks.

If deferred discovery selects an image-only decision route after a provisional
all-media choice, explicitly review image-only inputs/tests, turn media off, or
cancel. The revised proposal is shown before another test/save decision; no
unsupported inputs are silently dropped. Explicit incompatible declarations
require an affirmative correction or their existing validation errors prevent
saving; known-style and headless validation remains unchanged.

No inference POST is sent before **Test now**. Media advertisements never
silently enable inputs: keep the current/default choices or explicitly choose
formats/tests. `--no-probe` removes **Test now**, skips all model calls, and still
allows metadata GETs. For an entirely offline exact-ID setup, also supply
`--no-catalogue` and the endpoint/interface; editing an existing alias does not
fetch metadata. Omitting the model ID requests the model-list picker (a GET),
even with `--no-catalogue`, which skips public and automatic limit lookups.

To prefill the connection details, pass the answers as flags. The wizard still
asks about media choices, testing, changes, and saving unless you use `--yes`:

```sh
uv run nooa connect your-model --as my-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY
```

- `your-model` is the exact model ID the endpoint expects. Omit it to choose
  from the endpoint's model list.
- `--as` is the alias your agents use.
- `--api-style` selects one known interface: `chat` (OpenAI-compatible Chat
  Completions), `responses` (OpenAI Responses), `anthropic` (Anthropic Messages),
  `systemone` (System One Decisions), or `openai` (OpenAI Decisions, not Chat).
  An explicit style checks only that style. In guided setup, omit it to choose
  **Discover request interfaces** or a known interface before model listing.
  After **Test now**, Discover compares chat interfaces plus System One when
  a known or explicit decision endpoint is available. It sends the exact same
  model ID to every candidate; it never translates a completion model into a
  decision model.
- `--api-key-env` is the name of the environment variable that holds the key,
  never the key itself. Use `--api-key-env ''` for a local server with no key.
- `--prompt-key` lets you paste a masked key instead (even with `--yes`). At the
  end Connect offers to save it in NOOA's `secrets.yaml`; persisting a key
  requires separate approval, not just `--yes`.
- `--yes` approves the test requests and saving without asking. It needs
  `MODEL`, `--as`, and either `--provider` or `--endpoint` plus `--api-style`.

Run `uv run nooa connect --help` for all options.

### Guided interaction

- Use the arrow keys and Enter to pick from menus. Type to filter long lists.
- Ctrl-C cancels. Nothing is saved until you confirm the final **Save model**
  step.
- Results show as **Passed**, **Not available**, **Skipped**, or **Failed**.
- Add `--show-config` to see the full entry before saving, and
  `--show-diagnostics` to see full details when a check fails.

### Decision models

Decision models use `client_type: decision`. Give the exact model ID and the
full decisions request URL.

For a System One endpoint:

```sh
uv run nooa connect typesafe/jev-1.13 --as decisions \
  --endpoint https://openrouter.ai/api/alpha/decisions \
  --api-style systemone --api-key-env OPENROUTER_API_KEY
```

For OpenAI's Decisions API:

```sh
uv run nooa connect gpt-6-luna --as openai-decisions \
  --endpoint https://api.openai.com/v1/decisions \
  --api-style openai --api-key-env OPENAI_API_KEY
```

`--probe minimal` sends one small typed decision request to check routing.
`--probe all` also runs three fresh smoke cases: one multi-field boolean,
literal choice and ordinal score result, followed by positive and negative
boolean cases. These use the actual Agent `DecideStrategy`, including schema
validation and typed reconstruction. A valid response and a correct answer
are recorded separately. Finite probabilities, normalized distributions and
a score consistent with its weighted distribution are part of the contract.
Add `--no-probe` to save the connection as not tested, without model requests.

Native decision media checks support **image only**. OpenAI/Hub Decisions
send a named `Image` parameter as native `input_image` content. Direct TypeSafe
needs explicit `--decision-image-transport typesafe_multipart` and
`--typesafe-danger-zone-images` as well as image permission. The hostname never
selects the transport. System One JSON blocks image requests before sending. To test images, use
OpenAI Decisions or an approved direct TypeSafe multipart connection, or leave
images off. Enabling images manually on System One JSON still blocks image calls. Decision audio,
video and PDF are not implemented.

Image checks run after text/state decision checks, using a generated red PNG.
A pass requires the attachment on the actual HTTP request, a native typed
access result, and the correct known color. A reported-access answer with the wrong color fails the check. Each fresh test makes at most one request, with no retries and a
30-second deadline, sharing the approved token reservations. Routing acceptance
can be reused during setup; fresh accuracy and image tests cannot.

```sh
uv run nooa connect decision-model --stage decision --as decisions \
  --endpoint https://hub.example/v1/decisions --api-style openai --api-key-env MODEL_KEY

uv run nooa connect decision-model --stage media --as decisions \
  --endpoint https://hub.example/v1/decisions --api-style openai --api-key-env MODEL_KEY \
  --test-media-inputs image

uv run nooa connect typesafe-model --as decisions --endpoint https://typesafe.example/systemone \
  --api-style systemone --api-key-env MODEL_KEY --test-media-inputs image \
  --decision-image-transport typesafe_multipart --typesafe-danger-zone-images --yes
```

The permission (`media_inputs`), transport (`decision_image_transport`) and
explicit TypeSafe opt-in (`typesafe_danger_zone_images`) survive saving and
reloading. Checks and usage are evidence, kept out of runtime YAML. Changing
the model, endpoint, interface or image transport resets inherited permissions.
No fixtures, encoded images, raw replies or credentials are saved.

The Hub preset knows its System One route at
`https://inference-api.nvidia.com/v1/systemone`. Select System One in the early
menu, or use `--provider hub --api-style systemone`. An explicit `--endpoint`
always overrides the preset for a known style. Discover on Hub compares that
known System One route with the chat-family routes and lists from the shared
`/v1/models` catalogue regardless of your generation style.

For mixed discovery on a custom chat server, pass `--systemone-endpoint` with
the **full System One POST URL**. Connect never guesses a decision suffix from
a custom base. Without this option, custom discovery compares chat interfaces
only. `--stage interfaces` accepts the same option; omit `--api-style` for a
comparison, or specify it to check just one interface. If a mapped interface
has a different origin (scheme, host, or effective port), sharing a nonempty
key or configured key variable requires explicit `--share-interface-key`. This
also applies to Hub discovery when overriding the chat base to another origin.
Same-origin paths need no opt-in; without a key, cross-origin checks are allowed.
The flag does not authorize catalogue credentials: `--discovery-api-key-env`
retains its separate safety rules.

A successful decision interface check is reused during setup, rather than
sending another routing request. The fresh smoke cases are still run when approved. Results show accepted client types.
Mixed decision/chat results do not automatically favor chat merely because it
returned reasoning; chat-only comparisons retain their reasoning preference.

Model listing is independent of the inference interface. A server such as NVIDIA
Inference Hub can advertise chat and decision models in the same `/models` list.
For a custom decision route, supply `--models-url` with the **full catalogue GET
URL** to use the picker or `--stage discover`; Connect never appends `/models` to
the full decisions request URL. Without a listing URL, enter the exact model ID.
The wizard's `--provider hub` preset uses its shared `/v1/models` catalogue while
keeping your explicit `--endpoint` unchanged.

```sh
uv run nooa connect --stage discover \
  --endpoint https://hub.example/v1/decisions --api-style systemone \
  --models-url https://hub.example/v1/models --listing-style chat \
  --api-key-env HUB_KEY > models.json

uv run nooa connect decision-model --stage plan --as decisions \
  --endpoint https://hub.example/v1/decisions --api-style systemone \
  --models-url https://hub.example/v1/models --discovery-file models.json \
  --api-key-env HUB_KEY
```

`--listing-style` selects listing auth and pagination (`chat`/`responses` use
Bearer auth, `anthropic` uses its key header and pagination), not the inference
interface. `openai` remains a **decision inference** style, not a listing style.
Listing normally uses the inference key on the same origin. For a separate key,
use `--discovery-api-key-env CATALOGUE_KEY`; use `-` for no authentication.
A different catalogue origin requires this explicit credential choice; Connect
never automatically forwards the inference key there.

`--discovery-file` works for decision routes with `--models-url`, and is checked
against that exact listing source. Advertised identity, interface/media hints,
and limits remain in discovery output and transient plan provenance for review;
they do not verify support, enable media, or add chat limits to a decision entry.
As before, saving writes runtime settings only, not provenance. Decision checks
support `plan`, `interfaces`, `routing`, `decision`, `media`, `all`, and `save`.
A discovery file supplies metadata, never credentials.

## What Connect tests

For chat-family routes, the default `--probe all` runs:

1. **Connection.** Discover sends one routing request per candidate API style
   (and reasoning checks for chat styles); a known interface is checked alone.
2. **Text baseline.** Asks the model for `17 * 19` as a typed NOOA result. The
   model must return `323` as structured output. Without this pass, the
   wizard does not save.
3. **Tools.** Checks that the model makes a requested tool call. The call is
   never run.
4. **Reasoning levels.** Checks each level you set up.
5. **Conversation.** Three requests that check caching and that reasoning carries
   over between turns.
6. **Media**, only if you choose to. See [below](#choose-media-inputs-during-setup).

`--probe minimal` runs the typed arithmetic check using your settings. It skips
tools, reasoning, conversation, and media checks. Connect may first try the API
styles if you did not choose one.

`--no-probe` (or `--probe none`) skips model requests and saves an **unverified**
entry. It can still fetch model lists and catalogue details. To stay offline,
supply the exact model, endpoint, and API style plus `--no-probe --no-catalogue`.

For chat-family routes, a failed connection or configured text baseline blocks
wizard saving. Tool, reasoning, caching, or media results that need attention do not.

Native decision routes instead require accepted decision routing to save in the
wizard. With `--probe all`, three fresh typed decision smoke cases follow routing.
Their failures remain visible and prevent image testing, but do not erase routing
acceptance or block saving that route. `--probe minimal` checks decision routing
only. Explicit `--stage save` can write even a failed report with a warning;
inspect its checks before choosing to save.

### Configured typed text baseline

The baseline uses your exact settings and one request with no retry. It fails if
the answer is wrong, cut off, or not in the structured format. If your endpoint
cannot return structured output, save it without checks (`--no-probe`) instead.

## Token budget and reply limit

- `--budget-tokens N` limits the estimated total tokens for all checks. The
  default is no limit. It is an estimate, not a spending cap.
- A check that does not fit the remaining budget is skipped. Connect does not
  lower limits or ask for more budget.
- For chat-family routes, `--max-tokens N` (also `--reply-tokens`) sets the reply limit saved as
  `max_tokens`. It covers reasoning and the answer. Without it, Connect suggests
  the published recommendation, or 32,768. It must be less than the context
  window. Native decision calls have no reply-token cap and do not save chat
  limits or reasoning levels.
- `--context-window N` records a known context window.

## Reasoning levels

Connect uses published reasoning levels when they exist. To supply your own:

```sh
uv run nooa connect gateway/model --as work-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY --no-catalogue \
  --reasoning-template effort --levels low,medium,high
```

Templates are `effort`, `adaptive`, `budget`, `toggle`, and `thinking`. For exact
request settings, use `--levels-file` with a YAML mapping from each level name to
its request settings. Quote names like `"off"`. See
[reasoning levels](reasoning-levels.md).

Model details such as context window and reasoning levels come from OpenRouter's
public model list and exact endpoint listing facts, with server facts taking
priority. Use `--catalogue-model ID` to pick the public entry, or `--no-catalogue`
to skip public and automatic limit lookups. Missing limits stay unknown in the
proposal; continue with NOOA defaults or choose **Change settings** to enter
known limits. The proposal names its source; edits are your settings, not
verified capabilities.

## Known routes: configure from metadata without model calls

For a known OpenRouter, native OpenAI or native Anthropic chat-family route,
choose the exact model ID and interface explicitly. You do not need full Hub
interface discovery or paid checks just to save settings. The wizard's
`--no-probe` mode reads allowed metadata, lets you review settings/media and
saves an unverified entry without inference requests.

Agents can use a public metadata GET, an offline proposal, then an approved save:

```sh
# Public OpenRouter metadata, no inference key or generation request.
uv run nooa connect openai/example-model --stage catalogue > catalogue.json

# Exact native model and an explicit OpenRouter metadata mapping; no network.
uv run nooa connect example-model --stage plan --as work \
  --endpoint https://api.openai.com/v1 --api-style responses \
  --api-key-env OPENAI_API_KEY --catalogue-file catalogue.json \
  --catalogue-model openai/example-model > model-plan.json

# Save only after reviewing/approving the proposal. No model calls.
uv run nooa connect --stage save --input model-plan.json --output llm_config.yaml
```

These IDs are examples, not universal profiles. `--catalogue-file` is opt-in and
works in plan/check stages, including decision routing, decision and media checks.
It accepts a successful version-1 catalogue report (`data.source` and `data.models`) or an explicitly sourced raw
object: `{"source": "https://openrouter.ai/api/v1/models", "models": [...]}`.
The source must be that OpenRouter URL. Without `--catalogue-model`, the exact
inference ID must occur once in the file; there is no fuzzy or suffix guessing.
A native ID different from the OpenRouter ID needs an explicit mapping.

For native decision models, mapped OpenRouter details are metadata only: bounded
model descriptions, media advertisements and explicit source/ID provenance.
Exact endpoint media facts take priority. They never initialize chat caps,
context, reasoning, tools, sessions, store/include settings, image permissions or
transport opt-ins, and add no inference checks. `typesafe/jev-router` is not an
implicit match for Hub `nvidia/alexwortega/openjev-4b`. The wizard uses an exact
OpenRouter ID or a match you explicitly select; `--no-catalogue` skips lookup,
and editing and stage `plan` never fetch metadata.

For chat-family models, review context limits, any initialized reasoning settings,
the published ceiling
in `provenance.catalogue_limits`, and the proposed `max_tokens` reply cap. The
recommendation is conservatively clipped to known limits; the ceiling is not an
allocation. `--max-tokens` overrides the proposed cap explicitly. Pricing,
supported-parameter lists and miscellaneous default parameters are not copied
into requests. Check your server's limits if they differ from OpenRouter's.
Cached `--discovery-file` metadata from the selected native
endpoint/model takes preference for reported limits. Missing limits remain
unknown. Review `provenance.media_suggestions` for listed formats and their
source, then explicitly choose which inputs to enable before saving.

Checks are optional. With permission for those calls, repeat the same metadata
and settings options for the chosen check stage; do not automatically run `all`.
Saving to an existing alias still requires replacement approval and `--yes` in
stage mode. The canonical [Connect skill](../skills/nooa-connect/SKILL.md) covers
permissions, manual suggestions versus generated checks, source validation and
headless review. Full checks remain available for unknown shared hubs.

## Independent stages for agents

Agents should follow the [nooa-connect skill](../skills/nooa-connect/SKILL.md)
for the complete headless workflow. Get the user's permission to configure the
model and agree on checks and writes before running Connect.

`--stage NAME` runs one step and prints one JSON report to stdout. It does not
ask for spending approval; obtain permission before paid stages. Only `save`
writes the registry. Stages normally do not prompt, but `--prompt-key` asks for
a masked credential on endpoint checks; use `--api-key-env` for unattended use.

| Stage | What it does |
|---|---|
| `discover` | List the endpoint's models. |
| `catalogue` | Read public model details. |
| `interfaces` | Find which API styles work. |
| `plan` | Build an unsaved entry. No network. |
| `routing` | Chat: configured typed text baseline. Native decisions: routing acceptance. |
| `tools` | Chat-family routes only: check tool calls. |
| `reasoning` | Chat-family routes only: check levels from `--levels-file`. |
| `media` | Run the text baseline and selected media tests; native decisions run the typed decision baseline first. |
| `decision` | Native decisions only: fresh routing and three bounded typed decision smoke cases. |
| `session` | Chat-family routes only: check caching and reasoning across turns. |
| `all` | Chat: baseline, tools, declared reasoning, conversation and optional media. Native decisions: routing, fresh typed smoke cases and optional image checks. |
| `save` | Write an entry from a plan or result file. |

Example:

```sh
uv run nooa connect your-model --stage routing --as work \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY > result.json
uv run nooa connect --stage save --input result.json --output llm_config.yaml
# Register this explicit output path for the application:
export NEMO_OO_LLM_CONFIG="$PWD/llm_config.yaml"
```

- Pass `--endpoint`, `--api-style`, and `--api-key-env` explicitly.
- Save `--stage discover` output and reuse it with `--discovery-file` to avoid
  another listing request.
- `--stage save` needs `--yes` to replace an existing alias.
- Exit code `0` means the stage passed, `1` means it failed or was inconclusive,
  and `2` means the options were invalid.
- The report's `diagnostic_prompt` describes the failure for an agent to fix.
  It never contains key values.

## When something fails

| Problem | What to do |
|---|---|
| No API style works | Check the endpoint URL, key, and model ID. The wizard offers to change them, retry with a 120-second timeout, or cancel. |
| Timeout | Completion is unknown. Check the endpoint and server status; approve a retry within the remaining budget. |
| Text baseline fails | Make sure the endpoint supports structured output. Otherwise save with `--no-probe`. |
| Endpoint rejects `include` or `encrypted_content` (Responses) | Set `include: []` in the entry, then rerun the checks. Saving stays blocked until the baseline passes. |
| Reply cut off | Raise `--max-tokens` and rerun the affected stage. |
| Reasoning not shown | Some servers hide reasoning. Try another API style or check server settings. |
| Saved alias has no effect | A later config file overrides it. Check `nooa.llm_config.llm_config_chain()`. |
| Budget ran out | Rerun with a larger `--budget-tokens`. |

Add `--show-diagnostics` in the wizard to see full details.

## Files and reconnecting

- Connect saves to `llm_config.yaml` in NOOA's user configuration directory.
  Use `--output FILE` for another file, or `-w DIR` for the project file
  `DIR/.nooa/llm_config.yaml`.
- Other aliases and comments are kept. Connect asks before replacing an alias.
- The entry stores the key's variable name (`api_key_env`), never the key.
- A key saved in `secrets.yaml` is overridden by an exported environment
  variable with the same name.
- To edit a saved model, run `uv run nooa connect --edit-model NAME`. Omit NAME
  to pick from a list. Edits normally go back into the file that defines the
  model. Use `--output FILE` to choose another file.
- After saving to a custom file, set `NEMO_OO_LLM_CONFIG` to its path or load it
  with `reload_registry(Path(...))`. `-w DIR` saves to `DIR/.nooa/llm_config.yaml`.
- Editing a saved model runs fresh checks, which may cost money. Keep diagnostic
  reports separately if you need a record of earlier results.
- To load a config file from another path, add it to `NEMO_OO_LLM_CONFIG`.

## Choose media inputs during setup

A model receives images, audio, video, or PDFs only if its entry lists them in
`media_inputs`. NOOA assumes new entries are text-only (`media_inputs: []`)
until you choose formats. See the [media guide](media.md) for examples and
[migration](model-configuration.md#explicit-media-inputs-migration).

After selecting the model and its request interface, Connect reads the allowed
public/cached model information **before** offering media configuration. It shows
advertised formats and the source for only the selected model/route. OpenRouter
uses `architecture.input_modalities`; Anthropic uses the exact boolean flags
`capabilities.image_input.supported` and `capabilities.pdf_input.supported`.
For PDFs, use the specific PDF flag or a generated PDF check; a generic `file`
listing leaves PDF support unknown. Connect reads only the allowlisted fields,
not names, descriptions or arbitrary nested flags.

When suggestions are available, choose **Keep current media settings** (the default),
**Enable listed inputs without testing**, **Change inputs manually**, or
**Turn all media inputs off**. An optional **Test listed inputs before enabling**
choice appears only when
generated checks are selected and implemented for that interface. The guided
wizard still waits for **Test now** before sending any of those requests.
Suggestions never enable anything themselves, never replace explicit current choices, and
`--yes` never accepts suggestions without explicit media options. Missing/custom
metadata uses the existing manual/test choice without an extra suggestion prompt.
The proposed configuration and test/save/change decision follow media selection.
`--no-probe --no-catalogue` with an exact model/interface remains offline; editing does not fetch new metadata or
replace existing settings.

Decision suggestions allow **image only**, even if a shared model list includes
other inputs. System One JSON reports that image testing is unavailable; use
OpenAI Decisions, or direct TypeSafe with explicit multipart transport and
experimental opt-in, or leave images off. Responses/Anthropic audio and video
listings can be shown; use Chat to test those formats, or enable them without
testing. No transport is chosen from a model listing.

The decision image menu offers **Leave images off** (the new-entry default),
**Enable images without testing**, and **Test images before enabling**. An entry
with existing images defaults to **Keep current image settings**. TTY menus use
arrows and Enter; piped input accepts displayed numbers or `text`, `manual`,
`test`, and `keep` when available. Image tests use a small generated sample,
no user files, and the approved check budget.

When there is no advertised suggestion, chat setup keeps its existing choices:

- **Test all media inputs (generated samples).** Connect sends one small
  generated sample per format to check that this model accepts it through this
  connection. Formats that pass are enabled for this model entry. This uses the
  budget you already approved. Check the model's answers on your own examples
  before relying on it.
- **Choose media inputs manually.** No requests. The choice is saved as
  unverified.

Before saving, you can keep the result or change it by hand. A failed retest
keeps a previously enabled format. The summary shows **Images: off**,
**enabled (test passed)**, **enabled (not confirmed by this test)** for retained
failed inputs, or **enabled (not tested)** for manual/skipped inputs. These
settings are separate from the recorded check outcome.

Test limits:

- For chat-family interfaces, image and PDF are tested on all three API styles.
- Connect tests audio and video only with `--api-style chat`. Elsewhere it says
  “testing isn't available for this API format.” Use Chat to test those inputs,
  or enable them without testing. Responses audio request mapping remains
  unresolved here. Decision checks remain image-only.
- **Not available** means the check did not show that this model accepts that
  format through this connection.

From the command line:

```sh
# Set formats by hand (unverified, no requests):
uv run nooa connect your-model --as vision --endpoint https://gateway.example/v1 \
  --api-style chat --api-key-env MODEL_KEY --media-inputs image,pdf \
  --no-probe --no-catalogue --yes

# Test formats and enable those that pass:
uv run nooa connect your-model --as vision --endpoint https://gateway.example/v1 \
  --api-style chat --api-key-env MODEL_KEY --test-media-inputs image,audio,video,pdf \
  --yes

# Same, as a stage:
uv run nooa connect your-model --stage media --endpoint https://gateway.example/v1 \
  --api-style chat --api-key-env MODEL_KEY --test-media-inputs image,pdf \
  > result.json
uv run nooa connect --stage save --input result.json --output llm_config.yaml
# Register this explicit output path for the application:
export NEMO_OO_LLM_CONFIG="$PWD/llm_config.yaml"
```

Rules:

- `--media-inputs ''` or `-` means text-only.
- `--test-media-inputs` cannot be combined with `--media-inputs`, `--no-probe`,
  or `--probe minimal` / `--probe none`.
- In stage mode, `--test-media-inputs` works only with `plan`, `media`, and `all`.

## Library calls from NOOA agents

The CLI is a front end for `nooa.unifiedllm.connect`. Agent code can call it
directly:

```python
from nooa.unifiedllm import connect

proposal = connect.plan(
    "work", "your-model", "chat", "https://gateway.example/v1", "MODEL_KEY",
    budget_tokens=65536,
)
result = await connect.check_stage(proposal, "routing")
# Save only when you mean to:
# connect.write(result.entry, registry_path, alias=result.alias)
```

`plan()` sends no requests. `run()`, `run_steps()`, and `check_stage()` send the
test requests. `discover()`, `catalogue()`, `check_interfaces()`, and `write()`
match the stages above. The library never prompts or prints.

### Mixed interface discovery

The library defaults remain the three chat-family styles. Explicit per-style
`endpoints` overrides support mixed attempts; every decision style in a mixed
comparison requires its own full URL. All routes are validated before any
request, and all attempts share one token budget, bounded deadlines, and zero
network retries. Decision styles use their own client/validator and never
receive chat reasoning settings. With a nonempty `api_key` or configured
`api_key_env`, a mapped endpoint on another origin requires explicit
`allow_cross_origin_credentials=True`; the default is `False` and refusal occurs
before any interface request. Origins compare normalized scheme, host and
effective port, not paths. Only opt in for servers you trust to receive the key.

```python
async for event in connect.check_interfaces(
    "candidate", "exact-model-id", "https://inference-api.nvidia.com/v1", "HUB_KEY",
    styles=("chat", "responses", "anthropic", "systemone"),
    endpoints={"systemone": "https://inference-api.nvidia.com/v1/systemone"},
    reasoning_template="effort", budget_tokens=8192,
):
    if isinstance(event, connect.InterfaceResult):
        print(event.accepted, event.accepted_client_types)
        selected = event.results["systemone"].entry
        # Pass selected as plan(existing_entry=selected) to reuse routing evidence.
```

Each result retains its actual `endpoint` (decision) or `api_base` (chat), plus
transient `provenance.interface_endpoint` describing the selected style/URL and
whether it came from `endpoints` or the base argument. Accepted interfaces
record which connections passed. Test media separately and review answers on
your own examples before relying on the model.
