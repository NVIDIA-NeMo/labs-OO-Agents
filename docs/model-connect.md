# Configure a model with NOOA Connect

`nooa connect` asks for approval of API-call costs upfront, checks the selected model with bounded
requests, then asks before saving its registry entry. The same `nooa.unifiedllm.connect` library is available to
the TUI: frontends supply consent and display; the library supplies the plan,
UnifiedLLM checks and registry updates.

## CLI

Start the guided setup with no arguments:

```sh
uv run nooa connect
```

### Guided interaction

On a terminal, finite decisions are visible labelled menus: provider, API format,
model-settings actions, default reasoning level, reply-limit presets, and media
setup/review. Use arrow keys and Enter; a prompt without a default starts at
**Choose an option…** and requires a selection. Media declarations offer all
16 combinations of the four supported formats, including **Text only**.

Model IDs from a server are explicitly **search advertised model IDs** inputs
with Tab completion. Endpoints, key-variable names, masked keys, aliases,
numeric limits and custom reasoning-level lists remain editable. Reply-limit
**Custom…** opens a separate numeric input. Yes/no confirmations still ask
explicitly before checks, replacement, entry writes and optional key storage.
Piped scripts retain their existing token answers and defaults; media mode/review
also accept the existing numbered answers. `--yes` bypasses menus, and
`--no-probe` keeps manual setup unverified without generation calls.

Check rows use one vocabulary: **Passed**, **Not available**, **Skipped**, or
**Failed**, followed by elapsed time when measured. Explanations appear below
instead of extending the header. Not-available attachment access and optional
cache misses are distinct from failed requests; retained manual permissions are
still shown separately. Reasoning cost/opaque-state sizes remain available with
`--show-diagnostics`, not in the normal installer flow. This is a presentation
change, not a change to qualification criteria or permission/save policy.

## Independent stages for agents

Endpoint `/models` limits take priority over public catalogue limits, with sources
shown per field. An endpoint's `max_input_tokens` is retained separately and used
as a conservative context-management bound when needed; it is not added to the
output limit to invent a total context size. Explicit user edits still win.
When a context bound is unknown, both library provenance and stage JSON warn.

Save discovery JSON and reuse it with `--discovery-file discovery.json` on the
wizard or a plan/check stage. The file must name the same normalized endpoint;
limits are selected by exact model ID. This keeps `--stage plan` offline:

```sh
uv run nooa connect --stage discover --endpoint https://gateway.example/v1 \
  --api-key-env MODEL_KEY > discovery.json
uv run nooa connect your-model --stage plan --api-style chat \
  --endpoint https://gateway.example/v1 --discovery-file discovery.json > model-plan.json
```

Cache checks report both continuation readings. Substantial reuse on either
confirms caching; a later miss does not erase that evidence. A stable request
with implicit caching (Chat), or explicit cache markers but no reported reuse,
produces a warning rather than making a working entry fail. Missing required
markers or an unstable prefix still need attention. Reasoning rows show puzzle
correctness separately from whether reasoning was returned.

After an interface failure the wizard offers **retry one interface with a
120-second timeout**. This is an explicit new routing check, charged against the
same approved budget, not an automatic network retry. Both the SDK read timeout
and whole-check deadline use the longer limit. Successful discovery plus timed-out
model calls is described as a possibly slow route, not proof of bad credentials.

`--stage` runs without prompts and writes one JSON result to stdout. It never
saves implicitly. A generation stage runs directly within its configured limits;
there is no separate spending-approval dialogue. The interactive wizard remains
available when `--stage` is absent.

| Stage | Purpose |
|---|---|
| `discover` | List the endpoint's models (not proof of authentication). |
| `catalogue` | Read public model metadata; optional MODEL filters candidates. |
| `interfaces` | Check API formats and report which worked. |
| `plan` | Build an unsaved configuration and proposed requests; no network. |
| `routing` | Run one configured typed NOOA text baseline with the chosen interface. |
| `tools` | Check whether the model produces the requested tool call; never execute it. |
| `reasoning` | Check declared levels supplied by `--levels-file`. |
| `session` | Check cache reuse and reasoning replay across turns. |
| `all` | Run the basic and conversation checks. |
| `save` | Write the entry from a JSON plan/result to a registry. |

For example:

```sh
uv run nooa connect --stage discover \
  --endpoint https://gateway.example/v1 --api-key-env MODEL_KEY

uv run nooa connect your-model --stage routing --as work \
  --endpoint https://gateway.example/v1 --api-style chat --api-key-env MODEL_KEY

uv run nooa connect your-model --stage plan --as work \
  --endpoint https://gateway.example/v1 --api-style chat --api-key-env MODEL_KEY \
  > model-plan.json

uv run nooa connect --stage save --input model-plan.json --output llm_config.yaml
```

The diagnostic handoff names the target file, working directory, active registry
files and effective alias source, installed version, credential availability
(never the value), caps, timeouts and remaining budget. Interface failures include
a quoted CLI reproduction command capped at three basic checks when budget remains.
Timeouts are `not_confirmed`, with elapsed time and exception-chain class names;
they do not prove that the server received the request. The handoff includes
same-run discovery status, proxy-variable presence (never values), safe planned
request settings, and the running package location. It resolves absolute local
skill/doc paths from that installation, or supplies a checkout command pinned
to its recorded commit or release version. Unknown revisions are stated, never
silently replaced by main. A target outside the current registry chain is called
out explicitly. The handoff is not new permission
to spend money or overwrite other aliases.

Use explicit endpoint/interface/key-variable options in stage mode; wizard
presets and `--no-probe` do not apply. Stages normally never prompt; explicitly
passing `--prompt-key` enables one masked credential prompt on stderr, leaving
JSON on stdout. Pasted keys are not persisted or included in reproduction commands.
`--budget-tokens` defaults to
unlimited; it is only capped when the flag is passed explicitly.
`--output-tokens` controls only initial interface discovery.
Configured routing, tools, reasoning and conversation checks send the saved
`--max-tokens` value, or the selected level's cap. `--reasoning-output-tokens`
is retained for command compatibility but no longer overrides configured caps.
Redirect stdout to keep stage reports; `--output` is
the registry target for `save` only. Replacing an existing alias requires `--yes`
and prints a warning to stderr. Saving an untested plan does not validate it.

Reports have `version`, `stage`, `ok`, `data`, `checks`, `error`, `run_context`, `warnings`, and
`diagnostic_prompt`. Exit 0 means the stage met its criterion; 1 means failure
or missing evidence, not proof of unsupported features; 2 means invalid stage
options. Request acceptance alone is not success for tools or enabled reasoning
checks. A session requires retained reasoning; a provider cache miss with a stable
prefix is a warning, not a broken connection.
The diagnostic prompt includes safe route/credential-variable names and outcomes,
not key values, raw provider error bodies, or returned reasoning. It asks an
agent to investigate, repair, and rerun the affected stage within configured
limits. Wizard failures show a short explanation and point to
`--show-diagnostics`, which prints the full library-generated handoff. Stage JSON still includes the handoff
without an extra flag. Neither frontend launches another agent automatically.
Rerunning checks for diagnostics can incur charges; the flag only changes display.

`data` contains the full library result, including the synthetic prompts in the
probe plan/provenance; `checks` omits those request bodies. Neither contains
returned reasoning or credential values. Local file failures retain their path;
YAML failures identify the file, line and column without echoing file contents.

## Library calls from NOOA agents

The CLI is a frontend to `nooa.unifiedllm.connect`, not a subprocess requirement. Inside
an async NOOA agent method, use:

```python
from nooa.unifiedllm import connect

proposal = connect.plan(
    "work", "your-model", "chat", "https://gateway.example/v1", "MODEL_KEY",
    budget_tokens=65536,
    reasoning_levels={"high": {"reasoning_effort": "high"}},
)
result = await connect.check_stage(proposal, "reasoning")
checks = result.entry["provenance"]["probes"]
handoff = connect.diagnostic_prompt("reasoning", result.entry, checks)
# Explicit persistence, when wanted:
# connect.write(result.entry, registry_path, alias=result.alias)
```

`check_stage` runs selected checks afresh and leaves the input plan unchanged.
Other public stages are `discover`, `catalogue`, `match_models`, `plan`,
`check_interfaces`, and `write`. Use `run_steps` for progress events on an approved
plan. The library never prints, prompts, reads stdin, or launches a diagnostic/repair
agent. The configured text baseline uses a transient NOOA agent with a typed
Predict method (see below); frontends decide presentation and persistence.

## Interactive setup

Choose NVIDIA (build.nvidia.com), OpenAI, Anthropic, Google (Gemini), OpenRouter,
or a custom endpoint. Presets fill in the public server URL and key variable;
model names still come from the endpoint, not a bundled list. The order is
server, credentials, model selection, then automatic interface checks. It tries
Chat Completions, Responses and Anthropic Messages once each and offers only
interfaces that returned the expected response format. One success is selected
automatically; multiple successes give a choice. A timeout, authentication error
or rejected request is not labelled “unsupported.” If no check succeeds, the wizard
offers to change the key, edit the server and model, retry, or cancel without saving.
Failed attempts remain charged to the original approved budget; correcting the
connection does not increase it or ask for another spending approval. Exhausting
that budget ends setup without saving. Scripted `--yes` runs still fail without
prompting. A successful model listing is not described as authenticated: some
servers list their models publicly. Provider help banners are hidden during CLI
checks, while safe authentication, routing and timeout explanations remain visible.
A server listing models is not evidence that every model on it uses the same
interface. `--api-style` supplies an explicit choice and skips interface detection;
scripted `--yes --provider ...` setup uses the preset default if none is supplied.
The wizard shows published model details
before asking to use them: context window, maximum reply length, reasoning levels,
and default reasoning level. Missing values say “Not listed.”
These come from OpenRouter's model listing; server limits may differ. The setup
check's output cap is shown separately and does not become the model's default.
Choose **use**, **edit**, **skip**, or **cancel**. Editing lets you change the
context window, maximum output, reasoning levels and default reasoning level;
Enter keeps each suggestion and `-` leaves a field unknown. The revised details
are shown again before you accept them, and saved edits are attributed to you.
Confirmed edits replace command-line limit/level settings; skipping the details
keeps any explicit command-line settings. Skip means continue without these
published details, not cancel. Cancel stops setup without saving.
It then shows the remaining checks and budget. The initial warning explains that setup
makes paid API calls and asks once for approval before any generation; there are no repeated approval prompts for these checks.
Each check shows progress and its result as it runs. Saving is a separate
confirmation; Ctrl-C cancels setup. A pasted key stays in memory unless you
approve storing it in NOOA's secrets file at final save. The model entry names
its environment variable, not its value.

After the checks finish, the final **Save model** step asks “Save this model as”.
The suggested name, existing-name completion and overwrite confirmation all live
here. `--as` supplies the name without prompting; `--yes` still requires it.
Naming or declining a replacement does not rerun checks. Cancelling at this step
leaves the registry unchanged; any already-completed API calls have still occurred.

The wizard replaces the active check line on a terminal and prints compact
results with elapsed time, then counts passed, inconclusive and skipped checks.
Conversation checks are numbered 1/3 through 3/3; cache reuse and reasoning
retention have separate result rows. Piped output stays readable without cursor
controls, and `NO_COLOR` disables colors. The full YAML is hidden by default;
`--show-config` previews the final entry before the save question. The agent
`--stage` JSON interface is unchanged.

The displayed **reported reply ceiling** is capability metadata held under the
session's `provenance.catalogue_limits.max_completion_tokens`, not a request default;
it is never persisted to the registry.
Every entry also has an actual **reply budget**, saved as `max_tokens` for all
three interfaces. Responses translates that to `max_output_tokens` on the wire.
Press Enter to accept the recommendation, choose a higher budget for high reasoning
(65,536) or extended reasoning (131,072), choose the model's declared maximum output
directly via "Model maximum" (no `--custom` needed to hit that ceiling exactly),
choose a smaller 8,192- or 2,048-token budget, or choose Custom to edit the number.
Higher options — including "Model maximum" — appear only above the recommendation
and within known model limits; they do not change the reasoning level itself. High/
extended presets are truncated against "Model maximum" when the model's declared
ceiling is lower. Smaller options appear only when
below the recommendation. Connect offers the catalogue's output recommendation when available,
otherwise 32,768 (labelled NOOA default), bounded by the known ceiling and half the
known context window. Explicit caps may exceed half the window but must leave
room for input: a cap at or above the window is rejected before saving.
Existing entries keep their current setting. The budget includes thinking
and the final answer; short replies use fewer tokens. The custom editor shows the
known upper limit as a constraint, not as a suggested allocation.
`--max-tokens N` (also `--reply-tokens N`) sets it in scripted or interactive
mode; `--output-tokens` controls only initial interface discovery.
The library's `configure_entry()` and `write()` enforce the same defaults,
including on stage-save input created before these fields were required.
Keep reply caps, and Responses `store`/`include`, at the entry's top level:
Connect rejects copies in `extra_body` that would override those settings.
A large ceiling can be valid while being unsuitable
as a default: input, reasoning and the answer must fit together in the context
window. The wizard highlights ceilings close to that window. Published
thinking on/off information is shown even when named effort levels are absent;
this display does not invent request mappings. A declared thinking-token budget
that would leave no room for the answer raises only that level's reply cap,
with a warning. It must still fit the model limit; checks cannot exceed their
approved cap to test such a level.

If enabled reasoning-level checks succeed but return neither reasoning fields
nor reported reasoning tokens, Connect warns once before saving and lists the
affected levels. It suggests another API format or checking the server settings.
Disabled reasoning settings are exempt, as are rejected or skipped checks. A
server may hide reasoning information, so this warning does not claim that
reasoning is off. The saved results keep acceptance and reasoning observation
separate. The TUI can use `connect.unobserved_reasoning_levels(entry)` for the
same warning. This does not add calls or prove reasoning quality; the setup
question remains a small connection check.

Full setup also runs a three-call conversation check through the saved entry's
UnifiedLLM client and the default cached renderer. It seeds a longer reference
prompt, replays the actual reply, then repeats that saved history with a different
final question. It does not execute tools or invent reasoning state. It reports:

- Cache reads as a fraction of input tokens, whether explicit markers were sent,
  and whether the stable request prefix matched. Confirmation requires cache
  reads covering at least half of the seed's input; a small tool-schema-only hit
  is not enough.
- Whether reasoning appeared in replies, whether its readable/native state reached
  the follow-up's reasoning fields unchanged, and whether the selected reasoning
  controls survived on the wire. Ordinary text fallback does not count as retained
  reasoning. Missing evidence says “not confirmed,” not “reasoning is disabled.”

These calls send the configured reply cap, including the selected reasoning
level's override. The three-turn reservation is `3 * (8192 + 3 * cap)`, accounting
for reusable input and earlier replies. If that does not fit the remaining approved
budget, the conversation check is skipped; Connect never silently lowers the cap.
For example, a 32,768 cap requires a 319,488-token conversation reservation;
if an explicit `--budget-tokens` is set below that, choose a larger approved
budget to run it — the default shared budget is unlimited and always covers
it. Routing/tool/level checks each reserve their configured cap plus 512.
There are no automatic length retries: truncation at the configured cap is
inconclusive, and the partial response is never replayed. To try a larger cap,
edit the configuration and rerun the affected checks within your allowance.
Detailed JSON retains configured/tested caps, stop reasons and usage. A saved cap
without an accepted wire-verified check is labelled unverified.
These are estimates, not billing limits: servers can ignore caps. Use
`--budget-tokens` before setup to choose another budget. Small context windows,
incompatible reply caps, failures or insufficient budget leave the conversation
check unconfirmed or untested. Results include sanitized counts and flags only;
raw responses, signatures, encrypted state and captured bodies stay in memory.
Responses entries use `store: false` and request
`include: [reasoning.encrypted_content]` by default, with source `connect` recorded
in provenance. The wizard explains that encrypted reasoning carries reasoning
context between turns without requesting stored responses. This is a compatibility
request, not a guarantee of reasoning availability; current native OpenAI APIs may
return that state automatically. See [OpenAI's reasoning guide](https://developers.openai.com/api/docs/guides/reasoning).

A 400/422 explicitly rejecting `include` or `encrypted_content` records a
sanitized request rejection. A configured typed baseline is one-shot: it does
not disable the option or send another candidate automatically, and wizard
saving is blocked. To deliberately opt out, configure `include: []` and run a
fresh approved baseline (or choose manual unverified setup). The saved opt-out
omits the field on the wire, including on native endpoints. A session rejection
can disable the option, stops the conversation check and leaves retention
unconfirmed; it does not rerun the session. Authentication, rate limits, server
failures and invalid input-history errors do not change the option. Reconnecting
to the same route preserves a recorded opt-out; changing routes tests the default
again. Cache settings are still tested without overrides.
`--no-probe` disables these calls too; `--yes` explicitly approves the selected
checks as well as saving. Library frontends opt in with `plan(..., session_checks=True)`.

If you do not want paid checks, use `--no-probe` or follow
[manual model configuration](model-configuration.md), optionally with the
`nooa-model-configuration` skill. With `--no-probe`, interface selection is manual.

In a terminal, type part of a model name to filter a scrolling completion menu;
Tab selects a match. Large catalogues do not print every model into the prompt.
Provider, endpoint, API-format, alias, catalogue and confirmation prompts also
complete their available choices. Server URL suggestions include every valid
`api_base` in the existing target configuration file, as well as provider presets;
duplicates and URLs with embedded credentials are excluded. The URL menu opens
immediately: use arrow keys and Enter to pick a server, or type to filter or enter
a different URL. Key-variable completion uses environment
variable **names only**, never their values. Secret entry is masked and has no
completion or history. Defaults appear as faint suggestions: Enter accepts one,
typing replaces it, and Right Arrow brings it into the editor. Arrow keys,
Home/End, Backspace and Delete edit the current answer. Existing alias names in
the target file show a warning while typing; replacement still requires confirmation.
The terminal groups setup into four steps, with a provider menu and F1 help.
Piped input uses plain line prompts.

The initial approval warns: “Checks may incur charges. The token budget is an
estimate, not a spending cap.” It shows the token budget and initial per-call
limits. After approval, normal output does not repeat charge warnings or price
estimates; it shows progress, saved reply limits and the remaining token budget.
Budget exhaustion and skipped checks remain visible.
Normal progress uses concise labels such as `Connection: Passed`. A failed
answer, incomplete reply, request error or skipped check is still shown; detailed
request evidence is available in stage JSON or with `--show-diagnostics`.
Manual setup is labeled `Connection: Unverified (manual setup; no model checks)`
before saving. This does not change any check criteria or retry/budget rules.

Flags can prefill answers or support scripted setup:

```sh
uv run nooa connect --provider nvidia

uv run nooa connect gateway/model --as work-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY
```

The model argument is the exact ID used by the endpoint, without the additional
LiteLLM routing prefix. API styles are `chat`, `responses`, and `anthropic`.
Omit MODEL to list and select from the endpoint's models. Discovery tries `/models`
and then `/v1/models` if a root endpoint returns 404. For Anthropic, explicitly
name the appropriate key environment variable. `--prompt-key` reads a masked key
for this setup; Connect offers to save it at final confirmation and never changes
process environment variables. A project or explicit override secrets file can
shadow the user secrets file, just as an exported shell value can.
An empty `--api-key-env ''` supports local servers without authentication.
In the wizard, enter `-` at the key-variable prompt for no authentication.

Preset endpoints follow the public connection guides for
[NVIDIA](https://docs.api.nvidia.com/nim/docs/api-quickstart),
[OpenAI](https://developers.openai.com/api/reference/overview),
[Anthropic](https://platform.claude.com/docs/en/api/overview),
[Google's OpenAI-compatible API](https://ai.google.dev/gemini-api/docs/openai), and
[OpenRouter](https://openrouter.ai/docs/quickstart). They do not certify that a
particular model supports every optional feature; the approved probes check that
endpoint. Frontends can reuse the defaults through `connect.PROVIDERS`.

OpenRouter metadata supplies candidate model names, context and output limits,
prices and reasoning levels where present. Confirm the candidate; a matching
name does not prove that a gateway exposes the same capabilities. Ambiguous
matches require a choice, including with `--yes`. Use `--catalogue-model` for an
explicit choice or `--no-catalogue` to leave it unknown.

Use `--no-probe` to save without model calls, or `--probe minimal` for routing
only (up to three interface attempts unless `--api-style` is supplied).
The default plan also checks tools and each proposed reasoning level. The
selected interface's discovery acceptance is not a configured baseline and cannot
be reused as one. A fresh configured typed baseline is required before wizard saving.
Initial interface discovery uses 200 output tokens and a 30-second deadline.
Configured checks use the saved cap and a 120-second deadline;
conversation-check calls also have 120 seconds per attempt. Only truncated
conversation replies do not retry automatically. The CLI uses
the fixed budget approved at the beginning. Each basic request reserves its
output cap plus 512 estimated input tokens. An explicit `--budget-tokens` limits
the entire setup, including interface detection; it is never increased. Connect
warns before the remaining checks if that limit is too small. Larger reported
usage increases the charge. The next request is skipped if the budget would be
exceeded, and a warning lists checks left undone. If the configured text baseline
was skipped (including budget exhaustion), failed or incomplete, the wizard stops
before any save/key prompt or write, even with `--yes`. Optional tools, reasoning
or media checks needing attention do not by themselves block saving after a
valid baseline. To deliberately save a manual **unverified** configuration, rerun
with `--no-probe` or `--probe none`; supply an explicit model and `--no-catalogue`
to avoid discovery/metadata requests too. See [manual setup](model-configuration.md).
These are estimates, not billing caps: a gateway may ignore an output limit. There are no
context-window or maximum-output capacity probes.

When metadata has no levels, they remain unknown. You can supply candidates:

```sh
nooa connect gateway/model --as work-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY --no-catalogue \
  --reasoning-template effort --levels low,medium,high
```

Other templates are `adaptive`, `budget`, `toggle`, and `thinking`. They are
candidate request shapes, not provider support tables. For exact route settings,
`--levels-file` accepts a YAML mapping of labels to whole request blocks instead.
Quote labels such as `"off"` that YAML otherwise treats as booleans. A template
whose output allocation exceeds the approved cap is left untested; Connect does
not silently increase the budget. `--context-window` supplies a known limit with
user provenance; an unknown limit remains absent and the current runtime's
fallback applies.

The saved entry uses the current `model_name`/`client_type` schema and the
registry's reasoning-level mechanism. Every generation check uses UnifiedLLM,
constructed from the unsaved entry through the same factory as `get_llm_client`.
That tests runtime routing, settings translation and response parsing, not just
whether the server accepts a hand-built request. Context and maximum-output limits are
metadata, not permission to generate that many tokens on each call.

## Files and reconnecting

To edit a model, run `uv run nooa connect --edit-model NAME`. With no name,
`uv run nooa connect --edit-model` opens a registry selector. This skips server
listing and catalogue discovery and opens the settings editor. Existing custom
request fields and reasoning-level bodies are preserved; adding a new level
requires its complete request settings in `--levels-file`. Checks use the edited
entry, and changes are written only after save confirmation. The defining user,
project or override file is edited; bundled defaults are copied into the user
registry. `--output` chooses a different destination and `--as` renames the copy.

When an endpoint already appears in the registry, Connect selects its saved key
variable automatically. Multiple saved variables open a selector; an explicit
`--api-key-env` always wins. URL completion also uses all registry layers.

You can choose `new` at the key-variable prompt, or use `--prompt-key`, to enter a
masked key without first exporting an environment variable. At final save,
Connect offers to store that key in NOOA's user `secrets.yaml`, under `env:`.
The file is plain text with owner-only permissions (`0600`), written atomically;
existing secret values are preserved but YAML formatting may change. The model
entry contains only `api_key_env`, never the key. Cancelling before save writes
neither file. `--yes` does not implicitly consent to storing a pasted key.
NOOA loads the secrets file on startup; an exported shell value takes precedence,
so unset or update an old exported value when replacing a key.

Declared reasoning parameters are explicitly allowed through legacy parameter
filtering. Level checks also inspect the serialized HTTP body: a request that
succeeds after losing its settings is not a confirmed check. Session reports
distinguish settings reaching the server from reasoning state surviving replay.
Neither observation proves the server honoured an effort value.

The default is `llm_config.yaml` in NOOA's user configuration directory. Connect
warns before overwriting an existing alias, including a hand-written one, and
asks for confirmation before replacing it. All selected checks run before the
local alias is chosen. Names are reread at this step to catch entries added while
checks ran. `--yes` skips save/overwrite confirmation;
it still prints the overwrite warning. Other aliases and surrounding
comments stay intact. Writes replace the file atomically, follow existing symlinks,
and retain an existing registry's mode and CRLF newlines. Flow-style mappings may
lose internal comments when rewritten; other alias values remain unchanged.
Cooperating writers serialize the complete update through a retained adjacent
`.FILENAME.lock` file. Files containing YAML anchors/aliases are refused with
instructions to expand them first; Connect does not silently flatten them and
discard comments. Literal credential fields are rejected at any nesting depth
by the shared library, not only the JSON frontend.

Unchanged accepted UnifiedLLM probes are reused when reconnecting. Older direct-HTTP
checks are repeated: they did not test the runtime. Changing the route or
level declarations changes which probe requests can be reused; unchanged requests
on the same route remain reusable. `--output` chooses another
file; load custom paths with `NEMO_OO_LLM_CONFIG` or `reload_registry(path)`.
Later configuration layers can override the alias; inspect `llm_config_chain()`
if a saved entry does not take effect. The wizard and save stage warn when a
higher-priority file shadows the saved alias.

### Saved routing and evidence fields

New entries include `transport: direct`, selecting the SDK transport once direct
support (#337) is installed. Earlier runtimes ignore this field and continue using
LiteLLM; checks on those runtimes are not evidence of direct-transport behavior.
Each completed check records the runtime's actual `transport` separately.
If the runtime bypasses Connect's owned HTTP pool (for example an unauthenticated
legacy fallback), accepted requests are labelled unobserved, not as settings
proven missing from the wire.
Edits preserve explicit `transport: litellm` choices. `api_style` identifies the
wire interface for Connect and the forthcoming direct runtime. Connect does not
infer a `replay_vendor` from an interface or model name.

`provenance` contains diagnostic evidence and metadata, not runtime request
settings. It lives in the connect session and in stage JSON, never in the
saved registry: `connect.write` persists the discovered settings only, so the
config stays the smallest capture of what Connect learned. Catalogue identity
belongs under `provenance.catalogue.id`, not `underlying_model`. Successful
tools are recorded in `provenance.probes.tools`; Connect does not write an
unused `tools: true` capability switch.

## Library interface for the TUI

```python
from nooa.unifiedllm import connect

# Optional, before selecting the model (no generation calls):
discovery = await connect.discover("https://gateway.example", api_key=temporary_key)
# The frontend selects an ID from discovery.models.
proposal = connect.plan(
    "work-model", selected_model, "chat", discovery.api_base, "MY_MODEL_KEY",
    catalogue=None,
)
# Show proposal.entry, proposal.probes and its estimates in the frontend.
# Frontends own the cost warning or approval. plan() above performs no I/O;
# run() below sends the approved requests and may incur charges.
result = await connect.run(proposal, approved="minimal", api_key=temporary_key)
# Separately obtain approval to save.
connect.write(result.entry, destination, alias=result.alias)
```

`run` accepts `all`, `minimal` or `none`. Cancelling its task cancels the HTTP call.
For inline feedback, consume `run_steps()` instead: it yields `ProbeUpdate`
objects before and after each check, followed by the final `ConnectResult`.
Use `contextlib.aclosing` if the frontend may stop reading early. `run()` uses
the same iterator internally, so CLI and TUI checks cannot diverge.
For interface detection, consume `check_interfaces(alias, model, api_base,
api_key_env, ...)`. It yields `ProbeUpdate` events named for each interface and
then an `InterfaceResult` containing `accepted`, the per-interface `results`,
and `tokens_charged_to_budget`. Offer only `accepted`, pass the chosen result's
entry to `plan(existing_entry=...)`, and deduct that charge from the remaining
budget before running additional probes. Discovery tests plain interface
acceptance, not the typed baseline. Only identical, complete configured typed
baseline evidence with the actual settings observed can be reused. Level and
tool requests still need their own checks.
There are no callbacks, terminal imports, prompts or execution of returned tools.
The text baseline does instantiate a transient NOOA Predict agent; it does not
launch a diagnostic/repair agent or an iterative CodeAct session. Discovery uses
HTTPX; generation checks lazily load UnifiedLLM and the current runtime
(LiteLLM by default). No temporary registry entries or global
registry changes are needed. Each checked client is closed even if its call fails.
The TUI keeps model selection, confirmation, secret persistence and switching;
it can call these async functions directly without invoking Click or a subprocess.
Its existing Ollama-specific adapter remains separate from these three API styles.

## What the observations mean

### Configured typed text baseline

Configured `routing`, `media` and `all` checks use an actual NOOA generation
method `compute() -> int` with `PredictStrategy`, constructed from the unsaved
entry via the runtime client factory. It computes `17 * 19`: success requires
validated integer **323**, an observed final answer (reasoning alone is not
sufficient), complete `stop` finish, accepted outcome, and the configured reply
cap/selected settings observed on the actual request. Missing or legacy evidence
and plain discovery acceptance cannot establish this baseline.

This is the runtime's native **scalar JSON value contract**: the integer return
is represented by the object-root structured output schema `{"value": 323}`.
Bare `323`, prose, wrong integers, invalid types, and truncated output do not
pass. Endpoints must support the runtime's structured output request in addition
to accepting plain text; for Anthropic the adapter may use a schema tool for that
contract. This is distinct from the tools capability probe; returned capability
probe tools are never executed. The independent `tools` stage remains tools-only
and does not require or run a typed baseline.

The baseline is bounded to one outbound generation request: no Predict repair
retry, automatic alternate include candidate, reduced cap, or context-window
recovery request. It shares the approved budget. Transient answer/reasoning/raw
response values are not retained in agent history or exported as diagnostic
facts. A failure reports sanitized evidence and requires fixing the endpoint or
explicit manual unverified setup; it is not proof that plain text or optional
media is unsupported. Wizard `run_checks` and a defensive `save_model` guard
both prevent ordinary default-yes/`--yes` saves without this evidence when checks
were approved. `--no-probe`/`--probe none` still permit manual unverified saves.
Low-level `connect.write` and explicit `--stage save` are manual persistence
operations, not an automatic wizard baseline approval gate.


Each record distinguishes successful runtime calls and observed reasoning.
The stored request is the planned input, not a claim that every field survived
runtime translation unchanged. Reasoning observations come from actual returned reasoning content/parts, not
reported reasoning-token usage alone. A successful call alone does not establish
that a setting had an effect.
Reasoning-level checks use an eight-job scheduling puzzle and request only the
final order. `answer_correct` scores that public answer independently of
`reasoning_observed`; a correct answer alone does not prove reasoning was enabled.
The puzzle exists to elicit reasoning, not to prove the model can solve it: a
wrong answer with reasoning genuinely observed does not flag the check for
attention. Only a missing reasoning signal does.
Per-call input, output and reasoning-token counts are included when usage is
available. The final answer and reasoning text are not saved. The default
reasoning-check cap is the configured reply limit; a `length` finish is inconclusive.
An agent can explicitly revise it with `--stage reasoning --max-tokens 65536` within its approved
`--budget-tokens` limit if more room is needed. There is no automatic retry.
HTTP 400 is recorded as rejected, not unsupported. Auth, timeout and transient
failures remain untested; failed routing or auth stops subsequent calls within
that interface's plan. Interface detection still tries the other styles within
the same budget, because their authentication conventions differ. Reasoning
not being visible can be normal, especially at a disabled level. Returned tool
calls are inspected as data and never executed. Raw model responses, server error
bodies and credential headers are not retained. Limits and defaults keep their
catalogue source and are explicitly marked as not probed.
An endpoint speaking Responses does not imply it accepts explicit cache fields;
the session checks inspect markers and reuse under the runtime's cache defaults.
Responses entries use `store: false` and request `reasoning.encrypted_content`;
an explicit `include: []` opts out. Checks use the same client as an agent.

## Code walkthrough: what and why

- `src/nooa/unifiedllm/connect/__init__.py`: plans data first so either frontend can obtain consent;
  runs bounded UnifiedLLM calls through the registry's shared client factory;
  updates one alias while retaining other entries and comments.
- `packages/nooa-cli/src/nooa_cli/commands/connect.py` and its `_connect_*`
  helpers: argument parsing, interactive prompts, progress, preview and approval.
  Click and prompt-toolkit belong to `nooa-cli`, not NOOA core. The CLI imports
  the framework lazily; the Connect library never imports the CLI. The TUI can
  call the library directly without these presentation dependencies.
- `_connect_wizard.py` separates connection, model selection, interfaces,
  metadata, configuration, checks and save steps. Both frontends use the library's
  `verdict(entry)` policy; `ProbeRecord` in `connect/_records.py` documents optional
  evidence fields. Missing fields mean unknown evidence, not zero or unsupported.
  `public_record()` excludes private requests from public check summaries.
  After changing an entry in a plan, call `refresh_plan()` to rebuild detached
  requests and reservations before running checks.
- `tests/unifiedllm/connect/`: library budgets, exact HTTP bodies, targeted writes
  and the real registry on main, including a frontend-dependency isolation check.
- `packages/nooa-cli/tests/test_connect*.py`: wizard approval, prompts, progress,
  key privacy, cancellation and scripted stages.

This is an internal organization, not a separately installable UnifiedLLM package.

Request-shape references: [OpenAI Responses](https://developers.openai.com/api/reference/python/resources/responses/methods/create)
and [OpenRouter model metadata](https://openrouter.ai/docs/api/api-reference/models/list-all-models-and-their-properties).

## Choose media inputs during setup

New entries default to `media_inputs: []` (text-only). Connect never infers these
permissions from model names or catalogue capabilities. PDF means exactly
`application/pdf`, not every file format. See the
[MIME mapping and migration guide](model-configuration.md#explicit-media-inputs-migration).

The wizard asks for **one approval at the beginning** covering setup checks and
optional generated-media tests within the same budget. No inference checks run
before that approval. **How would you like to configure media inputs?** immediately
shows **Test all media inputs (generated samples)** and **Choose media inputs manually**.
Enter accepts the selected default: tests for new setup, manual selection when editing.
Use arrow keys then Enter to change it; Ctrl-C cancels without saving. Piped input
accepts Enter for that default, `1`/`test` for tests, or `2`/`manual` for manual selection.
TEST automatically plans `image`, `audio`, `video`, and `pdf`; there is no follow-on
format-entry prompt.
`--no-probe`/`--probe none` never run media tests. Manual
selection sends no media requests; it saves explicit unverified declarations.
Editing keeps existing declarations unless explicitly changed. Retesting alone
does not revoke existing declarations: inconclusive/skipped/rejected retests leave
the prior explicit permission in place, labeled unverified by that test. Before saving, the wizard shows the effective enabled list. After
testing you can keep that list or explicitly choose unverified inputs manually.

Guided TEST mode makes at most one media request per format when a portable
sample is available and the approved budget permits it, following a
correct, complete [configured typed text baseline](#configured-typed-text-baseline).
All tests appear in the plan and share the
approved setup budget. Each reserves the saved reply cap (reasoning and answer
share that cap) plus 4,096 estimated input tokens for the small generated sample.
Checks that do not fit are skipped; no reduced caps, media retries, additional
approval prompts or automatic budget expansion. Usage can increase the charge;
input estimates and output caps are **not hard billing limits**. Cost depends on
the endpoint; catalogue price estimates are shown only when available.

Samples are generated locally in memory or bundled from synthetic frames; no
user files, uploads or external fixture downloads:

- **Image:** valid 32×32 red PNG, passed as a native NOOA `Image` argument.
- **Audio (Chat only):** original mono PCM WAV with two separated tones, passed
  as `Audio`. Responses and Anthropic have no portable native inline WAV mapping
  here and are **not tested**; another API may support audio.
- **PDF:** valid one-page PDF containing MAPLE, passed as `File` with
  `application/pdf`. Image/PDF keep their Chat, Responses and Anthropic mappings.
- **Video (compatible Chat routes only):** original bundled 1-second MP4,
  H.264/yuv420p, 64×64, four frames, no audio, 1,675 bytes, passed as `Video`.
  The inline `video_url` mapping is a backend extension, not a portable OpenAI
  contract. Responses and Anthropic have no native video mapping here and are
  **not tested**. No frame conversion, transcription or other-model substitution.

All four methods ask whether the model can access and interpret the attachment,
and instruct it to return false if it cannot. This is **model-reported access,
not an accuracy benchmark**: no tone counting, color grading or word quiz.
Each returns a strict boolean through the real Agent/Predict runtime using
`PredictConfig(max_retries=1)` (one attempt), the configured client/transport,
selected settings, saved reply cap and approved reservation. Predict's native
collector attaches the original argument once. The object-root schema wraps
`{"value": true}` or `{"value": false}`; the native method returns the boolean.
Strings, integers, prose and malformed responses are not coerced or manually
salvaged. Endpoints must support both the native schema and attachment mapping.

Public stage evidence separates `request_accepted` (successful observed HTTP,
even if parsing fails), `typed_result_validated`, optional boolean
`access_reported`, `final_answer_observed`, `attachment_observed` and
`output_settings_observed`. Schema failures do not imply false access. Completion
and provider usage survive parse failures; raw replies, reasoning, provider
error text and attachment bytes are not diagnostic evidence. Only a completed
public final answer counts, not reasoning-only or unfinished/tool-limited replies.

Validated **true**, with a successful request, complete final answer, original
attachment and configured settings observed, is **passed** and may newly enable
that format. Validated **false** with the same evidence is **unavailable**, an
ordinary result: `Not available · Model cannot access this input`. It is not a
request error or a wrong answer. Explicit provider media rejection is
**rejected**; malformed, incomplete, missing-evidence and request failures are
**inconclusive** and display **Failed** with their specific cause. Neither enables
new inputs. Skipped formats display **Skipped**.

Existing explicit permissions survive false or failed retests unless changed
manually. Rows distinguish disabled inputs from retained manual choices; review
shows enabled and unavailable formats and lets users continue with enabled
inputs or choose manually if known supported. The review menu asks **How would
you like to continue?** and immediately shows **Continue with text only** (or
**Continue with image, pdf**, for example) and **Choose media inputs manually**.
The first choice is selected: Enter continues unchanged, arrow keys then Enter
choose an action, and Ctrl-C cancels without saving. Only the manual action opens
the labelled format-combination menu on a terminal (comma-separated input for pipes);
the final write confirmation remains separate.
Piped input shows numbered, labelled choices: Enter or `1`/`keep` continues;
`2`/`manual` opens manual selection. `--yes` skips this menu and keeps enabled inputs.
Unavailable inputs may need another
model or API, without initiating extra requests. `--yes` introduces no prompts
or extra charges after the initial approval. One test is not permanent capability
proof. Observations stay in Connect reports, separate from saved declarations.

Unattended choices are explicit and mutually exclusive:

```sh
# Manual declaration; no inference (listing/metadata are also disabled here).
uv run nooa connect your-model --as vision --endpoint https://gateway.example/v1 \
  --api-style chat --api-key-env MODEL_KEY --media-inputs image,pdf \
  --no-probe --no-catalogue --yes

# Test chosen formats and save only those that passed; requests may incur charges.
uv run nooa connect your-model --as vision --endpoint https://gateway.example/v1 \
  --api-style chat --api-key-env MODEL_KEY --test-media-inputs image,audio,video,pdf \
  --budget-tokens 100000 --max-tokens 4096 --no-catalogue --yes

# Offline plan preview; no credentials/network needed.
uv run nooa connect your-model --stage plan --endpoint https://gateway.example/v1 \
  --api-style chat --test-media-inputs image,pdf --max-tokens 4096 \
  --budget-tokens 100000 > plan.json

# Explicit paid media stage includes a fresh text baseline, no tools/session tests.
uv run nooa connect your-model --stage media --endpoint https://gateway.example/v1 \
  --api-style chat --api-key-env MODEL_KEY --test-media-inputs image,pdf \
  --max-tokens 4096 --budget-tokens 100000 > result.json
uv run nooa connect --stage save --input result.json --output models.yaml
```

`--media-inputs ''` or `-` clears declarations. With `--yes`, omission keeps an
existing entry's declarations or defaults a new entry to text-only. Tests require
`--probe all`, cannot combine with `--media-inputs` or `--no-probe`, and stage
mode accepts tests only for `plan`, `media`, `all`. `--stage save` preserves its
input entry's list unless `--media-inputs` explicitly overrides it. Invalid
format/flag combinations fail before requests or writes. Stage result JSON
contains sanitized test facts; plan JSON includes the generated sample payloads.

### Video fixture provenance and offline validation

`src/nooa/unifiedllm/connect/_video_fixture.py` bundles synthetic bytes under
the repository's Apache-2.0 license: no third-party/user media or download.
Generated locally using FFmpeg 8.0.1 / libx264 from RGB24 solid frames;
SHA-256: `dc8c43009ecf3887c4d9d4b5dd8be8adfefb0517e85d12b4aeeb3d0e59eb8c90`.
The raw input is `bytes((255,0,0))*4096*2 + bytes((0,0,255))*4096*2`.
Generation command (raw bytes on stdin; metadata/container timestamps are not
used as the correctness oracle):

```sh
ffmpeg -v error -f rawvideo -pixel_format rgb24 -video_size 64x64 \
  -framerate 4 -i pipe:0 -an -c:v libx264 -threads 1 -pix_fmt yuv420p \
  -movflags +faststart sample.mp4
```

Connect itself requires **no FFmpeg, ffprobe or encoder package**. The offline
container/integrity test always runs. With development FFmpeg/ffprobe on PATH,
the independent decode test verifies one H.264 video stream (no audio), 1-second
duration, 64×64, 4 fps and all four decoded RGB frames: red, red, blue, blue.
It checks high dominant-channel / low other-channel values, not just the hash
or an MP4 header. Without those tools, that development decode test explicitly
skips; it does not claim validation passed. No inference occurs in these tests.

```sh
uv run pytest tests/unifiedllm/connect/test_connect_media.py
# Normal paid setup: choose TEST all media formats after the initial approval,
# or explicitly test video on a Chat backend implementing video_url:
uv run nooa connect your-model --endpoint https://gateway.example/v1 \
  --api-style chat --api-key-env MODEL_KEY --test-media-inputs video \
  --max-tokens 4096 --budget-tokens 100000 --no-catalogue
```

No live inference was used to develop/validate this fixture; successful mocked
wire tests show serialization and scoring policy, not a model's video ability.
