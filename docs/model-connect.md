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
| `routing` | Test one basic request with the chosen interface. |
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
limits. Wizard failures print the same library-generated handoff; neither
frontend launches another agent automatically.

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
plan. The library never prints, prompts, reads stdin, or launches an agent;
frontends decide presentation and persistence.

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

The displayed **reported reply ceiling** is capability metadata saved under
`provenance.catalogue_limits.max_completion_tokens`, not a request default.
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

A 400/422 explicitly rejecting `include` or `encrypted_content` disables the option
and records a sanitized rejection. Authentication, rate limits, server failures,
and invalid input-history errors do not change it. The saved `include: []` opt-out
omits the field on the wire, including on native endpoints. A routing rejection
can lead to one new capped check without the option, charged to the original
budget; no unchanged request is retried. A session rejection stops the conversation
check and leaves retention unconfirmed. Reconnecting to the same route preserves
the recorded rejection; changing routes tests the default again. Cache settings
are still tested without overrides.
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
selected interface's successful routing request is reused, not sent twice.
Initial interface discovery uses 200 output tokens and a 30-second deadline.
Configured checks use the saved cap and a 120-second deadline;
conversation-check calls also have 120 seconds per attempt. Only truncated
conversation replies do not retry automatically. The CLI uses
the fixed budget approved at the beginning. Each basic request reserves its
output cap plus 512 estimated input tokens. An explicit `--budget-tokens` limits
the entire setup, including interface detection; it is never increased. Connect
warns before the remaining checks if that limit is too small. Larger reported
usage increases the charge. The next request is skipped if the budget would be
exceeded, and a warning lists checks left undone before saving. These are
estimates, not billing caps: a gateway may ignore an output limit. There are no
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
settings. Catalogue identity belongs under `provenance.catalogue.id`, not
`underlying_model`. Successful tools are recorded in `provenance.probes.tools`;
Connect does not write an unused `tools: true` capability switch.

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
budget before running additional probes. This reuses the exact accepted routing
request; level and tool requests still need their own checks.
There are no callbacks, terminal imports, prompts, agent instances or tool
execution. Discovery uses HTTPX; generation checks lazily load UnifiedLLM and
the current runtime (LiteLLM by default). No temporary registry entries or global
registry changes are needed. Each checked client is closed even if its call fails.
The TUI keeps model selection, confirmation, secret persistence and switching;
it can call these async functions directly without invoking Click or a subprocess.
Its existing Ollama-specific adapter remains separate from these three API styles.

## What the observations mean

Each record distinguishes successful runtime calls and observed reasoning.
The stored request is the planned input, not a claim that every field survived
runtime translation unchanged. Reasoning observations come from readable response
text or reported reasoning-token usage. A successful call alone does not establish
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
