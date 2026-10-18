# nooa-atom

`nooa-atom` is NOOA Atom: an interactive agent, served over ACP with
`nooa atom`, and the Session layer it runs on.

The Session layer serves any NOOA interactive agent: a `Session`
that owns one agent and its turn loop, a `SessionRegistry` that keeps the tree
of sessions (a root and the children it delegates to), the agent-side port
(`self.session`) through which an agent creates and talks to children, and a
headless in-process host (`open_tree`). The Atom agent runs on that layer:
`nooa_atom.agent:AtomAgent` is the agent spec a host passes in
`SessionOptions.agent_spec`.

Atom reads its settings from the `atom` section of `settings.yaml` (user and
workspace `.nooa/`). The TUI and `nooa-acp` read `coding`; the first time
Atom reads a settings file that has `coding` and no `atom`, it adds an `atom`
copy, and the two sections change independently from then on.

Status: pre-release. The package is part of the workspace but is not published
yet, and its API may change until the hosts switch over to it. The ACP
adapter and the benchmark host move onto it in later changes.

Sessions are stored per workspace, in `<workspace>/.nooa/sessions`
(`sessions_root(workspace)`), where the `nooa-acp` server and the TUI keep
theirs; sessions they wrote are listed and load, with their saved agent
state when it can be restored. `NOOA_SESSIONS_DIR`, or an explicit
`SessionOptions.sessions_dir` / `--sessions-dir`, names one shared directory
for all workspaces instead. A `SessionRegistry` serves one such directory;
a subagent's session goes in its root's.

`run_task(options, task)` runs one task unattended in a new tree: the root
runs `handle_batch` turns until the agent returns `Done`, follows a turn
that ends `Waiting` to the next one, and stops at `max_turns` or `timeout`.
It returns a `TaskRun` with the final `Done` and its `TaskResult`, the
reason it stopped otherwise, the turn count, the usage and the root agent's
events. `nooa-bench run` and the benchmark runner's `atom` agent type use
it.

A host never holds a session's agent (`Session._agent` is private). It
submits items (`submit`, `prompt`, `steer`, `withdraw`, `cancel`) and reads
and changes the session through data: `info`, `transcript()`, `channels()`,
`model_info()`, `set_model()`, `set_reasoning()`, `set_mode()`,
`commands()` and `invoke_command()`, `plan()` (the agent's plan entries,
such as the Atom agent's todos), and, before the first turn,
`prepare_tools()` and `register_tools()`. `subscribe()` delivers every
change as a pydantic update, in order: turns, items, title, mode, model,
reasoning level, commands, usage, children, close, and each of the agent's
own events (`AgentEventUpdate`). The ACP adapter uses only these; a test
checks that the `nooa_atom.acp` package reads no `agent` attribute.

Over ACP, a `session/prompt` sent during a turn is queued, not steered.
`_nooa/session/inject` queues or steers a message without a prompt request,
and `_nooa/session/revoke_inject` takes one back; see `docs/acp-router.md`
("Messages during a turn") and the ACP RFD for message injection
(agent-client-protocol PR #1261). Pool sends what the person types during a
turn with its own request, `_poolside/session_steer`, which the server
advertises. The message is queued for the next turn, not steered, until the
Pool team says which of the two the request means. The open prompt stays open
until those messages are handled; Stop withdraws the ones no turn took and
lists them.

### Questions and explicit forms

Module-visible constructors from `nooa.interactive` work in generated CodeAct
cells and tagged JSON/Predict outputs:

```python
from nooa.interactive import (
    NeedInput, NeedInputForm, FormResponse, TextQuestion,
    ChoiceQuestion, MultiChoiceQuestion, FormChoice,
)

# Next ordinary message answers; never a dialog/permission.
return_result(NeedInput(question="Which branch?"))

# Explicit ordered questions; ids are stable, unique answer keys.
return_result(NeedInputForm(
    heading="Release details", reason="Choose the destination before deploying.",
    questions=[
        TextQuestion(id="name", label="Release name?", help="A human-readable name"),
        ChoiceQuestion(id="branch", label="Branch?", choices=[
            FormChoice(value="main", title="Main branch"),
            FormChoice(value="dev", title="Development branch"),
        ]),
        ChoiceQuestion(id="target", label="Destination?", choices=[
            FormChoice(value="staging", title="Staging cluster"),
        ]),
        TextQuestion(id="notes", label="Notes?", required=False),
    ],
))

# Outcome on notification["user_messages"], or external host/parent submission:
await session.submit(FormResponse(action="accept", content={
    "name": "Aurora", "branch": "dev", "target": "another cluster", "notes": "",
}))
await child_question.answer(FormResponse(action="decline"))
FormResponse(action="cancel")
```

`NeedInputForm` has a heading, optional reason, and ordered questions with unique
ids. `NeedInput` is still only a conversational question and optional reason; it
never opens a dialog or carries options. `FormQuestion` discriminates `text`,
`integer`, `number`, `boolean`, `choice`, and `multi_choice`. Constructors supply
these tags; persisted JSON descriptors include them.

Every question has `id`, `label`, optional `help`, and `required` (default true).
The native descriptors are small, not a fluent DSL or arbitrary JSON Schema API:

- `TextQuestion`: `default`, `min_length`, `max_length`, `pattern`, `format`.
  Formats (email, uri, date, date-time, and unknown strings) are preserved as
  annotations, not rejection rules. Patterns use stdlib regular-expression
  syntax in an isolated stdlib worker; compilation/search have a one-second
  wall-clock budget. Pattern/value limits are 4,096/65,536 characters, forms have
  at most 64 questions, and at most four workers run process-wide (excess work
  fails safely for retry, rather than queueing unlimited tasks). Unix workers
  additionally cap virtual memory at 256 MiB where supported; other platforms
  retain size/time/concurrency guards, not an equivalent memory sandbox.
  External answer validation awaits workers without blocking the event loop,
  kills/reaps cancelled work and rechecks ownership afterward. A bounded cache
  stores only immutable pattern/value results, never executable/compiled data.
  Unsupported syntax or timeout fails validation safely.
- `IntegerQuestion` / `NumberQuestion`: inclusive `minimum`, `maximum`, `default`.
  Numbers must be finite. Booleans are not numbers; no string-to-number parsing.
- `BooleanQuestion`: `default`.
- `ChoiceQuestion`: `choices`, optional string `default`. Each
  `FormChoice(value=..., title=None, description=None)` stores the literal value.
  Omit all titles for an untitled enum, or supply all titles for a titled enum;
  option descriptions require titles.
- `MultiChoiceQuestion`: the same choices, list `default`, `min_items`, `max_items`.
  Both native answers and defaults contain listed values and at most one
  written Other value. Aggregate bounds apply to the native list.

Every choice allows listed values **or written Other**; there is no strict picker
constructor. Each public `FormResponse` contains **one native answer per authored
id**: `str`, `int`, `float`, `bool`, or `list[str]`. `doc(NeedInputForm)` keeps
this primary contract compact; request `doc(TextQuestion)` (or another control)
on demand, or combine the six controls with `FormChoice` and `FormResponse` for
complete constructors. Missing optional answers remain
omitted. Required means presence, not nonblank: express text length/pattern rules
explicitly. Defaults are UI suggestions and are never inserted as missing answers.
Text and raw conversational fallback replies are preserved verbatim. Agents should
validate domain meaning and ask targeted follow-ups, not repeat a whole wizard.

```python
from nooa.interactive import (
    BooleanQuestion, IntegerQuestion, MultiChoiceQuestion, FormChoice,
    FormResponse, NeedInputForm,
)
need = NeedInputForm(heading="Deploy", questions=[
    IntegerQuestion(id="replicas", label="Replicas?", minimum=1, maximum=10, default=2),
    BooleanQuestion(id="dry_run", label="Dry run?", default=True),
    MultiChoiceQuestion(id="regions", label="Regions?", choices=[
        FormChoice(value="east"), FormChoice(value="west"),
    ], required=False, max_items=3),
])
need.validate_response(FormResponse(action="accept", content={
    "replicas": 2, "dry_run": True, "regions": ["east", "custom region"],
}))
```

`InputRequest` discriminates question/form; turn results are
`Done | InputRequest | Waiting`. Both input results are forbidden in unattended
`handle_batch`. The host renders requests, so do not also send them via `message()`.

### Capabilities and failure guarantees

Standard ACP uses advertised non-null `elicitation.form`, session-scoped
`mode="form"`, and restricted native `requestedSchema` fields. Untitled single
choices use `enum`, titled choices `oneOf` with `const`/`title`/optional description;
multiselect items use untitled string enums or titled `anyOf` options.

For choice/Other, standard ACP sends an optional enum field and optional Other
string companion. Generated companion ids are reserved against every authored id
and are private to the transport. Responses are validated against the exact owning
form, then collapsed. **A nonempty written Other replaces a single selection**
(even if both arrive): it is an alternate answer, never a note. Empty Other is
unused; whitespace is preserved. Multiselect appends one nonempty Other to its
listed array, unless already selected. Custom-only produces a one-item list.
Invalid selected enum values are rejected even if Other is also provided. The
public API never exposes paired answers, generated ids, or a notes dictionary.

Two limitations prevent claiming complete wire/schema conformance:

1. SDK 0.12.1's actual request serializer strips the required root
   `requestedSchema.type="object"` through `exclude_defaults=True`. It exposes no
   supported serializer hook and marks its connection wrapper final. The strict
   sender test remains strictly xfailed. A minimal upstream correction is to
   serialize the elicitation request with aliases and non-null defaults retained;
   deployment must adopt a supported SDK release containing that correction and
   pass the real-RPC sender test before claiming sender conformance. No version
   is guessed here. No global patch, nonstandard RPC, or private transport bypass
   is used here.
2. Restricted ACP has no conditional/union language for required-either-field or
   aggregate list bounds across an enum field and its Other companion. Logical
   requiredness and final list bounds are validated locally; the listed array also
   retains its emitted bounds. Thus, with `min_items=2`, one listed item plus Other
   fails the picker field's minimum even though the final length is two. A supplied
   listed array must independently meet its emitted min/max bounds, and the final
   list must meet aggregate bounds. Lowering picker minItems by one would admit
   mixed answers but weaken that emitted constraint for listed-only answers; it
   cannot encode the conditional aggregate rule. We retain standard picker bounds,
   not exact schema equivalence. Split-default representability is checked at the
   standard adapter boundary, not in native descriptors. Empty written Other
   defaults fail there with a precise error (empty ordinary text remains valid).
   A mixed default must put Other last to preserve native order through folding;
   custom-only defaults omit an empty picker default when its minimum forbids it.
   Such unrepresentable forms use complete plain-text fallback. Patterns use Python
   syntax, not an unrestricted ECMAScript regular-expression engine.

Pool (`clientInfo.name="pool"`, inspected version 1.0.16) uses its own
`_poolside/elicitation` extension. Required choice/Other uses the known equivalent
picker-or-text `anyOf`; optional text is supported. Optional choices, native
numbers/booleans/multiselect, defaults, text constraints/formats, and option help
use **complete ordered plain-text fallback**, never a broken dialog or a fabricated
parsed answer. Optional choices are never forced required. Multiple supported
questions carry `_meta["poolside/field_order"]` in authored order. Text-only clients
also get the complete request as readable questions, help, constraints, defaults,
and choices including Other. Raw replies are not converted to `FormResponse`.

Old `pick_one`/`pick_one_or_text` descriptors migrate narrowly in durable decoding
to `choice`, retaining question ids, values and recovery/owner tokens. Old string
answers are never reinterpreted as selections or companion fields by coincidence.
New constructors reject old tags and arbitrary extras; no custom validators are
restored. URL mode, nested objects, general unions, and conditional JSON Schema
are deliberately outside scope. Do not request passwords, tokens, or other secrets
in forms or plain text; use a separate secure authentication flow.

Accept, decline, and cancel retain their separate `FormResponse` actions. Only
accept carries internal content. Standard wire decline/cancel content is ignored;
omitted/null accept content is empty input, checked against the owning request
(required questions still reject it). A Pool automatic unsupported-schema decline and a user's
Escape decline return the same `{"action":"decline"}` without a reason. The
adapter cannot infer intent from this wire action and does not label it a user
decision or reinterpret it as a pending request. The original heading/reason and
ordered questions, help, and choices are rendered as visible text before asking, and preserved
on replay; decline is still admitted once as the actual client-reported outcome.
No automatic reask of the whole form is performed. Stop is transport/prompt cancellation and submits no
answer, even if a client swallows cancellation. Ownership is checked without an
await before admission; superseded dialogs cannot admit stale responses. Successful
user-message admission claims/invalidates the form once, even before dispatch starts.
Withdrawal does not resurrect the dialog. ChildQuestion carries an internal durable
request token; its answer method rejects superseded same-id forms. Manual session
submit targets the current request; hosts retaining a request can supply request_id. Responses
are recorded before queueing, echoed once, and replayed without reopening dialogs.
Unknown SDK actions are preserved as opaque durable `Metadata` transport records,
not accepted answers. Their content, metadata and vendor fields are not printed,
echoed, or included in transcript/model context. Raw history remains untrusted data,
not a secure authentication store. Known actions keep their existing semantics.
Malformed client payloads fail closed **once** with actionable fallback guidance;
there is no whole-form retry or manufactured answer. Failed Pool extension calls
disable the extension for that connection. Unsupported/text-only hosts preserve
the ordered questions as text; raw text stays raw,
not an accepted form outcome. No URL/nested-object widget support is claimed.

`FormResponse` validates its three-action envelope and native-value dictionary.
The owning request's `validate_response()` checks authored ids, required key
presence, native types and descriptor bounds, without parsing or inserting defaults.
Only the standard listed selection field checks enum membership before folding Other. External hosts should use `session.submit()`; `admit()` is
record-before-queue plumbing, not an external form-validation API. Child requests
carry descriptors as data; the owning child validates at submission.

### Migration and durable records

Replace draft `NeedInputForm(question=..., answer_type=Model/options=...)` with
`NeedInputForm(heading=..., questions=[...])`. Old typed authoring and presentation
switches are rejected. No dynamic Pydantic answer class is needed or restored.

New durable forms store `outcome_kind="need_input_form"`, `kind="form"`, heading,
reason, ordered descriptors, and an internal ownership token directly; JSON roundtrips preserve them.
Questions retain `need_input`. Recovery preserves request intent without reopening
historical dialogs, and does not restore forms superseded by later admission or
an unfinished turn. Legacy child forms without ownership tokens require a fresh
request before they can be answered through ChildQuestion.answer(). Old explicitly untyped text/choice forms migrate to one
`answer` descriptor with string-dictionary content. Old records containing an
answer class or JSON answer schema remain unavailable original data: acceptance
is refused and the agent must request a new descriptor form, even if the old class
is importable. This does not claim recovery of old custom validators. Decline,
cancel, or explanatory raw text remain possible. Old conversational
`NeedInput`/`ChildQuestion` records drop obsolete `options` at durable decoding
only, preserving question/reason. New constructors reject `options`; form choices
and old form migration are unchanged. Old `ChildQuestion` form records drop only
the former null options carrier; non-null legacy form choices stay unavailable
data. Old schema data is retained only as legacy display data. Historical decline markers cannot recover lost action intent.

## Skills and MCP servers

The Atom agent reaches all its capabilities through one `SkillManager`,
`self.skills`: code skills (`Skill` classes from `nooa.skills` entry points,
workspace libraries and `.py` files in skill directories), text skills
(`SKILL.md` directories) and MCP servers (`.mcp.json`, `atom.mcp_servers`
in the settings files, and the servers an ACP client sends). The model uses:

- `search(query, limit=10)`: one line per match with name, kind, state and
  description. Searching loads nothing; an installed skill that is not
  loaded is described by its package's summary.
- `await activate(names)` / `await deactivate(names)`: a code skill is loaded
  and becomes `self.<name>`; an MCP server is connected and its tools are
  methods of `self.<server>` (deactivating keeps it connected); a text
  skill's instructions arrive on the `system_messages` channel at the start
  of the next turn.
- `read(name)`: a text skill's instructions, as text.
- `doc(name)`: the full description of one skill, with an MCP server's tools.

The prompt holds one small block, which changes only when something is
activated:

```text
<skills>
Active: libwriting, methodwriting, repo, shell, todo, workspace_settings, github (mcp)
38 more (code, text, MCP): self.skills.search('query'); await activate(['name']); read('name') for text skills
</skills>
```

`/skills list` shows every skill with its kind and state; `/skills activate`
and `/skills deactivate` save the choice for code skills in the workspace
settings. `/mcp` shows the MCP servers and their state. A server runs only
after the person approves its exact configuration with
`/mcp approve NAME` (review) and `/mcp approve NAME CODE`.

MCP sign-in uses the MCP SDK's OAuth support (discovery, client
registration, PKCE, token refresh). Because a sandbox cannot receive a
browser callback, sign-in is manual: the link is shown (as an agent message
when the agent activated the server, in the command output for
`/mcp approve`), the person opens it and signs in, then copies the address
the browser ends on (the page may fail to load) and runs
`/mcp auth NAME ADDRESS`. The agent is told when the server connects.
Tokens are kept in `~/.config/nooa/mcp_oauth.json`, readable only by the
user, so later connections need no sign-in. The `oauth_open_browser` and
`oauth_manual` server settings are accepted but no longer used.

A session saved before the skill manager loads normally; the old `<mcp>`
block is dropped and the agent is told what was not restored.

## Recovering a session marked in use

A session in use is marked by its lock file, so that no two processes write
it, including a sandbox and its host sharing the directory. A process that
crashes leaves the mark behind, and the session is then left out of the
session list and cannot be opened. In an ACP session, `/recover` lists the
sessions of the workspace that are marked in use, with their owner and the
time of their last write. `/recover <id, id prefix or title>` copies one
into a new session titled "<title> (recovered)", which `/resume` then
opens. The original file is never modified; if it is still in use
elsewhere, the copy is a branch from that point. Queued messages the
original never read and its subagent sessions are not carried over, and
the agent is told so. A damaged file is copied event by event, and the
reply says how many events could not be read. The store API is
`SessionStore.in_use()` and `SessionStore.fork()`.

The design is tracked in issue #388.


## Dispatch and durable input

Session supplies two core turn hooks: cancellable preparation for pending model
activation, and synchronous commit for durable input ownership. Initial
`ItemConsumed` rows and `TurnStarted` are appended in one SQLite transaction before
any dequeue or agent invocation. Core does not know Session IDs or storage.
`admit()` is the synchronous record-before-queue surface for registry event delivery;
`submit()` remains its async facade. `requeue()` owns `ItemRequeued` and the queue
transfer, without changing identity or replay policy. Lifecycle/client-sharing hooks
are constructor parameters (`before_close`, `llm_in_use`); `closing` is public.

A preparation/commit failure leaves inputs queued and pauses dispatch. Existing
prompt/outcome waiters fail with `TurnFailedError`, including a later outcome query;
new external admission is rejected while `dispatch_error` is set. Repair the fault,
then call `session.resume_dispatch()` explicitly, or close/reopen to replay eligible
input. Resume first requeues retained buffered steers. Already reported terminal
outcomes are not rewritten by retry: observe updates or submit a new prompt. A
preparation cancellation likewise pauses without consumption and resolves open
prompts as cancelled; explicit resume can retry the still-queued input. A normal
committed cancellation retains the existing cancelled-output/TurnCancelled contract.

Blocked dispatch rejects external submit/prompt/steer work, but internal registry
admission still records and queues child deliveries while paused. Finished children
are auto-closed only after successful delivery; if ledger admission itself fails,
the child remains live with its own durable result available for recovery.

Pause is live dispatch control, not durable suspension. Session close pauses before
children cleanup, then stops/settles the loop and closes owned resources. Recursive
close from cleanup (including callback child tasks) is safe; self-close from an
executing turn is rejected. Closed sessions resolve late outcome queries for queued
admissions too. Core loop details are in `src/nooa/runtime/README.md`.

Steers during preparation are ordinary queued messages. During execution, buffered
steers are consumed one at a time before the next model call; close/cancel/failure
leftovers are admitted as messages with the same IDs. Recording failures retain
remaining buffer ownership, fail relevant outcomes and block dispatch. Mid-turn
get/drain consumption keeps its identity until the ledger write succeeds; failure
moves consumed-but-unrecorded ownership into a separate orphan recovery map (not
queued identity tracking), retaining the object, channel and error. It fails the
relevant prompt and the turn even if the method returns Done. Submitting the same
object after resume creates a new, independently settled receipt. The raw
item has already reached the consumer at that point, so resume cannot magically
replay it in memory. Close/reopen can replay its never-recorded consumption; external
side effects may already have occurred. This is not exactly-once execution.

Model selections are persisted immediately; effective model info/options change
only on successful activation. Pending activation is attempted once, not implicitly
retried every turn. A failed activation restores the old client and retains the new
client for cleanup. Old/pending/failed clients remain owned across cancelled disposal
or metadata failure; clients shared with children are retired until children close.
A hung old-client close is cancellable preparation with input still queued and old
cleanup ownership retained. Final resource close requires cooperative clients; there
is no timeout/forced-disposal policy in this change.

### Compatibility and scope

The prototype `before_turn(notification)` was new, unpublished and had no external
callers in this checkout. It combined async preparation with recording after
consumption, so preserving that contract would defeat the approved boundary. It is
replaced by `prepare()` and synchronous `commit(batch)`, rather than a compatibility
shim that silently changes notification timing. Supplied hook tests migrate to these
APIs and explicit blocked-dispatch recovery; unrelated assertions stay intact.
`stop_starting()` remains an alias for pause. Existing race callers are unchanged.

Crash load still requeues only admitted items without recorded consumption,
withdrawal or discard. It does not resume Python coroutines or consumed Waiting
obligations. No new delivery IDs or replay policy are introduced. Commands/mode
remain host-neutral; ask mode still does not enforce permissions. Later reasoning,
MCP/auth, command serialization and adapter initialization are outside this change.
Shared reader guides and GitHub are not modified.


## macOS / Zed: isolated FAKE-provider form test branch

This is a **test branch**, `test/pr432-zed-forms`, containing the current native
form work plus `nooa_atom.acp.form_demo`. It is not a release or a claim that the
installed Zed GUI supports these forms. No model account/API key is needed.
**Do not run `nooa connect` or select Zed's paid/default agent for this test.**

### Fresh checkout and locked setup

Keep existing clones/worktrees and their changes intact. Use a new directory:

```sh
git clone --branch test/pr432-zed-forms --single-branch \
  https://github.com/NVIDIA-NeMo/labs-OO-Agents.git "$HOME/src/nooa-form-test"
ROOT="$(cd "$HOME/src/nooa-form-test" && pwd)"
git -C "$ROOT" rev-parse HEAD
```

If uv is missing, follow its [official installation guide](https://docs.astral.sh/uv/getting-started/installation/)
(the official installer is `curl -LsSf https://astral.sh/uv/install.sh | sh`), then
open a new terminal. Use uv-managed Python **3.13**, not macOS system Python or
3.14 (this workspace requires `>=3.12,<3.14`):

```sh
uv python install 3.13
uv sync --project "$ROOT" --package nooa-atom \
  --python 3.13 --managed-python --locked --no-default-groups
uv run --project "$ROOT" --package nooa-atom \
  --python 3.13 --managed-python --locked --no-default-groups \
  python -c 'import sys, importlib.metadata as m, nooa, nooa_atom, acp; print(sys.version); print(nooa.__file__); print(nooa_atom.__file__); print(acp.__file__); print(m.version("agent-client-protocol"))'
command -v uv
printf '%s\n' "$ROOT"
```

The lock pins official `agent-client-protocol` **0.12.1** and editable workspace
core/CLI/Atom. `--package nooa-atom` includes its transitive core/MCP/CLI/SDK
requirements; the runtime demo does not need pytest or dev groups. `--locked`
fails instead of changing dependencies. Keep Git metadata for dynamic package
versions. Initial setup downloads public dependencies/build tooling; a fresh
macOS sync has **not** been verified locally. Do not change the SDK or lock to
hide a failure. See uv's [workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/)
and [sync](https://docs.astral.sh/uv/concepts/projects/sync/) documentation.

The actual launch command, which waits for an ACP client on stdio, is:

```sh
uv run --project "$ROOT" --package nooa-atom \
  --python 3.13 --managed-python --locked --no-default-groups \
  python -m nooa_atom.acp.form_demo
```

Use Zed to launch it rather than typing ACP JSON by hand. No shell init,
project environment recreation, test-only fixture or `PYTHONPATH` hack is needed
on the Mac.

### Add one custom external agent (preserve other settings)

Follow [Zed's official Custom Agents instructions](https://zed.dev/docs/ai/external-agents#custom-agents):
**Agent Settings → External Agents → Add Agent → Add Custom Agent**. This opens
your settings file. Merge **only the named entry** into your existing
`agent_servers` object; do not replace the whole settings file or your other
agents/provider settings.

Replace the two absolute-path placeholders with the outputs above (`command -v
uv` must resolve to an absolute executable path). JSON does **not** interpolate
`$ROOT` or shell quotes. Each argument below is one string; paths containing
spaces need no embedded quotes. There is deliberately **no `cwd` setting**.

```json
{
  "agent_servers": {
    "NOOA Forms — FAKE": {
      "type": "custom",
      "command": "/ABSOLUTE/PATH/TO/uv",
      "args": [
        "run", "--project", "/ABSOLUTE/PATH/TO/nooa-form-test",
        "--package", "nooa-atom",
        "--python", "3.13", "--managed-python",
        "--locked", "--no-default-groups",
        "python", "-m", "nooa_atom.acp.form_demo"
      ],
      "env": {}
    }
  }
}
```

Open an **empty scratch project** in Zed (recommended even though the demo
ignores the client workspace), open the Agent Panel, and start an **external
agent** thread with **NOOA Forms — FAKE**. Record your exact installed Zed
version. No minimum stable/Preview release has been verified for this UI.

### Human checklist — fresh thread for each scenario

Send any harmless prompt. The two-cell fake script shows a form, then summarizes
your actual response; it never checks for a prechosen accepted payload. It has
strict exhaustion: repeat prompts in a finished thread fail visibly rather than
silently falling back. Start a new thread for the next case.

1. Inspect six authored types: optional **Name** string (max 80 chars), **Count**
   integer 1–5 (default 2), **Ratio** number 0–1 (default 0.5), **Enabled** boolean
   (default false), **Target** dev/prod (default dev), optional **Tags** string
   list fast/safe (default fast, max 3). Both choice controls allow **Other**;
   this mapper adds two companion text widgets, so six types need not mean six
   visible widgets.
2. Accept arbitrary valid values, e.g. count 4, ratio 0.25, enabled false,
   target prod, tags safe. Check the JSON/native-type summary: int, float (or int
   for a whole number), bool, string and list[str]; false is not a string.
3. Single choice: choose dev **and** write a custom Other. Custom text replaces
   the picker and is the **sole** `target` string. Multiselect: pick tags and
   write one Other; it appends once unless already selected. Final result has
   one authored answer per field, never `__acp_other_*` keys or notes. Empty
   Other is unused. Try custom-only answers as well.
4. Check defaults and optional fields. Defaults are UI suggestions, **not**
   inserted answers. Omitted optional keys stay omitted and are listed in
   `optional_omissions`; empty string/list are different from omission. Actual
   Zed omission behavior must be observed, not assumed. Missing required keys
   or invalid values are rejected, without automatic retries/coercion.
5. In separate threads, **decline** and **form cancel**: the summary must retain
   the distinct action and show `content: null`. If Zed does not offer these
   distinct controls, record that limitation rather than treating Stop as cancel.
6. Separately test thread **Stop** with a pending dialog. It stops the prompt
   (`cancelled`), does **not** manufacture a FormResponse, and must not print a
   normal native-answer summary.
7. Without advertised `clientCapabilities.elicitation.form`, questions appear
   conversationally. Your next plain-text message is reported as **raw input**,
   never parsed/coerced into a native FormResponse.

If no dialog appears, use the Command Palette action **`dev::OpenAcpLogs`**
([official debugging docs](https://zed.dev/docs/ai/external-agents#debugging-agents))
to inspect initialization capability and actual `elicitation/create` messages.
Keep logs local/redact personal values before sharing. **SDK 0.12.1 drops the
required root `requestedSchema.type="object"` while serializing defaults.** It
is intentionally unchanged here; the strict sender diagnostic remains xfailed.
Real Zed must establish acceptance/rendering, and that does not establish full
sender conformance. Do not configure a real model to "fix" a form/UI failure.

### What is isolated, and what this test does not prove

FakeLLM alone does **not** prevent workspace skills or forwarded MCP tools from
running in the normal server. This entrypoint specifically uses the official
SDK's `run_agent`, real `AtomACPAgent`, real `SessionRegistry`/`Session`, real
Atom CodeAct cells and the current real mapper/validation/admission path, but:

- A **demo-only adapter** rewrites session workspace/store selection to a newly
  allocated private scratch tree, skips workspace settings/model discovery and
  all session tool preparation, and ignores forwarded MCP servers with a visible
  warning. The factory constructs Atom directly without workspace skill/config
  setup. No external MCP launches/connections occur.
- It clears inherited credentials/model/config/trace overrides, sets private
  HOME/user/project/session/cache roots, disables automatic trace initialization,
  and disables only InteractiveAgent's eager import-time real-provider default
  in this process (a scoped bootstrap override, not a production server patch).
  Each session receives a fresh strict FakeLLMClient; no provider call is made.
- Unknown installed skill entry points cause a fail-closed startup error. Use the
  locked dedicated environment, not one with arbitrary custom plugins.
- A process-local stdlib audit guard rejects external subprocesses and network
  sockets (including loopback); stdout is reserved for ACP, diagnostics go to
  stderr. This is defense in depth for the fixed script, **not an OS security
  sandbox for malicious dependencies/arbitrary Python/untrusted clients**.
- Scratch state is deleted on normal shutdown. Forced kill/crash can leave a
  temporary `nooa-form-demo-*` directory. No supplied path is recursively deleted.
  Answers may remain in Zed's own thread history/logs: enter no sensitive data.
- The simpler **single-process** demo takes no CLI arguments and does not test
  production router/worker lifecycle, PTY/Pool, live providers, or Mac GUI
  rendering. Production server/SDK source is not monkeypatched by this demo.

Offline tests in `tests/acp_adapter/test_protocol.py` launch the **shipped
module** via the unchanged SDK, covering native schema/Other/defaults, arbitrary
accepts/omissions, decline/cancel, invalid content, fallback, Stop, scratch-only
workspace and an external MCP sentinel that must never run. For a developer
with an already provisioned dev environment:

```sh
uv run --project "$ROOT" --package nooa-atom \
  --python 3.13 --managed-python --locked --no-default-groups --group dev \
  pytest packages/nooa-atom/tests/acp_adapter/test_protocol.py \
         packages/nooa-atom/tests/acp_adapter/test_need_input.py \
         packages/nooa-atom/tests/acp_adapter/test_questions.py
```

Local offline Linux results and limitations are recorded in the publication
report; neither macOS dependency setup nor actual Zed widgets were run here.
