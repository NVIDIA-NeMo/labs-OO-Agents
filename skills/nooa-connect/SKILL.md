---
name: nooa-connect
description: Configure or diagnose NOOA model connections headlessly with permission-first stages, bounded checks, JSON evidence and explicit registry saves. Canonical agent-facing Connect workflow for chat-family and native decision routes.
---

# Run Connect headlessly as an agent

Use this skill for model onboarding and Connect diagnostic handoffs. Other
NOOA skills refer here for the complete headless workflow. Agents use
**`--stage NAME`**, not the interactive wizard, piped answers or `--yes` to
operate guided setup. Never pipe a prompt or use `--prompt-key` unattended.

## Permission comes before configuration

Ask before configuring a model unless the current scoped request already
explicitly authorizes it. Agree on the endpoint, exact model ID, alias and target
registry, and separately bound:

- Network metadata lookups and model checks, including destinations, request
  count, token allowance and timeouts.
- Media tests and the formats/samples that may be transmitted. Permission for
  synthetic samples is not permission to send private user media or signed URLs.
- Local report/file writes, alias creation, and replacement of an existing alias.
- Credential use and any credential persistence, separately. An available key
  is not permission to send it to an arbitrary endpoint.

For example: “May I configure alias `work` in this project's registry, run one
routing check with a 4,096-token allowance and a 1,024-token reply cap, and save
it if it passes?” Once approved, proceed within that scope without repeatedly
asking. Ask again before expanding checks, changing destinations, raising caps
or allowance, replacing an unapproved alias, or persisting credentials.
Permission to author an agent or inspect a diagnostic command is not permission
to configure/test models. Stage mode does not obtain permission for you.
**`--yes` on stage save only permits alias replacement; it is not user consent.**

## Inspect installation, configuration and credentials

Run from the user's project with its installed CLI:

```sh
uv run nooa connect --help
```

Check the available stages/options first. Use matching release/commit docs,
not examples from arbitrary development main. In a source checkout these links
are relative; copied skills need their siblings and matching source docs. A
wheel without docs needs the source references in `run_context` or the handoff.
See [registry semantics](../nooa-model-configuration/SKILL.md) for precedence.

```python
from nooa.llm_config import llm_config_chain

print(llm_config_chain())
```

Inspect target and effective alias sources, existing custom fields, and any
transport override before editing. Prefer an existing suitable alias.
Stage checks do not load an alias as an editable connection: supply its actual
endpoint, model ID and settings explicitly.

Use **key variable names only** with `--api-key-env MODEL_KEY`; never print or
paste values into commands, registry entries, reports or commits. Let the user
configure environment variables or layered `secrets.yaml`. Use
`--api-key-env ''` for a credentialless route. A configured variable must be
nonempty for endpoint checks; offline plan does not need its value. Stage save
never persists a key. Do not use masked-input flags in headless work.

## Stage and report contract

`--stage NAME` emits one JSON report to stdout; capture stderr separately.
Reports have `version`, `stage`, `ok`, `data`, `error`, `checks`, `warnings`,
`run_context` and `diagnostic_prompt`. Only save writes the registry; redirecting
reports is still a local write requiring an approved workspace.

| Stage | Operation |
|---|---|
| `discover` | GET the endpoint's advertised models; no inference test. |
| `catalogue` | GET public OpenRouter metadata; suggestions, not route verification. |
| `interfaces` | Compare Chat, Responses and Anthropic, or check the selected decision style. |
| `plan` | Build an unsaved proposal locally; no network. |
| `routing` | Chat-family configured typed integer baseline; decisions single Boolean routing probe. |
| `tools` | Chat-family tool-call observation; the tool is never executed. |
| `reasoning` | Chat-family checks for levels declared in `--levels-file`. |
| `media` | Chat-family routing baseline, then explicitly selected generated media access tests. |
| `session` | Chat-family three-turn continuation, caching and reasoning replay checks. |
| `all` | Routing, tools, declared levels, selected media, then session; decisions routing only. |
| `save` | Write one alias from a reviewed plan/result file; no network. |

Exit **0** means the stage met its criterion, **1** failure or missing evidence
(including skipped/inconclusive checks), and **2** invalid options. Inspect
`ok`, `checks`, `warnings`, `error` and `diagnostic_prompt`, not exit alone.
Handled stage failures are JSON; Click parsing/prevalidation errors may instead
be plain stderr with exit 2. Library validation errors can be handled as exit 1
(for example decision tools/media/session or reply-cap misuse); inspect `error`
rather than assuming every unsupported option combination returns 2. Catalogue can succeed with no matches; save success
is not route validation. Wizard exit status is not this stage contract: its
failed baseline can return exit 0 with “Nothing was saved”.

Stage mode rejects wizard flags `--provider`, `--catalogue-model`,
`--edit-model`, `--levels`, `--no-probe`, nondefault `--probe`, and
`--show-config`. `--no-catalogue` does not disable stage catalogue;
`--show-diagnostics` is unnecessary for stage failure handoffs.
`--reasoning-template` and `--reasoning-level` apply only to interfaces.
**`--input` is save-only**; `--output` and `--working-dir` are save-only in stage
mode. Redirect other reports. Checks build a **fresh plan from flags**, not from
prior plan JSON, and do not accumulate earlier evidence. Repeat the same
settings/files for each check and keep reports separately.

## Minimal proposal → check → review → save

Replace placeholders with the approved connection. This Chat example uses an
alias that does not already exist. The context size below is an example of a
supplied known limit, not a measured capacity; omit it if unknown.

### 1. Propose offline

```sh
uv run nooa connect your-model --stage plan --as work \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --context-window 16384 \
  --max-tokens 1024 --budget-tokens 4096 > model-plan.json
```

Review `data.entry`, `data.probes`, estimates and timeouts. Plan makes no
catalogue request; supply known limits or `--discovery-file` from an approved
listing. Public catalogue output cannot be fed to plan with a catalogue-file
flag in this CLI. Consult it manually for candidates or use library
`connect.plan(..., catalogue=selected_metadata)` after reviewing the mapping.
Plan does **not** enable session checks; its estimate is not the complete all
allocation. For an offline full preview use library `session_checks=True`.

### 2. Check the same settings within permission

```sh
uv run nooa connect your-model --stage routing --as work \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY --context-window 16384 \
  --max-tokens 1024 --budget-tokens 4096 > model-result.json
```

Chat routing requires a completed validated integer `323` for `17 * 19`, with
the configured settings observed on the wire. A truncated or merely accepted
request is not a pass. Inspect the actual entry and checks before choosing to
save. Do not chain save mechanically after a command or merely test exit 0.
**Save can persist an untested proposal or a failed report (with a warning).**
If unverified setup is intended, obtain that approval and label it unverified.

### 3. Save only the reviewed entry when authorized

```sh
uv run nooa connect --stage save --input model-result.json \
  --output .nooa/llm_config.yaml > model-save.json
```

Save accepts a report's `data` or a raw mapping containing `alias` and `entry`.
Use `--working-dir /absolute/project` instead of `--output` to write that
project's `.nooa/llm_config.yaml`; the project must exist and the options are
mutually exclusive. Save normalizes settings and rejects literal credentials.
It strips transient provenance from YAML; retain result JSON as evidence.

Existing alias: inspect it and get replacement permission before adding `--yes`
to save. Save preserves other aliases/comments, but **replaces the selected
alias as a whole**. Fresh plans do not merge its custom fields. Review and carry
forward approved fields explicitly; if changing tested settings, rerun affected
checks or mark them unverified. Never silently drop unrelated custom fields.

### 4. Confirm effective loading

Inspect save `data.path` and optional `data.shadowed_by`. An explicit output
may not join the discovery chain or may be shadowed. Load it deliberately:

```python
from pathlib import Path
from nooa.unifiedllm.registry import reload_registry
from nooa.unifiedllm import get_llm_client

reload_registry(Path(".nooa/llm_config.yaml"))
llm = get_llm_client("work")
# Construction/loading is not a model test. Pass llm to the application.
await llm.aclose()
```

Alternatively set `NEMO_OO_LLM_CONFIG` to the approved file for the application.
Use `get_decision_model` rather than `get_llm_client` for decision aliases.
Report the effective alias/file, verified checks, and unknown capabilities.

## Discover models and interfaces separately

If the ID is unknown, obtain permission for a model-list GET:

```sh
uv run nooa connect --stage discover \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY > models.json
```

Reuse with `--discovery-file models.json` on plan/check stages for the same
endpoint, repeating credential variable names. Listing may be public and does
not prove inference credential validity. Advertised limits/modalities are
metadata, not verified support; do not enable media from names alone.
Public metadata can be fetched separately with permission:

```sh
uv run nooa connect your-model --stage catalogue > catalogue.json
```

Review exact/suffix matches and optional fuzzy suggestions manually; never
silently change the inference ID to a catalogue suggestion.

For an unknown chat-family interface, compare all three using an explicit small
initial cap (the default is 2,048, not 200):

```sh
uv run nooa connect your-model --stage interfaces --as work \
  --endpoint https://gateway.example/v1 --api-key-env MODEL_KEY \
  --output-tokens 200 --budget-tokens 4096 > interfaces.json
```

**Even explicit chat/responses/anthropic `--api-style` on interfaces still
compares all three** in this CLI. For a known interface use routing instead.
The combined interfaces result is not a saveable single entry. Select an
accepted style, then plan/check its actual route. Optional interface reasoning
uses `--reasoning-template`/`--reasoning-level`; the compatibility option
`--reasoning-output-tokens` still controls that comparison's cap, not configured
reasoning-stage caps. Accepted settings may be ignored; observe evidence.

## Native decision routes

`systemone` and `openai` are decision styles; **openai does not mean Chat or
Responses**. Supply the complete POST URL and the exact endpoint model ID,
without adding a provider prefix. Only plan/interfaces/routing/all/save apply;
there is no decision stage. Native routing sends one Boolean predicate probe,
not arithmetic, tools, image/media, reasoning levels or session smoke tests.

```sh
uv run nooa connect decision-model --stage plan --as decisions \
  --endpoint https://decision.example/v1/systemone --api-style systemone \
  --api-key-env DECISION_KEY --budget-tokens 4096 > decision-plan.json
uv run nooa connect decision-model --stage routing --as decisions \
  --endpoint https://decision.example/v1/systemone --api-style systemone \
  --api-key-env DECISION_KEY --budget-tokens 4096 > decision-result.json
```

For OpenAI Decisions choose `--api-style openai` and its full decisions URL.
Do not supply reply-token fields, levels, media or discovery files. Native
endpoints do not list models through stage discover. There are no shared-listing,
models-url, listing-style, discovery-key, mixed-endpoint or image-transport
options in this CLI. Do not adapt instructions requiring unmerged flags.

## Keep budgets, timeouts and settings truthful

Always specify `--budget-tokens` for checks: default **`10**15`** is effectively
unlimited. Track allowance across invocations yourself: each starts a new
budget. Failed attempts consume allowance; inspect charges and skipped cases.
These are local estimates, not monetary guarantees or measured prompt lengths.
Insufficient budget skips a check, never silently lowers its cap.

- Unknown ordinary reply cap defaults to **32,768**, adjusted to known output
  and context limits. `--max-tokens` / `--reply-tokens` saves the cap and sends
  it on configured checks (Responses maps it to `max_output_tokens`). Thinking
  and answer share it; level budgets need answer headroom.
- Configured routing/tools reserve cap + 512 input tokens; selected media
  reserves cap + 4,096 per format. Configured probes use 120-second timeouts.
- Initial interfaces reserve output cap + 512 per style, with 30-second timeouts;
  default discovery output cap is 2,048. Optional reasoning costs extra.
- Session has three turns, each reserving `8192 + 3 * selected reply cap`,
  with 120-second timeouts and call pacing. All also runs other checks.
- Native routing reserves 512 tokens for one request with a 30-second timeout;
  no provider reply-token field. Bound count, input and time as well as estimates.
- Discovery/catalogue have 30-second overall deadlines. SDK retries are disabled
  for configured checks; an explicit Responses encrypted-include rejection can
  trigger an adapted compatibility attempt. Do not assume retries are free.

For reasoning, prepare an approved `--levels-file` YAML mapping from labels to
complete request settings. Plan and run reasoning with the same file and cap.
A correct puzzle alone does not prove reasoning; distinguish acceptance, wire
settings, observed reasoning and later replay. Do not print reasoning text,
signatures, encrypted state or raw provider errors.

For media, obtain format/sample permission before using
`--test-media-inputs image,pdf` on plan/media/all. Media runs baseline plus
strict typed Boolean access checks, not general accuracy tests. Audio/video
are tested only on Chat. `--media-inputs image,pdf` merely declares unverified
formats; empty or `-` means text-only. Test and declaration flags are exclusive.
Failed retests do not remove earlier declarations. See
[nooa-media](../nooa-media/SKILL.md) for typed inputs and privacy limits.

## Diagnose and hand off without widening scope

Inspect sanitized checks and local `run_context`: installed source/version,
registry chain/effective file, credential availability/source, transport/proxy
overrides and remaining allowance. Never treat a diagnostic command as consent.
Rerun only affected checks within the approved allowance; ask before raising
caps/budgets or changing routes. Timeout/server errors do not prove unsupported
APIs; absent reasoning is unknown. Cache misses may be warnings, not a broken
connection. Preserve explicit retention and transport choices.

Keep report artifacts private: key redaction and safe `checks` do not mean every
proposal/provenance field is free of request data. Plan deliberately includes
synthetic requests. Default Connect already suppresses the tracing banner;
quiet tracing is not an offline switch and does not disable viewer probing or
export. Tests must mock tracing separately from provider HTTP.

Hand off alias, target/effective file, approved changes, sanitized outcomes,
allowance used/remaining and unverified capabilities. For library plan/check/
write APIs and guided human setup see [Connect reference](../../docs/model-connect.md);
for settings see [model configuration](../../docs/model-configuration.md).
