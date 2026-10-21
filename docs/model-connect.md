# Configure a model with NOOA Connect

`nooa connect` sets up a model for NOOA. It finds the model on your endpoint,
runs a few small test requests, and saves a named entry (an *alias*) in
`llm_config.yaml`. Your agents then use the model by that alias.

Test requests can cost money. Connect asks once, before the first one. To set
up a model without any requests, see [Configure a model without paid
checks](model-configuration.md).

## CLI

Run the guided setup:

```sh
uv run nooa connect
```

It asks for a provider (NVIDIA, NVIDIA Inference Hub, OpenAI, Anthropic,
Google, OpenRouter, or a custom endpoint), your API key, and the model. It then tests the model and asks
before saving.

To skip the questions, pass the answers as flags:

```sh
uv run nooa connect your-model --as my-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY
```

- `your-model` is the exact model ID the endpoint expects. Omit it to choose
  from the endpoint's model list.
- `--as` is the alias your agents use.
- `--api-style` is `chat` (OpenAI-compatible Chat Completions), `responses`
  (OpenAI Responses), or `anthropic` (Anthropic Messages). Omit it and Connect
  tries all three.
- `--api-key-env` is the name of the environment variable that holds the key,
  never the key itself. Use `--api-key-env ''` for a local server with no key.
- `--prompt-key` lets you paste a key instead. At the end Connect offers to save
  it in NOOA's `secrets.yaml`.
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

Connect sends one small decision request to test the setup. Add `--no-probe` to
save without sending it. Decision models do not take `media_inputs`.

In stage mode, decision models support only `plan`, `interfaces`, `routing`,
`all`, and `save`. `--stage discover` and `--discovery-file` are rejected,
because decision endpoints do not list their models.

## What Connect tests

With the default `--probe all`, Connect runs:

1. **Connection.** One request per API style, to find which one works.
2. **Text baseline.** Asks the model for `17 * 19` as a typed NOOA result. The
   model must return `323` as structured output. Without this pass, the
   wizard does not save.
3. **Tools.** Checks that the model makes a requested tool call. The call is
   never run.
4. **Reasoning levels.** Checks each level you set up.
5. **Conversation.** Three requests that check caching and that reasoning carries
   over between turns.
6. **Media**, only if you choose to. See [below](#choose-media-inputs-during-setup).

`--probe minimal` runs only the connection check. `--no-probe` (or
`--probe none`) sends no test requests and saves the entry as **unverified**.

Only a failed connection or text baseline blocks saving. Tool, reasoning,
caching, or media results that need attention do not.

### Configured typed text baseline

The baseline uses your exact settings and one request with no retry. It fails if
the answer is wrong, cut off, or not in the structured format. If your endpoint
cannot return structured output, save it without checks (`--no-probe`) instead.

## Token budget and reply limit

- `--budget-tokens N` limits the estimated total tokens for all checks. The
  default is no limit. It is an estimate, not a spending cap.
- A check that does not fit the remaining budget is skipped. Connect does not
  lower limits or ask for more budget.
- `--max-tokens N` (also `--reply-tokens`) sets the reply limit saved as
  `max_tokens`. It covers reasoning and the answer. Without it, Connect suggests
  the published recommendation, or 32,768. It must be less than the context
  window.
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
public model list. Use `--catalogue-model ID` to pick the entry, or
`--no-catalogue` to skip it.

## Independent stages for agents

`--stage NAME` runs one step without prompts and prints one JSON report to
stdout. Only `save` writes a file.

| Stage | What it does |
|---|---|
| `discover` | List the endpoint's models. |
| `catalogue` | Read public model details. |
| `interfaces` | Find which API styles work. |
| `plan` | Build an unsaved entry. No network. |
| `routing` | Run the text baseline. |
| `tools` | Check tool calls. |
| `reasoning` | Check levels from `--levels-file`. |
| `media` | Run the text baseline and the media tests. |
| `session` | Check caching and reasoning across turns. |
| `all` | Run the basic and conversation checks. |
| `save` | Write an entry from a plan or result file. |

Example:

```sh
uv run nooa connect your-model --stage routing --as work \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MODEL_KEY > result.json
uv run nooa connect --stage save --input result.json --output llm_config.yaml
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
| Timeout | The route may be slow. Retry, or check the endpoint. A timeout does not mean the key is wrong. |
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
  to pick from a list.
- When you reconnect, unchanged tests that already passed are reused.
- To load a config file from another path, add it to `NEMO_OO_LLM_CONFIG`.

## Choose media inputs during setup

A model receives images, audio, video, or PDFs only if its entry lists them in
`media_inputs`. New entries are text-only (`media_inputs: []`). Connect never
guesses this from the model name or published details. See
[media inputs](model-configuration.md#explicit-media-inputs-migration).

In the wizard, choose one:

- **Test all media inputs (generated samples).** Connect sends one small
  generated sample per format to check that this model accepts it through this
  connection. Formats that pass are enabled for this model entry. This uses the
  budget you already approved. It does not check how well the model understands
  the media.
- **Choose media inputs manually.** No requests. The choice is saved as
  unverified.

Before saving, you can keep the result or change it by hand. A failed retest
does not remove a format you enabled before.

Test limits:

- Image and PDF are tested on all three API styles.
- Audio and video are tested only with `--api-style chat`. On other styles they
  are skipped, not proven unsupported.
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
```

Rules:

- `--media-inputs ''` or `-` means text-only.
- `--test-media-inputs` cannot be combined with `--media-inputs`, `--no-probe`,
  or `--probe minimal`.
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
