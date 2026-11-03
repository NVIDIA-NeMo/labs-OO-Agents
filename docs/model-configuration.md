# Model configuration

Start with [nooa connect](model-connect.md) to configure the reply/output token
limit, known context window, media formats, and other model settings, and test
your route. This page explains the saved settings and how configuration files
are loaded.

You can also edit `llm_config.yaml` directly, or use the
[nooa-model-configuration](../skills/nooa-model-configuration/SKILL.md) skill.

Get the exact model ID, API format, and limits from the endpoint's
documentation. Leave out anything you do not know.

## Write the configuration

Add an entry under `models:` and keep any existing entries:

```yaml
models:
  my-model:
    model_name: openai/your-model
    client_type: completion
    api_base: https://gateway.example/v1
    api_key_env: MY_MODEL_KEY
    max_tokens: 8192
```

| Setting | What it means |
|---|---|
| `my-model` | The alias your agents use. |
| `model_name` | A prefix (`openai/` or `anthropic/`, see below), then the exact model ID the endpoint expects. The prefix is not sent. |
| `client_type` | `completion` or `responses`, see below. |
| `api_base` | The API base URL, not the full request URL. Leave off `/chat/completions`, `/responses`, or `/messages`. |
| `api_key_env` | The environment variable that holds the key. Never put the key itself in this file. Use `''` for a local server with no key. |
| `max_tokens` | The reply limit per request, including reasoning. Use what you need, not the model's maximum. Connect suggests a published recommendation or 32,768, adjusted to known limits. |
| `context_window` | Optional. The model's context window, if documented. |
| `media_inputs` | Optional. Which media the model can receive. See [below](#explicit-media-inputs-migration). |

Pick `model_name` and `client_type` by the endpoint's API:

| Endpoint API | `model_name` | `client_type` |
|---|---|---|
| OpenAI-compatible Chat Completions | `openai/your-model` | `completion` |
| OpenAI Responses | `openai/your-model` | `responses` |
| Anthropic Messages | `anthropic/your-model` | `completion` |

For Anthropic, set `api_base` to the server root without the final `/v1`.

For Responses entries, also add:

```yaml
    store: false
    include: [reasoning.encrypted_content]
```

This keeps the model's reasoning between turns. If the endpoint rejects
`include`, set `include: []`.

To add reasoning levels, see [reasoning levels](reasoning-levels.md). Only add
request settings whose exact form you know.

### Decision models

Decision models go in the same `models:` list, with `client_type: decision`:

```yaml
models:
  decisions:
    model_name: typesafe/jev-1.13
    client_type: decision
    api_style: systemone
    endpoint: https://openrouter.ai/api/alpha/decisions
    api_key_env: OPENROUTER_API_KEY
```

- `endpoint` is the full decisions request URL. NOOA adds nothing to it.
- `api_style` is `systemone` for System One endpoints, or `openai` for OpenAI's
  Decisions API (for example `endpoint: https://api.openai.com/v1/decisions`
  with `model_name: gpt-6-luna`).
- Use the alias with `decision_model="decisions"` on an agent or method.
  `get_llm_client("decisions")` does not work: decision models do not chat.
- Decision models take no `media_inputs`.

## Check the file locally

This loads the file and checks that the alias exists. It sends no requests:

```python
from pathlib import Path
from nooa.unifiedllm.registry import reload_registry

entries = reload_registry(Path("llm_config.yaml"))
assert "my-model" in entries
print("Model configuration loaded; endpoint and credentials have not been tested.")
```

This does not test the key or the endpoint. Use Connect to check the route.

## Configuration discovery and precedence

Automatic discovery reads existing files in this order (later aliases win):

1. Bundled defaults from installed packages.
2. User configuration: normally `~/.config/nooa/llm_config.yaml`; honors
   `XDG_CONFIG_HOME` or the explicit `NEMO_OO_USER_DIR` directory override.
3. Project configuration: normally `<project-root>/.nooa/llm_config.yaml`;
   `NEMO_OO_PROJECT_DIR` overrides the directory containing this file.
4. Comma-separated paths in `NEMO_OO_LLM_CONFIG`, in the order supplied.

Later files replace the **whole model entry**. Call
`nooa.llm_config.llm_config_chain()` to see which files your application reads.
When running from source, NOOA uses the source checkout as the project root.
For a file elsewhere, use one of the explicit loading methods below.

For the standalone file above, either explicitly reload it in each application
as shown, or make it discoverable before launching the application:

```sh
export NEMO_OO_LLM_CONFIG="$PWD/llm_config.yaml"
```

`reload_registry(Path(...))` loads only the supplied file(s); call
`reload_registry()` without arguments to return to automatic discovery.
Connect's `--output FILE` chooses where to write, not what Python will load.

## Skipping checks (reference)

`nooa connect` can write the entry for you without test requests. Add
`--no-probe` to skip model checks and `--no-catalogue` to skip public model
details:

```sh
uv run nooa connect your-model --as my-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY --no-probe --no-catalogue
```

For a decision model:

```sh
uv run nooa connect typesafe/jev-1.13 --as decisions \
  --endpoint https://openrouter.ai/api/alpha/decisions \
  --api-style systemone --api-key-env OPENROUTER_API_KEY \
  --no-probe --no-catalogue
```

These examples use the default user configuration destination. Supplying the
exact model, endpoint, and API style avoids model discovery; `--no-catalogue`
avoids metadata lookup. The entry is saved as **unverified**. Connect still asks
before saving (skip with `--yes`). See [Connect](model-connect.md) to test the model later.

When asking an agent to change configuration, specify the target alias and file.
Keep other entries and user settings, and use an environment variable for the key.

## Explicit media inputs (migration)

A model receives media only if its entry lists the kinds in `media_inputs`.
Without `media_inputs`, or with `media_inputs: []`, the model is text-only.
Media used to be sent without this setting, so add it to existing entries for
models that accept media.

```yaml
models:
  vision-model:
    model_name: openai/your-model
    media_inputs: [image, pdf]
  text-model:
    model_name: openai/your-text-model
    media_inputs: []
```

| Value | Sends |
|---|---|
| `image` | Images |
| `audio` | Audio |
| `video` | Video |
| `pdf` | Files with MIME type exactly `application/pdf` |

- List only kinds the endpoint really supports. NOOA does not check.
- `pdf` covers only `application/pdf`. Other files, such as Word documents, are
  never sent. A `.pdf` file name alone is not enough.
- A model with no `media_inputs` setting logs a one-time warning that it is
  text-only. Add `media_inputs: []` to silence it.
- Each entry has its own list. A config file that redefines an alias replaces
  the whole entry, including `media_inputs`.

For Python clients, pass the same list:

```python
from nooa.unifiedllm import CompletionClient, ResponsesClient

chat = CompletionClient("openai/your-model", media_inputs=["image", "audio"])
responses = ResponsesClient("openai/your-model", media_inputs=["image", "pdf"])
```

`get_llm_client("vision-model", media_inputs=[])` replaces the configured list.

### What happens to media the model cannot receive

Pass media as method arguments or use `show()` while CodeAct runs. When a
format is missing from `media_inputs`, NOOA sends a short note instead of the
attachment, for example:

- `[An image was removed here because this model is not set up to receive images.]`
- `[A file was removed here. Only PDF files can be sent to a model.]`

Stored history keeps the original attachments. Using that history with a model
configured for those formats can send them again. See the
[media guide](media.md#text-only-requests-and-replay).

To let a model receive it, add the kind to its entry, for example
`media_inputs: [video]`, and keep any kinds already listed.

Connect can test media for you with generated samples. See
[Connect media setup](model-connect.md#choose-media-inputs-during-setup).
