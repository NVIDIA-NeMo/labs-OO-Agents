# Configure a model without paid checks

You can write a model configuration yourself, or ask an agent using the
[nooa-model-configuration](../skills/nooa-model-configuration/SKILL.md) skill to do it.
Neither requires a model API call.
Use the endpoint's documentation for its exact model ID, API format and limits;
leave anything unknown unset rather than guessing.

## Write the configuration

Add an entry to `llm_config.yaml`, preserving any existing models. Replace the
example endpoint and model ID with yours:

```yaml
models:
  my-model:
    model_name: openai/your-model
    client_type: completion
    api_base: https://gateway.example/v1
    api_key_env: MY_MODEL_KEY
    max_tokens: 8192
```

`my-model` is the local name your agents use. `your-model` is the exact ID the
endpoint expects. The extra leading `openai/` selects the current runtime's
OpenAI-compatible transport; it is not part of the model ID sent to the endpoint.
Keep a route prefix that is part of the endpoint's actual ID after this extra
prefix. The API format determines these two fields:

| Endpoint API | `model_name` | `client_type` |
|---|---|---|
| OpenAI-compatible Chat Completions | `openai/your-model` | `completion` |
| OpenAI Responses | `openai/your-model` | `responses` |
| Anthropic Messages | `anthropic/your-model` | `completion` |

Use the API base URL, not the `/chat/completions`, `/responses`, or `/messages`
request URL. A model's name alone does not tell you which interface its server
accepts. Keep the key in the named environment variable or NOOA's secrets
configuration, never as a literal value in this YAML. Unauthenticated local
servers can use `api_key_env: ''`.

For the current Anthropic client, use the server root without the final `/v1`:
the client appends `/v1/messages`. Connect normalizes this when saving the entry.

If documented for your route, add `context_window` as a capacity hint. It is
not an output-token allocation. Set `max_tokens` to the reply budget you want,
not the model's advertised maximum; 32,768 is Connect's fallback starting budget
when no recommendation is available. Keep it within the model's documented
limits. Responses clients translate this to `max_output_tokens` on the wire.
For stateless Responses entries, also set `store: false` and
`include: [reasoning.encrypted_content]` to carry reasoning between turns.
If the endpoint rejects that include field, explicitly set `include: []` and
record that reasoning retention has not been confirmed. Connect checks this
automatically when probes are approved.
Add reasoning-level request blocks only when
you know their exact shape; see [reasoning levels](reasoning-levels.md).
Do not infer optional encrypted-reasoning or explicit-cache support from an
OpenAI-compatible URL. Unknown capabilities remain untested.

## Check the file locally

This loads the file and confirms that the alias exists without calling a model:

```python
from pathlib import Path
from nooa.unifiedllm.registry import reload_registry

entries = reload_registry(Path("llm_config.yaml"))
assert "my-model" in entries
print("Model configuration loaded; endpoint and credentials have not been tested.")
```

To make a custom file available to subsequent processes, include its path in
`NEMO_OO_LLM_CONFIG`. The normal user file is `llm_config.yaml` in NOOA's user
configuration directory. To inspect the active file chain, call
`nooa.llm_config.llm_config_chain()`. Later files can override earlier aliases.
Local loading does not verify credentials, endpoint availability or provider
support. A later generation call will use the provider and may incur charges.

## Use Connect without generation calls

The wizard's `--no-probe` (or `--probe none`) option disables interface, typed
text baseline, tool, reasoning and media generation checks.
It may still fetch a model list and public catalogue metadata. To avoid even
those requests, supply the model and use `--no-catalogue` as well:

```sh
uv run nooa connect your-model --as my-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY --no-probe --no-catalogue \
  --output llm_config.yaml
```

This is an explicit **manual, unverified** setup. The save review labels it
`Connection: Unverified (manual setup; no model checks)`. You still confirm before saving
or replacing an alias (or explicitly use `--yes`). With checks approved, the
wizard instead requires a correct, complete configured typed NOOA `-> int`
baseline before saving; failed/missing/incomplete evidence or budget exhaustion
before that baseline stops registry and key writes, even with `--yes`. Plain
interface acceptance is insufficient. The baseline uses the runtime's native
scalar JSON value contract (`{"value": 323}` for `17 * 19`), requiring structured
output support beyond plain text acceptance (Anthropic may use a schema tool).
It is a bounded one-shot Predict call, not an iterative repair agent. If your
endpoint lacks that contract, deliberately use the manual command above rather
than approving paid checks and accepting an ordinary save prompt.
For agent-assisted setup,
ask: “Use the `nooa-model-configuration` skill to configure this model without API
calls. Preserve my other entries, use an environment-variable name for the key,
and validate the file locally only.” See [Connect](model-connect.md) for the
automatic checks if you later want to test the endpoint.

## Explicit media inputs (migration)

**New default: text-only.** Previously attachments were forwarded without an
explicit declaration. Existing multimodal entries and direct Python clients
must opt in to keep sending media; no model names, catalogue metadata, endpoint
responses or unrequested probes enable inputs implicitly. In Connect's explicitly
approved **TEST** mode, successful generated-media self-reports add permissions
to the reviewed list before saving; the library's probe results alone never
mutate declarations. See [Connect media setup](model-connect.md#choose-media-inputs-during-setup).

```yaml
models:
  vision-model:
    model_name: openai/your-model
    media_inputs: [image, pdf]
  text-model:
    model_name: openai/your-text-model
    media_inputs: []
```

```python
from nooa.unifiedllm import CompletionClient, ResponsesClient

chat = CompletionClient("openai/your-model", media_inputs=["image", "audio"])
responses = ResponsesClient("openai/your-model", media_inputs=["image", "pdf"])
```

Connect's optional TEST mode asks the model whether it can access and interpret
one original generated attachment per selected format, using strict boolean
Agent/Predict methods. This is a self-report, not an accuracy benchmark. Only a
validated true with a complete final answer and observed attachment/settings can
newly enable an input. A valid false displays **Not available**; existing manual
permissions remain unless you change them. See [media tests](model-connect.md#choose-media-inputs-during-setup)
for native API mapping limits and the shared approval/budget contract.

`media_inputs` accepts a list of distinct `image`, `audio`, `video`, `pdf`
values (Python constructors also accept a tuple). Omission, null and `[]` all
mean text-only. Unknown values, scalar strings and duplicates are errors.
It is a **client setting**, not a provider parameter: `call`/`acall` and
`extra_body` cannot set it. Factory overrides such as
`get_llm_client("vision-model", media_inputs=[])` replace, not union, the list.
Later YAML layers replace an entire alias; a replacement entry without this
field is text-only. Existing clients keep their own declaration until replaced.

| Input | Recognized attachment blocks | MIME mapping |
|---|---|---|
| `image` | `image_url`, `input_image`, `image` | `image/*` |
| `audio` | `input_audio`, `audio`, `output_audio` | `audio/*` |
| `video` | `video_url`, `input_video`, `video` | `video/*` |
| `pdf` | `file`, `input_file`, `document` | **Exactly `application/pdf`** |

These are configuration categories, **not guarantees of provider/codec
support**. Image/audio/video block kinds suffice for remote URLs or opaque
references without MIME metadata; when a MIME hint is present it must match
the category. MIME is read only from documented `mime_type`/`media_type`/`format`
hints or a data URL in the attachment slot, never by fetching a URL or decoding
content. A file must carry an exact PDF MIME hint or
`data:application/pdf;base64,...`; a `.pdf` filename/URL or an untyped `file_id`
is not sufficient. Word documents, generic binary files and other file types
are not enabled by `pdf`. No conversion is attempted.

During model execution, `show()` raises `ValueError` immediately if the current
client does not enable the attachment type, including when `media_inputs` is
missing. The error explains how to enable it in this model's YAML entry, through
Connect, or on a Python client. For example, video requires:

```yaml
media_inputs: [video]  # Include any other inputs you want to keep.
```

For a Python client, pass `media_inputs=["video"]`. Only enable types the endpoint
supports. A rejected `show()` does not capture the attachment or print a success
acknowledgement. This applies to in-process execution and worker-local `show()`
in the sandbox; it does not change the separate sandbox restrictions on callable
skill helpers. Enabling the type permits sending it; it does not verify provider
support or guarantee inspection before the method returns.

Outgoing fresh and historical attachments outside the explicit list are replaced
with a short **attachment not inspected** notice on a request-owned view.
Stored events, Python media objects, caller dictionaries, tool order and native
reasoning remain unchanged; switching to an enabled client can send the original
history again. Model/endpoint/provider/custom-client overrides do not inherit the
old route's permissions: construct a separately configured client for that route.
Changing credentials alone does not change the declaration. Opaque signed
assistant media is not recursively rewritten: an unsupported disabled native
media part raises an actionable replay error rather than corrupting its envelope.
