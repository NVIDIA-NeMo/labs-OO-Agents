# Configure a model without paid checks

You can write a model configuration yourself, or ask an agent using the
`nooa-agent-authoring` skill to do it. Neither requires a model API call.
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

The wizard's `--no-probe` option disables interface, tool and reasoning checks.
It may still fetch a model list and public catalogue metadata. To avoid even
those requests, supply the model and use `--no-catalogue` as well:

```sh
uv run nooa connect your-model --as my-model \
  --endpoint https://gateway.example/v1 --api-style chat \
  --api-key-env MY_MODEL_KEY --no-probe --no-catalogue \
  --output llm_config.yaml
```

You still confirm before saving or replacing an alias. For agent-assisted setup,
ask: “Use the `nooa-agent-authoring` skill to configure this model without API
calls. Preserve my other entries, use an environment-variable name for the key,
and validate the file locally only.” See [Connect](model-connect.md) for the
automatic checks if you later want to test the endpoint.
