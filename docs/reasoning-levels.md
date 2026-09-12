# Reasoning levels

UnifiedLLM exposes the choices a configured route supports. Model names do not
determine those choices: registry YAML maps each label to exact request parameters.
Selecting a label applies `params.update(level_settings)` before dispatch.

```python
from pathlib import Path

from nooa.unifiedllm import get_llm_client
from nooa.unifiedllm.registry import reload_registry

# From the repository root; this example registry is not loaded automatically.
reload_registry(Path("examples/reasoning_levels/llm_config.yaml"))
client = get_llm_client("gpt-5.6-sol")
print(client.reasoning_levels)  # tuple of labels; None = unknown, () = unsupported
print(client.reasoning_default)  # documented default, or None when unknown
response = await client.acall(messages, reasoning_level="high")
```

The constructor also accepts `reasoning_level` as a persistent selection. A
per-call selection overrides it; explicit `reasoning_level=None` uses the raw
base configuration for that call. `reasoning_default` is metadata only: declaring
it does not add parameters, change costs or override existing provider settings.

## Registry declarations

```yaml
models:
  my-route:
    model_name: openai/my-model
    reasoning: {effort: medium, context: all_turns}
    reasoning_default: medium
    reasoning_levels:
      low: {reasoning: {effort: low, context: all_turns}}
      medium: {reasoning: {effort: medium, context: all_turns}}
      high: {reasoning: {effort: high, context: all_turns}}
```

Write the complete nested block for each level. There is no deep merge or
inheritance: selecting a level replaces the base value at each key it sets.
The declarations are trusted configuration, just like the rest of the registry;
they are not restricted to a list of provider fields maintained by NOOA.
Client routing and framework controls are reserved: `model`, `api_base`,
`base_url`, `api_key`, `custom_llm_provider`, `messages`, `input`, `extra_body`,
and the three `reasoning_*` configuration fields. They cannot appear inside a
level's settings. A level changes effort, not the endpoint, credentials or history.

Omit `reasoning_levels` (or use null) when support is unknown. An empty mapping
explicitly declares selection unsupported. Unknown support, unsupported selection
and invalid labels produce distinct errors; an invalid label lists valid choices.
Without a managed selection, raw provider parameters continue working as before.

A selected level cannot be combined with a per-call setting of the same key,
including inside `extra_body`. Choose the label or the raw settings, not both.
Declarations belong on the constructor, not per-call kwargs or `extra_body`.
Changing model or endpoint while using a managed level requires a new client:
one route's declared choices must not be applied to another route.

## Where the data comes from

The [example registry](../examples/reasoning_levels/llm_config.yaml) covers three
explicit Hub routes, based on the [GPT-5.6 Sol model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
[Claude effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort)
and [Gemini's OpenAI-compatible API](https://ai.google.dev/gemini-api/docs/openai).
It is not auto-loaded. Load it explicitly with `reload_registry(Path(...))`, or
copy reviewed entries to your own registry. The optional NVIDIA configuration
package can adopt the same fields without changing NOOA's provider-free defaults.

The Sonnet example omits `xhigh`: LiteLLM 1.97.0 rejects that value for the
Hub's model ID before sending a request, despite native Sonnet supporting it.
The advertised choices describe the usable route, including the installed SDK.

Provider documentation describes provider APIs, not all gateway routes. Mocked
HTTP tests check that the example's settings survive the installed transport;
the opt-in live test checks route acceptance, not reasoning quality or every
level's behavior. Do not infer support merely from a successful HTTP response
if a gateway silently ignores parameters.

Do not enable LiteLLM's global `drop_params` when verifying a declaration: it can
discard fields for gateway IDs it does not recognize, even with per-call
`drop_params=False`. The HTTP tests explicitly disable that global flag and check
the serialized fields. NOOA does not change process-global SDK configuration.

LangChain/Pi data can inform maintenance, but neither is a runtime dependency or
an automatic build input. Updating a route means reviewing its small declaration
and request tests, rather than importing hundreds of profiles.

## Scope and architecture

- `unifiedllm/reasoning.py` validates declarations and applies the chosen settings.
  The same function serves synchronous and asynchronous Chat and Responses calls.
- Registry fields are passed into UnifiedLLM and consumed before provider dispatch.
  Renderers, events and middleware do not translate reasoning levels.
- No selection changes existing behavior. Stored reasoning, compatibility gates,
  session archives and replay remain unchanged. This is not a retention toggle.
- Effort labels are provider-local, not comparable units of intelligence or cost.
  Changing effort can invalidate a provider's cached prefix. This PR does not add
  cache-preserving mid-turn steering, TUI controls, or selection persistence.

## Tests

Run `uv run pytest tests/unifiedllm/test_reasoning_levels.py tests/unifiedllm/test_reasoning_levels_wire.py`.
The tests check configuration ownership, invalid selections, route changes,
unchanged defaults and the serialized HTTP requests for the example routes.

For paid route probes, supply `NVIDIA_INFERENCE_API_KEY` (or
`NVIDIA_INTERNAL_API_KEY`) and run:

```sh
NOOA_RUN_REASONING_LEVELS_LIVE=1 uv run pytest \
  tests/integration/test_reasoning_levels_live.py -m integration -q -s
```

There is one low-effort request per example route, capped at 256 output tokens,
with retries disabled. All three passed on 2026-09-12 at `3e672ae0`:

| Hub route | Input tokens | Output tokens | Reasoning tokens (included in output) |
|---|---:|---:|---:|
| GPT-5.6 Sol | 19 | 5 | 0 |
| Claude Sonnet 5 | 24 | 3 | 0 |
| Gemini 3.1 Pro Preview | 15 | 145 | 142 |

The first Gemini probe exposed unsigned thinking that the base capture code
rejected. That was corrected separately in #318 before this run; unsigned text
is portable and neighboring signatures remain attached to their own parts.
These probes verify route acceptance and the outgoing settings, not that every
effort label changes model behavior.
