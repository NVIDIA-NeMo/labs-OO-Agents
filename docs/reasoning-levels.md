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
`base_url`, `api_key`, `custom_llm_provider`, `client`, `messages`, `input`, `extra_body`,
and the three `reasoning_*` configuration fields. They cannot appear inside a
level's settings. A level changes effort, not the endpoint, credentials or history.

Omit `reasoning_levels` (or use null) when support is unknown. An empty mapping
explicitly declares selection unsupported. Unknown support, unsupported selection
and invalid labels produce distinct errors; an invalid label lists valid choices.
Without a managed selection, raw provider parameters continue working as before.

A selected level cannot be combined with a per-call setting of the same key,
including inside `extra_body`. Choose the label or the raw settings, not both.
Constructor defaults in `extra_body` are replaced by the selected settings,
just like top-level defaults. Unrelated defaults are preserved.
Declarations belong on the constructor, not per-call kwargs or `extra_body`.
Changing model or endpoint while using a managed level requires a new client:
one route's declared choices must not be applied to another route.
Supplying an SDK client per call is also rejected with a managed level because
that client can choose a different endpoint. When overriding a registry alias's
route or client type, inherited levels, default and selection are cleared;
declare replacement levels explicitly, or leave support unknown.

## Where the data comes from

The [example registry](../examples/reasoning_levels/llm_config.yaml) illustrates
three request shapes, based on the [GPT-5.6 Sol model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
[Claude effort documentation](https://platform.claude.com/docs/en/build-with-claude/effort)
and [Gemini's OpenAI-compatible API](https://ai.google.dev/gemini-api/docs/openai).
Its endpoints are placeholders. Replace them, the model IDs and the declared
choices with settings for your own route before making calls. It is not
auto-loaded. Load your configured file with `reload_registry(Path(...))`.
Private endpoint settings and credentials belong in private configuration, not
this public example.

Provider documentation describes provider APIs, not all gateway routes. Mocked
HTTP tests check that the example's settings survive the installed transport;
the opt-in live test checks your configured route's acceptance, not reasoning quality or every
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

For paid probes, configure registry aliases with a `low` level and credentials
through their normal `api_key_env` settings. Then opt in and name the aliases:

```sh
NOOA_RUN_REASONING_LEVELS_LIVE=1 NOOA_REASONING_TEST_MODELS=my-route uv run pytest \
  tests/integration/test_reasoning_levels_live.py -m integration -q -s
```

The alias list is comma-separated. Missing aliases are skipped. Each configured
alias gets one request capped at 256 output tokens, with retries disabled.
The test checks outgoing settings and route acceptance, not that every effort
label changes model behavior. Provider-specific results belong with the
configuration used to run them.
