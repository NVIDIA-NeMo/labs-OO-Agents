# Local Models

NOOA uses LiteLLM model strings through `get_llm_client()`. The two common
local backends use different shapes:

- Ollama: `ollama_chat/<model>`, base URL without `/v1`, no API key.
- vLLM: `hosted_vllm/<model>`, base URL with `/v1`, optional API key.

## Ollama

```bash
ollama serve
```

```bash
# another shell
ollama pull qwen3:1.7b
```

```python
from nooa.unifiedllm.registry import get_llm_client

llm = get_llm_client(
    "ollama_chat/qwen3:1.7b",
    api_base="http://localhost:11434",
)
```

Use a model name shown by `ollama list`. Do not pass an API key for normal local
Ollama.

## vLLM

```bash
vllm serve Qwen/Qwen3-1.7B --host 0.0.0.0 --port 8000
```

```bash
# another shell
curl http://localhost:8000/v1/models
```

```python
from nooa.unifiedllm.registry import get_llm_client

llm = get_llm_client(
    "hosted_vllm/Qwen/Qwen3-1.7B",
    api_base="http://localhost:8000/v1",
)
```

Use the id from `/v1/models` after `hosted_vllm/`. If vLLM was started with
`--api-key token-abc123` or `VLLM_API_KEY`, pass the same key:

```python
llm = get_llm_client(
    "hosted_vllm/Qwen/Qwen3-1.7B",
    api_base="http://localhost:8000/v1",
    api_key="token-abc123",
)
```

## Aliases

Put repeated local config in `.nooa/llm_config.yaml`:

```yaml
models:
  local-ollama:
    model_name: ollama_chat/qwen3:1.7b
    api_base: http://localhost:11434

  local-vllm:
    model_name: hosted_vllm/Qwen/Qwen3-1.7B
    api_base: http://localhost:8000/v1
    # api_key_env: VLLM_API_KEY  # only when vLLM requires auth
```

Then call:

```python
llm = get_llm_client("local-ollama")
llm = get_llm_client("local-vllm")
```

References: [LiteLLM Ollama](https://docs.litellm.ai/docs/providers/ollama),
[LiteLLM vLLM](https://docs.litellm.ai/docs/providers/vllm).
