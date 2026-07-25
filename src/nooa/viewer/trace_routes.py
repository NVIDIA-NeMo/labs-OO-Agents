# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trace viewer API routes.

Uses otlp_store for trace storage/retrieval (OTLP JSON in SQLite).
"""

import json
import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import litellm

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nooa.unifiedllm.registry import resolve_api_key_from_config

from . import otlp_store
from .trace_models import TraceGroup

log = logging.getLogger(__name__)

router = APIRouter()

# Models config: look in cwd or env-provided path
_MODELS_CONFIG_ENV = os.environ.get("NEMO_OO_MODELS_CONFIG")
if _MODELS_CONFIG_ENV is None:
    # Deprecated: AGENT006_MODELS_CONFIG carried the internal codename into the
    # public config surface. Honored for one release; prefer NEMO_OO_MODELS_CONFIG.
    _legacy_models_config_env = os.environ.get("AGENT006_MODELS_CONFIG")
    if _legacy_models_config_env is not None:
        log.warning(
            "AGENT006_MODELS_CONFIG is deprecated and will be removed; "
            "use NEMO_OO_MODELS_CONFIG instead."
        )
        _MODELS_CONFIG_ENV = _legacy_models_config_env
MODELS_CONFIG_FILE = Path(_MODELS_CONFIG_ENV) if _MODELS_CONFIG_ENV else Path.cwd() / "models.yaml"

CUSTOM_MODELS_FILE = Path.cwd() / "custom_models.json"

# Log requests that take longer than this threshold (ms).
_SLOW_REQUEST_MS = 200


def _log_timing(operation: str, elapsed_ms: float, details: str = ""):
    if elapsed_ms >= _SLOW_REQUEST_MS:
        log.warning("[slow] %s  %.0fms  %s", operation, elapsed_ms, details)
    else:
        log.info("%s  %.0fms  %s", operation, elapsed_ms, details)


# ============================================================================
# Playground Models and Configuration
# ============================================================================


class InferenceRequest(BaseModel):
    """Request model for LLM inference."""

    messages: list[dict[str, Any]]
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096


class CustomModel(BaseModel):
    """Custom model configuration."""

    name: str
    model_id: str
    endpoint: str | None = None
    api_key_env: str | None = None


def load_models_config() -> dict:
    """Load models configuration from YAML file."""
    if MODELS_CONFIG_FILE.exists():
        with open(MODELS_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    return {}


def get_builtin_models() -> list[dict]:
    """Get built-in models from config."""
    config = load_models_config()
    models = []

    for model_id, model_info in config.get("models", {}).items():
        models.append(
            {
                "id": model_id,
                "name": model_info.get("name", model_id),
                "provider": model_info.get("provider", "unknown"),
                "endpoint": model_info.get("endpoint"),
                "api_key_env": model_info.get("api_key_env"),
            }
        )

    return models


def get_known_api_key_patterns() -> list[str]:
    """Get known API key patterns from config."""
    config = load_models_config()
    api_keys = set()

    for model_info in config.get("models", {}).values():
        if model_info.get("api_key_env"):
            api_keys.add(model_info["api_key_env"])

    api_keys.update(
        [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "NVIDIA_API_KEY",
            "GOOGLE_API_KEY",
            "AZURE_OPENAI_API_KEY",
            "TOGETHER_API_KEY",
            "GROQ_API_KEY",
            "MISTRAL_API_KEY",
            "COHERE_API_KEY",
        ]
    )

    return sorted(api_keys)


# ============================================================================
# Playground Helper Functions
# ============================================================================


def load_custom_models() -> list[dict]:
    """Load custom models from file."""
    if CUSTOM_MODELS_FILE.exists():
        with open(CUSTOM_MODELS_FILE) as f:
            data = json.load(f)
            if isinstance(data, list):
                return [model for model in data if isinstance(model, dict)]
    return []


def save_custom_models(models: list[dict]):
    """Save custom models to file."""
    with open(CUSTOM_MODELS_FILE, "w") as f:
        json.dump(models, f, indent=2)


def _normalize_model_pair(model: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return a normalized ``(endpoint, api_key_env)`` pair from model config."""
    endpoint = model.get("endpoint")
    api_key_env = model.get("api_key_env")
    return (
        endpoint.strip() if isinstance(endpoint, str) and endpoint.strip() else None,
        api_key_env.strip() if isinstance(api_key_env, str) and api_key_env.strip() else None,
    )


def _trusted_playground_pairs() -> set[tuple[str | None, str | None]]:
    """Pairs declared by the server-side models file are trusted for playground use.

    Custom models are browser-controlled state. They may alias a pair already
    approved by the local operator in ``models.yaml`` or use provider defaults,
    but they must not invent a destination or env-var name.
    """
    pairs = {(None, None)}
    config = load_models_config()
    for model_info in config.get("models", {}).values():
        if isinstance(model_info, Mapping):
            pairs.add(_normalize_model_pair(model_info))
    return pairs


def _sanitize_custom_model(model: Mapping[str, Any]) -> dict[str, Any]:
    """Drop untrusted endpoint/key pairs from persisted browser-controlled models."""
    sanitized = dict(model)
    pair = _normalize_model_pair(model)
    if pair not in _trusted_playground_pairs():
        log.warning(
            "Ignoring untrusted playground model endpoint/api_key_env pair for model %r.",
            model.get("model_id"),
        )
        sanitized["endpoint"] = None
        sanitized["api_key_env"] = None
    else:
        sanitized["endpoint"], sanitized["api_key_env"] = pair
    return sanitized


def _validate_custom_model_pair(model: Mapping[str, Any]) -> None:
    pair = _normalize_model_pair(model)
    if pair not in _trusted_playground_pairs():
        raise HTTPException(
            status_code=400,
            detail=(
                "Custom models may only use endpoint/api_key_env pairs already declared "
                "in the server-side models config."
            ),
        )


# ============================================================================
# Trace API Endpoints (backed by otlp_store)
# ============================================================================


@router.get("/api/version")
async def get_version():
    """Get API version information."""
    return {
        "version": "3.0.0",
        "api_format": "otlp",
        "description": "OTLP JSON trace storage",
    }


class PaginatedTraceResponse(BaseModel):
    """Paginated response for trace groups."""

    traces: list[TraceGroup]
    total: int
    page: int
    limit: int
    has_more: bool


_TRACE_SORT_KEYS: dict[str, str] = {
    "name": "name",
    "event_count": "event_count",
    "size": "size",
    "modified": "modified",
}


@router.get("/api/traces")
def list_traces(
    page: int = 1,
    limit: int = 100,
    search: str | None = None,
    experiment: str | None = None,
    batch_id: str | None = None,
    sort_by: str | None = None,
    sort_dir: str = "desc",
) -> PaginatedTraceResponse:
    """List trace sessions with pagination."""
    overall_start = time.time()
    try:
        limit = max(1, min(limit, 500))
        page = max(1, page)

        sessions = otlp_store.list_sessions(experiment=experiment, batch_id=batch_id)
        sizes = otlp_store.get_session_sizes()

        groups = [
            TraceGroup(
                id=s["id"],
                name=s["name"],
                modified=s["modified"],
                size=sizes.get(s["id"], 0),
                event_count=s["span_count"],
                batch_id=s.get("batch_id"),
            )
            for s in sessions
        ]

        sort_field = _TRACE_SORT_KEYS.get(sort_by or "", "modified")
        reverse = sort_dir != "asc"

        def _sort_key(g: TraceGroup):
            val = getattr(g, sort_field)
            if sort_field == "modified":
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return 0.0
            if isinstance(val, (int, float)):
                return val
            return str(val).lower()

        groups.sort(key=_sort_key, reverse=reverse)

        if search:
            search_lower = search.lower()
            groups = [g for g in groups if search_lower in g.name.lower()]

        total = len(groups)
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated = groups[start_idx:end_idx]

        _log_timing(
            "/api/traces TOTAL",
            (time.time() - overall_start) * 1000,
            f"{len(paginated)}/{total} groups (page {page})",
        )
        return PaginatedTraceResponse(
            traces=paginated,
            total=total,
            page=page,
            limit=limit,
            has_more=end_idx < total,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error listing traces: {str(e)}") from e


@router.delete("/api/traces/{session_id:path}", status_code=204)
def delete_trace(session_id: str):
    """Delete a single trace (session) and its spans and annotations."""
    deleted = otlp_store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Trace not found: {session_id}")


@router.delete("/api/traces", status_code=200)
def delete_all_traces(confirm: bool = False, batch_id: str | None = None):
    """Delete traces. Use ?batch_id=X to delete a batch, or ?confirm=true to delete all."""
    if batch_id:
        count = otlp_store.delete_sessions_by_batch(batch_id)
        return {"deleted": count, "batch_id": batch_id}
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must pass confirm=true to delete all traces (e.g. DELETE /api/traces?confirm=true)",
        )
    stats = otlp_store.delete_all_sessions()
    return {"deleted": stats}


@router.get("/api/experiments")
def list_experiments_api() -> list[str]:
    """List known experiment names."""
    return otlp_store.list_experiments()


@router.get("/api/trace-count")
def get_trace_count(session_id: str) -> dict:
    """Get span count for a specific session."""
    try:
        sessions = otlp_store.list_sessions()
        for s in sessions:
            if s["id"] == session_id:
                return {"path": session_id, "event_count": s["span_count"]}
        raise FileNotFoundError(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from e


@router.get("/api/trace")
def get_trace(session_id: str, limit: int | None = None, offset: int = 0) -> dict:
    """Load a specific session's OTLP spans."""
    overall_start = time.time()
    try:
        spans = otlp_store.get_session_spans(session_id)
        total_count = len(spans)

        if limit is not None:
            spans = spans[offset : offset + limit]
            has_more = offset + limit < total_count
        else:
            has_more = False

        result = {
            "format": "otlp",
            "path": session_id,
            "events": spans,
            "total_count": total_count,
            "has_more": has_more,
        }

        _log_timing(
            "/api/trace TOTAL",
            (time.time() - overall_start) * 1000,
            f"session={session_id}, returned={len(spans)}/{total_count}",
        )
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from e
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error loading session: {str(e)}") from e


@router.get("/api/trace/export")
async def export_trace(session_id: str):
    """Export a trace as a downloadable .jsonl file (OTLP format + annotations)."""
    try:
        bodies = otlp_store.export_session_otlp(session_id)
        annotations = otlp_store.list_annotations(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    def generate():
        for body in bodies:
            yield json.dumps(body, separators=(",", ":")) + "\n"
        if annotations:
            yield json.dumps({"annotations": annotations}, separators=(",", ":")) + "\n"

    safe_name = (
        session_id.replace("/", "_")
        .replace("\\", "_")
        .replace('"', "_")
        .replace("\r", "")
        .replace("\n", "")
    )
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.jsonl"'},
    )


@router.get("/api/trace/resource")
def get_trace_resource(session_id: str) -> dict:
    """Get OTLP resource attributes for a session."""
    try:
        resource = otlp_store.get_session_resource(session_id)
        return resource
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}") from e


@router.get("/api/experiment/{experiment_id}/traces")
def get_experiment_traces(experiment_id: str) -> dict:
    """Return all spans for all sessions in an experiment, grouped by session."""
    sessions = otlp_store.list_sessions(experiment=experiment_id)
    if not sessions:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")

    result_sessions = []
    for s in sessions:
        spans = otlp_store.get_session_spans(s["id"])
        result_sessions.append(
            {
                "session_id": s["id"],
                "span_count": s["span_count"],
                "spans": spans,
            }
        )

    return {
        "experiment": experiment_id,
        "sessions": result_sessions,
    }


@router.get("/api/config")
def get_config() -> dict:
    """Get current configuration."""
    stats = otlp_store.get_stats()
    return {
        "provider": "otlp_sqlite",
        "db_path": str(otlp_store.DB_PATH),
        "experiments": otlp_store.list_experiments(),
        "stats": stats,
    }


@router.get("/api/provider/status")
def get_provider_status() -> dict[str, Any]:
    """Get current provider status."""
    stats = otlp_store.get_stats()
    return {
        "provider_type": "otlp_sqlite",
        "sources": ["otlp"],
        "db_path": str(otlp_store.DB_PATH),
        "stats": stats,
    }


@router.get("/api/config/discovered")
def get_discovered_directories():
    """Get store info (legacy compat endpoint)."""
    return {
        "db_path": str(otlp_store.DB_PATH),
        "experiments": otlp_store.list_experiments(),
    }


# ============================================================================
# Playground API Endpoints
# ============================================================================


@router.get("/api/playground/models")
def get_playground_models():
    """Get available models for playground."""
    known_keys = get_known_api_key_patterns()
    available_keys = [key for key in known_keys if os.environ.get(key)]

    custom_models = [_sanitize_custom_model(model) for model in load_custom_models()]
    builtin_models = get_builtin_models()

    return {
        "builtin": builtin_models,
        "custom": custom_models,
        "available_api_keys": available_keys,
    }


@router.post("/api/playground/models")
def add_custom_model(model: CustomModel):
    """Add a custom model configuration."""
    models = load_custom_models()

    _validate_custom_model_pair(model.model_dump())

    if any(m["model_id"] == model.model_id for m in models):
        raise HTTPException(status_code=400, detail="Model with this ID already exists")

    models.append(_sanitize_custom_model(model.model_dump()))
    save_custom_models(models)

    return {"status": "success", "message": f"Model '{model.name}' added"}


@router.delete("/api/playground/models/{model_id:path}")
def delete_custom_model(model_id: str):
    """Delete a custom model configuration."""
    models = load_custom_models()

    original_count = len(models)
    models = [m for m in models if m["model_id"] != model_id]

    if len(models) == original_count:
        raise HTTPException(status_code=404, detail="Model not found")

    save_custom_models(models)

    return {"status": "success", "message": "Model deleted"}


def get_model_config(model_id: str) -> dict | None:
    """Look up model configuration by ID."""
    config = load_models_config()
    if model_id in config.get("models", {}):
        model_info = config["models"][model_id]
        return {
            "endpoint": model_info.get("endpoint"),
            "api_key_env": model_info.get("api_key_env"),
        }

    custom_models = load_custom_models()
    for model in custom_models:
        if model.get("model_id") == model_id:
            model = _sanitize_custom_model(model)
            return {
                "endpoint": model.get("endpoint"),
                "api_key_env": model.get("api_key_env"),
            }

    return None


DEFAULT_SANDBOX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute Python code and return the result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The Python code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_result",
            "description": "Return the final result to the user",
            "parameters": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "The result to return",
                    }
                },
                "required": ["result"],
            },
        },
    },
]


def normalize_messages_for_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transform messages to the format expected by OpenAI/litellm API."""
    normalized = []
    for msg in messages:
        new_msg = {"role": msg.get("role", "user")}

        content = msg.get("content")
        if content is not None:
            new_msg["content"] = content
        elif msg.get("role") == "assistant" and msg.get("tool_calls"):
            new_msg["content"] = None

        if msg.get("tool_calls") and msg.get("role") == "assistant":
            api_tool_calls = []
            for tc in msg["tool_calls"]:
                if "function" in tc and "type" in tc:
                    api_tool_calls.append(tc)
                else:
                    args = tc.get("arguments", {})
                    if isinstance(args, dict):
                        args_str = json.dumps(args)
                    else:
                        args_str = str(args)

                    api_tool_calls.append(
                        {
                            "id": tc.get("id", f"call_{len(api_tool_calls)}"),
                            "type": "function",
                            "function": {
                                "name": tc.get("name", "unknown"),
                                "arguments": args_str,
                            },
                        }
                    )
            new_msg["tool_calls"] = api_tool_calls

        if msg.get("tool_call_id"):
            new_msg["tool_call_id"] = msg["tool_call_id"]

        normalized.append(new_msg)

    return normalized


async def _collect(
    response: "litellm.ModelResponse | litellm.CustomStreamWrapper",
) -> "litellm.ModelResponse":
    """Consume a streaming or non-streaming litellm response, always returning ModelResponse."""
    import litellm

    if isinstance(response, litellm.CustomStreamWrapper):
        chunks = [chunk async for chunk in response]  # type: ignore
        result = litellm.stream_chunk_builder(chunks)
        if result is None:
            raise ValueError("stream_chunk_builder returned None for empty stream")
        if not isinstance(result, litellm.ModelResponse):
            raise TypeError(f"Expected ModelResponse, got {type(result)}")
        return result
    return response


@router.post("/api/playground/inference")
async def run_inference(request: InferenceRequest):
    """Run LLM inference with the specified model and messages."""
    try:
        import litellm

        model_config = get_model_config(request.model)
        normalized_messages = normalize_messages_for_api(request.messages)

        kwargs = {
            "model": request.model,
            "messages": normalized_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        has_tool_calls = any(
            msg.get("tool_calls") for msg in normalized_messages if isinstance(msg, dict)
        )
        if has_tool_calls:
            kwargs["tools"] = DEFAULT_SANDBOX_TOOLS

        if model_config and model_config.get("endpoint"):
            kwargs["api_base"] = model_config["endpoint"]
            kwargs["custom_llm_provider"] = "openai"

        if model_config:
            api_key = resolve_api_key_from_config(
                request.model,
                model_config,
                allowed_env_vars=get_known_api_key_patterns(),
            )
            if api_key:
                kwargs["api_key"] = api_key

        response = await litellm.acompletion(**kwargs)
        raw_response = await _collect(response)  # type: ignore[arg-type]

        choice = raw_response.choices[0]
        if not isinstance(choice, litellm.Choices):
            raise TypeError(f"Expected Choices, got {type(choice)}")
        message = choice.message
        reasoning_content = getattr(message, "reasoning_content", None)
        usage = getattr(raw_response, "usage", None)
        return {
            "status": "success",
            "response": {
                "role": message.role,
                "content": message.content,
                "tool_calls": (
                    [tc.model_dump() for tc in message.tool_calls] if message.tool_calls else None
                ),
                "reasoning_content": reasoning_content,
            },
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage is not None else None,
                "completion_tokens": usage.completion_tokens if usage is not None else None,
                "total_tokens": usage.total_tokens if usage is not None else None,
            },
            "model": raw_response.model,
        }
    except ImportError as e:
        raise HTTPException(
            status_code=500, detail="litellm not installed. Run: pip install litellm"
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}") from e
