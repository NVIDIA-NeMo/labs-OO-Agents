# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Current-configuration trust, separate from historical evidence and permissions.

Snapshots are versioned, detached, JSON-only descriptions of effective behavior.
They contain credential *declarations*, never credential values or their hashes.
No environment reads, client construction, discovery or other I/O happens here.
"""

import hashlib
import json
from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit

from nooa.unifiedllm.reasoning import ReasoningConfig, apply_reasoning_level
from nooa.unifiedllm.retry_config import RetryConfig

# Match registry.client_from_config's forwarded behavior fields. Declarations,
# dormant levels, catalogue provenance and media permission lists are not calls.
_PARAMS = (
    "temperature",
    "top_p",
    "timeout",
    "chat_template_kwargs",
    "custom_llm_provider",
    "max_tokens",
    "reasoning",
    "thinking",
    "reasoning_effort",
    "allowed_openai_params",
    "additional_drop_params",
    "extra_body",
    "store",
    "include",
    "output_config",
)
_SECRETS = {
    "api_key",
    "apikey",
    "x_api_key",
    "authorization",
    "proxy_authorization",
    "access_token",
    "client_secret",
    "password",
}


def _safe(value):
    if isinstance(value, dict):
        if any(str(k).lower().replace("-", "_") in _SECRETS for k in value):
            raise ValueError("Verification snapshots cannot contain credentials")
        return {k: _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError("Verification snapshots require JSON configuration, not client objects")


def _route(entry):
    if entry.get("client_type") == "decision":
        value = entry.get("endpoint")
    elif entry.get("client_type") == "responses":
        # Responses forwards base_url as api_base at dispatch.
        value = entry.get("base_url") or entry.get("api_base")
    else:
        value = entry.get("api_base") or entry.get("base_url")
    if not value:
        return None
    parts = urlsplit(value.strip())
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("Verification routes cannot contain credentials, query or fragment")
    path = parts.path.rstrip("/")
    if entry.get("api_style") == "anthropic":
        path = path.removesuffix("/v1")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def effective_route_scope(entry: dict) -> dict:
    """Non-secret endpoint-fact identity, independent of behavioral parameters.

    Never derive an inference route from a catalogue/models URL. Scope is the
    current model/client/normalized inference route and credential declaration.
    """
    if entry.get("api_key") is not None or entry.get("client") is not None:
        raise ValueError("Route scope requires a credential declaration and no client object")
    return _safe(
        {
            "model": entry.get("model_name"),
            "route": _route(entry),
            "client_type": entry.get("client_type", "completion"),
            "api_style": entry.get("api_style", "chat"),
            "custom_llm_provider": entry.get("custom_llm_provider"),
            "credential_scope": {"api_key_env": entry.get("api_key_env") or None},
        }
    )


def _retry_policy(entry):
    # Mirror registry and client defaults, not the check's no-retry override.
    value = entry.get("retry_config")
    if "retry_config" in entry and (value is None or value is False):
        retry = RetryConfig(max_retries=0, rate_limit_extra_retries=0)
    elif isinstance(value, dict):
        retry = RetryConfig(**value)
    else:
        retry = RetryConfig()  # Registry ignores invalid chat-policy declarations.
    if retry.on_retry is not None:
        raise ValueError("Verification requires JSON retry configuration, not callbacks")
    policy = retry.model_dump(
        exclude={"retryable_exceptions", "retryable_status_codes", "on_retry"}
    )
    policy["retryable_status_codes"] = sorted(retry.retryable_status_codes)
    policy["retryable_exceptions"] = [
        f"{exc.__module__}.{exc.__qualname__}" for exc in retry.retryable_exceptions
    ]
    return policy


def effective_config_snapshot(entry: dict, check: str = "routing") -> dict:
    """Resolve a configured candidate exactly once, without resolving credentials.

    reasoning_default documents provider defaults; only reasoning_level selects.
    A level check selects its own patch independently of the baseline selection.
    Whole patch values replace constructor defaults (including shadowed extra_body).
    Media checks force their own one-format permission at execution, so neither
    baseline trust nor independent media evidence depends on media_inputs grants.
    Call configure_entry first to resolve Connect's safe cap/context/retention defaults.
    """
    if entry.get("api_key") is not None or entry.get("client") is not None:
        raise ValueError("Verification requires a credential declaration and no client object")
    decision = entry.get("client_type") == "decision"
    level = check[6:] if check.startswith("level:") else entry.get("reasoning_level")
    params = {k: deepcopy(entry[k]) for k in _PARAMS if k in entry}
    if decision:
        params = {k: params[k] for k in ("timeout",) if k in params}
        level = None
    else:
        params = apply_reasoning_level(
            ReasoningConfig(
                levels=entry.get("reasoning_levels"), default=entry.get("reasoning_default")
            ),
            entry.get("model_name", ""),
            params,
            {},
            level,
        )
        # The registry constructs CompletionClient/ResponsesClient (LiteLLM),
        # not DirectSDKClient. The declared transport label does not select that
        # backend. Its OpenAI/Azure SDK dispatch and Responses HTTP adapter
        # flatten extra_body last; native Anthropic keeps it nested. Preserve
        # layers conservatively for other, unverified provider adapters.
        provider = (
            entry.get("custom_llm_provider") or (entry.get("model_name") or "").split("/", 1)[0]
        )
        if entry.get("client_type") == "responses" or provider in {"openai", "azure"}:
            extra = params.pop("extra_body", {}) or {}
            params.update(extra)
    snapshot = {
        "version": 1,
        "model": entry.get("model_name"),
        "route": _route(entry),
        "client_type": entry.get("client_type", "completion"),
        "api_style": entry.get("api_style", "chat"),
        "transport": entry.get("transport", "direct"),
        "credential_scope": {"api_key_env": entry.get("api_key_env") or None},
        "level": level,
        "params": params,
        "context_window": entry.get("context_window"),
        "structured_output_mode": entry.get("structured_output_mode"),
        "cache_breakpoint": None if decision else entry.get("cache_breakpoint", "auto"),
        "drop_params": entry.get("drop_params", True),
        "retry_policy": _retry_policy(entry),
    }
    if decision and check.startswith("media:"):
        snapshot["decision_media"] = {
            "decision_image_transport": entry.get("decision_image_transport", "json"),
            "typesafe_danger_zone_images": entry.get("typesafe_danger_zone_images", False),
        }
    return _safe(snapshot)


def effective_config_fingerprint(entry: dict, check: str = "routing") -> str:
    """SHA-256 of canonical non-secret effective configuration (not key material)."""
    return _fingerprint(effective_config_snapshot(entry, check))


def _fingerprint(snapshot):
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def stamp_verification(
    record: dict, entry: dict, check: str, *, timeout_seconds: float | None = None
) -> None:
    """Stamp configured dependencies at execution, never on import/reconfiguration.

    Timeout and retry policy are save-identity dependencies, not claims that the
    configured policies were exercised: checks use bounded timeouts and zero
    retries. Their execution policy is separate evidence.
    """
    snapshot = effective_config_snapshot(entry, check)
    record["verification_config"] = snapshot
    record["verification_fingerprint"] = _fingerprint(snapshot)
    record["verification_current"] = True
    record["verification_execution"] = {
        "endpoint_retries": 0,
        "timeout_seconds": timeout_seconds,
        "configured_retry_policy_tested": False,
        "configured_timeout_policy_tested": False,
    }


def verification_is_current(record: dict, entry: dict, check: str = "routing") -> bool:
    """Compare the tested dependencies to CURRENT candidate, not a saved boolean."""
    snapshot = record.get("verification_config")
    if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
        return False  # Unstamped legacy evidence is historical, even unchanged.
    try:
        current = effective_config_snapshot(entry, check)
        return snapshot == current and record.get("verification_fingerprint") == _fingerprint(
            current
        )
    except (ValueError, TypeError, KeyError):
        return False


def invalidate_verification(entry: dict) -> dict:
    """Return a detached candidate with per-check current flags recomputed.

    Keep independent/historical facts, including false access evidence. Never
    stamp old records or rewrite their outcomes to turn permission into proof.
    """
    result = deepcopy(entry)
    provenance = result.get("provenance") or {}
    for group in ("probes", "session_checks"):
        for name, record in (provenance.get(group) or {}).items():
            if isinstance(record, dict) and record.get("outcome") != "not_probed":
                record["verification_current"] = verification_is_current(record, result, name)
    probes = provenance.get("probes") or {}
    for aggregate, predicate in (
        ("requests_accepted", lambda r: r.get("outcome") == "accepted"),
        ("reasoning_observed", lambda r: r.get("reasoning_observed") is True),
    ):
        if aggregate in provenance:
            provenance[aggregate] = [
                name
                for name, record in probes.items()
                if isinstance(record, dict)
                and record.get("verification_current") is True
                and predicate(record)
            ]
    return result


def routing_allows_save(tested_entry: dict, candidate: dict) -> bool:
    """Fresh genuine baseline (or decision routing) for the exact save candidate.

    Manual/budget-untested saving remains a frontend policy, not fresh trust.
    """
    from ._records import text_baseline_passed

    record = tested_entry.get("provenance", {}).get("probes", {}).get("routing", {})
    if not verification_is_current(record, candidate):
        return False
    if candidate.get("client_type") == "decision":
        return record.get("client") == "decision" and record.get("outcome") == "accepted"
    return text_baseline_passed({**record, "verification_current": True})
