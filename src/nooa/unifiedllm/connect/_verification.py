# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decide whether a test result still applies to a model entry's current settings.

When a test call runs, its record is "stamped" with a *configuration
snapshot*: a JSON description of everything that affects how requests are
sent (model, endpoint URL, client type, parameters after applying the
reasoning level, retry policy, ...), plus a SHA-256 fingerprint of it. Later,
the snapshot is rebuilt from the entry; if it differs, the result is out of
date ("not current"): it is kept as history but no longer counts as passing.
Whether media inputs are permitted is a separate setting and does not affect
the snapshot.

Snapshots name the environment variable holding the API key, never the key or
a hash of it. This module reads no environment variables, builds no clients
and does no I/O.

The "route scope" (``effective_route_scope``) is a smaller snapshot that
identifies just the endpoint and model; it decides whether stored model
listing facts still apply.
"""

import hashlib
import json
from copy import deepcopy
from urllib.parse import urlsplit, urlunsplit

from nooa.unifiedllm.reasoning import ReasoningConfig, apply_reasoning_level
from nooa.unifiedllm.retry_config import RetryConfig

# Entry fields that registry.client_from_config passes on to requests. Other
# fields (key declarations, unselected reasoning levels, catalogue records,
# media permissions) do not change requests and are not part of the snapshot.
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
    """Return a JSON-only copy of ``value``; raise ValueError on secrets or objects.

    Dicts with a credential-like key (``api_key``, ``authorization``, ...) and
    values that are not plain JSON types are rejected.
    """
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
    """Return the normalised request URL of ``entry``, or None if it has none.

    Uses the field the entry's client type actually sends to. Scheme and host
    are lower-cased and trailing slashes removed; for the Anthropic style a
    trailing ``/v1`` is removed too, since both spellings reach the same API.
    Raises ValueError if the URL contains user info, a query or a fragment.
    """
    if entry.get("client_type") == "decision":
        value = entry.get("endpoint")
    elif entry.get("client_type") == "responses":
        # The Responses client sends base_url as api_base, so prefer it.
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
    """Return the route scope of ``entry``: the endpoint and model it sends requests to.

    Contains the model name, the normalised request URL, client type, API
    style, custom provider and the name of the API key environment variable.
    Request parameters (temperature, reasoning, ...) are not included, so
    changing them keeps stored endpoint facts valid. The URL is always the
    request URL, never a catalogue or model listing URL.

    Raises ValueError if the entry holds an API key value or a client object.
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
    """Return the entry's retry policy as JSON, resolved the way the registry does.

    This is the policy normal use would apply, not the no-retry setting the
    test calls themselves use. Raises ValueError if it has a callback.
    """
    value = entry.get("retry_config")
    if "retry_config" in entry and (value is None or value is False):
        retry = RetryConfig(max_retries=0, rate_limit_extra_retries=0)
    elif isinstance(value, dict):
        retry = RetryConfig(**value)
    else:
        retry = RetryConfig()  # The registry also ignores invalid values here.
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
    """Return the configuration snapshot of ``entry`` for the test ``check``.

    The snapshot describes how requests would be sent (see the module
    docstring); the API key itself is never read. Notes:

    - The reasoning level applied is ``entry["reasoning_level"]``, or for a
      ``level:<name>`` check, that level. ``reasoning_default`` only records
      what the provider does by default and selects nothing.
    - A level's settings replace the matching parameters as whole values
      (an ``extra_body`` from the level replaces the entry's ``extra_body``).
    - Media checks enable just their own media kind when they run, so
      ``media_inputs`` is not part of the snapshot.
    - Decision entries keep only ``timeout`` and, for media checks, their
      image transport settings.

    Call ``configure_entry`` first so Connect's default reply limit, context
    window and reasoning replay settings are filled in. Raises ValueError if
    the entry holds an API key value, a client object or non-JSON values.
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
        # The registry builds CompletionClient/ResponsesClient (LiteLLM), not
        # DirectSDKClient, whatever the entry's "transport" label says. For
        # OpenAI/Azure and the Responses API, LiteLLM merges extra_body into
        # the top level of the request last, so do the same here; Anthropic
        # keeps it nested. Other providers have not been verified, so their
        # extra_body is kept nested as written.
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
    """Return the SHA-256 hex digest of the configuration snapshot (contains no keys)."""
    return _fingerprint(effective_config_snapshot(entry, check))


def _fingerprint(snapshot):
    """Return the SHA-256 hex digest of ``snapshot`` as canonical JSON."""
    return hashlib.sha256(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def stamp_verification(
    record: dict, entry: dict, check: str, *, timeout_seconds: float | None = None
) -> None:
    """Mark ``record`` as a current result of ``check`` for ``entry`` (in place).

    Call this only when the test call has actually run, never when importing
    or editing an entry. Stores the snapshot, its fingerprint and
    ``verification_current = True``.

    The snapshot includes the configured timeout and retry policy so that
    changing them makes the result out of date; that does not mean they were
    tested. Test calls run with their own timeout (``timeout_seconds``) and no
    retries, which is recorded separately in ``verification_execution``.
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
    """Return True if ``record`` was stamped with ``entry``'s current configuration.

    The snapshot is rebuilt from the entry and compared; a stored
    ``verification_current`` flag is not trusted. Returns False for
    unstamped records or entries whose snapshot cannot be built.
    """
    snapshot = record.get("verification_config")
    if not isinstance(snapshot, dict) or snapshot.get("version") != 1:
        return False  # Older records without a snapshot never count as current.
    try:
        current = effective_config_snapshot(entry, check)
        return snapshot == current and record.get("verification_fingerprint") == _fingerprint(
            current
        )
    except (ValueError, TypeError, KeyError):
        return False


def invalidate_verification(entry: dict) -> dict:
    """Return a copy of ``entry`` with each test result's "current" flag recomputed.

    Also recomputes the summary lists ``requests_accepted`` and
    ``reasoning_observed`` from current results only. Outcomes and other
    facts are kept as they are (including negative results such as a model
    reporting it could not read an attachment); old records are never
    re-stamped, so permitting a feature never turns into a passed test.
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
    """Return True if ``tested_entry``'s routing test passed for exactly ``candidate``.

    The routing result must be current for ``candidate``'s configuration. For
    decision entries it must be an accepted decision call; otherwise the
    typed text test call must pass (``text_baseline_passed``). Saving without
    a passing test (manually, or when the budget did not allow testing) is
    decided by the user interface, not here.
    """
    from ._records import text_baseline_passed

    record = tested_entry.get("provenance", {}).get("probes", {}).get("routing", {})
    if not verification_is_current(record, candidate):
        return False
    if candidate.get("client_type") == "decision":
        return record.get("client") == "decision" and record.get("outcome") == "accepted"
    return text_baseline_passed({**record, "verification_current": True})


def save_verification(entry: dict) -> tuple[dict, dict]:
    """Summarise how well the entry about to be saved has been tested.

    Returns ``(summary, checks)``. ``summary["state"]`` is:

    - "verified": the typed text test is current and passed, and every other
      recorded or required check (selected media kinds and reasoning level)
      passed;
    - "partial": the text test passed but some other check is missing or failed;
    - "failed": a current routing result exists but did not pass;
    - "untested": there is no current routing result.

    This describes the recorded checks only; it is not a statement about the
    whole model, nor about whether the save itself succeeded. Results that
    are out of date or have no snapshot never count, even after an import.
    """
    from ._records import check_status

    candidate = invalidate_verification(entry)
    provenance = candidate.get("provenance") or {}
    checks = {
        **(provenance.get("probes") or {}),
        **(provenance.get("session_checks") or {}),
    }
    checks = {name: record for name, record in checks.items() if isinstance(record, dict)}
    current = {
        name: record
        for name, record in checks.items()
        if verification_is_current(record, candidate, name)
        and record.get("outcome") not in {"not_probed", "running"}
    }
    passed = [
        name
        for name, record in current.items()
        if check_status(
            name,
            record,
            entry=candidate,
            missing_reasoning=name.startswith("level:")
            and record.get("reasoning_observed") is not True,
        )
        == "passed"
    ]
    baseline = routing_allows_save(candidate, candidate)
    declared = [f"media:{kind}" for kind in candidate.get("media_inputs", [])]
    if candidate.get("reasoning_level"):
        declared.append("level:" + candidate["reasoning_level"])
    gaps = list(dict.fromkeys(name for name in [*checks, *declared] if name not in passed))
    state = (
        ("partial" if gaps else "verified")
        if baseline
        else "failed"
        if "routing" in current
        else "untested"
    )
    return {
        "state": state,
        "verified": state == "verified",
        "baseline_verified": baseline,
        "current_fingerprint": effective_config_fingerprint(candidate),
        "verified_scopes": passed,
        "unverified_scopes": gaps,
        "scope": "recorded checks only; not whole-model certification",
        "checks": checks,
    }, checks
