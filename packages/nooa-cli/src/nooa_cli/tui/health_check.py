# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup health check for the LLM endpoint.

Sends a minimal completion request at TUI launch to surface misconfiguration
early — before the user types their first message and gets a cryptic traceback.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

logger = logging.getLogger(__name__)

# Timeout for the probe call — fast enough to not annoy, slow enough
# for cold-start endpoints (serverless, NIM behind an autoscaler).
_PROBE_TIMEOUT_SECONDS = 30

# Some providers/models reject a one-token output cap as too small to finish
# even a tiny health-check response. Keep this cheap without being pathological.
_PROBE_MAX_TOKENS = 32

# Known provider → env var mapping (covers the common cases)
_PROVIDER_API_KEY_ENV: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "vertex_ai": "GOOGLE_APPLICATION_CREDENTIALS",
    "azure": "AZURE_API_KEY",
    "cohere": "COHERE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "together_ai": "TOGETHERAI_API_KEY",
    "fireworks_ai": "FIREWORKS_AI_API_KEY",
    "nvidia_nim": "NVIDIA_API_KEY",
}

_NO_MODELS_CONFIGURED_MESSAGE = "No LLM connected. Run `/connect` to configure one."


@dataclass
class HealthCheckResult:
    """Outcome of the LLM health probe."""

    ok: bool
    error_message: str | None = None
    fix_hint: str | None = None
    # Permanent configuration failures should prevent agent turns until the
    # user successfully changes models. Transient network/rate-limit failures
    # remain warnings so a real prompt may still succeed moments later.
    blocking: bool = False
    # Startup can render before the real provider probe finishes. While pending,
    # normal agent prompts stay blocked, but slash and shell commands remain usable.
    pending: bool = False


def _detect_provider(model: str) -> str | None:
    """Best-effort provider detection from model string."""
    try:
        import litellm

        # LiteLLM prints its provider documentation URL to stdout/stderr on a
        # registry miss. Detection is an internal branch decision, so keep that
        # library noise out of the TUI and render our actionable diagnosis.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _, provider, _, _ = litellm.get_llm_provider(model)
        return provider
    except Exception:
        return None


def unresolved_model_health(model: str) -> HealthCheckResult:
    """Return the permanent, actionable result for an unresolved model alias."""
    return HealthCheckResult(
        ok=False,
        error_message=(
            f"No LLM registry entry is loaded for model '{model}', "
            "and its provider cannot be inferred."
        ),
        fix_hint=(
            "  • Add .nooa/llm_config.yaml, pass --llm-config FILE, or install "
            "your organization's registry extension\n"
            "  • In the TUI, run `/connect <server-url>` to save an alias\n"
            "  • Run `nooa config show` to inspect locally discovered registry layers\n"
            "  • Or choose a provider-qualified model with --model <provider/model>"
        ),
        blocking=True,
    )


def no_models_configured_health() -> HealthCheckResult:
    """Return the first-run result when no model aliases are configured."""
    return HealthCheckResult(
        ok=False,
        error_message=_NO_MODELS_CONFIGURED_MESSAGE,
        blocking=True,
    )


def is_no_models_configured_health(result: HealthCheckResult | None) -> bool:
    """Return True when *result* is the first-run no-models state."""
    return result is not None and result.error_message == _NO_MODELS_CONFIGURED_MESSAGE


def _get_expected_env_var(llm: UnifiedLLM) -> str | None:
    """Determine which API key env var the LLM expects."""
    # Check registry config first (explicit api_key_env)
    registry_cfg = getattr(llm, "_registry_config", None) or {}
    if api_key_env := registry_cfg.get("api_key_env"):
        return api_key_env

    # Fall back to provider detection
    model = getattr(llm, "model", "")
    provider = _detect_provider(model)
    if provider:
        return _PROVIDER_API_KEY_ENV.get(provider)
    return None


def _has_llm_config_yaml() -> bool:
    """Check if a **user-supplied** llm_config.yaml is discoverable.

    Returns ``True`` only when the user has added a config file via
    one of the user-controlled layers (user dir, ``NEMO_OO_LLM_CONFIG``,
    project dir). The bundled defaults that ship in the wheel are
    deliberately excluded — they're always present on every install,
    so including them here would steer every connection / auth /
    model-not-found failure toward "check your llm_config.yaml" hints
    even for users who haven't created one.
    """
    try:
        from nooa.llm_config import (
            bundled_config_paths,
            llm_config_chain,
        )

        bundled_set = {p.resolve() for p in bundled_config_paths()}
        return any(p not in bundled_set for p in llm_config_chain())
    except ImportError:
        # Helper not installed (e.g. running against a stripped-down
        # wheel). Treat as "no config" — caller will fall through to
        # generic hints.
        return False
    except Exception:
        # Unexpected failure inside discovery (broken paths.py
        # dependency, OSError walking dirs, etc.). Don't crash the
        # health-check; log it so the real error is recoverable.
        logger.debug("llm_config_chain() raised unexpectedly", exc_info=True)
        return False


def _has_project_config() -> bool:
    """Check if a settings.yaml exists in any layer (user / project / env)."""
    try:
        from .settings import settings_present

        return settings_present()
    except Exception:
        return Path(".nooa/settings.yaml").exists()


def _classify_error(exc: Exception, llm: UnifiedLLM) -> HealthCheckResult:
    """Map a raw exception to a user-friendly diagnosis with fix instructions."""
    model = getattr(llm, "model", "unknown")
    msg = str(exc).lower()
    exc_type = type(exc).__name__

    has_yaml = _has_llm_config_yaml()
    has_config = _has_project_config()

    # --- Registry alias / provider cannot be resolved ---
    # LiteLLM's raw error is both noisy (it repeats the provider-doc URL) and
    # misleading for registry users: an organization-local alias is not fixed
    # by guessing a public API-key variable. Keep this message generic so the
    # public TUI can point at the extension boundary without knowing anything
    # about a private registry package or its contents.
    if "llm provider not provided" in msg or (
        "provider not provided" in msg and "pass in the llm provider" in msg
    ):
        return unresolved_model_health(model)

    expected_env = _get_expected_env_var(llm)

    # --- Authentication / API key issues ---
    if any(
        k in msg
        for k in (
            "401",
            "unauthorized",
            "invalid api key",
            "invalid x-api-key",
            "authentication",
            "api key not valid",
            "incorrect api key",
            "invalid_api_key",
        )
    ):
        hint_lines = [f"Authentication failed for model '{model}'."]
        fix_lines = []

        if expected_env:
            env_val = os.environ.get(expected_env)
            if env_val is None:
                fix_lines.append(
                    f"  • {expected_env} is NOT set — export it in your shell or .env file"
                )
            else:
                fix_lines.append(
                    f"  • {expected_env} is set but the key appears invalid — "
                    "check it hasn't expired or been revoked"
                )
        else:
            fix_lines.append(
                "  • Check the API key env var for your provider "
                "(e.g. ANTHROPIC_API_KEY, OPENAI_API_KEY)"
            )

        if has_yaml:
            fix_lines.append(
                "  • If using llm_config.yaml, verify 'api_key_env' points to the right variable"
            )

        return HealthCheckResult(
            ok=False,
            error_message=hint_lines[0],
            fix_hint="\n".join(fix_lines),
            blocking=True,
        )

    # --- Model not found / doesn\'t exist ---
    if any(
        k in msg
        for k in (
            "404",
            "not found",
            "model not found",
            "does not exist",
            "no such model",
            "model_not_found",
            "invalid model",
            "decommissioned",
        )
    ):
        fix_lines = [
            f"  • Verify the model name is correct (currently: '{model}')",
            "  • Run with --model <name> to try a different model",
        ]
        if has_config:
            fix_lines.append("  • Or edit 'default_model' in .nooa/settings.yaml")
        else:
            fix_lines.append(
                "  • Or run `/config set model <valid-model-name>` to write settings.yaml"
            )
        if has_yaml:
            fix_lines.append(
                "  • If using a custom endpoint via llm_config.yaml, ensure the model is deployed"
            )

        return HealthCheckResult(
            ok=False,
            error_message=f"Model '{model}' was not found by the API provider.",
            fix_hint="\n".join(fix_lines),
            blocking=True,
        )

    # --- Permission / access denied ---
    if any(
        k in msg
        for k in (
            "403",
            "forbidden",
            "permission",
            "access denied",
            "insufficient_quota",
            "billing",
        )
    ):
        return HealthCheckResult(
            ok=False,
            error_message=f"Access denied for model '{model}'.",
            fix_hint=(
                "Your account doesn't have access to this model. Check:\n"
                "  • Your API plan includes access to this model\n"
                "  • Billing is active and quota is not exhausted\n"
                "  • Organization/project permissions are correct"
            ),
            blocking=True,
        )

    # --- Connection refused / DNS / network ---
    if any(
        k in msg
        for k in (
            "connection",
            "connect",
            "refused",
            "unreachable",
            "dns",
            "name or service not known",
            "nodename nor servname",
            "network",
            "errno",
            "gaierror",
        )
    ):
        fix_lines = ["  • Check your internet connection"]
        if has_yaml:
            fix_lines.append("  • Verify 'api_base' in llm_config.yaml is correct and reachable")
        else:
            # If no llm_config.yaml, the endpoint is the provider default
            fix_lines.append("  • The provider's API may be temporarily down — try again")
        fix_lines.append("  • Check VPN or proxy settings if behind a firewall")

        return HealthCheckResult(
            ok=False,
            error_message=f"Cannot connect to the LLM endpoint for model '{model}'.",
            fix_hint="\n".join(fix_lines),
        )

    # --- Timeout ---
    if any(k in msg for k in ("timeout", "timed out", "deadline")):
        fix_lines = [
            f"The endpoint did not respond within {_PROBE_TIMEOUT_SECONDS}s.",
            "  • The service may be overloaded or cold-starting — try again",
            "  • A firewall may be silently dropping packets",
        ]
        if has_yaml:
            fix_lines.append("  • Check 'api_base' in llm_config.yaml is correct")

        return HealthCheckResult(
            ok=False,
            error_message=f"LLM endpoint timed out for model '{model}'.",
            fix_hint="\n".join(fix_lines),
        )

    # --- Output cap too small ---
    if any(
        k in msg
        for k in (
            "max_tokens or model output limit was reached",
            "model output limit was reached",
            "finish the message because max_tokens",
            "output limit was reached",
        )
    ):
        return HealthCheckResult(
            ok=False,
            error_message=f"LLM health check output limit was too small for model '{model}'.",
            fix_hint=(
                "The startup probe reached the model output cap before completing.\n"
                "  • Try a different model temporarily with --model <name>\n"
                "  • Upgrade nooa if a newer version is available\n"
                "  • If this persists on the latest version, file a bug with the "
                "model name and full startup error"
            ),
        )

    # --- Rate limit ---
    if any(k in msg for k in ("429", "rate limit", "too many requests", "rate_limit")):
        return HealthCheckResult(
            ok=False,
            error_message=f"Rate limited by the API provider for model '{model}'.",
            fix_hint=(
                "You've hit the rate limit. This is usually temporary:\n"
                "  • Wait a moment and try again\n"
                "  • Check your plan's rate limits at the provider dashboard"
            ),
        )

    # --- Generic / unknown ---
    fix_lines = ["Troubleshooting steps:"]
    if expected_env:
        env_status = "set" if os.environ.get(expected_env) else "NOT set"
        fix_lines.append(f"  • API key env var: {expected_env} ({env_status})")
    else:
        fix_lines.append(
            "  • Check env vars for API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)"
        )
    fix_lines.append(f"  • Current model: '{model}' — try --model gpt-4o-mini")
    if has_yaml:
        fix_lines.append("  • Check llm_config.yaml for api_base / api_key_env")
    if has_config:
        fix_lines.append("  • Check .nooa/settings.yaml for model name")

    return HealthCheckResult(
        ok=False,
        error_message=f"LLM health check failed for model '{model}': {exc_type}: {exc}",
        fix_hint="\n".join(fix_lines),
    )


async def probe_llm(llm: UnifiedLLM) -> HealthCheckResult:
    """Send a minimal request to verify the LLM endpoint is reachable and authenticated.

    This sends a tiny completion request ("hi") with max_tokens=1 so it\'s fast
    and cheap. The goal is to exercise the full auth + routing path without
    burning meaningful tokens.

    Returns HealthCheckResult with ok=True on success, or a user-friendly
    error_message + fix_hint on failure.
    """
    try:
        await asyncio.wait_for(
            llm.acall(
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=_PROBE_MAX_TOKENS,
            ),
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        return HealthCheckResult(ok=True)

    except TimeoutError:
        return _classify_error(Exception("Request timed out"), llm)
    except Exception as exc:
        return _classify_error(exc, llm)
