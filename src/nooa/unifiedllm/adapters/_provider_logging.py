# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-owned durable Logging handoff using the installed SDK's own setup."""

from datetime import datetime
from inspect import signature
from uuid import uuid4

from nooa._provider_ownership import _OWNED_LOGGING_KEY


def _prepare_owned_litellm_call(kwargs: dict, *, owner, call_type: str) -> dict:
    """Run the supported SDK's own setup before injecting its call-bound logger.

    This is called only at actual provider entry, after successful admission.
    It is not an adapter hook, retry executor, callback registration or generic
    tracing switch. Missing/incompatible private SDK setup fails before dispatch;
    setup/rule rejection is deliberately outside any best-effort tracing catch.
    """
    from nooa.unifiedllm._observations import _ObservationOwner

    if not isinstance(owner, _ObservationOwner):
        raise TypeError("Explicit one-send provider owner required")
    if call_type not in {"completion", "acompletion", "responses", "aresponses"}:
        raise ValueError("Unsupported owned LiteLLM call type")
    if kwargs.get("litellm_logging_obj") is not None:
        raise ValueError("Owned calls require a fresh call-bound SDK logger")
    try:
        from litellm.litellm_core_utils.rules import Rules
        from litellm.utils import function_setup

        compatible = tuple(signature(function_setup).parameters) == (
            "original_function",
            "rules_obj",
            "start_time",
            "args",
            "kwargs",
        )
    except (ImportError, ValueError, TypeError):
        compatible = False
    if not compatible:
        raise RuntimeError("Unsupported LiteLLM call-bound logging setup")
    prepared = dict(kwargs)
    prepared.setdefault("litellm_call_id", str(uuid4()))
    # Normal setup partitions these lists in-place; do not mutate caller aliases.
    for name in ("success_callback", "failure_callback"):
        if isinstance(prepared.get(name), list):
            prepared[name] = list(prepared[name])
    logger, prepared = function_setup(call_type, Rules(), datetime.now(), **prepared)
    if not isinstance(prepared, dict) or not isinstance(
        getattr(logger, "model_call_details", None), dict
    ):
        raise RuntimeError("Unsupported LiteLLM call-bound Logging contract")
    logger.model_call_details[_OWNED_LOGGING_KEY] = True
    prepared["litellm_logging_obj"] = logger
    return prepared
