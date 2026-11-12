# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SDK-free effective settings/facts seam for exact built-in loaded views.

No model map, tokenization, estimation or transport discovery. A selected cap is
not an advertised server ceiling. Legacy/custom limit resolvers stay separate.
"""

from nooa.unifiedllm.errors import FailureKind, LLMFailure
from nooa.unifiedllm.limits import REPLY_CAP_KEYS, ContextLimits

from ._binding import _thaw


def _effective_config(adapter):
    from ._native import _BUILTINS

    if type(adapter) in _BUILTINS:
        return _thaw(adapter._settings.selected)
    if hasattr(adapter, "_bound_settings"):
        from nooa.unifiedllm.reasoning import ReasoningConfig, apply_reasoning_level

        settings = adapter._bound_settings
        return apply_reasoning_level(
            ReasoningConfig(levels=_thaw(settings.reasoning_levels)),
            adapter._client.model,
            _thaw(settings.config),
            {},
            settings.reasoning_level,
        )
    raise LLMFailure(FailureKind.CONFIG, "Effective limits require a built-in loaded settings view")


def _adapter_limits(adapter, facts, fallback_reserve=0):
    config = _effective_config(adapter)
    # The extra-body layer wins on the wire, including native cap aliases.
    extra = config.get("extra_body")
    layers = [config] + ([extra] if isinstance(extra, dict) else [])
    cap = None
    for layer in layers:
        selected = [layer[key] for key in sorted(REPLY_CAP_KEYS) if layer.get(key) is not None]
        if len(selected) > 1:
            raise LLMFailure(FailureKind.CONFIG, "Conflicting selected reply caps")
        if selected:
            cap = selected[0]
    if cap is not None:
        if type(cap) is not int or cap <= 0:
            raise LLMFailure(FailureKind.CONFIG, "Selected reply cap must be a positive integer")
        if facts.context_window is not None and cap >= facts.context_window:
            raise LLMFailure(FailureKind.CONFIG, "Selected reply cap leaves no input context")
        if facts.max_output_tokens is not None and cap > facts.max_output_tokens:
            raise LLMFailure(FailureKind.CONFIG, "Selected reply cap exceeds known output ceiling")
    return ContextLimits(
        facts.context_window, cap if cap is not None else fallback_reserve, cap is None
    )


def _loaded_limits(llm, params, fallback_reserve):
    from ._loader import _check_derivable

    if params is not None and (type(params) is not dict or params):
        raise LLMFailure(
            FailureKind.CONFIG, "Loaded limits do not accept overrides; derive settings first"
        )
    if type(fallback_reserve) is not int or fallback_reserve < 0:
        raise LLMFailure(FailureKind.CONFIG, "Fallback reserve must be a non-negative integer")
    with llm._view_lock:
        _check_derivable(llm)
        return _adapter_limits(llm._adapter, llm._capabilities, fallback_reserve)
