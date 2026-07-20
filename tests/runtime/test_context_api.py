# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ContextApi — dict-like Skill wrapping agent.context_manager."""

import pytest

from nooa import Agent
from nooa.context_blocks import DynamicContext
from nooa.context_blocks.exceptions import DynamicNotResolvedError, ProtectedBlockError
from nooa.runtime.context import ContextApi
from nooa.unifiedllm import FakeLLMClient

_llm = FakeLLMClient()


class _TestAgent(Agent, llm=_llm):
    pass


def _ctx() -> ContextApi:
    return _TestAgent().context


# ── setitem ────────────────────────────────────────────────────────────────────


def test_setitem_and_getitem():
    ctx = _ctx()
    ctx["key"] = "value"
    assert ctx["key"] == "value"


def test_setitem_raises_on_protected_key():
    ctx = _ctx()
    with pytest.raises(ProtectedBlockError):
        ctx["system_prompt"] = "overwrite"


def test_setitem_accepts_dynamic_context_value():
    ctx = _ctx()
    ctx["dynamic"] = DynamicContext("'expr'")
    assert "dynamic" in ctx._context


def test_set_static_expr_lands_in_static_partition():
    ctx = _ctx()
    ctx.set_static("live", expr="1 + 1")
    # Re-evaluated (DynamicContext marker) yet in the static/cacheable partition.
    assert ctx._context.is_static("live") is True
    assert isinstance(ctx._context._blocks["live"], DynamicContext)
    assert ctx._context._blocks["live"].expr == "1 + 1"


def test_set_static_rejects_both_value_and_expr():
    ctx = _ctx()
    with pytest.raises(TypeError, match="both"):
        ctx.set_static("k", "v", expr="1")


def test_set_static_rejects_neither():
    ctx = _ctx()
    with pytest.raises(TypeError, match="requires"):
        ctx.set_static("k")


# ── getitem ────────────────────────────────────────────────────────────────────


def test_getitem_raises_for_missing_key():
    ctx = _ctx()
    with pytest.raises(KeyError):
        _ = ctx["missing"]


def test_getitem_dynamic_raises_before_resolve():
    ctx = _ctx()
    ctx.set_dynamic("dyn", "'hello'")
    with pytest.raises(DynamicNotResolvedError):
        _ = ctx["dyn"]


def test_getitem_dynamic_returns_after_resolve():
    ctx = _ctx()
    ctx.set_dynamic("dyn", "'hello'")
    ctx._context._update_resolved({"dyn": "hello"})
    assert ctx["dyn"] == "hello"


# ── set_dynamic ────────────────────────────────────────────────────────────────


def test_set_dynamic_raises_on_protected_key():
    ctx = _ctx()
    with pytest.raises(ProtectedBlockError):
        ctx.set_dynamic("system_prompt", "'expr'")


# ── delitem ────────────────────────────────────────────────────────────────────


def test_delitem():
    ctx = _ctx()
    ctx["key"] = "value"
    del ctx["key"]
    assert "key" not in ctx


def test_delitem_raises_for_missing_key():
    ctx = _ctx()
    with pytest.raises(KeyError):
        del ctx["missing"]


def test_delitem_raises_for_protected_key():
    ctx = _ctx()
    ctx._context._blocks["guarded"] = "value"
    ctx._context.protected_keys.add("guarded")
    with pytest.raises(ProtectedBlockError):
        del ctx["guarded"]


# ── contains / len / iter / repr ───────────────────────────────────────────────


def test_contains():
    ctx = _ctx()
    ctx["a"] = 1
    assert "a" in ctx
    assert "b" not in ctx


def test_len_and_iter():
    ctx = _ctx()
    ctx["a"] = 1
    ctx["b"] = 2
    assert len(ctx) == 2
    assert set(ctx) == {"a", "b"}


def test_repr():
    ctx = _ctx()
    ctx["x"] = 1
    assert "1 block" in repr(ctx)


# ── keys / _raw_items ──────────────────────────────────────────────────────────


def test_keys():
    ctx = _ctx()
    ctx["a"] = 1
    ctx["b"] = 2
    assert set(ctx.keys()) == {"a", "b"}


# ── get ────────────────────────────────────────────────────────────────────────


def test_get_returns_value():
    ctx = _ctx()
    ctx["key"] = "value"
    assert ctx.get("key") == "value"


def test_get_returns_default_for_missing():
    ctx = _ctx()
    assert ctx.get("missing", "default") == "default"
    assert ctx.get("missing") is None


# ── pop ────────────────────────────────────────────────────────────────────────


def test_pop_static_block():
    ctx = _ctx()
    ctx["key"] = "value"
    assert ctx.pop("key") == "value"
    assert "key" not in ctx


def test_pop_dynamic_block():
    ctx = _ctx()
    ctx._context._blocks["dyn"] = DynamicContext("'hello'")
    ctx._context._dynamic_cache["dyn"] = "hello"
    assert ctx.pop("dyn") == "hello"
    assert "dyn" not in ctx


def test_pop_returns_default_for_missing():
    ctx = _ctx()
    assert ctx.pop("missing", "default") == "default"


def test_pop_raises_for_missing_without_default():
    ctx = _ctx()
    with pytest.raises(KeyError):
        ctx.pop("missing")


def test_pop_raises_for_protected_key():
    ctx = _ctx()
    ctx._context._blocks["guarded"] = "value"
    ctx._context.protected_keys.add("guarded")
    with pytest.raises(ProtectedBlockError):
        ctx.pop("guarded")
