# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Endpoint context ceilings are facts about a route, not global model metadata."""

from copy import deepcopy

import pytest

from nooa.unifiedllm import connect


def bounded_entry():
    return connect.plan(
        "local",
        "model",
        "chat",
        "https://a.test/v1",
        "A_KEY",
        context_window=128000,
        reply_tokens=1000,
        endpoint_model={"id": "model", "max_input_tokens": 64000},
        endpoint_models_url="https://catalogue.test/shared/models",
    ).entry


@pytest.mark.parametrize(
    "patch",
    [
        {"api_base": "https://b.test/v1"},
        {"api_key_env": "B_KEY"},
        {"model_name": "openai/other"},
        {"client_type": "responses", "api_style": "responses"},
        {"api_style": "anthropic"},
        {"custom_llm_provider": "other"},
    ],
)
def test_endpoint_fact_scope_mismatch_cannot_reclamp_explicit_context(patch):
    original = bounded_entry()
    assert original["context_window"] == 64000
    assert connect.scoped_endpoint_limits(original)["max_input_tokens"] == 64000
    entry = deepcopy(original)
    entry.update(patch, context_window=128000)
    changed = connect.configure_entry(entry)
    assert changed["context_window"] == 128000
    assert connect.scoped_endpoint_limits(changed) == {}
    assert changed["provenance"]["context_window"]["effective"] == 128000
    assert changed["provenance"]["context_window"]["effective_source"] == "user"
    assert original["context_window"] == 64000
    assert (
        original["provenance"]["endpoint_limits"]["url"] == "https://catalogue.test/shared/models"
    )


@pytest.mark.parametrize(
    "patch",
    [
        {"temperature": 0.5},
        {"thinking": {"type": "adaptive"}},
        {"retry_config": False},
        {"cache_breakpoint": None},
        {"media_inputs": ["image"]},
        {"api_base": "https://A.TEST/v1/"},
    ],
)
def test_behavior_edits_and_normalized_same_route_keep_endpoint_facts(patch):
    entry = bounded_entry()
    entry.update(patch, context_window=128000)
    changed = connect.configure_entry(entry)
    assert changed["context_window"] == 64000
    assert connect.scoped_endpoint_limits(changed)["max_input_tokens"] == 64000


def test_unstamped_endpoint_facts_are_historical_and_do_not_reclamp():
    entry = bounded_entry()
    entry["provenance"].pop("endpoint_limit_scope")
    entry["context_window"] = 128000
    changed = connect.configure_entry(entry)
    assert changed["context_window"] == 128000
    assert connect.scoped_endpoint_limits(changed) == {}


def test_route_change_does_not_resurrect_a_larger_previous_declaration():
    entry = bounded_entry()
    entry["api_base"] = "https://b.test/v1"
    # 64K is now the candidate's literal declaration, even though old provenance says128K.
    changed = connect.configure_entry(entry)
    assert changed["context_window"] == 64000
    assert changed["provenance"]["context_window"]["declared"] == 64000
    assert connect.scoped_endpoint_limits(changed) == {}


def test_fresh_explicit_endpoint_facts_rebind_only_current_scope():
    entry = bounded_entry()
    entry.update(api_base="https://b.test/v1", api_key_env="B_KEY")
    changed = connect.apply_context_window(
        entry, 128000, model="model", endpoint_model={"id": "model", "max_input_tokens": 96000}
    )
    changed = connect.configure_entry(changed)
    assert changed["context_window"] == 96000
    assert connect.scoped_endpoint_limits(changed)["max_input_tokens"] == 96000
    assert changed["provenance"]["endpoint_limit_scope"]["route"] == "https://b.test/v1"
    assert changed["provenance"]["endpoint_limit_scope"]["credential_scope"] == {
        "api_key_env": "B_KEY"
    }
    # Never derive B's inference base or listing URL from the old custom catalogue.
    assert "url" not in changed["provenance"]["endpoint_limits"]


def test_endpoint_output_ceiling_is_not_reused_on_another_route():
    entry = connect.plan(
        "local",
        "model",
        "chat",
        "https://a.test/v1",
        "A_KEY",
        context_window=128000,
        reply_tokens=1000,
        endpoint_model={"id": "model", "max_input_tokens": 64000, "max_output_tokens": 2000},
    ).entry
    assert entry["provenance"]["catalogue_limits"]["max_completion_tokens"] == 2000
    changed = connect.configure_entry(
        {**entry, "api_base": "https://b.test/v1", "context_window": 128000}, reply_tokens=3000
    )
    assert changed["context_window"] == 128000 and changed["max_tokens"] == 3000
    with pytest.raises(ValueError, match="configured model limit"):
        connect.configure_entry(entry, reply_tokens=3000)
