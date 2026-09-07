# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider identity and capability contract tests.

Freezes the public schema of the provider-contract types (golden
fixtures) and pins the normalization, key-derivation, and fail-closed
behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nooa.unifiedllm.contracts import (
    DEFAULT_REASONING_CAPABILITIES,
    OPAQUE_REPLAY_KEY_VERSION,
    ModelCompatGroup,
    NormalizedModel,
    ProviderIdentity,
    ReasoningCapabilities,
    ReasoningKind,
    ReasoningRecord,
    ReasoningReplayMode,
    UnknownProviderIdentityError,
    compat_group_for,
    derive_opaque_replay_key,
    get_reasoning_capabilities,
    parse_model_string,
    register_compat_group,
    register_reasoning_capabilities,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "contracts_golden.json"


# --- Golden schema fixtures ------------------------------------------------


def test_golden_fixture_enums_frozen() -> None:
    golden = json.loads(GOLDEN.read_text())
    assert golden["enums"]["ReasoningKind"] == ["opaque", "text", "checkpoint"]
    assert golden["enums"]["ReasoningReplayMode"] == [
        "off",
        "auto",
        "native_only",
        "text_context",
    ]
    assert ReasoningKind("opaque") is ReasoningKind.OPAQUE
    assert ReasoningReplayMode("text_context") is ReasoningReplayMode.TEXT_CONTEXT


def test_golden_fixture_schemas_frozen() -> None:
    golden = json.loads(GOLDEN.read_text())
    schemas = golden["schemas"]
    assert schemas == {
        "ProviderIdentity": ProviderIdentity.model_json_schema(),
        "ReasoningRecord": ReasoningRecord.model_json_schema(),
        "ReasoningCapabilities": ReasoningCapabilities.model_json_schema(),
        "ModelCompatGroup": ModelCompatGroup.model_json_schema(),
        "NormalizedModel": NormalizedModel.model_json_schema(),
    }


def test_golden_fixture_examples_round_trip() -> None:
    golden = json.loads(GOLDEN.read_text())
    examples = golden["examples"]
    identity = ProviderIdentity.model_validate(examples["provider_identity"])
    assert identity.provider == "openai"
    assert identity.opaque_replay_key is not None
    # The frozen digest must still be reproducible from the frozen identity:
    # if a normalization rule changes without bumping OPAQUE_REPLAY_KEY_VERSION,
    # this catches the drift instead of the presence-only assertion staying green.
    assert identity.opaque_replay_key == derive_opaque_replay_key(
        provider=identity.provider,
        api_style=identity.api_style,
        model=identity.model,
        endpoint_id=identity.endpoint_id,
        account_scope=identity.account_scope,
    )
    record = ReasoningRecord.model_validate(examples["reasoning_record"])
    assert record.kind is ReasoningKind.OPAQUE
    assert record.redaction_class == "opaque"
    # Round-trip through JSON again: no SDK objects, plain NOOA-owned JSON.
    assert json.loads(record.model_dump_json()) == examples["reasoning_record"]
    caps = ReasoningCapabilities.model_validate(examples["capabilities_openai"])
    assert caps.replay_field is None
    assert caps.terminal_backfill is True


def test_opaque_replay_key_version_recorded_in_fixture() -> None:
    golden = json.loads(GOLDEN.read_text())
    assert golden["opaque_replay_key_version"] == OPAQUE_REPLAY_KEY_VERSION
    assert OPAQUE_REPLAY_KEY_VERSION == "nooa.opaque-replay-key.v1"


# --- Gateway alias fixtures (NVIDIA gateway) ---------------------------------


@pytest.mark.parametrize(
    ("model_string", "provider", "model"),
    [
        ("openai/nvidia/zai-org/glm-5.3", "glm", "glm-5.3"),
        ("openai/nvidia/moonshotai/kimi-k2.6", "kimi", "kimi-k2.6"),
        ("openai/azure/openai/gpt-5.6-sol", "openai", "gpt-5.6-sol"),
        ("anthropic/claude-sonnet-4-5", "anthropic", "claude-sonnet-4-5"),
        ("claude-3-5-sonnet", "anthropic", "claude-3-5-sonnet"),
        ("openai/nvidia/nemotron-3-super-v3", "nvidia", "nemotron-3-super-v3"),
        ("openai/nvidia/deepseek-r1", "deepseek", "deepseek-r1"),
        ("openai/nvidia/qwen3.5-35b-a3b", "qwen", "qwen3.5-35b-a3b"),
    ],
)
def test_gateway_aliases_resolve_to_logical_provider(
    model_string: str, provider: str, model: str
) -> None:
    parsed = parse_model_string(model_string)
    assert parsed.provider == provider
    assert parsed.model == model


def test_gateway_prefixes_never_determine_provider() -> None:
    # Routing prefixes are stripped regardless of stacking; the logical
    # provider comes from the model family, not the gateway brand.
    assert parse_model_string("openai/nvidia/zai-org/glm-5.3").provider == "glm"
    assert parse_model_string("nvidia/zai-org/glm-5.3").provider == "glm"
    # The routing prefix alone (no known family) must fail closed.
    assert parse_model_string("openai/someprivate-internal-model").provider is None


# --- Tier and alias normalization ----------------------------------------------


@pytest.mark.parametrize(
    ("model_string", "model", "tier"),
    [
        ("kimi-k3:free", "kimi-k3", "free"),
        ("deepseek-v4-flash:cloud", "deepseek-v4-flash", "cloud"),
        ("openai/nvidia/moonshotai/kimi-k3:free", "kimi-k3", "free"),
        ("qwen3.5-35b-a3b", "qwen3.5-35b-a3b", None),
    ],
)
def test_tier_suffixes_stripped_before_comparison(
    model_string: str, model: str, tier: str | None
) -> None:
    parsed = parse_model_string(model_string)
    assert parsed.model == model
    assert parsed.tier == tier


def test_provider_aliases_normalized() -> None:
    assert parse_model_string("claude/foo").provider == "anthropic"
    assert parse_model_string("zai/glm-5.3").provider == "glm"
    assert parse_model_string("moonshot/kimi-k2.6").provider == "kimi"
    # Bedrock-style embedded vendor ids also normalize.
    assert parse_model_string("anthropic.claude-3-5-sonnet").provider == "anthropic"


def test_empty_model_string_fails_closed() -> None:
    parsed = parse_model_string("")
    assert parsed.provider is None
    assert parsed.model == ""


# --- Key derivation ------------------------------------------------------------


def _openai_key(**overrides: object) -> str | None:
    kwargs: dict[str, object] = {
        "provider": "openai",
        "api_style": "responses",
        "model": "gpt-5.6-sol",
        "endpoint_id": "ep-123",
        "account_scope": "acct-abc",
    }
    kwargs.update(overrides)
    return derive_opaque_replay_key(**kwargs)  # type: ignore[arg-type]


def test_same_identity_derives_same_key() -> None:
    assert _openai_key() == _openai_key()
    # Normalization happens inside derivation: prefix and tier variants of the
    # same identity derive the same key.
    assert _openai_key() == _openai_key(model="openai/azure/openai/gpt-5.6-sol")


def test_key_differs_across_identity_parts() -> None:
    base = _openai_key()
    assert base is not None
    assert _openai_key(api_style="chat-completions") != base
    assert _openai_key(endpoint_id="ep-other") != base
    assert _openai_key(account_scope="acct-other") != base
    assert _openai_key(endpoint_id=None, account_scope=None) != base


def test_key_undeclared_model_fails_closed() -> None:
    # gpt-4o is a real OpenAI model but has no declared compat group: no key.
    assert _openai_key(model="gpt-4o") is None
    # No provider / empty model: never a speculative key.
    assert _openai_key(model="") is None
    assert derive_opaque_replay_key(provider="", api_style="responses", model="gpt-5.6-sol") is None


def test_key_transport_neutral() -> None:
    # transport is deliberately excluded from compatibility.
    litellm_key = derive_opaque_replay_key(
        provider="openai", api_style="responses", model="gpt-5.6-sol"
    )
    identity = ProviderIdentity.from_model_string(
        "openai/gpt-5.6-sol", api_style="responses", transport="anyllm"
    )
    assert identity.opaque_replay_key == litellm_key


def test_same_compat_group_members_derive_equal_keys() -> None:
    """The compat group is the model dimension of the key.

    Two models in the same hand-verified group must be replay-compatible:
    raw model ids are NOT mixed into the key payload, so model-id churn
    never changes the key.
    """
    k55 = derive_opaque_replay_key(provider="openai", api_style="responses", model="gpt-5.5")
    k56 = derive_opaque_replay_key(provider="openai", api_style="responses", model="gpt-5.6-sol")
    assert k55 is not None and k56 is not None
    assert k55 == k56
    # Different groups / providers still diverge.
    assert k56 != derive_opaque_replay_key(
        provider="anthropic", api_style="messages", model="claude-sonnet-4-5"
    )


def test_bedrock_dotted_id_derives_same_key_as_plain_form() -> None:
    """Two spellings of one identity must not diverge in key derivation."""
    dotted = derive_opaque_replay_key(
        provider="anthropic", api_style="messages", model="anthropic.claude-sonnet-4-5"
    )
    plain = derive_opaque_replay_key(
        provider="anthropic", api_style="messages", model="claude-sonnet-4-5"
    )
    assert dotted is not None
    assert dotted == plain


def test_key_is_prefixed_non_secret_digest() -> None:
    key = _openai_key()
    assert key is not None
    assert key.startswith("sha256:")
    assert len(key) == len("sha256:") + 64
    assert "gpt-5.6-sol" not in key  # digest, not a model echo


# --- Identity construction ----------------------------------------------------


def test_identity_from_model_string_uses_normalized_parts() -> None:
    identity = ProviderIdentity.from_model_string(
        "openai/nvidia/zai-org/glm-5.3",
        api_style="chat-completions",
        transport="litellm",
    )
    assert identity.provider == "glm"
    assert identity.model == "glm-5.3"


def test_identity_unknown_model_string_fails_closed() -> None:
    with pytest.raises(UnknownProviderIdentityError):
        ProviderIdentity.from_model_string(
            "totally-unknown-vendor/model-x", api_style="responses", transport="litellm"
        )


def test_identity_round_trips_through_json() -> None:
    identity = ProviderIdentity.from_model_string(
        "openai/gpt-5.6-sol", api_style="responses", transport="litellm"
    )
    restored = ProviderIdentity.model_validate(json.loads(identity.model_dump_json()))
    assert restored == identity


# --- Model compat groups --------------------------------------------------------


def test_compat_group_requires_declared_membership() -> None:
    assert compat_group_for("openai", "gpt-5.6-sol") is not None
    # Similar but undeclared model ids fail closed.
    assert compat_group_for("openai", "gpt-5.6-sonnet") is None
    assert compat_group_for("openai", "gpt-5.6-sol-mini") is None
    # Groups are scoped per logical provider.
    assert compat_group_for("glm", "gpt-5.6-sol") is None
    assert compat_group_for("openai", "claude-sonnet-4-5") is None


def test_injected_compat_groups_are_case_normalized() -> None:
    """Callers passing their own groups mapping get the same case handling
    as register_compat_group: mixed-case members must still match."""
    injected = {
        "custom-openai": ModelCompatGroup(
            name="custom-openai",
            provider="OpenAI",
            models=frozenset({"GPT-5.6-SOL"}),
        )
    }
    # Mixed-case provider and model both resolve through the injected mapping.
    group = compat_group_for("openai", "gpt-5.6-sol", groups=injected)
    assert group is not None and group.name == "custom-openai"
    # And key derivation honors the injected (mixed-case) declaration.
    key = derive_opaque_replay_key(
        provider="openai",
        api_style="responses",
        model="gpt-5.6-sol",
        compat_groups=injected,
    )
    assert key is not None


def test_register_compat_group_overrides_default() -> None:
    group = ModelCompatGroup(
        name="openai-gpt-5",
        provider="openai",
        models=frozenset({"gpt-5.6-sol", "gpt-5.6-sol-preview"}),
    )
    register_compat_group(group)
    try:
        assert "gpt-5.6-sol-preview" in compat_group_for("openai", "gpt-5.6-sol").models  # type: ignore[union-attr]
        # Newly declared members derive a key; nothing else changes.
        assert _openai_key(model="gpt-5.6-sol-preview") is not None
        assert _openai_key(model="gpt-5.5") is None
    finally:
        # Restore the default group so other tests are unaffected.
        from nooa.unifiedllm import contracts as c

        c._COMPAT_GROUPS.clear()
        c._COMPAT_GROUPS.update({group.name: group for group in c._DEFAULT_COMPAT_GROUPS})


# --- Capability profile -----------------------------------------------------------


def test_unknown_capability_fails_closed() -> None:
    assert get_reasoning_capabilities("some-unknown-provider") is None


def test_declared_capabilities_present() -> None:
    for provider in ("openai", "anthropic", "glm", "kimi", "deepseek", "qwen", "nvidia"):
        caps = get_reasoning_capabilities(provider)
        assert caps is not None, provider


def test_effort_map_null_means_unsupported() -> None:
    # OpenAI maps effort levels; the chat-family providers and Anthropic
    # declare None (unsupported) until a verified mapping exists.
    assert DEFAULT_REASONING_CAPABILITIES["openai"].effort_map["medium"] == "medium"
    for provider in ("glm", "kimi", "deepseek", "qwen", "nvidia", "anthropic"):
        assert provider in DEFAULT_REASONING_CAPABILITIES
        assert DEFAULT_REASONING_CAPABILITIES[provider].effort_map["medium"] is None


def test_register_reasoning_capabilities_overrides_catalog() -> None:
    custom = ReasoningCapabilities(
        capture_kinds=frozenset({ReasoningKind.TEXT}),
        native_replay_kinds=frozenset(),
        effort_map={"medium": "banana"},
        replay_field="thoughts",
    )
    register_reasoning_capabilities("bananaai", custom)
    try:
        assert get_reasoning_capabilities("bananaai") is custom
        assert get_reasoning_capabilities("BANANAai") is custom  # case-insensitive
        # Mixed-case registrations stay reachable (stored lowercased).
        assert get_reasoning_capabilities("BananaAI") is custom
    finally:
        # Restore the catalog so other tests are unaffected.
        from nooa.unifiedllm import contracts as c

        c._REASONING_CAPABILITIES.clear()
        c._REASONING_CAPABILITIES.update(dict(c.DEFAULT_REASONING_CAPABILITIES))


# --- No behavior change / additive surface --------------------------------------


def test_contracts_module_has_no_provider_sdk_imports() -> None:
    import nooa.unifiedllm.contracts as contracts

    source = Path(contracts.__file__).read_text()
    assert "import litellm" not in source
    assert "from litellm" not in source
    assert "from openai" not in source
    assert "from anthropic" not in source
