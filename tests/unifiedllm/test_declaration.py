# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for config-driven provider declarations at the registry edge.

An alias can declare ``provider`` / ``compat_group`` (and optional
``capabilities``) in ``llm_config.yaml``. These tests pin the lazy
registration flow, the fail-closed paths, and that declaration is metadata
only — the request path and undeclared aliases are untouched.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest

from nooa.unifiedllm import (
    CompletionClient,
    ResponsesClient,
    get_llm_client,
    reload_registry,
)
from nooa.unifiedllm import contracts as contracts
from nooa.unifiedllm.declaration import apply_alias_declaration

REPO = Path(__file__).resolve().parent.parent.parent
SCRIPT = REPO / "scripts/refresh_model_id_corpora.py"
FIXTURE = REPO / "tests/unifiedllm/fixtures/model_id_corpora.json"


@pytest.fixture()
def scratch_config(tmp_path, monkeypatch) -> Path:
    """Isolated registry pointing at one YAML file; restored afterwards.

    User/project dirs point at an empty temp dir (the real user's
    ``~/.config/nooa/llm_config.yaml`` must not leak in) and bundled-default
    entry-points are stubbed empty, mirroring test_model_registry.py.
    """
    user = tmp_path / "user"
    user.mkdir()
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user))
    monkeypatch.setenv("NEMO_OO_PROJECT_DIR", str(tmp_path / "proj"))
    monkeypatch.delenv("NEMO_OO_LLM_CONFIG", raising=False)
    monkeypatch.setattr("nooa.llm_config.bundled_config_paths", lambda: [])
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "llm_config.yaml"
    cfg.write_text("")
    yield cfg
    reload_registry()  # reset the registry for the next test


def write_models(cfg: Path, body: str) -> None:
    cfg.write_text(textwrap.dedent(body))


@pytest.fixture(autouse=True)
def _restore_contract_registries() -> Iterator[None]:
    """Declarations register into process-lifetime catalogs; restore them."""
    yield
    contracts._COMPAT_GROUPS.clear()
    contracts._COMPAT_GROUPS.update({g.name: g for g in contracts._DEFAULT_COMPAT_GROUPS})
    contracts._REASONING_CAPABILITIES.clear()
    contracts._REASONING_CAPABILITIES.update(dict(contracts.DEFAULT_REASONING_CAPABILITIES))
    from nooa.unifiedllm import declaration as declaration

    declaration._applied.clear()


# --- ModelConfig typed fields ----------------------------------------------


class TestModelConfigDeclarationFields:
    def test_provider_and_compat_group_are_typed_fields(self):
        from nooa.config import ModelConfig

        mc = ModelConfig.from_registry(
            "opaque-enterprise", {"provider": "openai", "compat_group": "openai-gpt-5"}
        )
        assert mc.provider == "openai"
        assert mc.compat_group == "openai-gpt-5"

    def test_declaration_fields_default_to_none(self):
        from nooa.config import ModelConfig

        mc = ModelConfig.from_registry("plain", {"model_name": "gpt-5.6"})
        assert mc.provider is None
        assert mc.compat_group is None

    def test_extra_allow_keeps_litellm_passthrough(self):
        from nooa.config import ModelConfig

        mc = ModelConfig.from_registry(
            "a", {"provider": "openai", "num_retries": 7, "extra_body": {"trace": True}}
        )
        assert mc.num_retries == 7  # type: ignore[attr-defined]
        assert mc.extra_body == {"trace": True}  # type: ignore[attr-defined]

    def test_non_string_declaration_rejected(self):
        from pydantic import ValidationError

        from nooa.config import ModelConfig

        with pytest.raises(ValidationError):
            ModelConfig.from_registry("bad", {"provider": 42})


# --- Direct declaration behavior -------------------------------------------


class TestApplyAliasDeclaration:
    def test_no_declaration_returns_none(self):
        assert (
            apply_alias_declaration(
                "some-unknown-model-xyz", {}, api_style="responses", transport="litellm"
            )
            is None
        )

    def test_declared_group_gives_opaque_id_a_key(self):
        identity = apply_alias_declaration(
            "enterprise-internal",
            {"provider": "openai", "compat_group": "openai-gpt-5"},
            api_style="responses",
            transport="litellm",
        )
        assert identity is not None
        assert identity.provider == "openai"
        assert identity.model == "enterprise-internal"
        # Key matches the one derived straight from the declared group.
        expected = contracts.derive_opaque_replay_key(
            provider="openai",
            api_style="responses",
            model="enterprise-internal",
            compat_groups={
                "openai-gpt-5": contracts.ModelCompatGroup(
                    name="openai-gpt-5",
                    provider="openai",
                    models=frozenset({"enterprise-internal"}),
                )
            },
        )
        assert identity.opaque_replay_key == expected
        assert identity.opaque_replay_key is not None

    def test_group_only_declaration_uses_parsed_provider(self):
        # No `provider` declared: the model string itself must resolve one,
        # which is what makes a compat_group-only declaration non-speculative.
        identity = apply_alias_declaration(
            "alias",
            {"model_name": "gpt-5.6-sol", "compat_group": "openai-gpt-5"},
            api_style="responses",
            transport="litellm",
        )
        assert identity is not None
        assert identity.provider == "openai"
        assert identity.model == "gpt-5.6-sol"

    def test_group_only_with_unresolvable_model_raises(self):
        with pytest.raises(ValueError, match="without a provider"):
            apply_alias_declaration(
                "enterprise-internal",
                {"compat_group": "my-group"},
                api_style="responses",
                transport="litellm",
            )

    def test_declared_provider_must_not_contradict_the_model_string(self):
        # The model string resolves to a provider the declaration disagrees
        # with; trusting the declaration would route replay artifacts across
        # providers, so it is rejected instead.
        with pytest.raises(ValueError, match="fix the declaration"):
            apply_alias_declaration(
                "alias",
                {
                    "model_name": "claude-opus-4-5",
                    "provider": "openai",
                    "compat_group": "openai-gpt-5",
                },
                api_style="responses",
                transport="litellm",
            )

    def test_alias_provider_is_canonicalized_like_the_key_derivation(self):
        identity = apply_alias_declaration(
            "a",
            {"provider": "ZAI", "compat_group": "glm-family"},
            api_style="responses",
            transport="litellm",
        )
        assert identity is not None
        assert identity.provider == "glm"
        # The group must have been registered under the canonical provider so
        # the case-insensitive catalog lookup in derive_opaque_replay_key
        # agrees about the scope.
        group = contracts.compat_group_for("glm", "a")
        assert group is not None and group.name == "glm-family"

    def test_registration_is_process_lifetime_and_declared_group_overrides_catalog(self):
        # "gpt-5.6" is in the built-in openai-gpt-5 group; the declaration
        # routes it to its own group, and the derived key reflects that.
        identity = apply_alias_declaration(
            "alias",
            {"model_name": "gpt-5.6", "provider": "openai", "compat_group": "my-gpt5"},
            api_style="responses",
            transport="litellm",
        )
        assert identity is not None
        assert identity.opaque_replay_key is not None
        group = contracts.compat_group_for("openai", "gpt-5.6")
        assert group is not None
        # Both the built-in and the declared group now claim the model;
        # the declaration wins for THIS alias, and catalog lookups resolve
        # deterministically (alphabetically first).
        assert group.name == "my-gpt5"

    def test_existing_group_members_are_extended_not_replaced(self):
        contracts.register_compat_group(
            contracts.ModelCompatGroup(
                name="verified-group", provider="openai", models=frozenset({"gpt-5.6"})
            )
        )
        apply_alias_declaration(
            "alias",
            {
                "model_name": "opaque-enterprise",
                "provider": "openai",
                "compat_group": "verified-group",
            },
            api_style="responses",
            transport="litellm",
        )
        group = contracts.compat_group_for("openai", "opaque-enterprise")
        assert group is not None
        assert {"gpt-5.6", "opaque-enterprise"} <= group.models

    def test_group_name_owned_by_other_provider_is_rejected(self):
        with pytest.raises(ValueError, match="already declared for provider"):
            apply_alias_declaration(
                "alias",
                {
                    "model_name": "claude-sonnet-4-5",
                    "provider": "anthropic",
                    "compat_group": "openai-gpt-5",
                },
                api_style="responses",
                transport="litellm",
            )

    def test_reapplication_with_same_declaration_is_a_noop(self):
        cfg = {"provider": "openai", "compat_group": "gpt-family"}
        first = apply_alias_declaration("a", cfg, api_style="responses", transport="litellm")
        second = apply_alias_declaration("a", cfg, api_style="responses", transport="litellm")
        assert first == second
        from nooa.unifiedllm import declaration as declaration

        assert declaration._applied["a"][1:] == ("openai", "gpt-family", None)

    def test_changed_declaration_reapplies_and_overrides(self):
        apply_alias_declaration(
            "a",
            {"provider": "openai", "compat_group": "group-one"},
            api_style="responses",
            transport="litellm",
        )
        # The registry was reloaded with different values for the same alias.
        identity = apply_alias_declaration(
            "a",
            {"provider": "openai", "compat_group": "group-two"},
            api_style="responses",
            transport="litellm",
        )
        assert identity is not None
        # The new declaration is registered and is authoritative for THIS
        # alias's key derivation. (The earlier group-one registration is
        # process-lifetime too, so the catalog now holds both; a fresh
        # process resolves only group-two.)
        assert "group-two" in contracts._COMPAT_GROUPS
        assert "a" in {m.lower() for m in contracts._COMPAT_GROUPS["group-two"].models}
        expected = contracts.derive_opaque_replay_key(
            provider="openai",
            api_style="responses",
            model="a",
            compat_groups={
                "group-two": contracts.ModelCompatGroup(
                    name="group-two", provider="openai", models=frozenset({"a"})
                )
            },
        )
        assert identity.opaque_replay_key == expected

    def test_capabilities_override_registered(self):
        caps = {
            "capture_kinds": ["text"],
            "native_replay_kinds": [],
            "effort_map": {"medium": "banana"},
            "replay_field": "thoughts",
        }
        apply_alias_declaration(
            "alias",
            {"provider": "openai", "compat_group": "openai-gpt-5", "capabilities": caps},
            api_style="responses",
            transport="litellm",
        )
        got = contracts.get_reasoning_capabilities("openai")
        assert got is not None
        assert "banana" in got.capture_kinds or got.effort_map["medium"] == "banana"

    def test_invalid_capabilities_dropped_with_warning(self, caplog):
        with caplog.at_level("WARNING"):
            apply_alias_declaration(
                "alias",
                {
                    "provider": "openai",
                    "compat_group": "openai-gpt-5",
                    "capabilities": {"bogus": True},
                },
                api_style="responses",
                transport="litellm",
            )
        assert any("capabilities" in r.message for r in caplog.records)
        # Catalog answer unchanged.
        assert (
            contracts.get_reasoning_capabilities("openai")
            == contracts.DEFAULT_REASONING_CAPABILITIES["openai"]
        )

    def test_none_declaration_fields_mean_undeclared(self):
        identity = apply_alias_declaration(
            "a",
            {"provider": None, "compat_group": None},
            api_style="responses",
            transport="litellm",
        )
        assert identity is None

    def test_capabilities_only_registers_override_but_no_identity(self):
        caps = {
            "capture_kinds": ["text"],
            "native_replay_kinds": [],
            "effort_map": {"medium": None},
        }
        identity = apply_alias_declaration(
            "a",
            {"model_name": "gpt-5.6-sol", "capabilities": caps},
            api_style="responses",
            transport="litellm",
        )
        # No provider/compat_group declared: the override applies, but the
        # alias has no identity declaration, so none is attached.
        assert identity is None
        # The declared profile replaces the catalog's wholesale.
        got = contracts.get_reasoning_capabilities("openai")
        assert got is not None
        assert got.effort_map == {"medium": None}
        assert "text" in got.capture_kinds

    def test_capabilities_only_without_resolvable_provider_is_dropped(self, caplog):
        with caplog.at_level("WARNING"):
            identity = apply_alias_declaration(
                "opaque-enterprise",
                {"capabilities": {"capture_kinds": ["text"], "effort_map": {}}},
                api_style="responses",
                transport="litellm",
            )
        assert identity is None
        assert any("without a resolvable provider" in r.message for r in caplog.records)


# --- Registry end-to-end (get_llm_client) ----------------------------------


class TestRegistryDeclaration:
    def test_opaque_alias_derives_group_key(self, scratch_config):
        write_models(
            scratch_config,
            """\
            models:
              enterprise-gpt:
                model_name: enterprise-internal-b9f2
                api_base: https://gw.example.com/v1
                api_key_env: MY_GATEWAY_KEY
                provider: openai
                compat_group: openai-gpt-5
            """,
        )
        reload_registry(scratch_config)
        client = get_llm_client("enterprise-gpt")
        assert client.model == "enterprise-internal-b9f2"
        identity = client.provider_identity
        assert identity is not None
        assert identity.provider == "openai"
        assert identity.model == "enterprise-internal-b9f2"
        assert identity.transport == "litellm"
        # Derived from the DECLARED provider+group, not from the opaque id.
        assert identity.opaque_replay_key is not None

    def test_same_alias_without_declaration_has_no_identity(self, scratch_config):
        write_models(
            scratch_config,
            """\
            models:
              enterprise-gpt:
                model_name: enterprise-internal-b9f2
            """,
        )
        reload_registry(scratch_config)
        client = get_llm_client("enterprise-gpt")
        assert client.provider_identity is None
        # And no speculative group was registered for the opaque id.
        assert contracts.compat_group_for("openai", "enterprise-internal-b9f2") is None

    def test_client_type_declares_api_style(self, scratch_config):
        write_models(
            scratch_config,
            """\
            models:
              enterprise-gpt:
                model_name: enterprise-internal-b9f2
                client_type: responses
                provider: openai
                compat_group: openai-gpt-5
            """,
        )
        reload_registry(scratch_config)
        client = get_llm_client("enterprise-gpt")
        assert isinstance(client, ResponsesClient)
        assert client.provider_identity is not None
        assert client.provider_identity.api_style == "responses"

    def test_declared_group_registered_for_process_lifetime(self, scratch_config):
        write_models(
            scratch_config,
            """\
            models:
              enterprise-gpt:
                model_name: enterprise-internal-b9f2
                provider: openai
                compat_group: openai-gpt-5
            """,
        )
        reload_registry(scratch_config)
        get_llm_client("enterprise-gpt")
        # The declaration registered the opaque id into the named group, so
        # even catalog lookups outside this alias now resolve it.
        group = contracts.compat_group_for("openai", "enterprise-internal-b9f2")
        assert group is not None
        assert group.name == "openai-gpt-5"

    def test_declaration_is_metadata_only(self, scratch_config, monkeypatch):
        """Identity is attached as an attribute, never as a request param."""
        captured: dict = {}

        class RecordingClient(CompletionClient):
            def __init__(self, **kwargs):
                captured.update(kwargs)
                super().__init__(**kwargs)

        # get_llm_client imports the client classes from the package at call
        # time, so patching the package attribute is enough.
        monkeypatch.setattr("nooa.unifiedllm.ResponsesClient", RecordingClient)
        write_models(
            scratch_config,
            """\
            models:
              enterprise-gpt:
                model_name: enterprise-internal-b9f2
                client_type: responses
                provider: openai
                compat_group: openai-gpt-5
            """,
        )
        reload_registry(scratch_config)
        client = get_llm_client("enterprise-gpt")
        assert isinstance(client, RecordingClient)
        assert client.provider_identity is not None
        assert "provider_identity" not in captured
        assert "provider" not in captured
        assert "compat_group" not in captured

    def test_undeclared_aliases_keep_exact_previous_config(self, scratch_config):
        write_models(
            scratch_config,
            """\
            models:
              my-alias:
                model_name: openai/my-org/my-model
                temperature: 0.3
            """,
        )
        reload_registry(scratch_config)
        client = get_llm_client("my-alias")
        assert client.provider_identity is None
        assert client.model == "openai/my-org/my-model"
        assert client.config["temperature"] == 0.3

    def test_invalid_declaration_fails_loudly_at_client_construction(self, scratch_config):
        """A contradictory declaration is a config error, not a silent no-op."""
        write_models(
            scratch_config,
            """\
            models:
              wrong:
                model_name: claude-opus-4-5
                provider: openai
                compat_group: openai-gpt-5
            """,
        )
        reload_registry(scratch_config)
        with pytest.raises(ValueError, match="fix the declaration"):
            get_llm_client("wrong")

    def test_non_string_declaration_rejected_at_client_construction(self, scratch_config):
        write_models(
            scratch_config,
            """\
            models:
              wrong:
                model_name: enterprise-internal
                provider: 42
            """,
        )
        reload_registry(scratch_config)
        with pytest.raises(ValueError, match="expected a string"):
            get_llm_client("wrong")


# --- Refresh script --check mode -------------------------------------------


@pytest.fixture(scope="module")
def script():
    """Load refresh_model_id_corpora.py, which lives in scripts/ (not importable)."""
    spec = importlib.util.spec_from_file_location("_refresh_model_id_corpora", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["_refresh_model_id_corpora"] = module
    spec.loader.exec_module(module)
    return module


class TestCorpusDriftLogic:
    def test_openrouter_entries_from_synthetic_payload(self, script):
        payload = {"data": [{"id": "openai/gpt-6-astra"}, {"id": "~z-ai/glm-latest"}]}
        assert script.openrouter_entries(payload) == [
            {"id": "openai/gpt-6-astra", "vendor": "openai"},
            {"id": "~z-ai/glm-latest", "vendor": "z-ai"},
        ]

    def test_nvidia_gateway_entries_from_synthetic_payload(self, script):
        payload = {
            "data": [
                {"id": "nvidia/zai-org/glm-5.3"},
                {"id": "gcp/google/gemini-omni-flash-preview"},
                {"id": "nvidia/nvidia/cosmos3-nano-reasoner"},
            ]
        }
        assert script.nvidia_gateway_entries(payload) == [
            {"id": "nvidia/zai-org/glm-5.3", "vendor": "zai-org"},
            {"id": "gcp/google/gemini-omni-flash-preview", "vendor": None},
            {"id": "nvidia/nvidia/cosmos3-nano-reasoner", "vendor": "nvidia"},
        ]

    def test_offline_fetch_failure_exits_nonzero_without_touching_fixture(
        self, script, monkeypatch, capsys
    ):
        """No network → non-zero exit, fixture untouched, clear message."""
        before = FIXTURE.read_text()
        monkeypatch.setattr(
            script.urllib.request,
            "urlopen",
            lambda *a, **k: (_ for _ in ()).throw(OSError("no network")),
        )
        assert script.main([]) == 1
        err = capsys.readouterr().err
        assert "needs network access" in err
        assert FIXTURE.read_text() == before

    def test_check_mode_reports_drift_without_writing(self, script, monkeypatch, tmp_path, capsys):
        """Drifted catalog + --check → exit 1, fixture untouched."""
        before = FIXTURE.read_text()
        fixture_copy = tmp_path / "model_id_corpora.json"
        fixture_copy.write_text(before)
        monkeypatch.setattr(script, "FIXTURE", fixture_copy)
        live = {"data": [{"id": "openai/gpt-6-astra"}]}  # one model: not the corpus
        monkeypatch.setattr(script, "fetch_catalog", lambda *a, **k: live)
        monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
        assert script.main(["--check"]) == 1
        assert "Drift detected" in capsys.readouterr().err
        assert fixture_copy.read_text() == before

    def test_check_mode_green_when_no_drift(self, script, monkeypatch, tmp_path, capsys):
        """--check against the current fixture's own openrouter section → 0."""
        import json

        current = json.loads(FIXTURE.read_text())
        fixture_copy = tmp_path / "model_id_corpora.json"
        fixture_copy.write_text(json.dumps(current, indent=1) + "\n")
        monkeypatch.setattr(script, "FIXTURE", fixture_copy)
        payload = {"data": [{"id": e["id"]} for e in current["openrouter"]]}
        monkeypatch.setattr(script, "fetch_catalog", lambda *a, **k: payload)
        monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
        assert script.main(["--check"]) == 0
        out = capsys.readouterr().out
        assert "No drift" in out
        # _comment is intentionally excluded from drift: it carries the fetch
        # date, which changes daily.
        assert fixture_copy.read_text() == json.dumps(current, indent=1) + "\n"

    def test_refresh_mode_rewrites_live_sections_only(self, script, monkeypatch, tmp_path):
        import json

        current = json.loads(FIXTURE.read_text())
        fixture_copy = tmp_path / "model_id_corpora.json"
        fixture_copy.write_text(FIXTURE.read_text())
        monkeypatch.setattr(script, "FIXTURE", fixture_copy)
        payload = {"data": [{"id": "openai/gpt-6-astra"}, {"id": "z-ai/glm-5.3"}]}
        monkeypatch.setattr(script, "fetch_catalog", lambda *a, **k: payload)
        monkeypatch.delenv("NVIDIA_INFERENCE_API_KEY", raising=False)
        assert script.main([]) == 0
        fresh = json.loads(fixture_copy.read_text())
        assert [e["id"] for e in fresh["openrouter"]] == ["openai/gpt-6-astra", "z-ai/glm-5.3"]
        # Non-live sections and the un-refreshed nvidia section are preserved.
        assert fresh["azure"] == current["azure"]
        assert fresh["vertex_ai"] == current["vertex_ai"]
        assert fresh["bedrock"] == current["bedrock"]
        assert fresh["nvidia_gateway"] == current["nvidia_gateway"]
