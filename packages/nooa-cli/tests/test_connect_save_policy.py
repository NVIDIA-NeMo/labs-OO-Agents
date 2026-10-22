# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save confidence, consent races and synthetic two-file failure semantics."""

import json
from copy import deepcopy

import click
import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa import secrets
from nooa.unifiedllm import connect, registry
from nooa.unifiedllm.connect._verification import (
    effective_config_snapshot,
    save_verification,
    stamp_verification,
)
from tests.unifiedllm.connect.test_connect_current_verification import fresh_media as _fresh_media

fresh_media = _fresh_media


def candidate():
    return connect.configure_entry(
        {
            "model_name": "openai/synthetic",
            "api_base": "https://api.test/v1",
            "api_key_env": "SYNTHETIC_SAVE_KEY",
            "api_style": "chat",
            "media_inputs": [],
        }
    )


def proof(entry, outcome="accepted"):
    record = {
        "outcome": outcome,
        "baseline_kind": "predict_int_v1",
        "typed_result_validated": True,
        "answer_correct": True,
        "final_answer_observed": True,
        "settings_sent": True,
        "finish_reason": "stop",
    }
    stamp_verification(record, entry, "routing")
    entry.setdefault("provenance", {}).setdefault("probes", {})["routing"] = record
    return entry


def stage_save(tmp_path, entry, *, ok=True, flags=(), checks=None, diagnostic=None):
    source, target = tmp_path / "report.json", tmp_path / "models.yaml"
    source.write_text(
        json.dumps(
            {
                "ok": ok,
                "data": {"alias": "local", "entry": entry},
                "checks": checks or {},
                "diagnostic_prompt": diagnostic,
            }
        )
    )
    result = CliRunner().invoke(
        command,
        [
            "--stage",
            "save",
            "--input",
            str(source),
            "--output",
            str(target),
            *flags,
        ],
    )
    return result, json.loads(result.stdout), target


@pytest.mark.parametrize("ack", [False, True])
def test_failed_stage_save_requires_separate_ack_and_retains_machine_handoff(tmp_path, ack):
    entry = proof(candidate(), "rejected")
    result, report, target = stage_save(
        tmp_path,
        entry,
        ok=False,
        diagnostic="Synthetic routing failed; review the URL",
        flags=["--yes", *(["--allow-unverified"] if ack else [])],
    )
    assert result.exit_code == (0 if ack else 2), result.output
    assert report["ok"] is ack and report["saved"] is ack and target.exists() is ack
    assert report["save_status"] == ("saved" if ack else "not_saved")
    assert report["verification"]["state"] == "failed"
    assert report["verification"]["verified"] is False
    assert report["checks"]["routing"]["outcome"] == "rejected"
    assert report["checks"]["routing"]["verification_current"] is True
    assert report["diagnostic_prompt"] == "Synthetic routing failed; review the URL"
    assert "verification_config" not in json.dumps(report["checks"])
    assert "verification_config" not in json.dumps(report["verification"])
    assert set(report) == {
        "version",
        "stage",
        "ok",
        "data",
        "error",
        "run_context",
        "warnings",
        "checks",
        "diagnostic_prompt",
        "saved",
        "save_status",
        "verification",
    }
    if not ack:
        assert report["error"]["type"] == "UsageError"
        assert "--allow-unverified" in report["error"]["message"]


def test_manual_save_has_no_verification_claim(tmp_path):
    result, report, _ = stage_save(tmp_path, candidate())
    assert result.exit_code == 0 and report["saved"] and report["ok"]
    assert report["verification"]["state"] == "untested"
    assert report["verification"]["verified"] is False
    assert report["verification"]["baseline_verified"] is False
    assert report["checks"] == {}


@pytest.mark.parametrize(
    "mutation",
    [
        {"temperature": 0.7},
        {"max_tokens": 777},
        {"api_base": "https://new.test/v1"},
        {"reasoning_levels": {"high": {"temperature": 0.1}}, "reasoning_level": "high"},
    ],
)
def test_mutated_proof_is_stale_not_passed(tmp_path, mutation):
    entry = proof(candidate())
    entry.update(mutation)
    result, report, _ = stage_save(tmp_path, entry)
    assert result.exit_code == 0, result.output
    assert report["verification"]["state"] == "untested"
    assert report["verification"]["verified"] is False
    assert report["checks"]["routing"]["verification_current"] is False


@pytest.mark.parametrize("optional", ["failed", "not_probed", "stale", "declared"])
def test_baseline_with_optional_gaps_is_partial(tmp_path, optional):
    entry = proof(candidate())
    if optional == "declared":
        entry["media_inputs"] = ["image"]
    else:
        record = {"outcome": "accepted" if optional == "stale" else optional}
        if optional != "stale":
            stamp_verification(record, entry, "tools")
        entry["provenance"]["probes"]["tools"] = record
    result, report, _ = stage_save(tmp_path, entry)
    assert result.exit_code == 0, result.output
    confidence = report["verification"]
    assert confidence["state"] == "partial" and confidence["baseline_verified"]
    assert confidence["verified"] is False
    assert confidence["verified_scopes"] == ["routing"]
    assert confidence["unverified_scopes"] == (
        ["media:image"] if optional == "declared" else ["tools"]
    )
    assert report["diagnostic_prompt"]


def test_current_baseline_save_and_exact_selected_path_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(registry, "reload_registry", lambda *a: pytest.fail("No global reload"))
    entry = proof(candidate())
    result, report, target = stage_save(tmp_path, entry)
    assert result.exit_code == 0, result.output
    assert report["verification"]["state"] == "verified"
    assert report["verification"]["verified"] is True
    loaded = registry.load_model_entries(target)["local"]
    assert effective_config_snapshot(loaded) == effective_config_snapshot(entry)
    assert "provenance" not in loaded
    assert save_verification(loaded)[0]["state"] == "untested"


async def test_actual_predict_verified_candidate_preserves_reload_identity(fresh_media, tmp_path):
    result, sent = fresh_media
    response, report, target = stage_save(tmp_path, result.entry)
    assert response.exit_code == 0, response.output
    assert report["verification"]["baseline_verified"]
    loaded = registry.load_model_entries(target)["local"]
    assert effective_config_snapshot(loaded) == effective_config_snapshot(result.entry)
    for field in ("max_tokens", "reasoning_levels", "media_inputs", "api_style", "transport"):
        assert loaded.get(field) == result.entry.get(field)
    assert len(sent) == 3  # Save/reload added no provider calls.


@pytest.mark.parametrize(
    "old,new",
    [
        (connect.ALIAS_MISSING, None),
        (None, {"model_name": "openai/other"}),
        ({"model_name": "openai/old"}, {"model_name": "openai/other"}),
        ({"model_name": "openai/old"}, connect.ALIAS_MISSING),
    ],
)
def test_locked_alias_cas_rejects_unreviewed_changes(tmp_path, old, new):
    path = tmp_path / "models.yaml"
    models = {} if new is connect.ALIAS_MISSING else {"local": new}
    path.write_text(yaml.safe_dump({"models": models}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Alias changed since review"):
        connect.write(candidate(), path, alias="local", expected_alias=old)
    assert path.read_bytes() == before
    assert path.with_name(".models.yaml.lock").exists()


@pytest.mark.parametrize("target", [connect.ALIAS_MISSING, None, {"model_name": "openai/old"}])
def test_cas_allows_reviewed_target_and_preserves_unrelated_updates(tmp_path, target):
    path = tmp_path / "models.yaml"
    models = {} if target is connect.ALIAS_MISSING else {"local": deepcopy(target)}
    expected = connect.alias_target(models, "local")
    models["unrelated"] = {"model_name": "openai/race"}
    path.write_text("# preserve comment\n" + yaml.safe_dump({"models": models}))
    connect.write(candidate(), path, alias="local", expected_alias=expected)
    assert registry.load_model_entries(path)["unrelated"] == models["unrelated"]
    assert "# preserve comment" in path.read_text()


@pytest.mark.parametrize("changed_alias", ["local", "unrelated"])
def test_stage_yes_is_cas_not_blanket_replacement_consent(tmp_path, monkeypatch, changed_alias):
    real_write = connect.write
    path = tmp_path / "models.yaml"
    path.write_text(yaml.safe_dump({"models": {"local": {"model_name": "openai/old"}}}))

    def race(entry, path, **kwargs):
        assert "expected_alias" in kwargs
        data = yaml.safe_load(path.read_text())
        data["models"][changed_alias] = {"model_name": "openai/race"}
        path.write_text(yaml.safe_dump(data))
        real_write(entry, path, **kwargs)

    monkeypatch.setattr(connect, "write", race)
    result, report, _ = stage_save(tmp_path, candidate(), flags=["--yes"])
    assert report["saved"] is (changed_alias == "unrelated")
    assert result.exit_code == (0 if changed_alias == "unrelated" else 1)
    assert registry.load_model_entries(path)[changed_alias] == {"model_name": "openai/race"}


def wizard_state(tmp_path, monkeypatch, *, key=False):
    from nooa_cli.commands import _connect_registry

    monkeypatch.setattr(_connect_registry, "shadowing_source", lambda *a, **k: None)
    monkeypatch.setattr(wizard, "shadowing_source", lambda *a, **k: None)
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(tmp_path))
    monkeypatch.delenv("SYNTHETIC_SAVE_KEY", raising=False)
    return wizard.WizardState(
        no_probe=True,
        approval="none",
        path=tmp_path / "models.yaml",
        alias="local",
        model="synthetic",
        api_style="chat",
        result=connect.ConnectResult("local", candidate()),
        api_key="synthetic-secret-canary" if key else None,
        api_key_env="SYNTHETIC_SAVE_KEY",
    )


@pytest.mark.parametrize("existing", [False, True])
def test_alias_appearing_or_changing_inside_final_confirm_is_rejected(
    tmp_path, monkeypatch, existing
):
    state = wizard_state(tmp_path, monkeypatch)
    if existing:
        state.path.write_text(yaml.safe_dump({"models": {"local": {"model_name": "openai/old"}}}))

    def confirm(label, **kw):
        if label.startswith("Write model"):
            state.path.write_text(
                yaml.safe_dump({"models": {"local": {"model_name": "openai/race"}}})
            )
        return True

    monkeypatch.setattr(prompts, "confirm", confirm)
    with pytest.raises(ValueError, match="Alias changed since review"):
        wizard.save_model(state)
    assert registry.load_model_entries(state.path)["local"] == {"model_name": "openai/race"}
    assert not (tmp_path / "secrets.yaml").exists()


@pytest.mark.parametrize("cancel", ["replace", "write", "interrupt"])
def test_decline_or_cancel_has_no_model_or_secret_mutation(tmp_path, monkeypatch, cancel):
    state = wizard_state(tmp_path, monkeypatch, key=True)
    state.path.write_text(yaml.safe_dump({"models": {"local": {"model_name": "openai/old"}}}))
    before = state.path.read_bytes()

    def confirm(label, **kw):
        if label.startswith("Replace") and cancel == "replace":
            return False
        if label.startswith("Write model"):
            if cancel == "interrupt":
                raise click.Abort()
            return False
        return True

    monkeypatch.setattr(prompts, "confirm", confirm)
    if cancel == "interrupt":
        with pytest.raises(click.Abort):
            wizard.save_model(state)
    else:
        wizard.save_model(state)
    assert state.path.read_bytes() == before
    assert not (tmp_path / "secrets.yaml").exists()


@pytest.mark.parametrize("failure", ["model", "secret", "race", "rollback"])
def test_human_key_model_failures_never_claim_success(tmp_path, monkeypatch, capsys, failure):
    state = wizard_state(tmp_path, monkeypatch, key=True)
    secret_path = tmp_path / "secrets.yaml"
    secret_path.write_text("# untouched neighbor\nenv:\n  NEIGHBOR: synthetic-neighbor\n")
    before = secret_path.read_bytes()
    monkeypatch.setattr(prompts, "confirm", lambda *a, **k: True)
    real_replace = secrets.os.replace

    def broken_model(*a, **kw):
        assert (
            yaml.safe_load(secret_path.read_text())["env"]["SYNTHETIC_SAVE_KEY"]
            == "synthetic-secret-canary"
        )
        if failure == "race":
            secret_path.write_text("env:\n  SYNTHETIC_SAVE_KEY: concurrent-synthetic-value\n")
        raise OSError("synthetic model failure")

    def broken_secret(*a, **kw):
        raise OSError("synthetic secret failure")

    if failure == "secret":
        monkeypatch.setattr(secrets, "_write_secret_env", broken_secret)
    else:
        monkeypatch.setattr(connect, "_write_entry", broken_model)
    if failure == "rollback":
        calls = 0

        def fail_second_replace(*a):
            nonlocal calls
            calls += 1
            if calls > 1:
                raise OSError("synthetic rollback rename failure")
            return real_replace(*a)

        monkeypatch.setattr(secrets.os, "replace", fail_second_replace)
    with pytest.raises((ValueError, OSError)) as exc:
        wizard.save_model(state)
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Saved model" not in output and "Saved key" not in output
    assert "synthetic-secret-canary" not in output + str(exc.value)
    assert not state.path.exists()
    if failure in {"race", "rollback"}:
        assert "partial state" in str(exc.value)
        if failure == "race":
            assert "concurrent-synthetic-value" in secret_path.read_text()
        else:
            assert "SYNTHETIC_SAVE_KEY" in secret_path.read_text()
    else:
        assert secret_path.read_bytes() == before


@pytest.mark.parametrize("original", [None, "env:\n  NEIGHBOR: synthetic\n"])
def test_secret_transaction_restores_owned_bytes_on_model_failure(tmp_path, original):
    path = tmp_path / "secrets.yaml"
    if original is not None:
        path.write_text(original)
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(ValueError, match="new key was rolled back"):
        with secrets.new_secret_env(path, "SYNTHETIC_SAVE_KEY", "synthetic-secret-canary"):
            raise OSError("synthetic model failure")
    assert (path.read_bytes() if path.exists() else None) == before
    assert path.with_name(".secrets.yaml.lock").exists()


@pytest.mark.parametrize("value", [None, "synthetic-existing"])
def test_new_secret_never_overwrites_unknown_variable(tmp_path, value):
    path = tmp_path / "secrets.yaml"
    path.write_text(yaml.safe_dump({"env": {"SYNTHETIC_SAVE_KEY": value}}))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        with secrets.new_secret_env(path, "SYNTHETIC_SAVE_KEY", "synthetic-secret-canary"):
            pytest.fail("Model write must not begin")
    assert path.read_bytes() == before


@pytest.mark.parametrize("flag", [True, False])
@pytest.mark.parametrize("listing", [None, [], ["image"], ["pdf"]])
def test_exact_server_image_advertisement_display_and_hint_share_priority(
    tmp_path, monkeypatch, capsys, flag, listing
):
    state = wizard_state(tmp_path, monkeypatch)
    state.model = "synthetic"
    state.candidate = None
    state.media_hints = {
        "id": "public",
        "input_modalities": ["image"],
        "media_source": connect.CATALOGUE_URL,
    }
    state.selected_endpoint_model = {"id": "synthetic"}
    if listing is not None:
        state.selected_endpoint_model["input_modalities"] = listing
    state.server_details = connect.ModelDetails(
        "synthetic",
        {"id": "synthetic"},
        (
            {
                "id": "synthetic",
                "schema": "litellm.model_group.info",
                "source": "https://api.test/model_group/info",
                "facts": {"supports_vision": flag},
            },
        ),
    )
    before = deepcopy(state.media_hints)
    hint = wizard.model_media_hints(state)
    advertised = connect.advertised_media_suggestions(hint)["advertised"]
    assert advertised == (listing if listing is not None else ["image"] if flag else [])
    wizard.show_model_metadata(state)
    output = " ".join(capsys.readouterr().out.split())
    assert ("Media types:" in output) is bool(advertised)
    assert output.count("Media types:") <= 1
    assert "Vision listed" not in output and "Images advertised" not in output
    assert state.media_hints == before and state.media_inputs is None
    assert "video" not in advertised and "audio" not in advertised


def test_model_post_commit_error_keeps_key_and_reports_partial_state(tmp_path, monkeypatch, capsys):
    state = wizard_state(tmp_path, monkeypatch, key=True)
    monkeypatch.setattr(prompts, "confirm", lambda *a, **k: True)
    real_write = connect.write

    def committed_then_failed(*a, **kw):
        real_write(*a, **kw)
        raise OSError("synthetic post-commit failure")

    monkeypatch.setattr(connect, "write", committed_then_failed)
    with pytest.raises(ValueError, match="partial state") as exc:
        wizard.save_model(state)
    assert registry.load_model_entries(state.path)["local"]["api_key_env"] == "SYNTHETIC_SAVE_KEY"
    assert (
        yaml.safe_load((tmp_path / "secrets.yaml").read_text())["env"]["SYNTHETIC_SAVE_KEY"]
        == "synthetic-secret-canary"
    )
    captured = capsys.readouterr()
    assert "Saved model" not in captured.out and "Saved key" not in captured.out
    assert "synthetic-secret-canary" not in captured.out + captured.err + str(exc.value)


def test_active_candidate_changed_inside_write_confirmation_is_rejected(tmp_path, monkeypatch):
    state = wizard_state(tmp_path, monkeypatch, key=True)

    def confirm(label, **kw):
        if label.startswith("Write model"):
            state.result.entry["temperature"] = 0.73
        return True

    monkeypatch.setattr(prompts, "confirm", confirm)
    with pytest.raises(ValueError, match="settings changed after review"):
        wizard.save_model(state)
    assert not state.path.exists() and not (tmp_path / "secrets.yaml").exists()


def test_acknowledged_failed_optional_report_is_partial_not_failed_baseline(tmp_path):
    entry = proof(candidate())
    record = {"outcome": "failed"}
    stamp_verification(record, entry, "tools")
    entry["provenance"]["probes"]["tools"] = record
    result, report, _ = stage_save(tmp_path, entry, ok=False, flags=["--allow-unverified"])
    assert result.exit_code == 0, result.output
    assert report["verification"]["state"] == "partial"
    assert report["verification"]["baseline_verified"] is True
    assert report["checks"]["tools"]["outcome"] == "failed"
    assert report["diagnostic_prompt"]


def test_concurrent_secret_change_immediately_after_create_is_not_owned(tmp_path, monkeypatch):
    path = tmp_path / "secrets.yaml"
    real_write = secrets._write_secret_env

    def changed_after_create(*a, **kw):
        owned = real_write(*a, **kw)
        path.write_text("env:\n  SYNTHETIC_SAVE_KEY: concurrent-synthetic-value\n")
        return owned

    monkeypatch.setattr(secrets, "_write_secret_env", changed_after_create)
    with pytest.raises(ValueError, match="partial state"):
        with secrets.new_secret_env(path, "SYNTHETIC_SAVE_KEY", "synthetic-secret-canary"):
            raise OSError("synthetic model failure")
    assert "concurrent-synthetic-value" in path.read_text()


@pytest.mark.parametrize("flag", [True, False])
def test_group_image_hint_preserves_public_pdf_source_and_no_duplicate_rows(
    tmp_path, monkeypatch, capsys, flag
):
    state = wizard_state(tmp_path, monkeypatch)
    state.media_hints = {
        "id": "synthetic",
        "input_modalities": ["pdf"],
        "media_source": connect.CATALOGUE_URL,
    }
    state.server_details = connect.ModelDetails(
        "synthetic",
        {"id": "synthetic"},
        (
            {
                "id": "synthetic",
                "schema": "litellm.model_group.info",
                "source": "https://api.test/model_group/info",
                "facts": {"supports_vision": flag},
            },
        ),
    )
    wizard.show_model_metadata(state)
    output = " ".join(capsys.readouterr().out.split())
    assert "Media types: " + ("Images, PDFs" if flag else "PDFs") in output
    assert output.count("Media types:") == 1
    hint = wizard.model_media_hints(state)
    assert connect.CATALOGUE_URL in hint["media_source"]
    assert ("https://api.test/model_group/info" in hint["media_source"]) is flag
    assert state.media_inputs is None


def test_secret_concurrent_change_during_staging_blocks_model_commit(tmp_path, monkeypatch):
    path = tmp_path / "secrets.yaml"
    path.write_text("env:\n  NEIGHBOR: original-synthetic\n")
    real_fsync = secrets.os.fsync

    def concurrent_fsync(fd):
        path.write_text("env:\n  NEIGHBOR: concurrent-synthetic\n")
        real_fsync(fd)

    monkeypatch.setattr(secrets.os, "fsync", concurrent_fsync)
    with pytest.raises(ValueError, match="changed during write"):
        with secrets.new_secret_env(path, "SYNTHETIC_SAVE_KEY", "synthetic-secret-canary"):
            pytest.fail("Model commit must not run")
    assert "concurrent-synthetic" in path.read_text()
    assert "SYNTHETIC_SAVE_KEY" not in path.read_text()


def test_concurrent_secret_change_during_successful_model_write_is_partial(
    tmp_path, monkeypatch, capsys
):
    state = wizard_state(tmp_path, monkeypatch, key=True)
    monkeypatch.setattr(prompts, "confirm", lambda *a, **k: True)
    real_write = connect.write

    def changed_after_model(*a, **kw):
        real_write(*a, **kw)
        (tmp_path / "secrets.yaml").write_text(
            "env:\n  SYNTHETIC_SAVE_KEY: concurrent-synthetic-value\n"
        )

    monkeypatch.setattr(connect, "write", changed_after_model)
    with pytest.raises(ValueError, match="partial state"):
        wizard.save_model(state)
    assert "concurrent-synthetic-value" in (tmp_path / "secrets.yaml").read_text()
    captured = capsys.readouterr()
    assert "Saved model" not in captured.out and "Saved key" not in captured.out


@pytest.mark.parametrize("outcome", ["failed", "accepted", "not_probed"])
def test_top_level_only_optional_records_are_historical_gaps_not_verified(tmp_path, outcome):
    entry = proof(candidate())
    result, report, _ = stage_save(
        tmp_path,
        entry,
        checks={"tools": {"outcome": outcome, "verification_current": True}},
    )
    assert result.exit_code == 0, result.output
    assert report["verification"]["state"] == "partial"
    assert report["verification"]["verified"] is False
    assert report["verification"]["baseline_verified"] is True
    assert report["verification"]["unverified_scopes"] == ["tools"]
    assert report["checks"]["tools"]["verification_current"] is False
    assert report["diagnostic_prompt"]


@pytest.mark.parametrize("unreadable", ["permission", "parse"])
def test_unknown_post_commit_read_keeps_key_and_reports_partial(
    tmp_path, monkeypatch, capsys, unreadable
):
    from pathlib import Path

    state = wizard_state(tmp_path, monkeypatch, key=True)
    monkeypatch.setattr(prompts, "confirm", lambda *a, **k: True)
    real_write, real_read = connect.write, Path.read_text

    def cannot_read_selected(path, *a, **kw):
        if path == state.path:
            if unreadable == "permission":
                raise PermissionError("synthetic selected file read error")
            return "models: [synthetic invalid YAML"
        return real_read(path, *a, **kw)

    def commit_then_fail(*a, **kw):
        real_write(*a, **kw)
        monkeypatch.setattr(Path, "read_text", cannot_read_selected)
        raise OSError("synthetic post-commit failure")

    monkeypatch.setattr(connect, "write", commit_then_fail)
    with pytest.raises(ValueError, match="partial state") as exc:
        wizard.save_model(state)
    assert state.path.exists()
    assert (
        yaml.safe_load(real_read(state.path))["models"]["local"]["api_key_env"]
        == "SYNTHETIC_SAVE_KEY"
    )
    assert (
        yaml.safe_load(real_read(tmp_path / "secrets.yaml"))["env"]["SYNTHETIC_SAVE_KEY"]
        == "synthetic-secret-canary"
    )
    captured = capsys.readouterr()
    assert "Saved model" not in captured.out and "Saved key" not in captured.out
    assert "synthetic-secret-canary" not in captured.out + captured.err + str(exc.value)


@pytest.mark.parametrize("move_alias", [False, True])
def test_concurrent_post_commit_cas_profile_edit_keeps_its_required_key(
    tmp_path, monkeypatch, capsys, move_alias
):
    state = wizard_state(tmp_path, monkeypatch, key=True)
    monkeypatch.setattr(prompts, "confirm", lambda *a, **k: True)
    real_write = connect.write

    def committed_then_changed(*a, **kw):
        real_write(*a, **kw)
        committed = registry.load_model_entries(state.path)["local"]
        changed = {**committed, "temperature": 0.73}
        real_write(
            changed,
            state.path,
            alias="other" if move_alias else "local",
            expected_alias=connect.ALIAS_MISSING if move_alias else committed,
        )
        if move_alias:
            # An unrelated alias may still require the newly created variable.
            real_write(
                candidate() | {"api_key_env": "DIFFERENT_SYNTHETIC_KEY"},
                state.path,
                alias="local",
                expected_alias=committed,
            )
        raise OSError("synthetic post-commit failure")

    monkeypatch.setattr(connect, "write", committed_then_changed)
    with pytest.raises(ValueError, match="partial state"):
        wizard.save_model(state)
    loaded = registry.load_model_entries(state.path)["other" if move_alias else "local"]
    assert loaded["temperature"] == 0.73 and loaded["api_key_env"] == "SYNTHETIC_SAVE_KEY"
    assert (
        yaml.safe_load((tmp_path / "secrets.yaml").read_text())["env"]["SYNTHETIC_SAVE_KEY"]
        == "synthetic-secret-canary"
    )
    captured = capsys.readouterr()
    assert "Saved model" not in captured.out and "Saved key" not in captured.out


def test_model_recovery_lock_is_held_through_secret_rollback(tmp_path, monkeypatch):
    import fcntl

    state = wizard_state(tmp_path, monkeypatch, key=True)
    path = tmp_path / "secrets.yaml"
    path.write_text("env:\n  NEIGHBOR: original-synthetic\n")
    before = path.read_bytes()
    monkeypatch.setattr(prompts, "confirm", lambda *a, **k: True)
    monkeypatch.setattr(
        connect,
        "_write_entry",
        lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic model failure")),
    )
    real_replace = secrets.os.replace
    replacements = 0

    def guarded_replace(*a):
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            with state.path.with_name(".models.yaml.lock").open("a") as other_writer:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other_writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return real_replace(*a)

    monkeypatch.setattr(secrets.os, "replace", guarded_replace)
    with pytest.raises(ValueError, match="new key was rolled back"):
        wizard.save_model(state)
    assert replacements == 2 and path.read_bytes() == before and not state.path.exists()
    # The stable sidecar is retained and released after recovery completes.
    with state.path.with_name(".models.yaml.lock").open("a") as other_writer:
        fcntl.flock(other_writer.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
