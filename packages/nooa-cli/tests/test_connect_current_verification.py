# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Frontends share exact current-candidate trust, not result booleans."""

import json
from copy import deepcopy

import pytest
from click.testing import CliRunner
from nooa_cli.commands import _connect_prompts as prompts
from nooa_cli.commands import _connect_wizard as wizard
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.test_connect_current_verification import MUTATIONS
from tests.unifiedllm.connect.test_connect_current_verification import fresh_media as _fresh_media

# Reuse the real below-SDK execution fixture, not synthetic frontend proof.
fresh_media = _fresh_media


@pytest.mark.parametrize("mutation", MUTATIONS, ids=lambda m: "+".join(m))
async def test_wizard_current_candidate_cannot_borrow_test_result(
    fresh_media, mutation, monkeypatch, tmp_path
):
    result, sent = fresh_media
    state = wizard.WizardState(
        approval="all",
        yes=True,
        path=tmp_path / "models.yaml",
        result=result,
        configured=deepcopy(result.entry),
    )
    assert wizard.baseline_allows_save(state)
    state.configured.update(mutation)
    # The result remains a genuinely passed old check, but it's not the candidate.
    assert result.entry["provenance"]["probes"]["routing"]["verification_current"] is True
    assert not wizard.baseline_allows_save(state)
    monkeypatch.setattr(
        prompts, "confirm", lambda *a, **k: pytest.fail("No save prompt for stale proof")
    )
    monkeypatch.setattr(connect, "write", lambda *a, **k: pytest.fail("No stale registry write"))
    assert not wizard.save_model(state)
    assert len(sent) == 3 and not state.path.exists()


async def test_wizard_result_edit_and_permission_only_baseline_policy(fresh_media):
    result, _ = fresh_media
    state = wizard.WizardState(approval="all", result=result, configured=deepcopy(result.entry))
    result.entry["media_inputs"] = ["image", "audio"]
    assert wizard.baseline_allows_save(state)
    assert "media:audio" not in result.entry["provenance"]["probes"]
    result.entry["temperature"] = 0.5
    assert not wizard.baseline_allows_save(state)
    state.no_probe = True  # Explicit manual/unverified mode, not a fresh pass.
    assert wizard.baseline_allows_save(state)
    assert not connect.routing_allows_save(result.entry, result.entry)


@pytest.mark.parametrize(
    "options,current",
    [
        ([], True),
        (["--media-inputs", "image,audio"], True),
        (["--reply-tokens", "2000"], False),
        (["--context-window", "64000"], False),
        (["--no-cache"], False),
        (["--structured-output-mode", "anthropic"], False),
    ],
)
async def test_stage_save_shared_gate_compares_tested_with_configured_current_candidate(
    fresh_media, monkeypatch, tmp_path, options, current
):
    result, sent = fresh_media
    source = tmp_path / "report.json"
    source.write_text(
        json.dumps(
            {
                "version": 1,
                "stage": "media",
                "ok": True,
                "data": {"alias": "local", "entry": result.entry},
            }
        )
    )
    target = tmp_path / "models.yaml"
    observed = []
    gate = connect.routing_allows_save

    def capture(tested, candidate):
        answer = gate(tested, candidate)
        observed.append((deepcopy(tested), deepcopy(candidate), answer))
        return answer

    monkeypatch.setattr(connect, "routing_allows_save", capture)
    response = CliRunner().invoke(
        command,
        ["--stage", "save", "--input", str(source), "--output", str(target), "--yes", *options],
    )
    assert response.exit_code == 0, response.output
    assert observed and observed[-1][2] is current
    tested, candidate, _ = observed[-1]
    assert tested["provenance"]["probes"]["routing"]["verification_current"] is True
    assert candidate["provenance"]["probes"]["routing"]["verification_current"] is current
    # Save stage still permits deliberate unverified imports with a warning;
    # success of the write is not advertised as success of the current checks.
    report = json.loads(response.stdout)
    record = report["data"]["entry"]["provenance"]["probes"]["routing"]
    assert record["verification_current"] is current
    assert ("current settings have not passed checks" in response.stderr) is (not current)
    assert target.exists() and len(sent) == 3


def test_stage_historical_good_record_is_unverified_but_manual_save_is_preserved(tmp_path):
    entry = connect.configure_entry(
        {
            "model_name": "openai/model",
            "provenance": {
                "probes": {
                    "routing": {
                        "outcome": "accepted",
                        "baseline_kind": "predict_int_v1",
                        "answer_correct": True,
                        "typed_result_validated": True,
                        "final_answer_observed": True,
                        "settings_sent": True,
                        "finish_reason": "stop",
                    }
                }
            },
        }
    )
    source = tmp_path / "legacy.json"
    source.write_text(json.dumps({"ok": True, "alias": "local", "entry": entry}))
    target = tmp_path / "models.yaml"
    response = CliRunner().invoke(
        command, ["--stage", "save", "--input", str(source), "--output", str(target), "--yes"]
    )
    assert response.exit_code == 0, response.output
    report = json.loads(response.stdout)
    candidate = report["data"]["entry"]
    assert "current settings have not passed checks" in response.stderr
    assert not connect.routing_allows_save(candidate, candidate)
    assert "routing" not in connect.verdict(candidate).passed
    assert "verification_config" not in candidate["provenance"]["probes"]["routing"]
