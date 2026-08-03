# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Contributed by CSOAI (csoai.org) — Council for the Safety of Artificial Intelligence.
"""Deterministic-core tests for the GSPC provision evaluator.

All tests run hermetically: the agent is constructed with FakeLLMClient, so
no API key or network access is needed. The generation method
(`explain_verdict`) is never exercised here — the deterministic layer does
not depend on the model.
"""

import pytest
from provision_evaluator import ProvisionEvaluator

from nooa.unifiedllm.fake import FakeLLMClient

HOSPITAL_SCENARIO = (
    "A hospital deploys an AI triage model in the EU that ranks ER patients "
    "by urgency to prioritise treatment order."
)

SOCIAL_SCORING_SCENARIO = (
    "A city council deploys social scoring software that gives every resident a "
    "trustworthiness score from their social media activity and benefits usage, "
    "used to rank access to public housing."
)

UNRELATED_SCENARIO = "A bakery uses a spreadsheet to track its flour inventory each week."


@pytest.fixture
def key_path(tmp_path):
    return tmp_path / "test_signing_key.hex"


@pytest.fixture
def agent(key_path):
    return ProvisionEvaluator(llm=FakeLLMClient(), signing_key_path=key_path)


def test_hospital_triage_matches_annex_iii_and_obligations(agent):
    record = agent.build_verdict(HOSPITAL_SCENARIO)

    assert record.verdict == "PERMITTED_WITH_CONDITIONS"
    # Annex III 5(a) matches directly; Art 9 / Art 14 attach statutorily.
    assert "EU-AI-ACT-ANNEX-III-5A" in record.provisions_matched
    assert "EU-AI-ACT-ART-9" in record.provisions_matched
    assert "EU-AI-ACT-ART-14" in record.provisions_matched
    # Evidence records the actual anchor terms that fired.
    assert "triage" in record.evidence["EU-AI-ACT-ANNEX-III-5A"]
    assert record.evidence["EU-AI-ACT-ART-9"] == ["attached via EU-AI-ACT-ANNEX-III-5A"]
    assert agent.verify_verdict(record)


def test_social_scoring_is_prohibited_risk(agent):
    record = agent.build_verdict(SOCIAL_SCORING_SCENARIO)

    assert record.verdict == "PROHIBITED_RISK"
    assert "EU-AI-ACT-ART-5" in record.provisions_matched
    assert agent.verify_verdict(record)


def test_unrelated_scenario_is_unmeasured(agent):
    record = agent.build_verdict(UNRELATED_SCENARIO)

    assert record.verdict == "UNMEASURED"
    assert record.provisions_matched == []
    assert record.evidence == {}
    assert agent.verify_verdict(record)


def test_same_scenario_same_signature(agent, key_path):
    first = agent.build_verdict(HOSPITAL_SCENARIO)
    # A second agent over the same key file must reproduce the exact record.
    other = ProvisionEvaluator(llm=FakeLLMClient(), signing_key_path=key_path)
    second = other.build_verdict(HOSPITAL_SCENARIO)

    assert first == second
    assert first.signature == second.signature


def test_tampering_changes_scenario_hash(agent):
    record = agent.build_verdict(HOSPITAL_SCENARIO)
    tampered = agent.build_verdict(HOSPITAL_SCENARIO + " The vendor is based outside the EU.")

    assert tampered.scenario_hash != record.scenario_hash
    assert tampered.signature != record.signature
    # And a forged record (verdict flipped, original signature kept) is rejected.
    forged = record.model_copy(update={"verdict": "UNMEASURED"})
    assert not agent.verify_verdict(forged)


def test_strict_narration_refuses_unmeasured(key_path):
    """strict_narration=True must refuse to narrate UNMEASURED verdicts.

    The deterministic core returns nothing to anchor on when no provision
    fires; in strict mode we surface that as a hard refusal rather than
    letting the LLM fill the gap. We assert the deterministic contract
    here (strict-mode guard fires before any LLM is invoked); end-to-end
    narration requires a real LLM client and is covered by manual runs.
    """
    import asyncio

    from provision_evaluator import VERDICT_UNMEASURED

    strict_agent = ProvisionEvaluator(
        llm=FakeLLMClient(), signing_key_path=key_path, strict_narration=True
    )
    permissive_agent = ProvisionEvaluator(
        llm=FakeLLMClient(), signing_key_path=key_path, strict_narration=False
    )

    # UNMEASURED -> strict mode raises BEFORE the LLM is invoked.
    unmeasured = strict_agent.build_verdict(UNRELATED_SCENARIO)
    assert unmeasured.verdict == VERDICT_UNMEASURED
    with pytest.raises(ValueError, match="strict_narration=True"):
        asyncio.run(strict_agent.explain_verdict(unmeasured))

    # Strict mode still works fine on measured verdicts at the deterministic
    # layer — the strict-mode guard only affects UNMEASURED verdicts.
    measured = strict_agent.build_verdict(HOSPITAL_SCENARIO)
    assert measured.verdict != VERDICT_UNMEASURED

    # Permissive mode (strict_narration=False) is the default and does not
    # raise on UNMEASURED — the deterministic layer is unchanged.
    permissive_unmeasured = permissive_agent.build_verdict(UNRELATED_SCENARIO)
    assert permissive_unmeasured.verdict == VERDICT_UNMEASURED

    # The strict-mode flag itself is observable on the agent instance.
    assert strict_agent.strict_narration is True
    assert permissive_agent.strict_narration is False
