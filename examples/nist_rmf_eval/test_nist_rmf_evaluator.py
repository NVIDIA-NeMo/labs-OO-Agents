# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Contributed by Council of AI (CSOAI Ltd, UK 16939677) — https://councilof.ai
"""Deterministic-core tests for the NIST AI RMF evaluator.

All tests run hermetically: the agent is constructed with FakeLLMClient, so
no API key or network access is needed. The generation method
(`explain_verdict`) is never exercised here — the deterministic layer does
not depend on the model.
"""

import pytest
from nist_rmf_evaluator import NistRmfEvaluator

from nooa.unifiedllm.fake import FakeLLMClient

GOVERNANCE_SCENARIO = (
    "Our AI programme documents all legal and regulatory requirements in a "
    "regulatory register, and assigns clear roles and responsibilities for "
    "every deployed model with an accountable owner."
)

SECURITY_SCENARIO = (
    "Before launch, the red team ran adversarial tests on the model and the "
    "security evaluation results were documented; we also maintain a "
    "mitigation plan and a kill switch to deactivate the system if needed."
)

FULL_COVERAGE_SCENARIO = (
    "The team documented the model's intended purpose and deployment context, "
    "maintains performance metrics on a public dashboard, ran red-team "
    "adversarial tests before launch, keeps a rollback plan to deactivate the "
    "system if it drifts, and publishes its roles and responsibilities."
)

UNRELATED_SCENARIO = "A bakery uses a spreadsheet to track its flour inventory each week."


@pytest.fixture
def key_path(tmp_path):
    return tmp_path / "test_signing_key.hex"


@pytest.fixture
def agent(key_path):
    return NistRmfEvaluator(llm=FakeLLMClient(), signing_key_path=key_path)


def test_governance_scenario_matches_govern_subcategories(agent):
    record = agent.build_verdict(GOVERNANCE_SCENARIO)

    assert record.verdict == "PARTIAL_FUNCTION_COVERAGE"
    assert "GOVERN-1.1" in record.subcategories_matched
    assert "GOVERN-2.1" in record.subcategories_matched
    assert "legal and regulatory requirements" in record.evidence["GOVERN-1.1"]
    assert agent.verify_verdict(record)


def test_security_scenario_matches_measure_and_manage_with_attachment(agent):
    record = agent.build_verdict(SECURITY_SCENARIO)

    # MEASURE-2.7 and MANAGE-1.3 / MANAGE-2.4 match directly.
    assert "MEASURE-2.7" in record.subcategories_matched
    assert "MANAGE-1.3" in record.subcategories_matched
    assert "MANAGE-2.4" in record.subcategories_matched
    # MAP-5.1 attaches via the framework-declared dependency: MANAGE-1.3's
    # own text addresses risks "as identified by the MAP function".
    assert "MAP-5.1" in record.subcategories_matched
    assert record.evidence["MAP-5.1"] == ["attached via MANAGE-1.3"]
    assert record.verdict == "PARTIAL_FUNCTION_COVERAGE"
    assert agent.verify_verdict(record)


def test_all_four_functions_is_full_coverage(agent):
    record = agent.build_verdict(FULL_COVERAGE_SCENARIO)

    assert record.verdict == "FULL_FUNCTION_COVERAGE"
    functions = {"GOVERN-2.1", "MAP-1.1", "MEASURE-1.1", "MEASURE-2.7", "MANAGE-2.4"}
    assert functions <= set(record.subcategories_matched)
    assert agent.verify_verdict(record)


def test_unrelated_scenario_is_unmeasured(agent):
    record = agent.build_verdict(UNRELATED_SCENARIO)

    assert record.verdict == "UNMEASURED"
    assert record.subcategories_matched == []
    assert record.evidence == {}
    assert agent.verify_verdict(record)


def test_same_scenario_same_signature(agent, key_path):
    first = agent.build_verdict(GOVERNANCE_SCENARIO)
    # A second agent over the same key file must reproduce the exact record.
    other = NistRmfEvaluator(llm=FakeLLMClient(), signing_key_path=key_path)
    second = other.build_verdict(GOVERNANCE_SCENARIO)

    assert first == second
    assert first.signature == second.signature


def test_tampering_changes_scenario_hash(agent):
    record = agent.build_verdict(GOVERNANCE_SCENARIO)
    tampered = agent.build_verdict(GOVERNANCE_SCENARIO + " We also track incidents weekly.")

    assert tampered.scenario_hash != record.scenario_hash
    assert tampered.signature != record.signature
    # And a forged record (verdict flipped, original signature kept) is rejected.
    forged = record.model_copy(update={"verdict": "FULL_FUNCTION_COVERAGE"})
    assert not agent.verify_verdict(forged)


def test_strict_narration_refuses_unmeasured(key_path):
    """strict_narration=True must refuse to narrate UNMEASURED verdicts.

    The deterministic core returns nothing to anchor on when no subcategory
    fires; in strict mode we surface that as a hard refusal rather than
    letting the LLM fill the gap. We assert the deterministic contract
    here (strict-mode guard fires before any LLM is invoked); end-to-end
    narration requires a real LLM client and is covered by manual runs.
    """
    import asyncio

    from nist_rmf_evaluator import VERDICT_UNMEASURED

    strict_agent = NistRmfEvaluator(
        llm=FakeLLMClient(), signing_key_path=key_path, strict_narration=True
    )
    permissive_agent = NistRmfEvaluator(
        llm=FakeLLMClient(), signing_key_path=key_path, strict_narration=False
    )

    # UNMEASURED -> strict mode raises BEFORE the LLM is invoked.
    unmeasured = strict_agent.build_verdict(UNRELATED_SCENARIO)
    assert unmeasured.verdict == VERDICT_UNMEASURED
    with pytest.raises(ValueError, match="strict_narration=True"):
        asyncio.run(strict_agent.explain_verdict(unmeasured))

    # Strict mode still works fine on measured verdicts at the deterministic
    # layer — the strict-mode guard only affects UNMEASURED verdicts.
    measured = strict_agent.build_verdict(GOVERNANCE_SCENARIO)
    assert measured.verdict != VERDICT_UNMEASURED

    # Permissive mode (strict_narration=False) is the default and does not
    # raise on UNMEASURED — the deterministic layer is unchanged.
    permissive_unmeasured = permissive_agent.build_verdict(UNRELATED_SCENARIO)
    assert permissive_unmeasured.verdict == VERDICT_UNMEASURED

    # The strict-mode flag itself is observable on the agent instance.
    assert strict_agent.strict_narration is True
    assert permissive_agent.strict_narration is False
