# SPDX-FileCopyrightText: Copyright (c) 2026 Sampoorn Nagpal
# SPDX-License-Identifier: Apache-2.0
"""Offline integration and boundary tests for the section retrieval example."""

import json
import sys
from pathlib import Path

import pytest

from nooa.unifiedllm import FakeLLMClient, LLMResponse

# Examples are repository sources, not an installed distribution. Restore the
# path after importing so this test does not change resolution for other tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    from examples.advanced.structured_retrieval.agent import (
        Answer,
        Citation,
        DocumentAgent,
        retrieve_and_answer,
        validate_answer,
    )
    from examples.advanced.structured_retrieval.demo import fixture
    from examples.advanced.structured_retrieval.index import DocumentIndex, Section, encode
finally:
    sys.path.pop(0)


def _ids(index):
    outline, complete = index.outline(100_000)
    assert complete
    return [item["section_id"] for item in json.loads(outline)]


def _agent(*responses):
    return DocumentAgent(
        llm=FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=json.dumps(response),
                    tool_calls=[],
                    finish_reason="stop",
                )
                for response in responses
            ]
        )
    )


def _answer(sid, quote="Run deploy rollback --production."):
    return {"text": quote, "citations": [{"section_id": sid, "quote": quote}], "abstained": False}


def test_repeated_titles_have_distinct_ids_and_ancestor_paths():
    index = fixture()
    outline, complete = index.outline(6000)
    rows = json.loads(outline)
    assert complete
    assert rows[1]["path"] == ["Production", "Rollback"]
    assert rows[3]["path"] == ["Staging", "Rollback"]
    assert rows[1]["section_id"] != rows[3]["section_id"]
    assert len(index.read(_ids(index), k=4, max_chars=6000)) == 4  # parent-owned prose


@pytest.mark.parametrize(
    "sections",
    [
        [Section("a", "A", "x"), Section("a", "B", "y")],
        [Section("a", "A", "x", "missing")],
        [Section("a", "A", "x", "a")],
        [Section("a", "A", "x", "b"), Section("b", "B", "y", "a")],
    ],
)
def test_invalid_hierarchies_rejected(sections):
    with pytest.raises(ValueError):
        DocumentIndex("doc", sections)


def test_snapshot_stability_and_stale_ids():
    index = fixture()
    assert _ids(index) == _ids(fixture())
    changed = DocumentIndex("deployment-handbook", [Section("production", "Production", "Changed")])
    with pytest.raises(ValueError, match="stale"):
        changed.read([_ids(index)[0]], k=1, max_chars=1000)


def test_membership_checked_before_dedup_and_limit():
    index = fixture()
    ids = _ids(index)
    assert index.validate_ids([ids[0], ids[0]], 1) == [ids[0]]
    with pytest.raises(ValueError, match="Too many"):
        index.validate_ids(ids, 1)
    with pytest.raises(ValueError, match="Unknown"):
        index.validate_ids([ids[0], "invalid"], 1)


@pytest.mark.parametrize("budget", [2, 50, 200, 300, 700, 2000])
def test_evidence_budget_counts_json_metadata_and_escaping(budget):
    index = DocumentIndex(
        "doc", [Section("a", "हिन्दी", '雪\n"' * 500), Section("b", "B", "second")]
    )
    evidence = index.read(_ids(index), k=2, max_chars=budget)
    assert len(encode(evidence)) <= budget
    for item in evidence:
        original = '雪\n"' * 500 if item.section_id.endswith(":a") else "second"
        assert item.text == original[item.start : item.end]
        assert item.truncated == (item.end < len(original))


def test_oversized_outline_explicitly_reports_incomplete():
    outline, complete = fixture().outline(2)
    assert outline == "[]"
    assert not complete


def test_empty_document_and_nonmatching_search():
    index = DocumentIndex("empty", [])
    assert index.outline(2) == ("[]", True)
    assert index.search("anything", 3) == []
    assert fixture().search("zzzz", 3) == []


def test_empty_parent_still_appears_in_child_path():
    index = DocumentIndex("doc", [Section("a", "Parent", ""), Section("b", "Child", "text", "a")])
    outline, _ = index.outline(1000)
    assert json.loads(outline)[0]["path"] == ["Parent", "Child"]
    assert len(_ids(index)) == 1


def test_citations_must_reference_supplied_excerpt():
    index = fixture()
    ids = _ids(index)
    evidence = index.read([ids[1]], k=1, max_chars=1000)
    validate_answer(Answer.model_validate(_answer(ids[1])), evidence)
    for sid, quote in [
        (ids[3], "Run deploy rollback --staging."),
        (ids[1], "invented"),
        (ids[1], " "),
    ]:
        with pytest.raises(ValueError):
            validate_answer(Answer.model_validate(_answer(sid, quote)), evidence)
    with pytest.raises(ValueError):
        validate_answer(Answer(text="unsupported", citations=[], abstained=False), evidence)


def test_citation_cannot_use_unread_suffix_of_selected_section():
    index = DocumentIndex("doc", [Section("a", "A", "prefix " * 100 + "UNREAD SUFFIX")])
    sid = _ids(index)[0]
    evidence = index.read([sid], k=1, max_chars=250)
    assert evidence[0].truncated
    with pytest.raises(ValueError, match="absent"):
        validate_answer(
            Answer(
                text="suffix",
                citations=[Citation(section_id=sid, quote="UNREAD SUFFIX")],
                abstained=False,
            ),
            evidence,
        )


async def test_predict_selection_and_citation_end_to_end():
    index = fixture()
    sid = _ids(index)[1]
    result = await retrieve_and_answer(
        _agent({"section_ids": [sid, sid]}, _answer(sid)), index, "rollback"
    )
    assert not result.answer.abstained
    assert not result.used_fallback
    assert len(result.evidence) == 1
    assert (result.selection_attempts, result.answer_attempts) == (1, 1)


async def test_invalid_selection_retries_then_content_fallback():
    index = fixture()
    sid = _ids(index)[1]
    result = await retrieve_and_answer(
        _agent({"section_ids": ["stale"]}, {"section_ids": ["unknown"]}, _answer(sid)),
        index,
        "rollback production",
        k=3,
    )
    assert result.used_fallback
    assert result.selection_attempts == 2
    assert not result.answer.abstained


async def test_schema_invalid_selection_uses_bounded_retry():
    index = fixture()
    sid = _ids(index)[1]
    result = await retrieve_and_answer(
        _agent({"section_ids": 42}, {"section_ids": [sid]}, _answer(sid)), index, "rollback"
    )
    assert result.selection_attempts == 2
    assert not result.used_fallback


async def test_incomplete_outline_skips_selector():
    index = fixture()
    sid = _ids(index)[1]
    result = await retrieve_and_answer(
        _agent(_answer(sid)), index, "rollback production", outline_chars=2
    )
    assert result.used_fallback
    assert result.selection_attempts == 0
    assert not result.answer.abstained


async def test_invalid_citation_retries_then_abstains():
    index = fixture()
    sid = _ids(index)[1]
    result = await retrieve_and_answer(
        _agent({"section_ids": [sid]}, _answer("unknown"), _answer(sid, "invented")),
        index,
        "rollback",
    )
    assert result.answer.abstained
    assert result.answer.citations == []
    assert result.answer_attempts == 2


async def test_no_evidence_never_calls_answer_model():
    result = await retrieve_and_answer(_agent(), DocumentIndex("empty", []), "question")
    assert result.answer.abstained
    assert result.selection_attempts == result.answer_attempts == 0


@pytest.mark.parametrize(
    "kwargs", [{"k": 0}, {"k": 21}, {"outline_chars": 1}, {"evidence_chars": 1}]
)
async def test_invalid_budgets_fail_before_model_call(kwargs):
    with pytest.raises(ValueError):
        await retrieve_and_answer(_agent(), fixture(), "question", **kwargs)
