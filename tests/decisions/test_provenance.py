# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nooa.decisions.client import BooleanQuestion, ChoiceQuestion, ScoreQuestion
from nooa.decisions.provenance import question_digest


def test_question_digest_is_independent_of_question_order() -> None:
    urgent = BooleanQuestion(instructions="Is it urgent?")
    team = ChoiceQuestion(instructions="Which team?", criteria={"a": "Billing", "b": "Support"})

    assert question_digest({"urgent": urgent, "team": team}) == question_digest(
        {"team": team, "urgent": urgent}
    )


def test_question_digest_changes_with_names_instructions_criteria_and_candidates() -> None:
    base = {"team": ChoiceQuestion(instructions="Which team?", criteria={"a": "Billing"})}
    variants = [
        {"queue": base["team"]},
        {"team": ChoiceQuestion(instructions="Which queue?", criteria={"a": "Billing"})},
        {"team": ChoiceQuestion(instructions="Which team?", criteria={"a": "Refunds"})},
        {"team": ChoiceQuestion(instructions="Which team?", criteria={"b": "Billing"})},
    ]

    digests = {question_digest(base), *(question_digest(variant) for variant in variants)}
    assert len(digests) == 1 + len(variants)


def test_question_digest_distinguishes_question_types_and_score_order() -> None:
    score = ScoreQuestion(instructions="Severity?", criteria=["Minor", "Major"])
    reordered = ScoreQuestion(instructions="Severity?", criteria=["Major", "Minor"])
    boolean = BooleanQuestion(instructions="Severity?")

    assert question_digest({"q": score}) != question_digest({"q": reordered})
    assert question_digest({"q": score}) != question_digest({"q": boolean})
    assert question_digest({"q": score}).startswith("sha256:")
