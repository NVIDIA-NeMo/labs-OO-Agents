# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A missing attribute on a work result says what the object is and what to use."""

import pytest
from nooa_atom.agent.work_subagent import Question, Subagent

from nooa.interactive import TaskResult


def test_a_question_says_to_read_its_text_and_answer_with_send():
    question = Question("Which branch?")
    with pytest.raises(AttributeError, match=r"Question has no attribute 'report'.*\.text.*send"):
        _ = question.report
    assert question.text == "Which branch?" and question.id == ""
    assert not hasattr(question, "_private")


def test_a_subagent_says_run_returns_the_task_result():
    worker = Subagent("fix parser")
    with pytest.raises(AttributeError, match=r"Subagent has no attribute 'report'.*\.run\(todo\)"):
        _ = worker.report
    assert worker.description == "fix parser" and worker.model is None
    assert not hasattr(worker, "__deepcopy__")


def test_a_task_result_lists_its_fields():
    result = TaskResult(solution_description="s", evidence="e", how_to_verify="h")
    with pytest.raises(AttributeError, match="no attribute 'status'; its fields are solution_"):
        _ = result.status
    assert result.report == "" and result.value == ""
    assert TaskResult.model_validate(result.model_dump()) == result
