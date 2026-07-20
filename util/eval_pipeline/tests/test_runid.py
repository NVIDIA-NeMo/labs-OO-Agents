# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for run_id feature (self-consistency scoring support)."""

import pytest

from eval_pipeline.models import Task


class TestTaskRunId:
    """Tests for Task.run_id field."""

    def test_task_default_run_id(self):
        """Task defaults run_id to 1."""
        task = Task(
            id="task_001",
            input=((), {"text": "hello"}),
            expected="result",
        )
        assert task.run_id == 1

    def test_task_explicit_run_id(self):
        """Task can have explicit run_id."""
        task = Task(
            id="task_001",
            input=((), {"text": "hello"}),
            expected="result",
            run_id=3,
        )
        assert task.run_id == 3

    def test_task_run_id_can_be_modified(self):
        """Task run_id can be modified after creation."""
        task = Task(
            id="task_001",
            input=((), {"text": "hello"}),
            expected="result",
        )
        task.run_id = 5
        assert task.run_id == 5

    def test_multiple_tasks_different_run_ids(self):
        """Multiple tasks can have different run_ids."""
        tasks = [
            Task(id="task_001", input=((), {}), expected="a", run_id=1),
            Task(id="task_002", input=((), {}), expected="b", run_id=2),
            Task(id="task_003", input=((), {}), expected="c", run_id=3),
        ]
        assert [t.run_id for t in tasks] == [1, 2, 3]

    def test_task_run_id_zero_allowed(self):
        """run_id of 0 is technically allowed (edge case)."""
        task = Task(
            id="task_001",
            input=((), {}),
            expected="result",
            run_id=0,
        )
        assert task.run_id == 0

    def test_task_with_all_fields(self):
        """Task with all fields including run_id works correctly."""
        task = Task(
            id="task_001",
            input=(("arg1",), {"key": "value"}),
            expected={"result": 42},
            metadata={"source": "test", "difficulty": "easy"},
            run_id=7,
        )
        assert task.id == "task_001"
        assert task.input == (("arg1",), {"key": "value"})
        assert task.expected == {"result": 42}
        assert task.metadata == {"source": "test", "difficulty": "easy"}
        assert task.run_id == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
