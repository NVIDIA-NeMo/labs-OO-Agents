# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the in-container NOOA benchmark runner."""

from __future__ import annotations

import json

from nooa_bench import runner


def test_write_result_preserves_activated_skills(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "LOGS_DIR", tmp_path)

    runner._write_result(
        {
            "success": True,
            "response": "pytest -q",
            "activated_skills": ["cmd.xlsx"],
            "n_input_tokens": 10,
            "n_output_tokens": 2,
        },
        model="openai/openai/openai/gpt-5.2",
        agent_type="bench",
    )

    payload = json.loads((tmp_path / "result.json").read_text())

    assert payload["success"] is True
    assert payload["activated_skills"] == ["cmd.xlsx"]
