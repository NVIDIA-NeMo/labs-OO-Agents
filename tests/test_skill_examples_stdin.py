# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skill instructions pass a script on standard input, not in a heredoc or ``python -c``."""

import re
from pathlib import Path

SKILLS = Path(__file__).resolve().parents[1] / "skills"
_QUOTED_CODE = re.compile(r"python3? (?:-c |<<)|<<-?\s*['\"]?EOF")


def test_refine_agent_prompt_runs_its_script_on_stdin():
    text = (SKILLS / "refine-agent-prompt" / "SKILL.md").read_text()
    assert 'sh.run("uv run python3 -", stdin=script)' in text
    assert "uv run python3 << 'EOF'" not in text


def test_no_skill_runs_code_inside_a_quoted_command_or_heredoc():
    found = [
        f"{path.relative_to(SKILLS)}:{number}"
        for path in sorted(SKILLS.glob("*/SKILL.md"))
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if _QUOTED_CODE.search(line)
    ]
    assert found == []
