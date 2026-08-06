# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""File-mention behavior for user-invocable slash commands."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from nooa_cli.tui.commands import CommandHandler, _UserSkill


class _Registry:
    def __init__(self, skill: _UserSkill, cwd):
        self._skill = skill
        self.agent = SimpleNamespace(cwd=cwd)

    def get_user_skill(self, name):
        return self._skill if name == self._skill.name else None

    def get_command(self, name):
        return None

    def get_all_command_classes(self):
        return {}


@pytest.mark.asyncio
async def test_text_skill_expands_mentions_in_generated_agent_message(tmp_path):
    target = tmp_path / "notes.md"
    target.touch()
    skill = _UserSkill(
        name="review",
        body="Review $ARGUMENTS",
        description="Review a target",
    )
    handler = CommandHandler(_Registry(skill, tmp_path), AsyncMock())

    result = await handler.handle("/review @notes.md")

    assert result.agent_message == f"Review [notes.md](<{target.resolve()}>)"


@pytest.mark.asyncio
async def test_python_skill_keeps_raw_args_and_expands_agent_text(tmp_path):
    target = tmp_path / "notes.md"
    target.touch()
    received = []

    def review(args: str):
        received.append(args)
        return f"Review {args}"

    skill = _UserSkill(
        name="review",
        body="",
        description="Review a target",
        _method=review,
    )
    handler = CommandHandler(_Registry(skill, tmp_path), AsyncMock())

    result = await handler.handle("/review @notes.md")

    assert received == ["@notes.md"]
    assert result.slash_result is not None
    assert result.slash_result.value == "Review @notes.md"
    assert result.slash_result.text == f"Review [notes.md](<{target.resolve()}>)"


@pytest.mark.asyncio
async def test_user_only_python_skill_does_not_rewrite_display_text(tmp_path):
    target = tmp_path / "notes.md"
    target.touch()

    def show(args: str):
        return f"Showing {args}"

    skill = _UserSkill(
        name="show",
        body="",
        description="Show a target",
        output_to_agent=False,
        _method=show,
    )
    handler = CommandHandler(_Registry(skill, tmp_path), AsyncMock())

    result = await handler.handle("/show @notes.md")

    assert result.slash_result is not None
    assert result.slash_result.text == "Showing @notes.md"
