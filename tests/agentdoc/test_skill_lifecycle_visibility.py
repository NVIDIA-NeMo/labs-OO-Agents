# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skill lifecycle hooks are framework plumbing, not model-facing tools."""

from nooa.agentdoc import doc
from nooa.skill import Skill


class ExampleSkill(Skill):
    """A useful example skill."""

    def useful(self, value: str) -> str:
        """Return a useful value."""
        return value


def test_skill_lifecycle_hooks_are_hidden_from_docs():
    rendered = doc(ExampleSkill)

    assert "def useful(" in rendered
    assert "def attach(" not in rendered
    assert "def detach(" not in rendered
