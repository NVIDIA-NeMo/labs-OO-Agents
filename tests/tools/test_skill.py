# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Skill base class / wrapper."""

import math

import pytest

from nooa.skill import Skill


class TestSkillAsBaseClass:
    def test_subclass_is_instance_of_skill(self):
        class MySkill(Skill):
            """My skill docs."""

        assert isinstance(MySkill(), Skill)

    def test_subclass_preserves_docstring(self):
        class MySkill(Skill):
            """My skill docs."""

        assert "My skill docs" in type(MySkill()).__doc__


class TestSkillAsWrapper:
    def test_wrapped_object_is_instance_of_skill(self):
        assert isinstance(Skill(math), Skill)

    def test_wrapped_module_delegates_doc(self):
        s = Skill(math)
        assert type(s).__doc__ == math.__doc__

    def test_wrapped_instance_delegates_doc(self):
        class Inner:
            """Inner docs."""

        s = Skill(Inner())
        assert "Inner docs" in type(s).__doc__

    def test_wrapped_object_with_no_doc_does_not_raise(self):
        class NoDocs:
            pass

        s = Skill(NoDocs())
        assert isinstance(s, Skill)

    def test_wrapped_object_exposes_attrs_via_dir(self):
        s = Skill(math)
        assert "sqrt" in dir(s)
        assert "pi" in dir(s)

    def test_wrapped_object_does_not_forward_attribute_access(self):
        s = Skill(math)
        with pytest.raises(AttributeError):
            _ = s.sqrt


class TestSkillWithContent:
    def test_content_skill_has_doc(self):
        s = Skill(content="One-liner.\n\nFull details here.")
        assert type(s).__doc__ == "One-liner.\n\nFull details here."

    def test_content_skill_is_instance_of_skill(self):
        s = Skill(content="Some skill content")
        assert isinstance(s, Skill)

    def test_content_skill_has_agentdoc_skip(self):
        s = Skill(content="Some skill content")
        assert type(s).__agentdoc_skip__ is True

    def test_no_args_raises(self):
        with pytest.raises(ValueError, match="requires one of"):
            Skill()

    def test_obj_and_content_together_raises(self):
        import math

        with pytest.raises(ValueError, match="exactly one of"):
            Skill(math, content="some content")
