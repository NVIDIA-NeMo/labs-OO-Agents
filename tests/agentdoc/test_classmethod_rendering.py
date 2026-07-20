# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for @classmethod rendering in doc()."""

from enum import Enum

from nooa.agentdoc import doc


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class MyService:
    """A service with classmethods."""

    def instance_method(self, x: int) -> str:
        """A regular instance method."""
        return str(x)

    @classmethod
    def create(cls, name: str) -> "MyService":
        """Factory classmethod."""
        return cls()

    @classmethod
    def from_env(cls, default: Color = Color.RED) -> "MyService":
        """Classmethod with enum default."""
        return cls()

    @staticmethod
    def version() -> str:
        """A static method."""
        return "1.0"


# ── @classmethod decorator ────────────────────────────────────────────────────


def test_classmethod_has_decorator_on_instance():
    result = doc(MyService())
    assert "@classmethod" in result


def test_classmethod_has_decorator_on_type():
    result = doc(MyService)
    assert "@classmethod" in result


def test_classmethod_decorator_precedes_def():
    result = doc(MyService())
    lines = result.splitlines()
    decorator_idx = next(i for i, ln in enumerate(lines) if "@classmethod" in ln)
    def_idx = next(i for i, ln in enumerate(lines) if "def create" in ln)
    assert decorator_idx == def_idx - 1


def test_classmethod_cls_in_signature():
    """cls must appear as the first parameter, consistent with self in instance methods."""
    result = doc(MyService())
    create_line = next(ln for ln in result.splitlines() if "def create" in ln)
    assert "cls" in create_line


def test_regular_method_no_classmethod_decorator():
    result = doc(MyService())
    lines = result.splitlines()
    instance_method_idx = next(i for i, ln in enumerate(lines) if "def instance_method" in ln)
    preceding = lines[instance_method_idx - 1]
    assert "@classmethod" not in preceding


# ── enum default in classmethod ───────────────────────────────────────────────


def test_classmethod_enum_default_renders_correctly():
    result = doc(MyService())
    assert "Color.RED" in result
