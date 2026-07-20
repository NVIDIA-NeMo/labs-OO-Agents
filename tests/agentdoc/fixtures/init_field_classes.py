# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixture classes for testing __init__ field extraction.

These classes are in a real file because inspect.getsource() doesn't work
for dynamically defined classes (in tests/REPL).
"""

from typing import Annotated


class SimpleCounter:
    """A simple counter with typed fields in __init__."""

    def __init__(self):
        self.count: int = 0
        self.name: str = "default"


class AnnotatedTool:
    """Tool with Annotated fields for descriptions."""

    def __init__(self, connection_string: str = "sqlite:///:memory:"):
        self.connection_string: Annotated[str, "Database connection string"] = connection_string
        self.query_count: Annotated[int, "Total queries executed"] = 0


class SecretClass:
    """Class with public and private fields."""

    def __init__(self):
        self.public: int = 1
        self._private: int = 2
        self.__dunder: int = 3


class MixedFields:
    """Class with fields at both class level and in __init__."""

    count: int = 0  # Class-level field

    def __init__(self):
        self.count: int = 10  # Same field assigned in __init__
        self.name: str = "test"  # New field only in __init__


# --- MRO / Inheritance test fixtures ---


class ParentWithFields:
    """Parent class that sets fields in __init__."""

    def __init__(self):
        self.parent_field: str = "from_parent"
        self.shared_field: int = 1
        self._private_parent: int = 99


class ChildWithOwnInit(ParentWithFields):
    """Child that defines its own __init__ and calls super()."""

    def __init__(self):
        super().__init__()
        self.child_field: float = 3.14
        self.shared_field: int = 42  # Override parent's field


class ChildWithoutInit(ParentWithFields):
    """Child with no __init__ — inherits everything from parent."""

    pass


class GrandchildClass(ChildWithOwnInit):
    """Grandchild to test multi-level MRO."""

    def __init__(self):
        super().__init__()
        self.grandchild_field: bool = True
