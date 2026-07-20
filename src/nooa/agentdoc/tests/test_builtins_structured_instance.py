# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bug: _is_structured_instance returns False for Pydantic/dataclass/attrs with __module__='builtins'.

Classes defined inside exec() or a CodeAct execute_python cell get __module__='builtins',
which makes the builtins guard bail out before checking model_fields / is_dataclass / __attrs_attrs__.
"""

from nooa.agentdoc._pformat import _is_structured_instance


def _make_builtins_pydantic():
    """Create a Pydantic model with __module__='builtins' (simulates exec/REPL)."""
    ns = {}
    exec(
        """
from pydantic import BaseModel
class Gaps(BaseModel):
    items: list[str]
    flag: bool
""",
        ns,
    )
    return ns["Gaps"]


def _make_builtins_dataclass():
    """Create a dataclass with __module__='builtins' (simulates exec/REPL)."""
    ns = {}
    exec(
        """
from dataclasses import dataclass
@dataclass
class Info:
    name: str
    count: int
""",
        ns,
    )
    return ns["Info"]


def _make_builtins_attrs():
    """Create an attrs class with __module__='builtins' (simulates exec/REPL)."""
    ns = {}
    exec(
        """
import attr
@attr.s(auto_attribs=True)
class Config:
    name: str
    value: int
""",
        ns,
    )
    return ns["Config"]


def test_pydantic_with_builtins_module():
    """Pydantic models defined in exec() (__module__='builtins') are recognized as structured."""
    Gaps = _make_builtins_pydantic()
    assert Gaps.__module__ == "builtins", "precondition: exec gives __module__='builtins'"
    gaps = Gaps(items=["a", "b", "c"], flag=True)
    assert _is_structured_instance(gaps), (
        "_is_structured_instance should return True for Pydantic models "
        "even when __module__='builtins'"
    )


def test_dataclass_with_builtins_module():
    """Dataclasses defined in exec() (__module__='builtins') are recognized as structured."""
    Info = _make_builtins_dataclass()
    assert Info.__module__ == "builtins", "precondition: exec gives __module__='builtins'"
    info = Info(name="test", count=42)
    assert _is_structured_instance(info), (
        "_is_structured_instance should return True for dataclasses even when __module__='builtins'"
    )


def test_attrs_with_builtins_module():
    """Attrs classes defined in exec() (__module__='builtins') are recognized as structured."""
    Config = _make_builtins_attrs()
    assert Config.__module__ == "builtins", "precondition: exec gives __module__='builtins'"
    cfg = Config(name="test", value=42)
    assert _is_structured_instance(cfg), (
        "_is_structured_instance should return True for attrs classes "
        "even when __module__='builtins'"
    )
