# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The visibility contract snapshot shares with agentdoc: ``nooa.agentdoc.markers``.

Snapshot reads the markers only through that public module (no agentdoc
underscore module), the module needs only the standard library, and every
marker form hides the same fields in both packages.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
from pathlib import Path
from typing import Annotated

import pydantic
import pytest

import nooa.agentdoc
import nooa.snapshot
from nooa.agentdoc import markers, spec
from nooa.agentdoc.visibility import is_hidden_field
from nooa.snapshot import Object, capture

_SNAPSHOT = Path(nooa.snapshot.__file__).parent


def _imports(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module)
    return names


def test_snapshot_imports_no_private_agentdoc_module():
    found = {
        (path.name, name)
        for path in _SNAPSHOT.glob("*.py")
        for name in _imports(path)
        if name.startswith("nooa.agentdoc") and name != "nooa.agentdoc.markers"
    }
    assert found == set()
    assert not any("_agentdoc_" in path.read_text() for path in _SNAPSHOT.glob("*.py"))


def test_markers_needs_only_the_standard_library():
    imported = _imports(Path(markers.__file__))
    assert {name.split(".")[0] for name in imported} <= set(sys.stdlib_module_names) | {
        "__future__"
    }


def test_agentdoc_hands_out_the_shared_markers():
    assert nooa.agentdoc.hidden is markers.hidden
    assert type(spec(hidden=True)) is markers.SpecAnnotation
    assert set(markers.__all__) == {
        "CLASS_METADATA_ATTR",
        "FIELD_METADATA_ATTR",
        "SpecAnnotation",
        "hidden",
    }


@dataclasses.dataclass
class _Data:
    shown: str = "ok"
    by_marker: Annotated[str, markers.hidden] = "secret-1"
    by_spec: Annotated[str, spec(hidden=True)] = "secret-2"
    by_override: str = "secret-3"


class _Model(pydantic.BaseModel):
    shown: str = "ok"
    by_marker: Annotated[str, markers.hidden] = "secret-1"
    by_spec: Annotated[str, spec(hidden=True)] = "secret-2"
    by_override: str = "secret-3"


class _Plain:
    by_marker: Annotated[str, markers.hidden]
    by_spec: Annotated[str, spec(hidden=True)]

    def __init__(self) -> None:
        self.shown = "ok"
        self.by_marker = "secret-1"
        self.by_spec = "secret-2"
        self.by_override = "secret-3"


for _cls in (_Data, _Model, _Plain):
    spec(_cls, "by_override", hidden=True)


@pytest.mark.parametrize("cls", [_Data, _Model, _Plain])
def test_every_marker_form_hides_the_field_in_both_packages(cls):
    value = cls()
    node = capture(value)
    assert type(node) is Object
    assert [name for name, _ in node.fields] == ["shown"]
    for name in ("by_marker", "by_spec", "by_override"):
        assert is_hidden_field(cls, name)
    assert not is_hidden_field(cls, "shown")


def test_an_instance_override_hides_the_field_in_both_packages():
    value = _Plain()
    spec(value, "shown", hidden=True)
    assert is_hidden_field(value, "shown")
    node = capture(value)
    assert type(node) is Object and node.fields == ()
