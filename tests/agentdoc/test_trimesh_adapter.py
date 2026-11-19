# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The trimesh doc() adapter: short views with the installed trimesh's signatures."""

from __future__ import annotations

import builtins
import importlib

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")

from nooa.agentdoc import doc  # noqa: E402
from nooa.agentdoc.adapters._libdoc import signature_parts  # noqa: E402
from nooa.agentdoc.registry import get_module_info_extractor, get_type_info_extractor  # noqa: E402


@pytest.fixture(autouse=True)
def adapter():
    import nooa.agentdoc.adapters.trimesh as _trimesh_adapter

    return importlib.reload(_trimesh_adapter)


def _resolve(root, dotted: str):
    obj = root
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def _assert_real_signatures(items, owner):
    assert items
    for item in items:
        func = _resolve(owner, item.name)
        assert item.signature in {
            signature_parts(func)[0],
            signature_parts(func, annotate=False)[0],
        }, item.name


class TestShortViews:
    def test_trimesh_class(self):
        out = doc(trimesh.Trimesh)
        assert out.startswith("class Trimesh(trimesh.Trimesh):")
        for name in ("vertices", "faces", "face_normals", "face_adjacency"):
            assert f"    {name}: " in out
        assert "def split(self" in out
        assert "## Referenced Types" not in out
        assert len(out.splitlines()) < 120

    def test_modules(self):
        assert len(doc(trimesh).splitlines()) < 60
        out = doc(trimesh.graph)
        assert "def connected_component_labels(edges, node_count=None):" in out
        assert len(out.splitlines()) < 60


class TestAccuracy:
    def test_signatures(self):
        _assert_real_signatures(
            get_type_info_extractor(trimesh.Trimesh)(trimesh.Trimesh).methods, trimesh.Trimesh
        )
        _assert_real_signatures(get_module_info_extractor(trimesh)(trimesh).functions, trimesh)
        _assert_real_signatures(
            get_module_info_extractor(trimesh.graph)(trimesh.graph).functions, trimesh.graph
        )

    def test_field_types(self):
        mesh = trimesh.creation.box()
        info = get_type_info_extractor(trimesh.Trimesh)(trimesh.Trimesh)
        for field in info.fields:
            value = getattr(mesh, field.name)
            if field.type.startswith("ndarray"):
                shape = field.type[field.type.index("(") + 1 : field.type.index(")")]
                dims = [d for d in shape.split(",") if d.strip()]
                assert isinstance(value, np.ndarray), field.name
                assert str(value.dtype) == field.type.rsplit(" ", 1)[1], field.name
                assert value.ndim == len(dims), field.name
                for dim, size in zip(dims, value.shape, strict=True):
                    if dim.strip().isdigit():
                        assert size == int(dim), field.name
            elif field.type in ("float", "int", "bool"):
                assert isinstance(value, (builtins.float, builtins.int, np.number, np.bool_))


class TestIdioms:
    def test_idioms_run(self, adapter):
        pytest.importorskip("scipy")  # trimesh's graph engine (or networkx)
        builtins.exec(adapter.TRIMESH_IDIOMS, {"np": np, "trimesh": trimesh})  # noqa: S102

    def test_idioms_shown_in_doc(self, adapter):
        out = doc(trimesh.Trimesh)
        for line in adapter.TRIMESH_IDIOMS.splitlines():
            assert line in out
