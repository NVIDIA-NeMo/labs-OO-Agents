# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The SciPy doc() adapter: csgraph, spatial and ndimage views with SciPy's signatures."""

from __future__ import annotations

import builtins
import importlib
import inspect

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

import scipy  # noqa: E402
import scipy.ndimage  # noqa: E402
import scipy.sparse.csgraph  # noqa: E402
import scipy.spatial  # noqa: E402

from nooa.agentdoc import doc  # noqa: E402
from nooa.agentdoc.adapters._libdoc import signature_parts  # noqa: E402
from nooa.agentdoc.registry import get_module_info_extractor, get_type_info_extractor  # noqa: E402

csgraph = scipy.sparse.csgraph
spatial = scipy.spatial
ndimage = scipy.ndimage


@pytest.fixture(autouse=True)
def adapter():
    import nooa.agentdoc.adapters.scipy as _scipy_adapter

    return importlib.reload(_scipy_adapter)


class TestShortViews:
    @pytest.mark.parametrize("module", [scipy, csgraph, spatial, ndimage])
    def test_modules(self, module):
        out = doc(module)
        assert out.startswith(f"# {module.__name__}")
        assert len(out.splitlines()) < 60

    @pytest.mark.parametrize("cls", [spatial.KDTree, spatial.cKDTree, spatial.ConvexHull])
    def test_classes(self, cls):
        out = doc(cls)
        assert out.startswith(f"class {cls.__name__}(")
        assert len(out.splitlines()) < 60

    @pytest.mark.parametrize(
        "func", [csgraph.connected_components, ndimage.label, spatial.distance.cdist]
    )
    def test_functions(self, func):
        out = doc(func)
        assert out.splitlines()[0].startswith(f"def {func.__name__}{inspect.signature(func)}")
        assert "Parameters:" in out and ">>>" not in out
        assert len(out.splitlines()) <= 20


class TestAccuracy:
    @pytest.mark.parametrize("module", [csgraph, ndimage])
    def test_module_signatures(self, module):
        for item in get_module_info_extractor(module)(module).functions:
            assert item.signature == str(inspect.signature(getattr(module, item.name)))

    @pytest.mark.parametrize("cls", [spatial.KDTree, spatial.cKDTree, spatial.ConvexHull])
    def test_class_signatures_and_fields(self, cls):
        info = get_type_info_extractor(cls)(cls)
        for item in info.methods:
            if item.name == "__init__":
                assert item.signature == "(self, " + signature_parts(cls)[0][1:]
            else:
                assert item.signature == signature_parts(getattr(cls, item.name))[0]
        points = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        instance = cls(points)
        for field in info.fields:
            assert hasattr(instance, field.name), field.name


class TestIdioms:
    @pytest.mark.parametrize("name", ["CSGRAPH_IDIOMS", "SPATIAL_IDIOMS", "NDIMAGE_IDIOMS"])
    def test_idioms_run(self, adapter, name):
        builtins.exec(getattr(adapter, name), {"np": np})  # noqa: S102 - our own idioms
