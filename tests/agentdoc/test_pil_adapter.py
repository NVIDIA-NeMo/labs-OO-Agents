# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Pillow doc() adapter: short curated views with the installed Pillow's signatures."""

from __future__ import annotations

import builtins
import importlib

import pytest

np = pytest.importorskip("numpy")
PIL = pytest.importorskip("PIL")

from PIL import Image, ImageDraw  # noqa: E402

from nooa.agentdoc import doc  # noqa: E402
from nooa.agentdoc.adapters import register_all  # noqa: E402
from nooa.agentdoc.adapters._libdoc import signature_parts  # noqa: E402
from nooa.agentdoc.registry import get_module_info_extractor, get_type_info_extractor  # noqa: E402


@pytest.fixture(autouse=True)
def adapter():
    import nooa.agentdoc.adapters.pil as _pil_adapter

    return importlib.reload(_pil_adapter)


class TestShortViews:
    def test_image_class(self):
        out = doc(Image.Image)
        assert out.startswith("class Image(PIL.Image.Image):")
        for name in ("convert", "resize", "crop", "save", "getpixel"):
            assert f"def {name}(self" in out
        assert "np.asarray(img)" in out
        assert len(out.splitlines()) < 120

    def test_image_instance_uses_the_view(self):
        out = doc(Image.new("RGB", (2, 2)))
        assert "def resize(self" in out and len(out.splitlines()) < 120

    def test_image_module(self):
        out = doc(Image)
        assert out.startswith("# PIL.Image")
        for name in ("open", "new", "fromarray"):
            assert f"def {name}(" in out
        assert "'RGBA'" in out
        assert len(out.splitlines()) < 80

    def test_imagedraw(self):
        assert len(doc(ImageDraw).splitlines()) < 40
        out = doc(ImageDraw.ImageDraw)
        assert "def rectangle(self" in out and len(out.splitlines()) < 40


class TestSignaturesMatchPillow:
    def _check(self, info, owner):
        items = info.methods if hasattr(info, "methods") else info.functions
        assert items
        for item in items:
            func = getattr(owner, item.name)
            assert item.signature in {
                signature_parts(func)[0],
                signature_parts(func, annotate=False)[0],
            }, item.name

    def test_image_methods(self):
        self._check(get_type_info_extractor(Image.Image)(Image.Image), Image.Image)

    def test_module_functions(self):
        self._check(get_module_info_extractor(Image)(Image), Image)
        self._check(get_module_info_extractor(ImageDraw)(ImageDraw), ImageDraw)

    def test_draw_methods(self):
        info = get_type_info_extractor(ImageDraw.ImageDraw)(ImageDraw.ImageDraw)
        self._check(info, ImageDraw.ImageDraw)

    def test_fields_exist_on_instances(self):
        img = Image.new("L", (3, 2))
        info = get_type_info_extractor(Image.Image)(Image.Image)
        for field in info.fields:
            assert hasattr(img, field.name), field.name
        assert img.size == (3, 2) and img.width == 3 and img.height == 2


class TestIdioms:
    def test_idioms_run(self, adapter, tmp_path):
        namespace = {"np": np, "Image": Image, "ImageDraw": ImageDraw, "path": tmp_path / "a.png"}
        builtins.exec(adapter.IMAGE_IDIOMS, namespace)  # noqa: S102 - our own idioms
        assert namespace["arr"].shape == (32, 64, 3)

    def test_idioms_shown_in_doc(self, adapter):
        out = doc(Image.Image)
        for line in adapter.IMAGE_IDIOMS.splitlines():
            assert line in out


def test_register_all_includes_pil():
    assert "pil" in register_all()
