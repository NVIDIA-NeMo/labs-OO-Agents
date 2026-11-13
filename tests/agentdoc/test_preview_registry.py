# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the value-preview extractor registry.

Preview extractors let ``pformat`` render structural, bounded previews for
opaque third-party values (numpy arrays, DataFrames, …) in place of the
truncated-repr fallback. These tests cover the registry mechanics with plain
Python types; the numpy/pandas adapters have their own test modules.
"""

import pytest

from nooa.agentdoc import pformat, spec
from nooa.agentdoc.ext import (
    PreviewBudget,
    clear_registry,
    get_preview_extractor,
    register_preview_extractor,
    unregister_preview_extractor,
)


class Blob:
    """Opaque sample type with a deliberately long repr."""

    def __init__(self, size: int = 1000):
        self.size = size

    def __repr__(self) -> str:
        return "Blob<" + "x" * self.size + ">"


class SubBlob(Blob):
    """Subclass of Blob."""


@pytest.fixture(autouse=True)
def clean_registry():
    """Clear registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


class TestRegistryMechanics:
    def test_register_and_get(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            return f"Blob(size={obj.size})"

        assert get_preview_extractor(Blob()) is blob_preview

    def test_get_returns_none_if_not_registered(self):
        assert get_preview_extractor(Blob()) is None

    def test_mro_lookup_covers_subclasses(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            return f"Blob(size={obj.size})"

        assert get_preview_extractor(SubBlob()) is blob_preview

    def test_unregister(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            return "preview"

        unregister_preview_extractor(Blob)
        assert get_preview_extractor(Blob()) is None

    def test_clear_registry_clears_previews(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            return "preview"

        clear_registry()
        assert get_preview_extractor(Blob()) is None

    def test_spec_define_preview_registers(self):
        @spec.define_preview(Blob)
        def blob_preview(obj, budget):
            return f"Blob(size={obj.size})"

        assert get_preview_extractor(Blob()) is blob_preview


class TestPformatIntegration:
    def test_preview_replaces_repr_fallback(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            return f"Blob(size={obj.size})"

        assert pformat(Blob(1000), max_string=100) == "Blob(size=1000)"

    def test_budget_reflects_call_site(self):
        seen: list[PreviewBudget] = []

        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            seen.append(budget)
            return "preview"

        pformat(Blob(), max_length=7, max_string=42)
        assert seen == [PreviewBudget(max_length=7, max_string=42)]

    def test_declining_falls_back_to_repr_marker(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            return None

        result = pformat(Blob(1000), max_string=100)
        # the truncated-repr fallback marker, unchanged
        assert result.startswith("Blob(repr_len=")

    def test_raising_extractor_falls_back_to_repr_marker(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            raise RuntimeError("boom")

        result = pformat(Blob(1000), max_string=100)
        assert result.startswith("Blob(repr_len=")

    def test_preview_applies_inside_containers(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            return f"Blob(size={obj.size})"

        result = pformat({"blob": Blob(1000)}, max_string=100)
        assert "Blob(size=1000)" in result

    def test_preview_applies_at_depth_limit(self):
        @register_preview_extractor(Blob)
        def blob_preview(obj, budget):
            return f"Blob(size={obj.size})"

        # Blob sits at max_depth, where the shallow formatter would
        # otherwise dump the full 1000-char repr.
        nested = {"a": {"b": Blob(1000)}}
        result = pformat(nested, max_string=100, max_depth=2)
        assert "Blob(size=1000)" in result
        assert "xxx" not in result

    def test_unregistered_types_unaffected(self):
        result = pformat(Blob(1000), max_string=100)
        assert result.startswith("Blob(repr_len=")
