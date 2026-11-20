# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Docs indexes: registration, BM25 search, package ranking, doc() fallback, hints."""

from __future__ import annotations

import subprocess
import sys

import pytest

from nooa.agentdoc import doc, docindex
from nooa.agentdoc.docindex import (
    DocEntry,
    DocIndex,
    get_doc_index,
    lookup_doc,
    name_tokens,
    register_doc_entries,
    register_doc_index,
    register_doc_resolver,
    search_doc_entries,
    search_docs,
    text_tokens,
    unregister_doc_index,
)
from nooa.agentdoc.registry import (
    get_attribute_hint,
    register_attribute_hint,
    register_attribute_hint_provider,
    unregister_attribute_hint,
    unregister_attribute_hint_provider,
)

GEO = [
    DocEntry("geo.Mesh", "class", "()", "A polygon mesh.", "A polygon mesh made of faces."),
    DocEntry(
        "geo.Subset.CreateFaceSubset",
        "staticmethod",
        "(mesh: geo.Mesh, name: str, indices: list[int]) | (mesh: geo.Mesh, name: str)",
        "Create a subset of mesh faces.",
        "Create a subset of mesh faces.\n\nUse it to bind materials per face.",
        "geo/subset.h:10",
    ),
    DocEntry("geo.Subset.ValidateFamily", "staticmethod", "(mesh: geo.Mesh)", "Check a family."),
    DocEntry("geo.Tokens.face", "token", " = 'face'", "Element type for face subsets."),
]


@pytest.fixture
def geo():
    register_doc_entries("geo", GEO, version="1.0", source="geo C++ docs", replace=True)
    yield get_doc_index("geo")
    unregister_doc_index("geo")
    unregister_doc_index("geowrap")


def test_tokens_split_names_and_stem_words():
    assert name_tokens("UsdGeom.Subset.CreateGeomSubset") == [
        "usd",
        "geom",
        "subset",
        "creat",
        "geom",
        "subset",
    ]
    assert text_tokens("the dependencies") == text_tokens("depends")
    assert text_tokens("validates families") == text_tokens("validate family")


def test_search_ranks_by_keywords(geo):
    hits = search_doc_entries("create face subsets for materials", packages=["geo"])
    assert hits[0].entry.name == "geo.Subset.CreateFaceSubset"
    out = search_docs("create face subsets", packages=["geo"], limit=2)
    lines = out.splitlines()
    assert lines[0] == (
        "geo.Subset.CreateFaceSubset(mesh: geo.Mesh, name: str, indices: list[int]) … "
        "— Create a subset of mesh faces."
    )
    assert len(lines) == 3  # two hits and the doc() hint
    assert lines[-1] == "Full text: doc(<name>), e.g. doc(geo.Subset.CreateFaceSubset)."


def test_search_without_match_says_so(geo):
    assert search_docs("zebra", packages=["geo"]) == (
        "No documentation matches 'zebra' (indexes searched: geo)."
    )


def test_listed_packages_rank_in_order(geo):
    register_doc_entries(
        "geowrap",
        [
            {
                "name": "self.geo.face_subset",
                "signature": "(mesh, faces)",
                "summary": "Make a face subset (wrapper).",
            }
        ],
    )
    hits = search_doc_entries("face subset", packages=["geowrap", "geo"])
    assert [h.package for h in hits][:2] == ["geowrap", "geo"]
    assert hits[0].entry.name == "self.geo.face_subset"
    reverse = search_doc_entries("face subset", packages=["geo", "geowrap"])
    assert reverse[-1].package == "geowrap"


def test_doc_search_searches_every_index(geo):
    out = doc.search("validate family")
    assert "geo.Subset.ValidateFamily(mesh: geo.Mesh) — Check a family." in out


def test_register_entries_merges_and_replaces(geo):
    register_doc_entries("geo", [DocEntry("geo.Mesh", "class", "", "Replaced.")])
    assert get_doc_index("geo").get("geo.Mesh").summary == "Replaced."
    assert len(get_doc_index("geo")) == 4
    register_doc_entries("geo", [DocEntry("geo.Only", "class")], replace=True)
    assert len(get_doc_index("geo")) == 1
    with pytest.raises(ValueError):
        register_doc_entries("geo", [{"summary": "no name"}])


def test_lazy_loader_runs_once():
    calls = []

    def load():
        calls.append(1)
        return DocIndex("lazy", [DocEntry("lazy.f", "function", "()", "Lazy f.")])

    register_doc_index("lazy", load)
    try:
        assert calls == []
        assert get_doc_index("lazy").get("lazy.f").summary == "Lazy f."
        get_doc_index("lazy")
        assert calls == [1]
    finally:
        unregister_doc_index("lazy")


class _Compiled:
    """Stand-in for a compiled function: resolved by a registered resolver."""


def test_resolver_maps_objects_to_entries(geo):
    marker = _Compiled()
    register_doc_resolver(
        __name__, lambda obj: ("geo", "geo.Subset.CreateFaceSubset") if obj is marker else None
    )
    try:
        index, entry = lookup_doc(marker)
        assert entry.name == "geo.Subset.CreateFaceSubset"
        text = docindex.entry_docstring(index, entry)
        assert text.splitlines()[0] == "Create a subset of mesh faces."
        assert "Overloads:\n    (mesh: geo.Mesh, name: str, indices: list[int])" in text
        assert text.endswith("Source: geo C++ docs, geo/subset.h:10")
        info = docindex.indexed_callable_info(marker)
        assert info.signature == "(mesh: geo.Mesh, name: str, indices: list[int])"
        assert info.docstring.startswith("Static. Create a subset")
        assert lookup_doc(object()) is None
    finally:
        docindex._resolvers[:] = [(p, r) for p, r in docindex._resolvers if p != __name__]


def test_index_note_is_shown_with_entries(geo):
    geo.note = "Docs index: geo 1.0; installed geo is 2.0, so details may differ."
    out = search_docs("mesh", packages=["geo"])
    assert geo.note in out.splitlines()


def test_official_addendum_only_when_it_adds_text(geo):
    marker = _Compiled()
    register_doc_resolver(
        __name__, lambda obj: ("geo", "geo.Subset.CreateFaceSubset") if obj is marker else None
    )
    try:
        assert docindex.official_addendum(marker, "Create a subset of mesh faces. Curated.") is None
        added = docindex.official_addendum(marker, "Curated recipe only.")
        assert added.startswith("Official description (geo C++ docs, geo/subset.h:10):")
        assert "Create a subset of mesh faces." in added
    finally:
        docindex._resolvers[:] = [(p, r) for p, r in docindex._resolvers if p != __name__]


class Holder:
    real_name = 1


def test_attribute_hint_provider_runs_after_static_hints():
    key = f"{__name__}.Holder"
    seen = []

    def provider(obj, name, owner_key):
        seen.append((name, owner_key))
        return f"{owner_key} has no {name}; try real_name"

    register_attribute_hint_provider(__name__, provider)
    register_attribute_hint(Holder, "static_guess", "static hint")
    try:
        with pytest.raises(AttributeError) as error:
            Holder.real_nam  # noqa: B018
        assert get_attribute_hint(error.value) == f"{key} has no real_nam; try real_name"
        with pytest.raises(AttributeError) as error:
            Holder.static_guess  # noqa: B018
        assert get_attribute_hint(error.value) == "static hint"
        assert seen == [("real_nam", key)]
    finally:
        unregister_attribute_hint_provider(__name__)
        unregister_attribute_hint(Holder, "static_guess")


def test_importing_agentdoc_loads_no_index_and_no_pxr():
    code = (
        "import sys\n"
        "from nooa.agentdoc import doc\n"
        "doc(len)\n"
        "assert 'pxr' not in sys.modules, 'pxr imported'\n"
        "assert 'nooa.agentdoc.docindex' not in sys.modules, 'docindex imported'\n"
        "assert 'nooa.agentdoc.adapters.pxr_index' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
