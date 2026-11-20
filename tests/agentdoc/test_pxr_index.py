# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pxr docs index: OpenUSD header docs in doc(), search and did-you-mean hints."""

from __future__ import annotations

import pytest

from nooa.agentdoc.adapters import pxr_index
from nooa.agentdoc.adapters.pxr_index import available_versions, choose_version


def test_both_openusd_indexes_ship():
    assert {(25, 11), (26, 8)} <= set(available_versions())
    for version in ((25, 11), (26, 8)):
        index = pxr_index.read_index_file(
            pxr_index.DATA_DIR / f"pxr-{pxr_index._label(version)}.json.gz"
        )
        assert len(index) > 5000
        entry = index.get("UsdGeom.Subset.CreateGeomSubset")
        assert entry.kind == "staticmethod"
        assert entry.signature.startswith(
            "(geom: UsdGeom.Imageable, subsetName: str, elementType: str"
        )
        assert entry.summary.startswith("Creates a new GeomSubset below the given geom")
        assert entry.source.startswith("usd/usdGeom/subset.h:")
        assert "\\p" not in entry.text and "\\ref" not in entry.text


def test_choose_version_prefers_exact_then_nearest():
    available = [(25, 11), (26, 8)]
    assert choose_version((26, 8), available) == ((26, 8), "")
    chosen, note = choose_version((26, 5), available)
    assert chosen == (26, 8)
    assert note == "Docs index: OpenUSD 26.08; installed usd-core is 26.5, so details may differ."
    assert choose_version((25, 8), available)[0] == (25, 11)
    assert choose_version(None, available)[0] == (26, 8)
    assert choose_version((26, 8), []) == (None, "")


pxr = pytest.importorskip("pxr")

from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils  # noqa: E402

from nooa.agentdoc import doc  # noqa: E402
from nooa.agentdoc.core import attribute_error_hint  # noqa: E402
from nooa.agentdoc.docindex import get_doc_index  # noqa: E402


@pytest.fixture(autouse=True)
def installed_index():
    # Sibling tests clear the registry; re-register the resolver and hint provider.
    pxr_index.install()


def _installed():
    version = Usd.GetVersion()
    return version[1], version[2]


def test_index_matches_installed_usd_core_or_says_so():
    index = get_doc_index("pxr")
    if _installed() in available_versions():
        assert index.version == pxr_index._label(_installed())
        assert index.note == ""
    else:
        assert "installed usd-core is" in index.note


def test_doc_shows_index_text_for_undocumented_function():
    out = doc(UsdUtils.ComputeAllDependencies)
    assert out.startswith("def UsdUtils.ComputeAllDependencies(assetPath: Sdf.AssetPath")
    assert "Recursively computes all the dependencies of the given asset" in out
    assert "Source: OpenUSD " in out and "usd/usdUtils/dependencies.h:" in out
    assert "No docstring or signature is available" not in out


def test_doc_shows_index_text_for_bound_method():
    stage = Usd.Stage.CreateInMemory()
    out = doc(stage.GetUsedLayers)
    assert out.startswith("def Usd.Stage.GetUsedLayers(self, includeClipLayers: bool = True)")
    assert "layers currently consumed by this stage" in out


def test_doc_of_uncurated_class_uses_index_for_class_and_methods():
    out = doc(UsdGeom.BBoxCache)
    assert "Caches bounds by recursively computing and aggregating bounds" in out
    assert "def ComputeWorldBound(self, prim: Usd.Prim)" in out
    assert "Compute the bound of the given prim in world space" in out


def test_curated_method_keeps_its_page_and_gains_the_official_description():
    out = doc(UsdGeom.Subset.CreateGeomSubset)
    curated = out.index("Static. Create (or update) geom.GetPath()/subsetName")
    official = out.index("Official description (OpenUSD ")
    assert curated < official < out.index("Other members: doc(UsdGeom.Subset)")
    assert "Creates a new GeomSubset below the given geom" in out


def test_search_finds_the_api_for_known_failure_cases():
    out = doc.search("create a face subset for material binding")
    assert "UsdShade.MaterialBindingAPI.CreateMaterialBindSubset(" in out
    out = doc.search("family type nonOverlapping validate subsets")
    assert "UsdGeom.Subset.SetFamilyType(" in out and "UsdGeom.Subset.ValidateFamily(" in out
    out = doc.search("all files a USD asset depends on")
    assert "Sdf.Layer.GetCompositionAssetDependencies(" in out
    assert out.splitlines()[-1].startswith("Full text: doc(<name>), e.g. doc(")
    assert len(out.splitlines()) <= 8 + 2


def _hint(code):
    with pytest.raises(AttributeError) as error:
        code()
    return attribute_error_hint(error.value)


def test_did_you_mean_names_the_same_token_on_another_module():
    hint = _hint(lambda: UsdGeom.Tokens.materialBind)
    assert hint.startswith(
        "UsdGeom.Tokens has no 'materialBind'. Did you mean UsdShade.Tokens.materialBind"
    )


def test_did_you_mean_finds_close_names_on_modules_and_classes():
    assert "Did you mean UsdGeom.BBoxCache" in _hint(lambda: UsdGeom.BBoxCompute)
    assert "UsdShade.MaterialBindingAPI.ComputeBoundMaterial" in _hint(
        lambda: UsdShade.MaterialBindingAPI.GetBoundMaterial
    )
    assert "Sdf.Layer.GetExternalAssetDependencies" in _hint(
        lambda: Sdf.Layer.GetExternalDependencies
    )
    assert "UsdShade.Output" in _hint(lambda: Usd.ShadeOutput)


def test_generic_suggestions_for_a_curated_hint_name():
    # UsdGeom.Subset.CreateSubset has a curated hint; the generic provider agrees.
    names = pxr_index.suggest("UsdGeom.Subset", dir(UsdGeom.Subset), "CreateSubset")
    assert names[0] == "UsdGeom.Subset.CreateGeomSubset"


def test_did_you_mean_on_an_instance():
    stage = Usd.Stage.CreateInMemory()
    hint = _hint(lambda: stage.GetExternalDependencies)
    assert "Sdf.Layer.GetExternalAssetDependencies" in hint
