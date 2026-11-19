# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pxr (OpenUSD) doc() adapter: curated views whose recipes really run."""

from __future__ import annotations

import builtins
import contextlib
import importlib
import io
import sys
import textwrap

import pytest

pxr = pytest.importorskip("pxr")

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt  # noqa: E402

from nooa.agentdoc import doc  # noqa: E402
from nooa.agentdoc.adapters import register_all  # noqa: E402


@pytest.fixture(autouse=True)
def adapter():
    # Reload to re-run the registrations even if a sibling test cleared the registry.
    import nooa.agentdoc.adapters.pxr as _pxr_adapter

    return importlib.reload(_pxr_adapter)


def _run(code: str, namespace: dict) -> str:
    """Execute documented recipe code and return what it printed."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        builtins.exec(code, namespace)  # noqa: S102 - running our own documented recipe
    return out.getvalue()


def _make_mesh_stage(path: str):
    """A file-backed stage with a two-triangle mesh and two UsdPreviewSurface materials."""
    stage = Usd.Stage.CreateNew(path)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3, 3]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 0, 2, 3]))
    materials = {}
    for name, color in (("Red", (1, 0, 0)), ("Blue", (0, 0, 1))):
        mat = UsdShade.Material.Define(stage, f"/World/Looks/{name}")
        sh = UsdShade.Shader.Define(stage, f"/World/Looks/{name}/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        materials[name] = mat
    return stage, mesh, materials


def _namespace(**extra) -> dict:
    return {
        "Usd": Usd,
        "UsdGeom": UsdGeom,
        "UsdShade": UsdShade,
        "Sdf": Sdf,
        "Gf": Gf,
        "Vt": Vt,
        **extra,
    }


class TestSubsetDoc:
    def test_create_geom_subset_signature_has_family_name(self):
        out = doc(UsdGeom.Subset)
        line = next(ln for ln in out.splitlines() if "def CreateGeomSubset(" in ln)
        assert "familyName: str = ''" in line
        assert "indices: Vt.IntArray" in line

    def test_doc_contains_recipes(self, adapter):
        out = doc(UsdGeom.Subset)
        for recipe in (adapter.MATERIAL_SUBSET_RECIPE, adapter.VERIFY_SUBSETS_RECIPE):
            assert textwrap.indent(recipe, " " * 8) in out
        assert "UsdShade.Tokens.materialBind" in out
        assert "SetDisplayName" in out

    def test_recipe_executes_and_reads_back_after_export(self, adapter, tmp_path):
        stage, mesh, mats = _make_mesh_stage(str(tmp_path / "scene.usda"))
        for subset_name, display_name, faces, mat in (
            ("wheel", "Front wheel", [0], mats["Red"]),
            ("body", "Car body", [1], mats["Blue"]),
        ):
            _run(
                adapter.MATERIAL_SUBSET_RECIPE,
                _namespace(
                    mesh=mesh,
                    material=mat,
                    subset_name=subset_name,
                    display_name=display_name,
                    face_indices=faces,
                ),
            )
        exported = str(tmp_path / "exported.usda")
        assert stage.Export(exported)

        # Read back from a fresh, independent layer the way a consumer would.
        fresh = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(exported))
        fresh_mesh = UsdGeom.Mesh.Get(fresh, "/World/Mesh")
        printed = _run(adapter.VERIFY_SUBSETS_RECIPE, _namespace(mesh=fresh_mesh))
        assert printed.splitlines() == [
            "wheel Front wheel materialBind face [0] /World/Looks/Red",
            "body Car body materialBind face [1] /World/Looks/Blue",
        ]
        subsets = UsdGeom.Subset.GetGeomSubsets(
            fresh_mesh, UsdGeom.Tokens.face, UsdShade.Tokens.materialBind
        )
        for s in subsets:
            bound, _rel = UsdShade.MaterialBindingAPI(s.GetPrim()).ComputeBoundMaterial()
            assert bound and bound.GetPath().GetParentPath() == Sdf.Path("/World/Looks")
            # display name is metadata, not a custom attribute
            assert not s.GetPrim().GetAttribute("displayName")
        assert UsdShade.MaterialBindingAPI(fresh_mesh.GetPrim()).GetMaterialBindSubsets()

    def test_subset_without_family_name_is_not_a_material_subset(self):
        # Why the doc insists on familyName: the default '' hides the subset from bindings.
        stage = Usd.Stage.CreateInMemory()
        mesh = UsdGeom.Mesh.Define(stage, "/M")
        UsdGeom.Subset.CreateGeomSubset(mesh, "s", UsdGeom.Tokens.face, Vt.IntArray([0]))
        assert (
            UsdGeom.Subset.GetGeomSubsets(mesh, UsdGeom.Tokens.face, UsdShade.Tokens.materialBind)
            == []
        )


class TestDocumentedPitfalls:
    def test_open_reuses_layer_until_reload(self, tmp_path):
        path = str(tmp_path / "a.usda")
        stage = Usd.Stage.CreateNew(path)
        prim = stage.DefinePrim("/A")
        prim.SetDisplayName("old")
        stage.GetRootLayer().Save()
        with open(path) as f:
            text = f.read()
        with open(path, "w") as f:
            f.write(text.replace('"old"', '"new"'))
        reopened = Usd.Stage.Open(path)
        assert reopened.GetPrimAtPath("/A").GetDisplayName() == "old"  # stale: shared layer
        fresh = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(path))
        assert fresh.GetPrimAtPath("/A").GetDisplayName() == "new"
        reopened.Reload()
        assert reopened.GetPrimAtPath("/A").GetDisplayName() == "new"

    def test_export_leaves_root_layer_unsaved(self, tmp_path):
        stage = Usd.Stage.CreateNew(str(tmp_path / "b.usda"))
        UsdGeom.Xform.Define(stage, "/X")
        assert stage.Export(str(tmp_path / "copy.usda"))
        assert stage.GetRootLayer().dirty
        assert stage.GetRootLayer().Save()
        assert not stage.GetRootLayer().dirty

    def test_removed_prim_handle_expires(self):
        stage = Usd.Stage.CreateInMemory()
        prim = stage.DefinePrim("/P")
        stage.RemovePrim("/P")
        assert not prim.IsValid()
        with pytest.raises(RuntimeError, match="expired"):
            prim.GetName()
        assert not stage.GetPrimAtPath("/P")

    def test_invented_names_do_not_exist(self):
        assert not hasattr(UsdGeom.Subset, "GetSubsetFamilyNameAttr")
        assert not hasattr(Usd.Prim, "CreateDisplayNameAttr")
        assert not hasattr(UsdGeom.Mesh, "CreateSubset")
        assert not hasattr(UsdGeom, "Material")
        assert not hasattr(UsdGeom, "MaterialBindingAPI")
        assert not hasattr(Usd, "Scope")

    def test_other_docstring_snippets_run(self):
        stage = Usd.Stage.CreateInMemory()
        # UsdGeom.Mesh / Xformable / Material / Gf.Matrix4d / Vt snippets
        mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
        mesh.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0)]))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
        assert mesh.GetFaceCount() == 1
        xf = UsdGeom.Xform.Define(stage, "/World/Obj")
        xf.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
        xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))
        xf.AddScaleOp().Set(Gf.Vec3f(2, 2, 2))
        world = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        assert world.ExtractTranslation() == Gf.Vec3d(1, 2, 3)
        m = Gf.Matrix4d(1.0)
        m.SetTranslate(Gf.Vec3d(1, 2, 3))
        assert m.Transform(Gf.Vec3d(0, 0, 0)) == Gf.Vec3d(1, 2, 3)
        mat = UsdShade.Material.Define(stage, "/World/Looks/Red")
        sh = UsdShade.Shader.Define(stage, "/World/Looks/Red/Shader")
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        assert mat.ComputeSurfaceSource()[0].GetPath() == sh.GetPath()
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        assert (
            UsdShade.MaterialBindingAPI(mesh.GetPrim()).ComputeBoundMaterial()[0].GetPath()
            == mat.GetPath()
        )
        cache = Usd.StageCache()
        with Usd.StageCacheContext(cache):
            pass
        assert cache.Size() == 0
        np = pytest.importorskip("numpy")
        assert list(Vt.IntArray.FromNumpy(np.array([1, 2, 3]))) == [1, 2, 3]
        assert len(Vt.Vec3fArray.FromNumpy(np.zeros((2, 3), np.float32))) == 2
        with pytest.raises(TypeError):
            Vt.IntArray(np.array([1, 2, 3], dtype=np.int64))


COVERED_TYPES = {
    Usd.Stage: "Usd.Stage",
    Usd.Prim: "Usd.Prim",
    Usd.Attribute: "Usd.Attribute",
    Usd.StageCache: "Usd.StageCache",
    UsdGeom.Mesh: "UsdGeom.Mesh",
    UsdGeom.Subset: "UsdGeom.Subset",
    UsdGeom.Tokens: "UsdGeom.Tokens",
    UsdGeom.Xformable: "UsdGeom.Xformable",
    UsdGeom.Xform: "UsdGeom.Xform",
    UsdShade.Material: "UsdShade.Material",
    UsdShade.Shader: "UsdShade.Shader",
    UsdShade.MaterialBindingAPI: "UsdShade.MaterialBindingAPI",
    UsdShade.Tokens: "UsdShade.Tokens",
    Sdf.Path: "Sdf.Path",
    Sdf.Layer: "Sdf.Layer",
    Sdf.ValueTypeNames: "Sdf.ValueTypeNames",
    Vt.IntArray: "Vt.IntArray",
    Vt.Vec3fArray: "Vt.Vec3fArray",
    Gf.Vec3f: "Gf.Vec3f",
    Gf.Vec3d: "Gf.Vec3d",
    Gf.Matrix4d: "Gf.Matrix4d",
}


class TestCoverage:
    def test_register_all_includes_pxr(self):
        assert "pxr" in register_all()

    @pytest.mark.parametrize("cls", list(COVERED_TYPES), ids=list(COVERED_TYPES.values()))
    def test_type_doc_is_curated(self, cls):
        out = doc(cls)
        assert out.startswith(f"class {COVERED_TYPES[cls]}:\n")
        assert '"""' in out  # curated docstring present (usd-core has none)
        assert "def Get(...)" not in out and "(...):" not in out  # no bare compiled stubs

    @pytest.mark.parametrize(
        "module", [pxr, Usd, UsdGeom, UsdShade, Sdf, Gf, Vt], ids=lambda m: m.__name__
    )
    def test_module_doc_is_curated(self, module):
        out = doc(module)
        assert out.startswith(f"# {module.__name__}\n")
        assert len(out.splitlines()) < 60
        assert "(...):" not in out  # no bare compiled stubs

    def test_xformable_subclass_shows_own_name(self):
        assert doc(UsdGeom.Sphere).startswith("class UsdGeom.Sphere:")

    def test_instance_doc_uses_curated_view(self):
        stage = Usd.Stage.CreateInMemory()
        assert "SetDisplayName" in doc(stage.DefinePrim("/A"))


def test_adapter_import_without_pxr_fails_only_inside_register_all(monkeypatch):
    """Without pxr, importing the adapter raises ImportError and register_all() skips it."""
    for name in [m for m in sys.modules if m == "pxr" or m.startswith("pxr.")]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "nooa.agentdoc.adapters.pxr", raising=False)
    real_import = builtins.__import__

    def _no_pxr(name, *args, **kwargs):
        if name == "pxr" or name.startswith("pxr."):
            raise ImportError("No module named 'pxr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_pxr)
    with pytest.raises(ImportError):
        importlib.import_module("nooa.agentdoc.adapters.pxr")
    assert "pxr" not in register_all()


def test_doc_loads_the_pxr_adapter_without_an_explicit_import():
    """An agent cell that imports pxr and calls doc() gets the curated view."""
    import subprocess
    import sys

    code = (
        "from pxr import UsdGeom\n"
        "from nooa.agentdoc import doc\n"
        "text = doc(UsdGeom.Subset)\n"
        "assert 'familyName' in text and 'materialBind' in text, text[:400]\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_doc_on_curated_boost_method_shows_signature_and_pitfalls():
    out = doc(UsdGeom.Subset.CreateGeomSubset)
    assert "No docstring or signature" not in out
    assert out.startswith("def UsdGeom.Subset.CreateGeomSubset(geom: UsdGeom.Imageable, ")
    params = ["geom", "subsetName", "elementType", "indices", "familyName", "familyType"]
    positions = [out.index(f"{name}: ") for name in params]
    assert positions == sorted(positions)
    assert "'materialBind' is never an elementType" in out
    assert "UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)" in out
    assert out.rstrip().endswith('Other members: doc(UsdGeom.Subset)\n    """'.rstrip())


def test_doc_on_get_geom_subsets_shows_its_curated_entry():
    out = doc(UsdGeom.Subset.GetGeomSubsets)
    assert out.startswith(
        "def UsdGeom.Subset.GetGeomSubsets(geom: UsdGeom.Imageable, elementType: str = '', "
        "familyName: str = '') -> list[UsdGeom.Subset]:"
    )
    assert "Child subsets filtered by elementType and familyName" in out
    assert "Other members: doc(UsdGeom.Subset)" in out


def test_doc_on_curated_method_bound_to_an_instance():
    stage = Usd.Stage.CreateInMemory()
    out = doc(stage.DefinePrim)
    assert out.startswith(
        "def Usd.Stage.DefinePrim(self, path: Sdf.Path | str, typeName: str = '')"
    )
    assert "Other members: doc(Usd.Stage)" in out


def test_concise_doc_on_curated_method_shows_only_its_line():
    out = doc(UsdGeom.Subset.CreateGeomSubset, concise=True)
    assert "elementType: str" in out
    assert "never an elementType" not in out


def test_uncurated_compiled_function_keeps_the_compiled_note():
    out = doc(Sdf.Path.IsAbsolutePath)
    assert "No docstring or signature is available for this compiled function." in out
    assert "doc(Path)" in out
