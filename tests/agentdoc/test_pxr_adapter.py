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


# --------------------------------------------------------------------------- #622 follow-ups


@pytest.mark.parametrize(
    ("function", "header"),
    [
        (lambda: Gf.Dot, "def pxr.Gf.Dot(a: Gf.Vec3f, b: Gf.Vec3f) -> float:"),
        (lambda: Gf.Cross, "def pxr.Gf.Cross(a: Gf.Vec3d, b: Gf.Vec3d) -> Gf.Vec3d:"),
        (
            lambda: UsdGeom.GetStageUpAxis,
            "def pxr.UsdGeom.GetStageUpAxis(stage: Usd.Stage) -> str:",
        ),
        (
            lambda: UsdGeom.SetStageUpAxis,
            "def pxr.UsdGeom.SetStageUpAxis(stage: Usd.Stage, upAxis: str) -> bool:",
        ),
        (
            lambda: UsdGeom.GetStageMetersPerUnit,
            "def pxr.UsdGeom.GetStageMetersPerUnit(stage: Usd.Stage) -> float:",
        ),
        (
            lambda: UsdGeom.SetStageMetersPerUnit,
            "def pxr.UsdGeom.SetStageMetersPerUnit(stage: Usd.Stage, metersPerUnit: float) -> bool:",
        ),
        (
            lambda: Vt.IntArray.FromNumpy,
            "def Vt.IntArray.FromNumpy(array: numpy.ndarray) -> Vt.IntArray:",
        ),
        (
            lambda: Vt.Vec3fArray.FromNumpy,
            "def Vt.Vec3fArray.FromNumpy(array: numpy.ndarray) -> Vt.Vec3fArray:",
        ),
    ],
)
def test_curated_module_functions_and_aliases_show_their_entry(function, header):
    out = doc(function())
    assert out.startswith(header)
    assert "No docstring or signature" not in out


def test_module_functions_run_as_documented():
    stage = Usd.Stage.CreateInMemory()
    assert UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    assert UsdGeom.GetStageUpAxis(stage) == "Z"
    assert UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(0.01)
    assert Gf.Dot(Gf.Vec3f(1, 2, 3), Gf.Vec3f(1, 1, 1)) == 6
    assert Gf.Cross(Gf.Vec3d(1, 0, 0), Gf.Vec3d(0, 1, 0)) == Gf.Vec3d(0, 0, 1)
    numpy = pytest.importorskip("numpy")
    assert list(Vt.IntArray.FromNumpy(numpy.array([1, 2], dtype=numpy.int64))) == [1, 2]
    assert len(Vt.Vec3fArray.FromNumpy(numpy.zeros((2, 3), dtype=numpy.float32))) == 2


@pytest.mark.parametrize(
    ("method", "header"),
    [
        (lambda: UsdGeom.Subset.GetPrim, "def Usd.SchemaBase.GetPrim(self) -> Usd.Prim:"),
        (lambda: UsdShade.Material.GetPath, "def Usd.SchemaBase.GetPath(self) -> Sdf.Path:"),
        (lambda: Usd.Attribute.GetName, "def Usd.Object.GetName(self) -> str:"),
        (lambda: Usd.Attribute.IsValid, "def Usd.Object.IsValid(self) -> bool:"),
        (lambda: UsdGeom.Xform.AddTranslateOp, "def UsdGeom.Xformable.AddTranslateOp(self, "),
    ],
)
def test_unbound_inherited_method_shows_its_defining_class(method, header):
    out = doc(method())
    assert out.startswith(header)
    # No subclass's notes: the access does not say which subclass was named.
    for subclass_note in ("Polygon mesh", "A scene node", "Typed value on a prim"):
        assert subclass_note not in out


def test_bound_inherited_method_shows_the_instance_class():
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/M")
    assert doc(mesh.GetPrim).startswith("def UsdGeom.Mesh.GetPrim(self) -> Usd.Prim:")
    assert doc(mesh.GetPrim().GetName).startswith("def Usd.Prim.GetName(self) -> str:")
    assert doc(mesh.GetPointsAttr().GetName).startswith("def Usd.Attribute.GetName(self) -> str:")


def test_every_curated_member_exists_in_usd_core():
    from nooa.agentdoc import registry

    missing = []
    for cls, extractor in list(registry._extractor_registry.items()):
        if not cls.__module__.startswith("pxr"):
            continue
        for method in extractor(cls).methods:
            if not hasattr(cls, method.name):
                missing.append(f"{cls.__name__}.{method.name}")
    for name, extractor in list(registry._module_extractor_registry.items()):
        if name.startswith("pxr"):
            module = sys.modules[name]
            missing += [f.name for f in extractor(module).functions if not hasattr(module, f.name)]
    assert missing == []


def test_new_curated_methods_run_with_their_documented_parameter_names():
    """Keyword calls check the curated parameter names against the bindings."""
    stage = Usd.Stage.CreateInMemory()
    mesh = UsdGeom.Mesh.Define(stage, "/W/Mesh")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3, 3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
    prim = mesh.GetPrim()

    # Usd.Object metadata and customData
    assert prim.SetCustomDataByKey(keyPath="tool:id", value=7) is None
    assert prim.GetCustomDataByKey(keyPath="tool:id") == 7
    assert prim.GetCustomData()["tool"] == {"id": 7}
    assert prim.HasCustomDataKey(keyPath="tool")
    assert prim.SetDocumentation(doc="d") and prim.GetDocumentation() == "d"
    assert "documentation" in prim.GetAllMetadata()
    # Usd.TimeCode, Usd.EditTarget, Usd.Relationship, Usd.Property, Usd.Prim
    assert Usd.TimeCode.Default().IsDefault() and Usd.TimeCode(3).GetValue() == 3
    assert stage.GetEditTarget().GetLayer() == stage.GetRootLayer()
    stage.SetEditTarget(stage.GetSessionLayer())
    assert stage.GetEditTarget().GetLayer() == stage.GetSessionLayer()
    stage.SetEditTarget(stage.GetRootLayer())
    rel = prim.CreateRelationship("r")
    assert rel.AddTarget(target="/W") and rel.GetTargets() == [Sdf.Path("/W")]
    assert rel.SetTargets(targets=["/W/Mesh"]) and rel.RemoveTarget(target="/W/Mesh")
    assert rel.ClearTargets(removeSpec=True)
    attr = prim.CreateAttribute("primvars:foo", Sdf.ValueTypeNames.Float)
    assert (attr.GetNamespace(), attr.GetBaseName(), attr.IsAuthored()) == ("primvars", "foo", True)
    assert prim.GetProperty(propName="points").GetName() == "points"
    assert "points" in prim.GetPropertyNames() and prim.GetRelationships()
    assert prim.GetPrimStack()
    assert "points" in UsdGeom.Mesh.GetSchemaAttributeNames(includeInherited=True)
    # UsdGeom: Scope, Boundable, Gprim, Imageable, Primvar(sAPI), XformOp, Subset, Mesh
    assert UsdGeom.Scope.Define(stage=stage, path="/W/Looks")
    extent = UsdGeom.Boundable.ComputeExtentFromPlugins(mesh, Usd.TimeCode.Default())
    assert mesh.GetExtentAttr().Set(extent) and len(mesh.GetExtentAttr().Get()) == 2
    assert mesh.GetOrientationAttr().Get() == UsdGeom.Tokens.rightHanded
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded)
    assert mesh.GetPurposeAttr().Get() == UsdGeom.Tokens.default_
    bbox = mesh.ComputeLocalBound(Usd.TimeCode.Default(), purpose1=UsdGeom.Tokens.default_)
    assert bbox.GetRange().GetMax() == Gf.Vec3d(1, 1, 0)
    world = mesh.ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)
    assert world.ComputeAlignedRange().GetSize() == Gf.Vec3d(1, 1, 0)
    mesh.MakeInvisible()
    assert mesh.ComputeVisibility() == UsdGeom.Tokens.invisible
    api = UsdGeom.PrimvarsAPI(prim)
    st = api.CreatePrimvar(
        name="st", typeName=Sdf.ValueTypeNames.TexCoord2fArray, interpolation="faceVarying"
    )
    assert st.Set(value=[(0, 0)] * 6) and st.SetIndices(indices=Vt.IntArray([0] * 6))
    assert (st.GetInterpolation(), st.GetPrimvarName(), st.IsIndexed()) == (
        "faceVarying",
        "st",
        True,
    )
    assert len(st.ComputeFlattened()) == 6 and list(st.GetIndices()) == [0] * 6
    assert api.GetPrimvar(name="st") and api.HasPrimvar(name="st") and api.GetPrimvars()
    assert mesh.GetFaceVaryingLinearInterpolationAttr().Get() == "cornersPlus1"
    xform = UsdGeom.Xform.Define(stage, "/W/X")
    op = xform.AddTranslateOp()
    assert op.Set(value=Gf.Vec3d(1, 2, 3)) and op.GetOpName() == "xformOp:translate"
    subset = UsdGeom.Subset.CreateGeomSubset(
        mesh, "s", UsdGeom.Tokens.face, Vt.IntArray([0, 1]), UsdShade.Tokens.materialBind
    )
    assert UsdGeom.Subset.ValidateSubsets(
        subsets=[subset], elementCount=2, familyType=UsdGeom.Tokens.partition
    )[0]
    # UsdShade: ConnectableAPI, Input, Output, MaterialBindingAPI
    material = UsdShade.Material.Define(stage, "/W/Looks/M")
    shader = UsdShade.Shader.Define(stage, "/W/Looks/M/S")
    texture = UsdShade.Shader.Define(stage, "/W/Looks/M/T")
    texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    color = shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f)
    assert UsdShade.ConnectableAPI.ConnectToSource(
        shadingAttr=color, source=texture.ConnectableAPI(), sourceName="rgb"
    )
    assert UsdShade.ConnectableAPI.HasConnectedSource(shadingAttr=color)
    sources, invalid = UsdShade.ConnectableAPI.GetConnectedSources(color)
    assert sources[0].sourceName == "rgb" and invalid == []
    assert UsdShade.ConnectableAPI.DisconnectSource(color) and not color.HasConnectedSource()
    assert color.ConnectToSource(source=texture.ConnectableAPI(), sourceName="rgb")
    surface = material.CreateSurfaceOutput()
    assert surface.ConnectToSource(shader.ConnectableAPI(), "surface")
    assert surface.HasConnectedSource()
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
    materials, rels = UsdShade.MaterialBindingAPI.ComputeBoundMaterials(prims=[prim])
    assert materials[0].GetPath() == material.GetPath()
    # Sdf
    src, dst = Sdf.Layer.CreateAnonymous(), Sdf.Layer.CreateAnonymous()
    Sdf.CreatePrimInLayer(src, "/A")
    assert Sdf.CopySpec(src, "/A", dst, "/B") and dst.GetPrimAtPath("/B")
    assert stage.GetRootLayer().GetCompositionAssetDependencies() == []


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        (
            "UsdGeom.Subset.CreateSubset",
            "UsdGeom.Subset has no CreateSubset; use UsdGeom.Subset.CreateGeomSubset(mesh, name, "
            "UsdGeom.Tokens.face, indices, UsdShade.Tokens.materialBind) - "
            "doc(UsdGeom.Subset.CreateGeomSubset)",
        ),
        ("mesh.CreateGeomSubset", "UsdGeom.Mesh has no CreateGeomSubset; use UsdGeom.Subset."),
        ("UsdGeom.GeomSubset", "UsdGeom has no GeomSubset; the class is UsdGeom.Subset"),
        ("UsdGeom.MaterialBindingAPI", "UsdGeom has no MaterialBindingAPI; use UsdShade."),
        ("Usd.Mesh", "Usd has no Mesh; use UsdGeom.Mesh"),
    ],
)
def test_invented_names_raise_an_attribute_error_naming_the_real_api(expression, expected):
    from nooa.errors.formatting import format_error_for_llm

    stage = Usd.Stage.CreateInMemory()  # keep the stage alive while using its prims
    mesh = UsdGeom.Mesh.Define(stage, "/M")
    owner, name = expression.rsplit(".", 1)
    target = mesh if owner == "mesh" else _namespace()[owner.split(".")[0]]
    for part in owner.split(".")[1:]:
        target = getattr(target, part)
    with pytest.raises(AttributeError) as info:
        getattr(target, name)
    hint = format_error_for_llm(info.value).rsplit("\n\nHint: ", 1)[1]
    assert hint.startswith(expected)


def test_attribute_hints_name_only_missing_attributes_and_real_replacements():
    import re

    hints = adapter_module()._ATTRIBUTE_HINTS
    namespace = _namespace()
    for owner, name, hint in hints:
        assert not hasattr(owner, name), name
        for call in re.findall(r"\b((?:Usd|UsdGeom|UsdShade|Vt)\.[\w.]+)", hint):
            target = namespace[call.split(".")[0]]
            for part in call.split(".")[1:]:
                if part in (name, "Tokens") or not part:
                    break
                assert hasattr(target, part), (hint, call)
                target = getattr(target, part)


def adapter_module():
    import nooa.agentdoc.adapters.pxr as module

    return module
