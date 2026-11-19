# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc adapter for OpenUSD (``pxr``): curated doc() views for the Python API.

Import to register:

    import nooa.agentdoc.adapters.pxr

usd-core ships its Boost.Python bindings without docstrings, so ``doc(UsdGeom.Subset)``
would otherwise list bare ``def Name(...)`` stubs. This adapter replaces that with the
correct idioms, recipes and the real Python signatures (taken from the C++ signatures
that Boost.Python reports, checked against usd-core 25.11). Doc extraction never calls
pxr functions; it only returns the static text below.

Signature conventions: ``self`` marks instance methods; methods without ``self`` are
static (call them on the class). TfToken parameters are plain ``str`` (use the
``*.Tokens`` constants); Sdf.Path parameters also accept a ``str`` path.
"""

import pxr
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from nooa.agentdoc import spec
from nooa.agentdoc.ext import CallableInfo, ModuleInfo, TypeInfo, register_attribute_hint

# Executable recipe (tests run it): bind a material to a set of mesh faces.
# Needs: mesh (UsdGeom.Mesh), material (UsdShade.Material), subset_name (valid
# identifier), display_name (any text), face_indices (list[int] of face indices).
MATERIAL_SUBSET_RECIPE = """\
subset = UsdGeom.Subset.CreateGeomSubset(
    mesh, subset_name, UsdGeom.Tokens.face, Vt.IntArray(face_indices),
    UsdShade.Tokens.materialBind)  # familyName='materialBind' is required for bindings
subset.GetPrim().SetDisplayName(display_name)  # displayName metadata, not an attribute
UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)"""

# Executable check (tests run it): read the subsets back the way a consumer does.
VERIFY_SUBSETS_RECIPE = """\
for s in UsdGeom.Subset.GetGeomSubsets(mesh, UsdGeom.Tokens.face, UsdShade.Tokens.materialBind):
    prim = s.GetPrim()
    mat_path = UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath()
    print(prim.GetName(), prim.GetDisplayName(), s.GetFamilyNameAttr().Get(),
          s.GetElementTypeAttr().Get(), list(s.GetIndicesAttr().Get()), mat_path)"""


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


def _m(name: str, signature: str, return_type: str, docstring: str) -> CallableInfo:
    return CallableInfo(
        name=name, signature=signature, return_type=return_type, docstring=docstring
    )


def _s(name: str, signature: str, return_type: str, docstring: str) -> CallableInfo:
    """A static method: called on the class, no ``self``."""
    return _m(name, signature, return_type, "Static. " + docstring)


def _register(target, name: str, docstring: str, methods=(), fields=()) -> None:
    def _extract(cls_or_instance) -> TypeInfo:
        return TypeInfo(
            name=name,
            base=None,
            fields=list(fields),
            methods=list(methods),
            docstring=docstring,
        )

    spec.define_doc(target)(_extract)


def _register_module(module, docstring: str, classes, functions=()) -> None:
    def _extract(mod) -> ModuleInfo:
        return ModuleInfo(
            name=module.__name__,
            docstring=docstring,
            functions=list(functions),
            classes=list(classes),
        )

    spec.define_doc(module)(_extract)


_T = "Usd.TimeCode = Usd.TimeCode.Default()"
_DEF = "defaultValue: object = None, writeSparsely: bool = False"

# --------------------------------------------------------------------------- modules


@spec.define_doc(pxr)
def _pxr_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="pxr",
        docstring=(
            "OpenUSD Python bindings. Import each submodule you use explicitly:\n"
            "    from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf, Vt\n"
            "Using UsdShade (or any other) without importing it raises NameError."
        ),
        functions=[],
        submodules=[
            ("pxr.Usd", "stages, prims, attributes (Usd.Stage, Usd.Prim, Usd.Attribute)"),
            ("pxr.UsdGeom", "geometry schemas: Mesh, Subset, Xform, Scope, Tokens"),
            ("pxr.UsdShade", "materials: Material, Shader, MaterialBindingAPI, Tokens"),
            (
                "pxr.Sdf",
                "paths, layers, value type names (Sdf.Path, Sdf.Layer, Sdf.ValueTypeNames)",
            ),
            ("pxr.Gf", "math values: Vec3f, Vec3d, Matrix4d, Range3d"),
            ("pxr.Vt", "typed arrays: IntArray, FloatArray, Vec3fArray, TokenArray"),
        ],
    )


_register_module(
    Usd,
    "Core scene API: a Usd.Stage composes layers into prims; prims hold attributes,\n"
    "relationships and metadata. Typed schemas (UsdGeom.Mesh, UsdShade.Material...) live\n"
    "in their own modules: there is no Usd.Scope, Usd.Mesh or Usd.Material.",
    [
        ("Stage", "open/create/save a scene; DefinePrim, GetPrimAtPath, RemovePrim, Traverse"),
        ("Prim", "a scene node: attributes, children, display name, applied schemas"),
        ("Attribute", "typed value on a prim: Get/Set (optionally per time code)"),
        ("Relationship", "targets other paths (e.g. material:binding)"),
        ("TimeCode", "Usd.TimeCode.Default() (non-animated) or a frame number"),
        ("Object", "base of Prim/Attribute/Relationship: name, path, metadata, customData"),
        ("Property", "base of Attribute and Relationship"),
        ("SchemaBase", "base of all schema classes: GetPrim(), GetPath()"),
        ("EditTarget", "stage.GetEditTarget().GetLayer(): where edits go"),
        ("StageCache", "explicit stage cache; used only inside Usd.StageCacheContext"),
        ("StageCacheContext", "with Usd.StageCacheContext(cache): Open() reuses cached stages"),
    ],
)

_register_module(
    UsdGeom,
    "Geometry schemas. Wrap a prim: UsdGeom.Mesh(prim); create: UsdGeom.Mesh.Define(stage, path).\n"
    "Materials are NOT here: use UsdShade.Material / UsdShade.MaterialBindingAPI.\n"
    "There is no UsdGeom.GeomSubset (use UsdGeom.Subset) or UsdGeom.MaterialBindingAPI.",
    [
        ("Mesh", "polygon mesh: points, faceVertexCounts, faceVertexIndices"),
        ("Subset", "GeomSubset: a named set of faces of a mesh (material assignment)"),
        ("Xform", "transformable group prim"),
        ("Xformable", "base with AddTranslateOp/AddRotateXYZOp/AddScaleOp"),
        ("XformCommonAPI", "simple SetTranslate/SetRotate/SetScale"),
        ("Scope", "untransformable grouping prim (UsdGeom.Scope, not Usd.Scope)"),
        ("Imageable", "visibility, purpose, ComputeWorldBound"),
        ("Boundable", "extent: GetExtentAttr, ComputeExtentFromPlugins"),
        ("Gprim", "displayColor, doubleSided, orientation"),
        ("PrimvarsAPI", "primvars (e.g. UVs: primvars:st)"),
        ("Primvar", "one primvar: Get/Set, interpolation, indices"),
        ("XformOp", "one transform op: Get/Set, GetOpName"),
        ("BBoxCache", "fast bounds: BBoxCache(time, [UsdGeom.Tokens.default_])"),
        ("Tokens", "string constants: face, partition, y, z, ..."),
    ],
    functions=[
        _m("GetStageUpAxis", "(stage: Usd.Stage)", "str", "'Y' or 'Z'."),
        _m(
            "SetStageUpAxis",
            "(stage: Usd.Stage, upAxis: str)",
            "bool",
            "Use UsdGeom.Tokens.y / .z.",
        ),
        _m("GetStageMetersPerUnit", "(stage: Usd.Stage)", "float", "Scene units in meters."),
        _m(
            "SetStageMetersPerUnit",
            "(stage: Usd.Stage, metersPerUnit: float)",
            "bool",
            "e.g. 0.01 for cm.",
        ),
    ],
)

_register_module(
    UsdShade,
    "Shading: Material prims contain Shader prims; MaterialBindingAPI binds materials to\n"
    "prims or to UsdGeom.Subset face sets. `from pxr import UsdShade` before use.",
    [
        ("Material", "material container; surface output connects to a Shader"),
        ("Shader", "shading node, e.g. info:id = 'UsdPreviewSurface'"),
        ("MaterialBindingAPI", "Apply(prim).Bind(material); ComputeBoundMaterial()"),
        ("Input", "shader/material input: Set(value), ConnectToSource(...)"),
        ("ConnectableAPI", "connections: ConnectToSource, HasConnectedSource"),
        ("Output", "shader/material output"),
        ("Tokens", "string constants: materialBind, surface, strongerThanDescendants, ..."),
    ],
)

_register_module(
    Sdf,
    "Scene description layer: paths, layers (files) and value type names.",
    [
        ("Path", "Sdf.Path('/World/Mesh'); most APIs also accept a str path"),
        ("Layer", "one file: FindOrOpen, OpenAsAnonymous, Save, Export, Reload"),
        ("ValueTypeNames", "attribute types for CreateAttribute/CreateInput: Float, Color3f, ..."),
        ("ValueTypeName", "type of each Sdf.ValueTypeNames entry"),
    ],
    functions=[
        _m(
            "CopySpec",
            "(srcLayer: Sdf.Layer, srcPath: Sdf.Path, dstLayer: Sdf.Layer, dstPath: Sdf.Path)",
            "bool",
            "Copy a prim/property spec and everything below it between layers (or within one); "
            "the destination parent must exist (Sdf.CreatePrimInLayer).",
        ),
    ],
)

_register_module(
    Gf,
    "Graphics math values. Suffix f=float32, d=float64, h=half, i=int.\n"
    "Points/colors: Gf.Vec3f; translations/transforms: Gf.Vec3d / Gf.Matrix4d.",
    [
        ("Vec3f", "Gf.Vec3f(x, y, z): float32 3-vector"),
        ("Vec3d", "Gf.Vec3d(x, y, z): float64 3-vector"),
        ("Matrix4d", "4x4 row-major transform; row vectors (p * M)"),
        ("Rotation", "Gf.Rotation(Gf.Vec3d(axis), degrees)"),
        ("Range3d", "axis-aligned box: GetMin(), GetMax(), GetSize()"),
        ("Quatf", "quaternion (Gf.Quatf(real, i, j, k))"),
        ("BBox3d", "box + transform from ComputeWorldBound: ComputeAlignedRange()"),
    ],
    functions=[
        _m("Dot", "(a: Gf.Vec3f, b: Gf.Vec3f)", "float", "Dot product (also Vec3d etc.)."),
        _m("Cross", "(a: Gf.Vec3d, b: Gf.Vec3d)", "Gf.Vec3d", "Cross product (also Vec3f)."),
    ],
)

_register_module(
    Vt,
    "Typed value arrays used for array attributes (int[], point3f[], ...).\n"
    "Build from lists (Vt.IntArray([0, 1])) or NumPy (Vt.IntArray.FromNumpy(arr)).\n"
    "Vt.IntArray(np_int64_array) raises TypeError: use FromNumpy or arr.tolist().",
    [
        ("IntArray", "int[] (faceVertexCounts, faceVertexIndices, subset indices)"),
        ("FloatArray", "float[]"),
        ("Vec3fArray", "point3f[] / normal3f[] / color3f[] values"),
        ("Vec2fArray", "texCoord2f[] (UVs)"),
        ("TokenArray", "token[]"),
        ("StringArray", "string[]"),
    ],
)

# --------------------------------------------------------------------------- Usd

_register(
    Usd.Stage,
    "Usd.Stage",
    "A composed scene. Keep a reference to the stage while using its prims/layers.\n"
    "    stage = Usd.Stage.CreateNew(path)   # new file; fails if that layer is already open\n"
    "    stage = Usd.Stage.Open(path)        # reuses an already-open layer, even if the file\n"
    "                                        # changed on disk: then call stage.Reload()\n"
    "    fresh = Usd.Stage.Open(Sdf.Layer.OpenAsAnonymous(path))  # independent read of disk\n"
    "    stage.GetRootLayer().Save()         # write edits to the same file (stage.Save() also)\n"
    "    stage.Export(other_path)            # write a flattened copy; root layer stays unsaved\n"
    "After RemovePrim(path), old Usd.Prim/schema objects for it are expired (RuntimeError\n"
    "'Accessed invalid expired prim'): re-fetch with GetPrimAtPath and check prim.IsValid().",
    methods=[
        _s(
            "Open",
            "(filePath: str, load: Usd.Stage.InitialLoadSet = Usd.Stage.LoadAll)",
            "Usd.Stage",
            "Open a file; also Open(rootLayer: Sdf.Layer). Missing file: Tf.ErrorException.",
        ),
        _s(
            "CreateNew",
            "(identifier: str, load: Usd.Stage.InitialLoadSet = Usd.Stage.LoadAll)",
            "Usd.Stage",
            "Create a new file-backed stage (file written on Save).",
        ),
        _s(
            "CreateInMemory",
            "(identifier: str = ..., load: Usd.Stage.InitialLoadSet = Usd.Stage.LoadAll)",
            "Usd.Stage",
            "Anonymous in-memory stage; identifier optional.",
        ),
        _m(
            "GetRootLayer",
            "(self)",
            "Sdf.Layer",
            "The root Sdf.Layer (Save/Export/Reload/identifier).",
        ),
        _m(
            "Save",
            "(self)",
            "None",
            "Save all dirty layers of the layer stack (not the session layer).",
        ),
        _m(
            "Export",
            "(self, filename: str, addSourceFileComment: bool = True, args: dict = {})",
            "bool",
            "Write the flattened composed stage to filename; does not save the root layer.",
        ),
        _m(
            "ExportToString",
            "(self, addSourceFileComment: bool = True)",
            "str",
            "Flattened stage as usda text.",
        ),
        _m("Reload", "(self)", "None", "Re-read all layers from disk, discarding unsaved edits."),
        _m(
            "DefinePrim",
            "(self, path: Sdf.Path | str, typeName: str = '')",
            "Usd.Prim",
            "Define (or get) a prim; typeName e.g. 'Xform', 'Scope', 'Mesh'.",
        ),
        _m(
            "GetPrimAtPath",
            "(self, path: Sdf.Path | str)",
            "Usd.Prim",
            "Prim at path; invalid (falsy) Usd.Prim if absent: check bool(prim) / prim.IsValid().",
        ),
        _m(
            "RemovePrim",
            "(self, path: Sdf.Path | str)",
            "bool",
            "Remove the prim spec from the edit target; existing Python handles expire.",
        ),
        _m(
            "Traverse",
            "(self)",
            "Iterable[Usd.Prim]",
            "Depth-first iteration over all active, loaded prims.",
        ),
        _m(
            "GetPseudoRoot",
            "(self)",
            "Usd.Prim",
            "The '/' prim; GetChildren() gives top-level prims.",
        ),
        _m("GetDefaultPrim", "(self)", "Usd.Prim", "The defaultPrim (invalid if unset)."),
        _m(
            "SetDefaultPrim",
            "(self, prim: Usd.Prim)",
            "None",
            "Set the root-layer defaultPrim metadata.",
        ),
        _m(
            "OverridePrim",
            "(self, path: Sdf.Path | str)",
            "Usd.Prim",
            "Author an 'over' (no type) at path.",
        ),
        _m(
            "GetEditTarget",
            "(self)",
            "Usd.EditTarget",
            "Where edits go; .GetLayer() is the target layer.",
        ),
        _m(
            "SetEditTarget",
            "(self, editTarget: Usd.EditTarget)",
            "None",
            "Redirect edits; an Sdf.Layer of the layer stack is accepted too.",
        ),
        _m("GetSessionLayer", "(self)", "Sdf.Layer", "The stage's session layer."),
    ],
)

_register(
    Usd.Prim,
    "Usd.Prim",
    "A scene node. Falsy when invalid (missing path or expired after RemovePrim).\n"
    "Display name is prim metadata: prim.SetDisplayName('Wheel'); prim.GetDisplayName().\n"
    "Do not create a custom attribute named 'displayName'; there is no CreateDisplayNameAttr.\n"
    "Wrap with a schema to use typed API: UsdGeom.Mesh(prim), UsdShade.Material(prim).",
    methods=[
        _m("IsValid", "(self)", "bool", "False for missing or expired prims (same as bool(prim))."),
        _m("GetPath", "(self)", "Sdf.Path", "Absolute path."),
        _m("GetName", "(self)", "str", "Last path element (the identifier, not the display name)."),
        _m("GetTypeName", "(self)", "str", "Schema type, e.g. 'Mesh', 'GeomSubset', 'Material'."),
        _m(
            "IsA", "(self, schemaType: type)", "bool", "Typed-schema check: prim.IsA(UsdGeom.Mesh)."
        ),
        _m(
            "HasAPI",
            "(self, schemaType: type)",
            "bool",
            "Applied-API check: prim.HasAPI(UsdShade.MaterialBindingAPI).",
        ),
        _m("GetDisplayName", "(self)", "str", "displayName metadata ('' if unset)."),
        _m(
            "SetDisplayName", "(self, name: str)", "bool", "Author displayName metadata (any text)."
        ),
        _m("ClearDisplayName", "(self)", "bool", "Remove displayName metadata."),
        _m("HasAuthoredDisplayName", "(self)", "bool", "True if displayName is authored."),
        _m("GetChildren", "(self)", "list[Usd.Prim]", "Active, loaded, defined children."),
        _m("GetChild", "(self, name: str)", "Usd.Prim", "Child by name (invalid if absent)."),
        _m("GetParent", "(self)", "Usd.Prim", "Parent prim."),
        _m("GetStage", "(self)", "Usd.Stage", "Owning stage."),
        _m(
            "GetAttribute",
            "(self, attrName: str)",
            "Usd.Attribute",
            "Attribute (invalid if absent; Get() -> None).",
        ),
        _m("GetAttributes", "(self)", "list[Usd.Attribute]", "All attributes."),
        _m("HasAttribute", "(self, attrName: str)", "bool", "Attribute exists."),
        _m(
            "CreateAttribute",
            "(self, name: str, typeName: Sdf.ValueTypeName, custom: bool = True, "
            "variability: Sdf.Variability = Sdf.VariabilityVarying)",
            "Usd.Attribute",
            "Create a (custom) attribute, e.g. typeName=Sdf.ValueTypeNames.Float.",
        ),
        _m("GetRelationship", "(self, relName: str)", "Usd.Relationship", "Relationship by name."),
        _m(
            "CreateRelationship",
            "(self, name: str, custom: bool = True)",
            "Usd.Relationship",
            "Create a relationship.",
        ),
        _m(
            "GetMetadata",
            "(self, key: str)",
            "object",
            "Metadata value, e.g. 'displayName', 'kind'.",
        ),
        _m("SetMetadata", "(self, key: str, value: object)", "bool", "Author metadata."),
        _m(
            "RemoveProperty",
            "(self, propName: str)",
            "bool",
            "Remove an attribute/relationship spec.",
        ),
        _m(
            "GetAppliedSchemas",
            "(self)",
            "list[str]",
            "Applied API schema names, e.g. ['MaterialBindingAPI'].",
        ),
        _m(
            "SetActive",
            "(self, active: bool)",
            "bool",
            "Deactivate instead of removing (non-destructive).",
        ),
        _m("GetRelationships", "(self)", "list[Usd.Relationship]", "All relationships."),
        _m(
            "GetProperty",
            "(self, propName: str)",
            "Usd.Property",
            "Attribute or relationship by name (invalid if absent).",
        ),
        _m(
            "GetPropertyNames",
            "(self, predicate: object = None)",
            "list[str]",
            "Names of all properties; predicate(name) -> bool filters.",
        ),
        _m(
            "GetPrimStack",
            "(self)",
            "list[Sdf.PrimSpec]",
            "Prim specs contributing to this prim, strongest first.",
        ),
    ],
)

_register(
    Usd.Attribute,
    "Usd.Attribute",
    "Typed value on a prim. attr.Get() returns None if unauthored with no fallback;\n"
    "attr.Set(value) converts lists/tuples (e.g. [0, 1] -> Vt.IntArray, (1, 0, 0) -> Gf.Vec3f).\n"
    "Invalid attributes (prim.GetAttribute('missing')) are falsy.",
    methods=[
        _m("Get", f"(self, time: {_T})", "object", "Value at time (None if no value)."),
        _m(
            "Set",
            f"(self, value: object, time: {_T})",
            "bool",
            "Author value (default time = non-animated).",
        ),
        _m("IsValid", "(self)", "bool", "Attribute exists."),
        _m("HasValue", "(self)", "bool", "Authored or fallback value exists."),
        _m("HasAuthoredValue", "(self)", "bool", "Authored value exists."),
        _m("GetName", "(self)", "str", "Full name, e.g. 'inputs:diffuseColor'."),
        _m("GetBaseName", "(self)", "str", "Name without namespace."),
        _m(
            "GetTypeName",
            "(self)",
            "Sdf.ValueTypeName",
            "Value type, e.g. Sdf.ValueTypeNames.IntArray.",
        ),
        _m("GetPrim", "(self)", "Usd.Prim", "Owning prim."),
        _m("GetPath", "(self)", "Sdf.Path", "Property path '/Prim.attr'."),
        _m("GetTimeSamples", "(self)", "list[float]", "Authored time codes."),
        _m("Clear", "(self)", "bool", "Remove authored default and time samples."),
        _m("Block", "(self)", "None", "Author a value block (attribute reads as no value)."),
    ],
)

_register(
    Usd.StageCache,
    "Usd.StageCache",
    "Explicit stage cache. Usd.Stage.Open only consults it inside a context:\n"
    "    cache = Usd.StageCache()\n"
    "    with Usd.StageCacheContext(cache):\n"
    "        stage = Usd.Stage.Open(path)   # same stage on repeated Open\n"
    "Plain Open() still reuses already-open layers (see Usd.Stage); use stage.Reload().",
    methods=[
        _m("Insert", "(self, stage: Usd.Stage)", "Usd.StageCache.Id", "Add a stage."),
        _m("Contains", "(self, stage: Usd.Stage)", "bool", "Also Contains(id)."),
        _m("Erase", "(self, stage: Usd.Stage)", "bool", "Drop a stage (also Erase(id))."),
        _m("Clear", "(self)", "None", "Drop all stages."),
        _m("Size", "(self)", "int", "Number of cached stages."),
        _m("GetAllStages", "(self)", "list[Usd.Stage]", "Cached stages."),
        _m(
            "FindAllMatching",
            "(self, rootLayer: Sdf.Layer)",
            "list[Usd.Stage]",
            "Stages with this root layer.",
        ),
    ],
)

# --------------------------------------------------------------------------- UsdGeom

_register(
    UsdGeom.Mesh,
    "UsdGeom.Mesh",
    "Polygon mesh. Faces: faceVertexCounts (verts per face) + faceVertexIndices into points.\n"
    "    mesh = UsdGeom.Mesh.Define(stage, '/World/Mesh')\n"
    "    mesh.CreatePointsAttr(Vt.Vec3fArray([(0, 0, 0), (1, 0, 0), (1, 1, 0)]))\n"
    "    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))\n"
    "    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))\n"
    "Face sets / per-face materials: UsdGeom.Subset.CreateGeomSubset (no mesh.CreateSubset).",
    methods=[
        _s(
            "Define",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdGeom.Mesh",
            "Define a Mesh prim at path.",
        ),
        _s(
            "Get",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdGeom.Mesh",
            "Wrap existing prim (falsy if none). Also UsdGeom.Mesh(prim).",
        ),
        _m("GetPrim", "(self)", "Usd.Prim", "Underlying prim."),
        _m("GetPath", "(self)", "Sdf.Path", "Prim path."),
        _m("GetPointsAttr", "(self)", "Usd.Attribute", "point3f[] points."),
        _m("CreatePointsAttr", f"(self, {_DEF})", "Usd.Attribute", "Create/set points."),
        _m("GetFaceVertexCountsAttr", "(self)", "Usd.Attribute", "int[] vertices per face."),
        _m(
            "CreateFaceVertexCountsAttr",
            f"(self, {_DEF})",
            "Usd.Attribute",
            "Create/set face vertex counts.",
        ),
        _m(
            "GetFaceVertexIndicesAttr",
            "(self)",
            "Usd.Attribute",
            "int[] point indices, face by face.",
        ),
        _m(
            "CreateFaceVertexIndicesAttr",
            f"(self, {_DEF})",
            "Usd.Attribute",
            "Create/set face vertex indices.",
        ),
        _m(
            "GetFaceCount",
            f"(self, timeCode: {_T})",
            "int",
            "Number of faces (len(faceVertexCounts)).",
        ),
        _m("GetNormalsAttr", "(self)", "Usd.Attribute", "normal3f[] normals."),
        _m("CreateNormalsAttr", f"(self, {_DEF})", "Usd.Attribute", "Create/set normals."),
        _m("GetDisplayColorAttr", "(self)", "Usd.Attribute", "primvars:displayColor (color3f[])."),
        _m(
            "CreateDisplayColorAttr",
            f"(self, {_DEF})",
            "Usd.Attribute",
            "Create/set display color.",
        ),
        _m(
            "GetSubdivisionSchemeAttr",
            "(self)",
            "Usd.Attribute",
            "'catmullClark' (default) or 'none'.",
        ),
        _m(
            "CreateSubdivisionSchemeAttr",
            f"(self, {_DEF})",
            "Usd.Attribute",
            "Set UsdGeom.Tokens.none for polygons.",
        ),
        _m(
            "GetFaceVaryingLinearInterpolationAttr",
            "(self)",
            "Usd.Attribute",
            "faceVarying primvar (UV) interpolation on subdivision: default 'cornersPlus1'.",
        ),
        _m(
            "CreateFaceVaryingLinearInterpolationAttr",
            f"(self, {_DEF})",
            "Usd.Attribute",
            "e.g. 'all' (UVs linear everywhere) or 'none'.",
        ),
        _m("CreateExtentAttr", f"(self, {_DEF})", "Usd.Attribute", "Bounds [(min), (max)]."),
        _s(
            "ComputeExtent",
            "(points: Vt.Vec3fArray)",
            "Vt.Vec3fArray",
            "[min, max] of points (for extent).",
        ),
    ],
)

_SUBSET_DOC = (
    "GeomSubset: a named set of element indices (faces) of a mesh; child prim of the mesh.\n"
    "Material face sets need elementType='face' AND familyName='materialBind' (the default\n"
    "familyName '' is ignored by material binding; 'materialBind' is never an elementType).\n"
    "Bind material to faces (recipe):\n"
    f"{_indent(MATERIAL_SUBSET_RECIPE)}\n"
    "Verify like a consumer:\n"
    f"{_indent(VERIFY_SUBSETS_RECIPE)}\n"
    "Subset.Define(stage, path) authors nothing else: set familyName, indices and display name\n"
    "yourself. Attribute getters: GetFamilyNameAttr, GetElementTypeAttr, GetIndicesAttr.\n"
    "Not in usd-core: Subset.CreateSubset, CreateFamily, GetGeomSubset(ByName), GetSubset,\n"
    "Mesh.CreateGeomSubset, UsdGeom.GeomSubset. Family type: Subset.SetFamilyType/GetFamilyType."
)

_register(
    UsdGeom.Subset,
    "UsdGeom.Subset",
    _SUBSET_DOC,
    methods=[
        _s(
            "CreateGeomSubset",
            "(geom: UsdGeom.Imageable, subsetName: str, elementType: str, indices: Vt.IntArray | list[int], "
            "familyName: str = '', familyType: str = '')",
            "UsdGeom.Subset",
            "Create (or update) geom.GetPath()/subsetName. elementType=UsdGeom.Tokens.face; "
            "familyName=UsdShade.Tokens.materialBind for materials.",
        ),
        _s(
            "CreateUniqueGeomSubset",
            "(geom: UsdGeom.Imageable, subsetName: str, elementType: str, indices: Vt.IntArray | list[int], "
            "familyName: str = '', familyType: str = '')",
            "UsdGeom.Subset",
            "Like CreateGeomSubset but appends _1, _2... if the name exists.",
        ),
        _s(
            "GetGeomSubsets",
            "(geom: UsdGeom.Imageable, elementType: str = '', familyName: str = '')",
            "list[UsdGeom.Subset]",
            "Child subsets filtered by elementType and familyName ('' = any).",
        ),
        _s(
            "GetAllGeomSubsets",
            "(geom: UsdGeom.Imageable)",
            "list[UsdGeom.Subset]",
            "All child subsets.",
        ),
        _s(
            "GetAllGeomSubsetFamilyNames",
            "(geom: UsdGeom.Imageable)",
            "list[str]",
            "Family names in use.",
        ),
        _s(
            "SetFamilyType",
            "(geom: UsdGeom.Imageable, familyName: str, familyType: str)",
            "bool",
            "UsdGeom.Tokens.partition | nonOverlapping | unrestricted (default).",
        ),
        _s(
            "GetFamilyType",
            "(geom: UsdGeom.Imageable, familyName: str)",
            "str",
            "Family type token.",
        ),
        _s(
            "ValidateFamily",
            "(geom: UsdGeom.Imageable, elementType: str = '', familyName: str = '')",
            "tuple[bool, str]",
            "(valid, reason); needs the mesh's faceVertexCounts authored.",
        ),
        _s(
            "ValidateSubsets",
            "(subsets: list[UsdGeom.Subset], elementCount: int, familyType: str)",
            "tuple[bool, str]",
            "(valid, reason) for these subsets against elementCount faces and a family type.",
        ),
        _s(
            "GetUnassignedIndices",
            "(geom: UsdGeom.Imageable, elementType: str, familyName: str, "
            "time: Usd.TimeCode = Usd.TimeCode.EarliestTime())",
            "Vt.IntArray",
            "Element indices not in any subset of the family.",
        ),
        _s(
            "Define",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdGeom.Subset",
            "Define an empty GeomSubset (elementType 'face', familyName '', no indices).",
        ),
        _s(
            "Get",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdGeom.Subset",
            "Wrap existing prim.",
        ),
        _m("GetPrim", "(self)", "Usd.Prim", "Underlying prim (SetDisplayName, binding API)."),
        _m("GetFamilyNameAttr", "(self)", "Usd.Attribute", "uniform token familyName."),
        _m("CreateFamilyNameAttr", f"(self, {_DEF})", "Usd.Attribute", "Set familyName."),
        _m("GetElementTypeAttr", "(self)", "Usd.Attribute", "uniform token elementType ('face')."),
        _m("CreateElementTypeAttr", f"(self, {_DEF})", "Usd.Attribute", "Set elementType."),
        _m("GetIndicesAttr", "(self)", "Usd.Attribute", "int[] element indices."),
        _m("CreateIndicesAttr", f"(self, {_DEF})", "Usd.Attribute", "Set indices."),
    ],
)

_register(
    UsdGeom.Tokens,
    "UsdGeom.Tokens",
    "String constants; each value equals its name (UsdGeom.Tokens.face == 'face'),\n"
    "except y/z = 'Y'/'Z' and default_ = 'default'. Use them where the API takes a token.\n"
    "    Subset elementType:  face (also point, edge)\n"
    "    Subset familyType:   partition (each face once) | nonOverlapping | unrestricted (default)\n"
    "    upAxis:              y | z            subdivisionScheme: none | catmullClark\n"
    "    purpose:             default_ | render | proxy | guide\n"
    "    primvar interpolation: constant | uniform | vertex | faceVarying\n"
    "Material family name is UsdShade.Tokens.materialBind (not in UsdGeom.Tokens).",
)

_XFORM_METHODS = [
    _m(
        "AddTranslateOp",
        "(self, precision: UsdGeom.XformOp.Precision = UsdGeom.XformOp.PrecisionDouble, "
        "opSuffix: str = '', isInverseOp: bool = False)",
        "UsdGeom.XformOp",
        "Append translate op; then op.Set(Gf.Vec3d(x, y, z)).",
    ),
    _m(
        "AddRotateXYZOp",
        "(self, precision: UsdGeom.XformOp.Precision = UsdGeom.XformOp.PrecisionFloat, "
        "opSuffix: str = '', isInverseOp: bool = False)",
        "UsdGeom.XformOp",
        "Append rotateXYZ op (degrees); op.Set(Gf.Vec3f(rx, ry, rz)).",
    ),
    _m(
        "AddScaleOp",
        "(self, precision: UsdGeom.XformOp.Precision = UsdGeom.XformOp.PrecisionFloat, "
        "opSuffix: str = '', isInverseOp: bool = False)",
        "UsdGeom.XformOp",
        "Append scale op; op.Set(Gf.Vec3f(sx, sy, sz)).",
    ),
    _m(
        "AddTransformOp",
        "(self, precision: UsdGeom.XformOp.Precision = UsdGeom.XformOp.PrecisionDouble, "
        "opSuffix: str = '', isInverseOp: bool = False)",
        "UsdGeom.XformOp",
        "Append matrix op; op.Set(Gf.Matrix4d(...)).",
    ),
    _m("GetOrderedXformOps", "(self)", "list[UsdGeom.XformOp]", "Ops in xformOpOrder."),
    _m("ClearXformOpOrder", "(self)", "bool", "Clear xformOpOrder (op attributes remain)."),
    _m(
        "GetLocalTransformation", f"(self, time: {_T})", "Gf.Matrix4d", "Local matrix from the ops."
    ),
    _m(
        "ComputeLocalToWorldTransform",
        "(self, time: Usd.TimeCode)",
        "Gf.Matrix4d",
        "World matrix; pass Usd.TimeCode.Default() (required argument).",
    ),
    _m("GetXformOpOrderAttr", "(self)", "Usd.Attribute", "token[] xformOpOrder."),
]

_XFORMABLE_DOC = (
    "Transformable prim (base of Xform, Mesh, Camera...). Ops are appended in order:\n"
    "    xf = UsdGeom.Xform.Define(stage, '/World/Obj')\n"
    "    xf.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))\n"
    "    xf.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, 0))     # degrees\n"
    "    xf.AddScaleOp().Set(Gf.Vec3f(2, 2, 2))\n"
    "Adding an op that already exists raises Tf.ErrorException; reuse GetOrderedXformOps() or ClearXformOpOrder().\n"
    "Simple alternative: UsdGeom.XformCommonAPI(prim).SetTranslate(Gf.Vec3d(...))."
)


# Curated views of pxr base classes, merged down the MRO: doc() on a subclass that
# has no view of its own (UsdGeom.Cube, UsdGeom.Sphere, ...) lists the API of all its
# curated bases under its own name.
_SCHEMA_VIEWS: dict[type, tuple[str, str, list[CallableInfo]]] = {}


def _pxr_name(cls: type) -> str:
    """'UsdGeom.Cube' for pxr.UsdGeom.Cube."""
    return f"{cls.__module__.removeprefix('pxr.')}.{cls.__qualname__}"


def _register_schema(target: type, name: str, docstring: str, methods=()) -> None:
    """Register a view whose subclasses inherit its methods (and those of curated bases)."""
    _SCHEMA_VIEWS[target] = (name, docstring, list(methods))

    def _extract(cls_or_instance) -> TypeInfo:
        cls = cls_or_instance if isinstance(cls_or_instance, type) else type(cls_or_instance)
        bases = [base for base in cls.__mro__ if base in _SCHEMA_VIEWS]
        methods: list[CallableInfo] = []
        seen: set[str] = set()
        for base in bases:
            for method in _SCHEMA_VIEWS[base][2]:
                if method.name not in seen:
                    seen.add(method.name)
                    methods.append(method)
        own_name, own_doc, _ = _SCHEMA_VIEWS[bases[0]]
        if cls is bases[0]:
            doc = own_doc
        else:
            own_name = _pxr_name(cls)
            curated = ", ".join(_SCHEMA_VIEWS[base][0] for base in bases)
            doc = f"{own_name}: only the API inherited from {curated} is curated here.\n{own_doc}"
        return TypeInfo(name=own_name, base=None, fields=[], methods=methods, docstring=doc)

    spec.define_doc(target)(_extract)


_register_schema(UsdGeom.Xformable, "UsdGeom.Xformable", _XFORMABLE_DOC, _XFORM_METHODS)


_register(
    UsdGeom.Xform,
    "UsdGeom.Xform",
    "Transformable group prim; children inherit its transform.\n" + _XFORMABLE_DOC,
    methods=[
        _s(
            "Define",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdGeom.Xform",
            "Define an Xform prim.",
        ),
        _s(
            "Get",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdGeom.Xform",
            "Wrap existing prim.",
        ),
        _m("GetPrim", "(self)", "Usd.Prim", "Underlying prim."),
        *_XFORM_METHODS,
    ],
)

# --------------------------------------------------------------------------- UsdShade

_register(
    UsdShade.Material,
    "UsdShade.Material",
    "Material prim (UsdShade.Material, not UsdGeom.Material); put its Shader prims under it.\n"
    "    mat = UsdShade.Material.Define(stage, '/World/Looks/Red')\n"
    "    sh = UsdShade.Shader.Define(stage, '/World/Looks/Red/Shader')\n"
    "    sh.CreateIdAttr('UsdPreviewSurface')\n"
    "    sh.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 0, 0))\n"
    "    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface')\n"
    "Bind: UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat).",
    methods=[
        _s(
            "Define",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdShade.Material",
            "Define a Material prim.",
        ),
        _s(
            "Get",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdShade.Material",
            "Wrap existing prim (falsy if none). Also UsdShade.Material(prim).",
        ),
        _m("GetPrim", "(self)", "Usd.Prim", "Underlying prim."),
        _m("GetPath", "(self)", "Sdf.Path", "Prim path."),
        _m(
            "CreateSurfaceOutput",
            "(self, renderContext: str = '')",
            "UsdShade.Output",
            "outputs:surface.",
        ),
        _m(
            "GetSurfaceOutput",
            "(self, renderContext: str = '')",
            "UsdShade.Output",
            "outputs:surface.",
        ),
        _m(
            "CreateDisplacementOutput",
            "(self, renderContext: str = '')",
            "UsdShade.Output",
            "outputs:displacement.",
        ),
        _m(
            "CreateInput",
            "(self, name: str, type: Sdf.ValueTypeName)",
            "UsdShade.Input",
            "Material interface input.",
        ),
        _m("GetInput", "(self, name: str)", "UsdShade.Input", "Input by name (without 'inputs:')."),
        _m(
            "ComputeSurfaceSource",
            "(self, renderContext: str = '')",
            "tuple[UsdShade.Shader, str, UsdShade.AttributeType]",
            "Shader connected to the surface output.",
        ),
    ],
)

_register(
    UsdShade.Shader,
    "UsdShade.Shader",
    "Shading node. Set the node type with CreateIdAttr('UsdPreviewSurface') (or SetShaderId).\n"
    "Inputs: sh.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(0.5);\n"
    "outputs: sh.CreateOutput('surface', Sdf.ValueTypeNames.Token). Connect a Material output\n"
    "with mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), 'surface').",
    methods=[
        _s(
            "Define",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdShade.Shader",
            "Define a Shader prim.",
        ),
        _s(
            "Get",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdShade.Shader",
            "Wrap existing prim.",
        ),
        _m("GetPrim", "(self)", "Usd.Prim", "Underlying prim."),
        _m(
            "CreateIdAttr", f"(self, {_DEF})", "Usd.Attribute", "info:id, e.g. 'UsdPreviewSurface'."
        ),
        _m("GetIdAttr", "(self)", "Usd.Attribute", "info:id attribute."),
        _m("SetShaderId", "(self, id: str)", "bool", "Same as CreateIdAttr(id)."),
        _m("GetShaderId", "(self)", "str | None", "info:id value."),
        _m(
            "CreateInput",
            "(self, name: str, type: Sdf.ValueTypeName)",
            "UsdShade.Input",
            "inputs:<name>; then .Set(value) or .ConnectToSource(...).",
        ),
        _m("GetInput", "(self, name: str)", "UsdShade.Input", "Input by name (without 'inputs:')."),
        _m("GetInputs", "(self, onlyAuthored: bool = True)", "list[UsdShade.Input]", "All inputs."),
        _m(
            "CreateOutput",
            "(self, name: str, type: Sdf.ValueTypeName)",
            "UsdShade.Output",
            "outputs:<name>.",
        ),
        _m("GetOutput", "(self, name: str)", "UsdShade.Output", "Output by name."),
        _m(
            "ConnectableAPI",
            "(self)",
            "UsdShade.ConnectableAPI",
            "Source argument for ConnectToSource.",
        ),
    ],
)

_register(
    UsdShade.MaterialBindingAPI,
    "UsdShade.MaterialBindingAPI",
    "Binds materials (UsdShade, not UsdGeom). Apply before binding; works on any prim\n"
    "including a GeomSubset prim (face set; its familyName must be 'materialBind'):\n"
    "    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)\n"
    "    UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()   # -> (material, rel)\n"
    "    UsdShade.MaterialBindingAPI(prim).GetDirectBinding().GetMaterialPath()\n"
    "Face subsets: see doc(UsdGeom.Subset) for the full recipe.",
    methods=[
        _s(
            "Apply",
            "(prim: Usd.Prim)",
            "UsdShade.MaterialBindingAPI",
            "Apply the API schema (adds apiSchemas metadata) and return it.",
        ),
        _s("CanApply", "(prim: Usd.Prim)", "bool", "Whether Apply is allowed."),
        _m(
            "Bind",
            "(self, material: UsdShade.Material, bindingStrength: str = 'fallbackStrength', "
            "materialPurpose: str = '')",
            "bool",
            "Author material:binding. Strength: UsdShade.Tokens.strongerThanDescendants to override children.",
        ),
        _m(
            "UnbindDirectBinding",
            "(self, materialPurpose: str = '')",
            "bool",
            "Block the direct binding.",
        ),
        _m("UnbindAllBindings", "(self)", "bool", "Remove all bindings."),
        _m(
            "ComputeBoundMaterial",
            "(self, materialPurpose: str = '', supportLegacyBindings: bool = True)",
            "tuple[UsdShade.Material, Usd.Relationship]",
            "Resolved material (falsy Material if none) and the rel.",
        ),
        _m(
            "GetDirectBinding",
            "(self, materialPurpose: str = '')",
            "UsdShade.MaterialBindingAPI.DirectBinding",
            "Authored binding; .GetMaterialPath(), .GetMaterial().",
        ),
        _m(
            "GetDirectBindingRel",
            "(self, materialPurpose: str = '')",
            "Usd.Relationship",
            "material:binding rel.",
        ),
        _m(
            "GetMaterialBindSubsets",
            "(self)",
            "list[UsdGeom.Subset]",
            "Child subsets with familyName 'materialBind' (call on the mesh prim).",
        ),
        _m(
            "CreateMaterialBindSubset",
            "(self, subsetName: str, indices: Vt.IntArray | list[int], elementType: str = 'face')",
            "UsdGeom.Subset",
            "Mesh-side shortcut: subset with familyName 'materialBind' (still Apply+Bind it).",
        ),
        _m(
            "SetMaterialBindSubsetsFamilyType",
            "(self, familyType: str)",
            "bool",
            "e.g. UsdGeom.Tokens.nonOverlapping or partition.",
        ),
        _m("GetMaterialBindSubsetsFamilyType", "(self)", "str", "Default 'unrestricted'."),
        _s(
            "ComputeBoundMaterials",
            "(prims: list[Usd.Prim], materialPurpose: str = '', supportLegacyBindings: bool = True)",
            "tuple[list[UsdShade.Material], list[Usd.Relationship]]",
            "Bound material and binding rel for each prim (batched ComputeBoundMaterial).",
        ),
    ],
)

_register(
    UsdShade.Tokens,
    "UsdShade.Tokens",
    "String constants; each value equals its name (UsdShade.Tokens.surface == 'surface').\n"
    "    materialBind               GeomSubset familyName for material face sets\n"
    "    surface, displacement, volume   material outputs / shader output names\n"
    "    strongerThanDescendants, weakerThanDescendants, fallbackStrength   Bind() strength\n"
    "    allPurpose (''), preview, full   materialPurpose",
)

# --------------------------------------------------------------------------- Sdf

_register(
    Sdf.Path,
    "Sdf.Path",
    "Scene path. Sdf.Path('/World/Mesh'); APIs taking a path also accept str.\n"
    "Prim names must be identifiers (letters, digits, _; no leading digit or spaces):\n"
    "check with Sdf.Path.IsValidIdentifier(name); put free text in prim.SetDisplayName().\n"
    "Properties: '/World/Mesh.points'. Attributes: .pathString, .name.",
    methods=[
        _m("__init__", "(self, path: str)", "None", "Parse a path string."),
        _m("AppendChild", "(self, childName: str)", "Sdf.Path", "/A + 'b' -> /A/b."),
        _m("AppendPath", "(self, path: Sdf.Path)", "Sdf.Path", "Append a relative path."),
        _m("AppendProperty", "(self, propName: str)", "Sdf.Path", "/A + 'x' -> /A.x."),
        _m("GetParentPath", "(self)", "Sdf.Path", "Parent path."),
        _m("GetPrimPath", "(self)", "Sdf.Path", "Strip the property part."),
        _m("ReplaceName", "(self, newName: str)", "Sdf.Path", "Same parent, new last element."),
        _m("HasPrefix", "(self, prefix: Sdf.Path)", "bool", "Path is prefix or below it."),
        _s("IsValidIdentifier", "(name: str)", "bool", "Valid prim name?"),
    ],
)

_register(
    Sdf.Layer,
    "Sdf.Layer",
    "One USD file. Layers are shared by identifier while any stage holds them, so a re-Open\n"
    "returns stale contents after an external file change: layer.Reload() or\n"
    "Sdf.Layer.OpenAsAnonymous(path) for an independent copy.",
    methods=[
        _s(
            "FindOrOpen",
            "(identifier: str, args: dict = {})",
            "Sdf.Layer",
            "Open (or reuse an open) layer.",
        ),
        _s(
            "OpenAsAnonymous",
            "(filePath: str = '', metadataOnly: bool = False, tag: str = '')",
            "Sdf.Layer",
            "Fresh, unshared read of the file; Usd.Stage.Open(layer).",
        ),
        _s(
            "CreateNew", "(identifier: str, args: dict = {})", "Sdf.Layer", "New file-backed layer."
        ),
        _s("CreateAnonymous", "(tag: str = '', args: dict = {})", "Sdf.Layer", "In-memory layer."),
        _m("Save", "(self, force: bool = False)", "bool", "Write to its file."),
        _m(
            "Export",
            "(self, filename: str, comment: str = '', args: dict = {})",
            "bool",
            "Write to another file.",
        ),
        _m("Reload", "(self, force: bool = False)", "bool", "Re-read from disk."),
        _m("ExportToString", "(self)", "str", "usda text."),
        _m(
            "GetCompositionAssetDependencies",
            "(self)",
            "list[str]",
            "Asset paths of sublayers, references and payloads, as authored.",
        ),
    ],
)

_register(
    Sdf.ValueTypeNames,
    "Sdf.ValueTypeNames",
    "Attribute value types (Sdf.ValueTypeName) for CreateAttribute / CreateInput /\n"
    "CreateOutput / CreatePrimvar. Each scalar also has an 'Array' form (Float -> FloatArray).\n"
    "    Bool, Int, Float, Double, String, Token, Asset (Sdf.AssetPath, e.g. textures)\n"
    "    Float2/Float3 (Gf.Vec2f/Vec3f), Double3 (Gf.Vec3d), Color3f (Gf.Vec3f), Matrix4d\n"
    "    IntArray (Vt.IntArray), Point3fArray / Normal3fArray / Color3fArray (Vt.Vec3fArray),\n"
    "    TexCoord2fArray (UVs, Vt.Vec2fArray)",
)

# --------------------------------------------------------------------------- Vt

_VT_DOC = (
    "{name}: typed array for {kind} attributes. Behaves like a sequence (len, index, iter).\n"
    "    Vt.{name}({example})\n"
    "    Vt.{name}.FromNumpy(arr)    # NumPy array{shape}\n"
    "    numpy.array(vt_array)        # back to NumPy\n"
    "Vt.{name}(numpy_array) can raise TypeError (numpy scalar types): use FromNumpy or .tolist()."
)

_register(
    Vt.IntArray,
    "Vt.IntArray",
    _VT_DOC.format(
        name="IntArray",
        kind="int[] (indices, counts)",
        example="[0, 1, 2]",
        shape=" (any int dtype)",
    ),
    methods=[
        _m(
            "__init__",
            "(self, values: Sequence[int])",
            "None",
            "From a list; IntArray(n) gives n zeros.",
        ),
        _s("FromNumpy", "(array: numpy.ndarray)", "Vt.IntArray", "Copy a 1-D NumPy array."),
    ],
)

_register(
    Vt.Vec3fArray,
    "Vt.Vec3fArray",
    _VT_DOC.format(
        name="Vec3fArray",
        kind="point3f[] / normal3f[] / color3f[]",
        example="[(0, 0, 0), (1, 0, 0)]",
        shape=" of shape (N, 3)",
    ),
    methods=[
        _m(
            "__init__",
            "(self, values: Sequence[Gf.Vec3f | tuple[float, float, float]])",
            "None",
            "From a list of 3-tuples or Gf.Vec3f.",
        ),
        _s("FromNumpy", "(array: numpy.ndarray)", "Vt.Vec3fArray", "Copy an (N, 3) NumPy array."),
    ],
)

# --------------------------------------------------------------------------- Gf

for _cls, _other, _prec, _use in (
    (Gf.Vec3f, "Vec3d", "float32", "points, colors, rotateXYZ/scale values"),
    (Gf.Vec3d, "Vec3f", "float64", "translations (AddTranslateOp), bounds"),
):
    _n = f"Gf.{_cls.__name__}"
    _register(
        _cls,
        _n,
        f"{_prec} 3-vector for {_use}.\n"
        f"{_n}(1, 2, 3); v[0]; v * 2; v + w; Gf.Dot(a, b); Gf.Cross(a, b); tuple(v).",
        methods=[
            _m(
                "__init__",
                "(self, x: float, y: float, z: float)",
                "None",
                f"Also {_n}(s) for (s, s, s) and {_n}(Gf.{_other}(...)) to convert.",
            ),
            _m("GetLength", "(self)", "float", "Euclidean length."),
            _m("GetNormalized", "(self)", _n, "Unit vector."),
            _m("Normalize", "(self)", "float", "Normalize in place; returns the old length."),
            _m("GetDot", f"(self, other: {_n})", "float", "Dot product."),
        ],
    )

_register(
    Gf.Matrix4d,
    "Gf.Matrix4d",
    "4x4 double matrix, row-vector convention (translation in the last row).\n"
    "    m = Gf.Matrix4d(1.0)                       # identity\n"
    "    m.SetTranslate(Gf.Vec3d(1, 2, 3))\n"
    "    p = m.Transform(Gf.Vec3d(0, 0, 0))         # transform a point\n"
    "World matrix of a prim: UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).",
    methods=[
        _m(
            "__init__",
            "(self, value: float = ...)",
            "None",
            "Matrix4d(1.0) identity; also 16 floats or Matrix4d(rotation, translation).",
        ),
        _m(
            "SetTranslate",
            "(self, translation: Gf.Vec3d)",
            "Gf.Matrix4d",
            "Set to a pure translation (in place).",
        ),
        _m(
            "SetRotate",
            "(self, rotation: Gf.Rotation)",
            "Gf.Matrix4d",
            "Set to a pure rotation (in place).",
        ),
        _m(
            "SetScale",
            "(self, scale: Gf.Vec3d | float)",
            "Gf.Matrix4d",
            "Set to a pure scale (in place).",
        ),
        _m("SetIdentity", "(self)", "Gf.Matrix4d", "Reset to identity."),
        _m("ExtractTranslation", "(self)", "Gf.Vec3d", "Translation part."),
        _m("ExtractRotation", "(self)", "Gf.Rotation", "Rotation part."),
        _m("GetInverse", "(self)", "Gf.Matrix4d", "Inverse matrix."),
        _m("Transform", "(self, point: Gf.Vec3d)", "Gf.Vec3d", "Transform a point (also Vec3f)."),
        _m(
            "TransformDir",
            "(self, vector: Gf.Vec3d)",
            "Gf.Vec3d",
            "Transform a direction (no translation).",
        ),
    ],
)


# --------------------------------------------------------------------------- base classes
# Signatures checked against the Boost.Python C++ signatures of usd-core 26.8.

_register_schema(
    Usd.Object,
    "Usd.Object",
    "Base of Usd.Prim and Usd.Property (Usd.Attribute, Usd.Relationship): name, path,\n"
    "validity and metadata. customData is a free-form dict; keyPath 'a:b' addresses\n"
    "nested keys: prim.SetCustomDataByKey('myTool:id', 7); prim.GetCustomDataByKey('myTool:id').",
    methods=[
        _m(
            "IsValid", "(self)", "bool", "False for missing or expired objects (same as bool(obj))."
        ),
        _m(
            "GetName",
            "(self)",
            "str",
            "Prim: last path element; property: full namespaced name, e.g. 'inputs:diffuseColor'.",
        ),
        _m("GetPath", "(self)", "Sdf.Path", "Prim path or property path '/Prim.attr'."),
        _m("GetPrim", "(self)", "Usd.Prim", "The prim itself, or a property's owning prim."),
        _m("GetStage", "(self)", "Usd.Stage", "Owning stage."),
        _m("GetMetadata", "(self, key: str)", "object", "Metadata value (None if unset)."),
        _m("SetMetadata", "(self, key: str, value: object)", "bool", "Author metadata."),
        _m("HasMetadata", "(self, key: str)", "bool", "Metadata authored or has a fallback."),
        _m("GetAllMetadata", "(self)", "dict", "All authored and fallback metadata."),
        _m("GetCustomData", "(self)", "dict", "customData dict ({} if unset)."),
        _m(
            "GetCustomDataByKey",
            "(self, keyPath: str)",
            "object",
            "One customData value; 'a:b' for nested dicts (None if unset).",
        ),
        _m(
            "SetCustomData",
            "(self, customData: dict)",
            "None",
            "Replace the whole customData dict.",
        ),
        _m(
            "SetCustomDataByKey",
            "(self, keyPath: str, value: object)",
            "None",
            "Author one customData entry; 'a:b' creates nested dicts. Returns None.",
        ),
        _m("HasCustomDataKey", "(self, keyPath: str)", "bool", "customData has keyPath."),
        _m("GetDocumentation", "(self)", "str", "'documentation' metadata ('' if unset)."),
        _m("SetDocumentation", "(self, doc: str)", "bool", "Author 'documentation' metadata."),
    ],
)

_register_schema(
    Usd.Property,
    "Usd.Property",
    "Base of Usd.Attribute and Usd.Relationship. Names may be namespaced with ':'\n"
    "(GetName() 'primvars:st', GetNamespace() 'primvars', GetBaseName() 'st').",
    methods=[
        _m("GetBaseName", "(self)", "str", "Name without namespace."),
        _m("GetNamespace", "(self)", "str", "Namespace part ('' if none)."),
        _m("IsAuthored", "(self)", "bool", "Has an authored spec in some layer."),
        _m("IsCustom", "(self)", "bool", "Custom (not from a schema)."),
        _m("IsDefined", "(self)", "bool", "Authored or defined by a schema."),
    ],
)

_register_schema(
    Usd.Relationship,
    "Usd.Relationship",
    "Targets other paths (e.g. material:binding, proxyPrim). Create with\n"
    "prim.CreateRelationship(name); targets are Sdf.Path values (str paths are accepted).",
    methods=[
        _m("GetTargets", "(self)", "list[Sdf.Path]", "Authored target paths."),
        _m(
            "GetForwardedTargets",
            "(self)",
            "list[Sdf.Path]",
            "Targets with relationships-to-relationships resolved.",
        ),
        _m(
            "AddTarget",
            "(self, target: Sdf.Path, position: Usd.ListPosition = Usd.ListPositionBackOfPrependList)",
            "bool",
            "Add one target.",
        ),
        _m("SetTargets", "(self, targets: list[Sdf.Path])", "bool", "Replace all targets."),
        _m("RemoveTarget", "(self, target: Sdf.Path)", "bool", "Remove one target."),
        _m(
            "ClearTargets",
            "(self, removeSpec: bool)",
            "bool",
            "Clear targets; removeSpec=True also removes the relationship spec.",
        ),
    ],
)

_register_schema(
    Usd.SchemaBase,
    "Usd.SchemaBase",
    "Base of every schema class (UsdGeom.Mesh, UsdShade.Material, UsdGeom.PrimvarsAPI...).\n"
    "A schema object wraps a prim: schema.GetPrim(); falsy if the prim is invalid.",
    methods=[
        _m("GetPrim", "(self)", "Usd.Prim", "The wrapped prim."),
        _m("GetPath", "(self)", "Sdf.Path", "Path of the wrapped prim."),
        _m("IsConcrete", "(self)", "bool", "Concrete typed schema (can be Define()d)."),
        _m("GetSchemaKind", "(self)", "Usd.SchemaKind", "e.g. Usd.SchemaKind.ConcreteTyped."),
        _s(
            "GetSchemaAttributeNames",
            "(includeInherited: bool = True)",
            "list[str]",
            "Attribute names the schema defines; call on the class, e.g. "
            "UsdGeom.Mesh.GetSchemaAttributeNames().",
        ),
    ],
)

_register(
    Usd.TimeCode,
    "Usd.TimeCode",
    "Time at which to read/write a value. Usd.TimeCode.Default() is the non-animated value\n"
    "(what Get()/Set() use without a time); Usd.TimeCode(24) is frame 24. A plain float also\n"
    "works where a time code is expected.",
    methods=[
        _m("__init__", "(self, time: float)", "None", "Time code at a frame; Usd.TimeCode() is 0."),
        _s("Default", "()", "Usd.TimeCode", "The default (non-animated) time."),
        _s("EarliestTime", "()", "Usd.TimeCode", "Earliest time sample (for range queries)."),
        _m("IsDefault", "(self)", "bool", "This is Usd.TimeCode.Default()."),
        _m("GetValue", "(self)", "float", "Frame number."),
    ],
)

_register(
    Usd.EditTarget,
    "Usd.EditTarget",
    "Which layer new edits go to: stage.GetEditTarget().GetLayer(). Author into another\n"
    "layer with stage.SetEditTarget(layer) (an Sdf.Layer of the stage's layer stack, e.g.\n"
    "stage.GetSessionLayer()); stage.SetEditTarget(stage.GetRootLayer()) switches back.",
    methods=[
        _m("GetLayer", "(self)", "Sdf.Layer", "The layer edits are written to."),
        _m("IsValid", "(self)", "bool", "Has a layer."),
    ],
)

_register_schema(
    UsdGeom.Imageable,
    "UsdGeom.Imageable",
    "Base of renderable prims (Xform, Scope, Mesh...): visibility, purpose and bounds.\n"
    "    bbox = UsdGeom.Imageable(prim).ComputeWorldBound(Usd.TimeCode.Default(), UsdGeom.Tokens.default_)\n"
    "    rng = bbox.ComputeAlignedRange()   # Gf.Range3d: GetMin(), GetMax(), GetSize()\n"
    "Many bounds: UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).",
    methods=[
        _m("GetVisibilityAttr", "(self)", "Usd.Attribute", "'inherited' or 'invisible'."),
        _m("MakeVisible", f"(self, time: {_T})", "None", "Make visible (and its ancestors)."),
        _m("MakeInvisible", f"(self, time: {_T})", "None", "Author visibility 'invisible'."),
        _m(
            "ComputeVisibility",
            f"(self, time: {_T})",
            "str",
            "'inherited' or 'invisible', taking ancestors into account.",
        ),
        _m(
            "GetPurposeAttr",
            "(self)",
            "Usd.Attribute",
            "purpose: UsdGeom.Tokens.default_ | render | proxy | guide.",
        ),
        _m("CreatePurposeAttr", f"(self, {_DEF})", "Usd.Attribute", "Create/set purpose."),
        _m(
            "ComputeWorldBound",
            "(self, time: Usd.TimeCode, purpose1: str = '', purpose2: str = '', "
            "purpose3: str = '', purpose4: str = '')",
            "Gf.BBox3d",
            "World-space bounds; pass at least one purpose, e.g. UsdGeom.Tokens.default_ "
            "(none gives an empty box).",
        ),
        _m(
            "ComputeLocalBound",
            "(self, time: Usd.TimeCode, purpose1: str = '', purpose2: str = '', "
            "purpose3: str = '', purpose4: str = '')",
            "Gf.BBox3d",
            "Bounds in the prim's local space (includes its own transform).",
        ),
    ],
)

_register_schema(
    UsdGeom.Boundable,
    "UsdGeom.Boundable",
    "Base of prims with an extent (Mesh, Points, Cube, ...): extent is the local bounding\n"
    "box [(min), (max)] as a float3[] attribute. Compute it from the geometry with\n"
    "    mesh.GetExtentAttr().Set(UsdGeom.Boundable.ComputeExtentFromPlugins(mesh, Usd.TimeCode.Default()))",
    methods=[
        _m("GetExtentAttr", "(self)", "Usd.Attribute", "float3[] extent [(min), (max)]."),
        _m("CreateExtentAttr", f"(self, {_DEF})", "Usd.Attribute", "Create/set extent."),
        _s(
            "ComputeExtentFromPlugins",
            "(boundable: UsdGeom.Boundable, time: Usd.TimeCode)",
            "Vt.Vec3fArray",
            "Extent [min, max] from the prim's geometry (also with a transform: Gf.Matrix4d).",
        ),
    ],
)

_register_schema(
    UsdGeom.Gprim,
    "UsdGeom.Gprim",
    "Base of geometric primitives (Mesh, Cube, Sphere...): display color/opacity,\n"
    "doubleSided, orientation ('rightHanded' default; UsdGeom.Tokens.leftHanded flips\n"
    "the face winding).",
    methods=[
        _m("GetDisplayColorAttr", "(self)", "Usd.Attribute", "primvars:displayColor (color3f[])."),
        _m(
            "CreateDisplayColorAttr",
            f"(self, {_DEF})",
            "Usd.Attribute",
            "Create/set display color.",
        ),
        _m("GetDoubleSidedAttr", "(self)", "Usd.Attribute", "bool doubleSided."),
        _m("CreateDoubleSidedAttr", f"(self, {_DEF})", "Usd.Attribute", "Create/set doubleSided."),
        _m(
            "GetOrientationAttr",
            "(self)",
            "Usd.Attribute",
            "uniform token orientation: 'rightHanded' (default) or 'leftHanded'.",
        ),
        _m(
            "CreateOrientationAttr",
            f"(self, {_DEF})",
            "Usd.Attribute",
            "Create/set orientation (UsdGeom.Tokens.rightHanded / leftHanded).",
        ),
    ],
)

_register_schema(
    UsdGeom.Scope,
    "UsdGeom.Scope",
    "Untransformable grouping prim (UsdGeom.Scope, not Usd.Scope); use it for /World/Looks\n"
    "and other organizational groups. Use UsdGeom.Xform when children need a transform.",
    methods=[
        _s(
            "Define",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdGeom.Scope",
            "Define a Scope prim.",
        ),
        _s(
            "Get",
            "(stage: Usd.Stage, path: Sdf.Path | str)",
            "UsdGeom.Scope",
            "Wrap existing prim.",
        ),
    ],
)

_register_schema(
    UsdGeom.PrimvarsAPI,
    "UsdGeom.PrimvarsAPI",
    "Create and read primvars (geometry data interpolated over a surface, e.g. UVs):\n"
    "    api = UsdGeom.PrimvarsAPI(mesh.GetPrim())   # also UsdGeom.PrimvarsAPI(mesh)\n"
    "    st = api.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)\n"
    "    st.Set(uvs); st.SetIndices(Vt.IntArray(uv_indices))   # optional indexing\n"
    "Names are given without the 'primvars:' prefix.",
    methods=[
        _m("__init__", "(self, prim: Usd.Prim)", "None", "Wrap a prim (or a schema object)."),
        _m(
            "CreatePrimvar",
            "(self, name: str, typeName: Sdf.ValueTypeName, interpolation: str = '', elementSize: int = -1)",
            "UsdGeom.Primvar",
            "Create primvars:<name>; interpolation constant|uniform|vertex|faceVarying.",
        ),
        _m(
            "GetPrimvar",
            "(self, name: str)",
            "UsdGeom.Primvar",
            "Primvar by name (falsy if absent).",
        ),
        _m("HasPrimvar", "(self, name: str)", "bool", "Primvar exists."),
        _m("GetPrimvars", "(self)", "list[UsdGeom.Primvar]", "All primvars, incl. schema-defined."),
        _m("GetAuthoredPrimvars", "(self)", "list[UsdGeom.Primvar]", "Authored primvars only."),
        _m("RemovePrimvar", "(self, name: str)", "bool", "Remove a primvar (and its indices)."),
    ],
)

_register(
    UsdGeom.Primvar,
    "UsdGeom.Primvar",
    "A primvar wraps an attribute 'primvars:<name>' plus interpolation and optional indices.\n"
    "Get it from UsdGeom.PrimvarsAPI(prim).GetPrimvar(name) or wrap UsdGeom.Primvar(attr).\n"
    "Indexed primvars: Get() returns the unique values; ComputeFlattened() expands them.",
    methods=[
        _m("__init__", "(self, attr: Usd.Attribute)", "None", "Wrap a 'primvars:' attribute."),
        _m("GetName", "(self)", "str", "Full attribute name, e.g. 'primvars:st'."),
        _m("GetPrimvarName", "(self)", "str", "Name without 'primvars:', e.g. 'st'."),
        _m("GetTypeName", "(self)", "Sdf.ValueTypeName", "Value type."),
        _m(
            "GetInterpolation",
            "(self)",
            "str",
            "constant | uniform | vertex | varying | faceVarying (default 'constant').",
        ),
        _m(
            "SetInterpolation",
            "(self, interpolation: str)",
            "bool",
            "Use UsdGeom.Tokens.faceVarying etc.",
        ),
        _m("Get", f"(self, time: {_T})", "object", "Authored values (unique values if indexed)."),
        _m("Set", f"(self, value: object, time: {_T})", "bool", "Author values."),
        _m("HasValue", "(self)", "bool", "Has a value."),
        _m("IsIndexed", "(self)", "bool", "Has authored indices."),
        _m("GetIndices", f"(self, time: {_T})", "Vt.IntArray", "Indices into Get() values."),
        _m(
            "SetIndices",
            f"(self, indices: Vt.IntArray, time: {_T})",
            "bool",
            "Author indices (makes the primvar indexed).",
        ),
        _m(
            "ComputeFlattened",
            f"(self, time: {_T})",
            "object",
            "Values expanded through the indices.",
        ),
        _m("GetAttr", "(self)", "Usd.Attribute", "The underlying attribute."),
    ],
)

_register(
    UsdGeom.XformOp,
    "UsdGeom.XformOp",
    "One transform operation of a UsdGeom.Xformable (from AddTranslateOp(), AddRotateXYZOp(),\n"
    "GetOrderedXformOps()...). op.Set(value) / op.Get(); its attribute is named like\n"
    "'xformOp:translate' (GetOpName).",
    methods=[
        _m("GetOpName", "(self)", "str", "Attribute name, e.g. 'xformOp:translate'."),
        _m("GetOpType", "(self)", "UsdGeom.XformOp.Type", "e.g. UsdGeom.XformOp.TypeTranslate."),
        _m("GetPrecision", "(self)", "UsdGeom.XformOp.Precision", "e.g. PrecisionDouble."),
        _m("Get", f"(self, time: {_T})", "object", "Op value, e.g. Gf.Vec3d for translate."),
        _m("Set", f"(self, value: object, time: {_T})", "bool", "Author the op value."),
        _m(
            "GetOpTransform",
            "(self, time: Usd.TimeCode)",
            "Gf.Matrix4d",
            "Matrix of this op alone.",
        ),
        _m("GetAttr", "(self)", "Usd.Attribute", "The underlying attribute."),
    ],
)

_CONNECT_DOC = (
    "Connect to a source; also ConnectToSource(sourceOutput|sourceInput|sourcePath). "
    "sourceName is the source output name without 'outputs:'. Returns True on success."
)

_register_schema(
    UsdShade.ConnectableAPI,
    "UsdShade.ConnectableAPI",
    "Connections between shading attributes. Usually call the method on the input/output:\n"
    "    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).ConnectToSource(\n"
    "        texture.ConnectableAPI(), 'rgb')\n"
    "    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')\n"
    "The static forms take the shading attribute (Usd.Attribute, UsdShade.Input or Output)\n"
    "first: UsdShade.ConnectableAPI.ConnectToSource(input, source, 'rgb').\n"
    "The source output must exist (texture.CreateOutput('rgb', Sdf.ValueTypeNames.Float3)).",
    methods=[
        _m(
            "__init__",
            "(self, prim: Usd.Prim)",
            "None",
            "Wrap a shader/material prim; also shader.ConnectableAPI().",
        ),
        _s(
            "ConnectToSource",
            "(shadingAttr: Usd.Attribute | UsdShade.Input | UsdShade.Output, "
            "source: UsdShade.ConnectableAPI, sourceName: str, "
            "sourceType: UsdShade.AttributeType = UsdShade.AttributeType.Output, "
            "typeName: Sdf.ValueTypeName = Sdf.ValueTypeName())",
            "bool",
            _CONNECT_DOC.replace("ConnectToSource(source", "ConnectToSource(shadingAttr, source"),
        ),
        _s(
            "HasConnectedSource",
            "(shadingAttr: Usd.Attribute | UsdShade.Input | UsdShade.Output)",
            "bool",
            "Attribute has a valid connection.",
        ),
        _s(
            "GetConnectedSources",
            "(shadingAttr: Usd.Attribute | UsdShade.Input | UsdShade.Output)",
            "tuple[list[UsdShade.ConnectionSourceInfo], list[Sdf.Path]]",
            "(valid sources, invalid source paths); info.source, info.sourceName.",
        ),
        _s(
            "DisconnectSource",
            "(shadingAttr: Usd.Attribute | UsdShade.Input | UsdShade.Output, "
            "sourceAttr: Usd.Attribute = Usd.Attribute())",
            "bool",
            "Remove one connection (all if sourceAttr is omitted).",
        ),
        _s(
            "ClearSources",
            "(shadingAttr: Usd.Attribute | UsdShade.Input | UsdShade.Output)",
            "bool",
            "Remove all connections.",
        ),
        _m("GetInputs", "(self, onlyAuthored: bool = True)", "list[UsdShade.Input]", "Inputs."),
        _m("GetOutputs", "(self, onlyAuthored: bool = True)", "list[UsdShade.Output]", "Outputs."),
    ],
)

_INPUT_OUTPUT_METHODS = [
    _m(
        "ConnectToSource",
        "(self, source: UsdShade.ConnectableAPI, sourceName: str, "
        "sourceType: UsdShade.AttributeType = UsdShade.AttributeType.Output, "
        "typeName: Sdf.ValueTypeName = Sdf.ValueTypeName())",
        "bool",
        _CONNECT_DOC,
    ),
    _m("HasConnectedSource", "(self)", "bool", "Has a valid connection."),
    _m(
        "GetConnectedSources",
        "(self)",
        "tuple[list[UsdShade.ConnectionSourceInfo], list[Sdf.Path]]",
        "(valid sources, invalid source paths).",
    ),
    _m(
        "DisconnectSource",
        "(self, sourceAttr: Usd.Attribute = Usd.Attribute())",
        "bool",
        "Remove one connection (all if omitted).",
    ),
    _m("GetAttr", "(self)", "Usd.Attribute", "The underlying attribute."),
    _m("GetBaseName", "(self)", "str", "Name without 'inputs:' / 'outputs:'."),
    _m("GetTypeName", "(self)", "Sdf.ValueTypeName", "Value type."),
]

_register(
    UsdShade.Input,
    "UsdShade.Input",
    "Shader or material input 'inputs:<name>' (from shader.CreateInput(name, type)).\n"
    "Set a value with Set(value) or connect it with ConnectToSource(source, 'outputName').",
    methods=[
        _m("Set", f"(self, value: object, time: {_T})", "bool", "Author the input value."),
        _m("Get", f"(self, time: {_T})", "object", "Authored value (None if unset)."),
        *_INPUT_OUTPUT_METHODS,
    ],
)

_register(
    UsdShade.Output,
    "UsdShade.Output",
    "Shader or material output 'outputs:<name>' (shader.CreateOutput(name, type),\n"
    "material.CreateSurfaceOutput()). Material outputs connect to a shader output:\n"
    "material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface').",
    methods=list(_INPUT_OUTPUT_METHODS),
)

_register(
    Gf.BBox3d,
    "Gf.BBox3d",
    "A box (Gf.Range3d) with a transform (Gf.Matrix4d), returned by ComputeWorldBound /\n"
    "ComputeLocalBound. Axis-aligned extent: bbox.ComputeAlignedRange().",
    methods=[
        _m("__init__", "(self, box: Gf.Range3d, matrix: Gf.Matrix4d)", "None", "Also BBox3d(box)."),
        _m("GetRange", "(self)", "Gf.Range3d", "The untransformed box."),
        _m("GetMatrix", "(self)", "Gf.Matrix4d", "The box's transform."),
        _m(
            "ComputeAlignedRange",
            "(self)",
            "Gf.Range3d",
            "Axis-aligned range of the transformed box.",
        ),
        _m("ComputeCentroid", "(self)", "Gf.Vec3d", "Center of the transformed box."),
    ],
)

_register(
    Gf.Range3d,
    "Gf.Range3d",
    "Axis-aligned box of Gf.Vec3d min/max. Empty when min > max (e.g. bounds of nothing).",
    methods=[
        _m("GetMin", "(self)", "Gf.Vec3d", "Minimum corner."),
        _m("GetMax", "(self)", "Gf.Vec3d", "Maximum corner."),
        _m("GetSize", "(self)", "Gf.Vec3d", "max - min."),
        _m("GetMidpoint", "(self)", "Gf.Vec3d", "Center."),
        _m("IsEmpty", "(self)", "bool", "min > max in some axis."),
    ],
)


# --------------------------------------------------------------------------- invented names
# Names agents guessed that do not exist in usd-core: the AttributeError they raise
# (shown to the model by NOOA's error formatting) names the real API.
_CREATE_SUBSET = (
    "UsdGeom.Subset.CreateGeomSubset(mesh, name, UsdGeom.Tokens.face, indices, "
    "UsdShade.Tokens.materialBind) - doc(UsdGeom.Subset.CreateGeomSubset)"
)
_GET_SUBSETS = (
    "UsdGeom.Subset.GetGeomSubsets(mesh, UsdGeom.Tokens.face, familyName) and filter by "
    "s.GetPrim().GetName(), or UsdGeom.Subset(mesh.GetPrim().GetChild(name)) - doc(UsdGeom.Subset)"
)
_ATTRIBUTE_HINTS = [
    (UsdGeom.Subset, "CreateSubset", f"UsdGeom.Subset has no CreateSubset; use {_CREATE_SUBSET}"),
    (
        UsdGeom.Subset,
        "CreateFamily",
        "UsdGeom.Subset has no CreateFamily: a family exists once a subset uses its familyName "
        "(CreateGeomSubset(..., familyName)); set its type with "
        "UsdGeom.Subset.SetFamilyType(mesh, familyName, UsdGeom.Tokens.partition) - doc(UsdGeom.Subset)",
    ),
    (
        UsdGeom.Subset,
        "GetGeomSubsetByName",
        f"UsdGeom.Subset has no GetGeomSubsetByName; use {_GET_SUBSETS}",
    ),
    (UsdGeom.Subset, "GetGeomSubset", f"UsdGeom.Subset has no GetGeomSubset; use {_GET_SUBSETS}"),
    (UsdGeom.Subset, "GetSubset", f"UsdGeom.Subset has no GetSubset; use {_GET_SUBSETS}"),
    (
        UsdGeom.Subset,
        "GetGeomSubsetFamilies",
        "UsdGeom.Subset has no GetGeomSubsetFamilies; use "
        "UsdGeom.Subset.GetAllGeomSubsetFamilyNames(mesh) - doc(UsdGeom.Subset)",
    ),
    (
        UsdGeom.Subset,
        "GetFamilyTypeAttr",
        "UsdGeom.Subset has no GetFamilyTypeAttr: the family type is stored on the mesh; use "
        "UsdGeom.Subset.GetFamilyType(mesh, familyName) / SetFamilyType(mesh, familyName, "
        "familyType) - doc(UsdGeom.Subset)",
    ),
    (
        UsdGeom.Mesh,
        "CreateGeomSubset",
        f"UsdGeom.Mesh has no CreateGeomSubset; use {_CREATE_SUBSET}",
    ),
    (UsdGeom.Mesh, "CreateSubset", f"UsdGeom.Mesh has no CreateSubset; use {_CREATE_SUBSET}"),
    (
        UsdGeom,
        "GeomSubset",
        "UsdGeom has no GeomSubset; the class is UsdGeom.Subset - doc(UsdGeom.Subset)",
    ),
    (
        UsdGeom,
        "MaterialBindingAPI",
        "UsdGeom has no MaterialBindingAPI; use UsdShade.MaterialBindingAPI "
        "(from pxr import UsdShade) - doc(UsdShade.MaterialBindingAPI)",
    ),
    (
        UsdGeom,
        "BindingAPI",
        "UsdGeom has no BindingAPI; use UsdShade.MaterialBindingAPI - doc(UsdShade.MaterialBindingAPI)",
    ),
    (
        UsdGeom,
        "Material",
        "UsdGeom has no Material; use UsdShade.Material - doc(UsdShade.Material)",
    ),
    (
        UsdShade,
        "BindingAPI",
        "UsdShade has no BindingAPI; use UsdShade.MaterialBindingAPI - doc(UsdShade.MaterialBindingAPI)",
    ),
    (UsdShade, "Subset", "UsdShade has no Subset; use UsdGeom.Subset - doc(UsdGeom.Subset)"),
    (
        Usd,
        "Mesh",
        "Usd has no Mesh; use UsdGeom.Mesh (from pxr import UsdGeom) - doc(UsdGeom.Mesh)",
    ),
    (Usd, "Scope", "Usd has no Scope; use UsdGeom.Scope - doc(UsdGeom.Scope)"),
    (Usd, "Xform", "Usd has no Xform; use UsdGeom.Xform - doc(UsdGeom.Xform)"),
    (Usd, "Material", "Usd has no Material; use UsdShade.Material - doc(UsdShade.Material)"),
    (
        Vt,
        "IntArrayFromPython",
        "Vt has no IntArrayFromPython; use Vt.IntArray(list_of_ints) or "
        "Vt.IntArray.FromNumpy(array) - doc(Vt.IntArray)",
    ),
]
for _owner, _name, _hint in _ATTRIBUTE_HINTS:
    register_attribute_hint(_owner, _name, _hint)
