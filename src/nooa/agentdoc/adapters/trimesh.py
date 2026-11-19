# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc adapter for trimesh: curated doc() views for trimesh, Trimesh and trimesh.graph.

Import to register:

    import nooa.agentdoc.adapters.trimesh

Signatures are read from the installed trimesh with ``inspect.signature``; long type
unions are omitted where they would hide the parameter names. Doc extraction never
calls trimesh functions or evaluates mesh properties.
"""

import trimesh
import trimesh.creation
import trimesh.graph
import trimesh.util
from trimesh import Trimesh

from nooa.agentdoc import spec
from nooa.agentdoc.adapters._libdoc import lib_callable
from nooa.agentdoc.ext import FieldInfo, ModuleInfo, TypeInfo

# Executable idioms (tests run them with np and trimesh).
TRIMESH_IDIOMS = """\
mesh = trimesh.creation.box(extents=(1.0, 1.0, 1.0))      # or trimesh.load(path, force="mesh")
assert mesh.vertices.shape == (8, 3) and mesh.faces.shape == (12, 3)
other = mesh.copy(); other.apply_translation((3.0, 0.0, 0.0))   # apply_* edit in place
both = trimesh.util.concatenate([mesh, other])            # one Trimesh, two pieces
parts = both.split(only_watertight=False)                 # connected pieces -> list[Trimesh]
assert len(parts) == 2
labels = trimesh.graph.connected_component_labels(both.face_adjacency, node_count=len(both.faces))
assert labels.shape == (24,) and labels.max() == 1        # one component id per face
up = both.faces[both.face_normals[:, 2] > 0.9]            # faces whose normal points +Z"""

_PITFALLS = """\
Pitfalls:
- Trimesh(vertices, faces) and trimesh.load merge duplicate vertices by default
  (process=True): vertex count and order can change. Pass process=False to keep them.
- trimesh.load returns a Scene for multi-object files (glb, obj with groups): use
  trimesh.load(path, force="mesh") or scene.dump(concatenate=True) for one Trimesh.
- Properties (face_normals, face_adjacency, ...) are cached and recomputed after edits;
  edit mesh.vertices / mesh.faces by assignment, not by in-place element writes.
- split() keeps only watertight pieces unless only_watertight=False.
- split() and graph.connected_components need scipy or networkx installed
  (ImportError 'no graph engines available!' otherwise)."""


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


@spec.define_doc(trimesh)
def _trimesh_module_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="trimesh",
        docstring=(
            "Triangle meshes: load/save, geometry queries, connectivity. `import trimesh`.\n"
            "print(doc(trimesh.Trimesh)) for the mesh object, doc(trimesh.graph) for\n"
            "connectivity. Build: trimesh.Trimesh(vertices=(n, 3) floats, faces=(m, 3) ints)."
        ),
        functions=[
            lib_callable(
                trimesh.load,
                "Load a file (stl, obj, ply, glb, off, ...): Trimesh or Scene;\n"
                "force='mesh' always returns one Trimesh.",
                annotate=False,
            ),
            lib_callable(
                trimesh.util.concatenate,
                "Merge meshes into one Trimesh (faces re-indexed).",
                name="util.concatenate",
                annotate=False,
            ),
            lib_callable(
                trimesh.creation.box, "Axis-aligned box mesh.", name="creation.box", annotate=False
            ),
            lib_callable(
                trimesh.creation.icosphere,
                "Sphere mesh.",
                name="creation.icosphere",
                annotate=False,
            ),
        ],
        classes=[
            ("Trimesh", "the mesh: vertices, faces, face_normals, face_adjacency, split"),
            ("Scene", "several named geometries with transforms (from multi-object files)"),
        ],
        submodules=[
            ("trimesh.graph", "connectivity: face_adjacency, connected components, split"),
            ("trimesh.creation", "primitives: box, icosphere, cylinder, extrude_polygon"),
            ("trimesh.util", "concatenate and helpers"),
            ("trimesh.proximity", "closest_point, signed_distance"),
            ("trimesh.repair", "fix_normals, fill_holes, fix_winding"),
        ],
    )


_FIELDS = [
    ("vertices", "ndarray (n, 3) float64", "Vertex positions; assign a new array to edit."),
    ("faces", "ndarray (m, 3) int64", "Vertex indices of each triangle."),
    ("face_normals", "ndarray (m, 3) float64", "Unit normal per face."),
    ("vertex_normals", "ndarray (n, 3) float64", "Area-weighted normal per vertex."),
    ("face_adjacency", "ndarray (k, 2) int64", "Pairs of face indices that share an edge."),
    ("face_adjacency_edges", "ndarray (k, 2) int64", "The shared edge (vertex indices) per pair."),
    ("face_adjacency_angles", "ndarray (k,) float64", "Angle between adjacent face normals."),
    ("edges_unique", "ndarray (e, 2) int64", "Each undirected edge once."),
    ("vertex_neighbors", "list[list[int]]", "Neighbouring vertex indices per vertex."),
    ("area", "float", "Total surface area."),
    ("area_faces", "ndarray (m,) float64", "Area per face."),
    ("volume", "float", "Enclosed volume (meaningful for watertight meshes)."),
    ("bounds", "ndarray (2, 3) float64", "[min_xyz, max_xyz]."),
    ("extents", "ndarray (3,) float64", "bounds[1] - bounds[0]."),
    ("centroid", "ndarray (3,) float64", "Area-weighted surface centre."),
    ("is_watertight", "bool", "Every edge is shared by exactly two faces."),
    ("body_count", "int", "Number of connected pieces."),
]


@spec.define_doc(Trimesh)
def _trimesh_doc(cls_or_instance) -> TypeInfo:
    return TypeInfo(
        name="Trimesh",
        base="trimesh.Trimesh",
        fields=[FieldInfo(name=n, type=t, description=d) for n, t, d in _FIELDS],
        methods=[
            lib_callable(
                Trimesh.__init__,
                "Trimesh(vertices, faces); process=False keeps vertices as given.",
                annotate=False,
            ),
            lib_callable(
                Trimesh.split,
                "Connected pieces as list[Trimesh]; kwargs: only_watertight=False,\n"
                "adjacency=..., engine=... (see trimesh.graph.split).",
            ),
            lib_callable(
                Trimesh.submesh, "New mesh(es) from lists of face indices.", annotate=False
            ),
            lib_callable(Trimesh.copy, "Independent copy."),
            lib_callable(Trimesh.apply_transform, "Apply a 4x4 matrix in place.", annotate=False),
            lib_callable(Trimesh.apply_translation, "Translate in place."),
            lib_callable(Trimesh.apply_scale, "Scale in place (float or per-axis)."),
            lib_callable(Trimesh.update_faces, "Keep only faces where mask is True (in place)."),
            lib_callable(Trimesh.remove_unreferenced_vertices, "Drop vertices no face uses."),
            lib_callable(
                Trimesh.merge_vertices, "Merge duplicate vertices in place.", annotate=False
            ),
            lib_callable(Trimesh.fix_normals, "Make winding and normals consistent (in place)."),
            lib_callable(Trimesh.contains, "Inside test per point (needs a watertight mesh)."),
            lib_callable(Trimesh.sample, "Random surface points (count, 3).", annotate=False),
            lib_callable(
                Trimesh.export,
                "Write to a path (format from extension) or return bytes/str.",
                annotate=False,
            ),
        ],
        docstring=("A triangle mesh.\n\nIdioms:\n" + _indent(TRIMESH_IDIOMS) + "\n\n" + _PITFALLS),
    )


@spec.define_doc(trimesh.graph)
def _graph_doc(mod) -> ModuleInfo:
    g = trimesh.graph
    return ModuleInfo(
        name="trimesh.graph",
        docstring=(
            "Mesh connectivity. Faces are graph nodes and mesh.face_adjacency gives edges:\n"
            "    labels = trimesh.graph.connected_component_labels(\n"
            "        mesh.face_adjacency, node_count=len(mesh.faces))  # component id per face\n"
            "    groups = trimesh.graph.connected_components(mesh.face_adjacency,\n"
            "        nodes=np.arange(len(mesh.faces)))  # list of face-index arrays; nodes=\n"
            "                                            # includes isolated faces\n"
            "Same with SciPy: scipy.sparse.csgraph.connected_components on a sparse matrix."
        ),
        functions=[
            lib_callable(g.face_adjacency, "(k, 2) pairs of faces sharing an edge."),
            lib_callable(
                g.connected_components,
                "Groups of connected nodes (list of index arrays; groups smaller than min_len dropped).",
                annotate=False,
            ),
            lib_callable(g.connected_component_labels, "Component label per node (ndarray)."),
            lib_callable(
                g.split, "Split a mesh into connected pieces (list[Trimesh]).", annotate=False
            ),
            lib_callable(g.vertex_adjacency_graph, "networkx.Graph of vertex connectivity."),
            lib_callable(g.shared_edges, "Edges shared by two sets of faces."),
        ],
    )
