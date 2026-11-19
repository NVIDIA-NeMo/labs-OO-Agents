# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc adapter for SciPy: the submodules agents use with meshes and point sets.

Import to register:

    import nooa.agentdoc.adapters.scipy

Covers scipy.sparse.csgraph (connected_components, shortest paths), scipy.spatial
(KDTree/cKDTree, ConvexHull, Delaunay, distance) and scipy.ndimage (label, morphology).
Each public function of these modules gets the condensed numpydoc view (signature,
summary, one line per parameter); KDTree/cKDTree/ConvexHull get curated views. Signatures
come from the installed SciPy. Doc extraction never calls SciPy functions.
"""

import inspect

import scipy
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial
import scipy.spatial.distance

from nooa.agentdoc import spec
from nooa.agentdoc.adapters._libdoc import lib_callable, signature_parts
from nooa.agentdoc.adapters._numpydoc import condensed_callable_info, summary_line
from nooa.agentdoc.ext import CallableInfo, FieldInfo, ModuleInfo, TypeInfo

csgraph = scipy.sparse.csgraph
spatial = scipy.spatial
ndimage = scipy.ndimage

# Executable idioms (tests run them with np, scipy and a trimesh-like faces array).
CSGRAPH_IDIOMS = """\
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components
pairs = np.array([[0, 1], [2, 3]])           # edges, e.g. mesh.face_adjacency
n = 5                                        # number of nodes (faces); node 4 is isolated
graph = coo_array((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
count, labels = connected_components(graph, directed=False)
assert count == 3 and labels.tolist() == [0, 0, 1, 1, 2]"""

SPATIAL_IDIOMS = """\
from scipy.spatial import KDTree, ConvexHull
pts = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0.2, 0.2, 0.2]])
tree = KDTree(pts)
dist, idx = tree.query([[0.9, 0, 0]], k=2)   # (1, 2) arrays: nearest first
assert idx[0, 0] == 1
near = tree.query_ball_point([0, 0, 0], r=0.5)   # list of indices within r
assert sorted(near) == [0, 4]
hull = ConvexHull(pts)                        # needs >= ndim + 1 points, not all coplanar
assert sorted(hull.vertices.tolist()) == [0, 1, 2, 3] and hull.simplices.shape == (4, 3)"""

NDIMAGE_IDIOMS = """\
from scipy import ndimage
mask = np.array([[1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]], dtype=bool)
labels, count = ndimage.label(mask)           # 4-connected by default in 2-D
assert count == 2 and labels[2, 3] == labels[1, 3]
labels8, count8 = ndimage.label(mask, structure=np.ones((3, 3)))   # 8-connected
sizes = ndimage.sum_labels(mask, labels, index=range(1, count + 1))  # pixels per label
assert sizes.tolist() == [2.0, 3.0]"""


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in text.splitlines())


def _summary_infos(module, names) -> list[CallableInfo]:
    infos = []
    for name in names:
        func = getattr(module, name)
        info = condensed_callable_info(func, name)
        infos.append(
            CallableInfo(
                name=name,
                signature=info.signature,
                return_type=info.return_type,
                docstring=summary_line(func),
            )
        )
    return infos


@spec.define_doc(scipy)
def _scipy_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="scipy",
        docstring=(
            "SciPy: scientific algorithms on NumPy arrays. Import submodules explicitly:\n"
            "    from scipy.sparse.csgraph import connected_components\n"
            "    from scipy.spatial import KDTree, ConvexHull\n"
            "    from scipy import ndimage"
        ),
        functions=[],
        submodules=[
            ("scipy.sparse", "sparse matrices/arrays: coo_array, csr_array (graphs as matrices)"),
            ("scipy.sparse.csgraph", "graph algorithms: connected_components, shortest_path"),
            ("scipy.spatial", "KDTree/cKDTree nearest neighbours, ConvexHull, Delaunay"),
            ("scipy.spatial.distance", "cdist (all pairs between two sets), pdist"),
            ("scipy.ndimage", "n-D image arrays: label, binary_dilation, distance_transform_edt"),
            ("scipy.optimize", "minimize, least_squares, linear_sum_assignment"),
            ("scipy.interpolate", "griddata, RBFInterpolator, splines"),
        ],
    )


@spec.define_doc(csgraph)
def _csgraph_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="scipy.sparse.csgraph",
        docstring=(
            "Graph algorithms on an (n, n) adjacency matrix (sparse array/matrix or dense).\n"
            "Mesh pieces from face adjacency:\n"
            + _indent(CSGRAPH_IDIOMS)
            + "\nPass directed=False for undirected graphs (default directed=True with\n"
            "connection='weak' gives the same components). shape must be (n, n) with n =\n"
            "number of nodes, or isolated trailing nodes are lost."
        ),
        functions=_summary_infos(
            csgraph,
            [
                "connected_components",
                "shortest_path",
                "dijkstra",
                "breadth_first_order",
                "minimum_spanning_tree",
                "laplacian",
            ],
        ),
    )


@spec.define_doc(spatial)
def _spatial_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="scipy.spatial",
        docstring=(
            "Spatial data structures and algorithms.\n" + _indent(SPATIAL_IDIOMS) + "\n"
            "All-pairs distances: scipy.spatial.distance.cdist(A, B) -> (len(A), len(B))."
        ),
        functions=[],
        classes=[
            ("KDTree", "nearest neighbours: query, query_ball_point, query_pairs"),
            ("cKDTree", "same API as KDTree (KDTree is now equally fast)"),
            ("ConvexHull", "hull of points: vertices, simplices, volume, area"),
            ("Delaunay", "triangulation: simplices, find_simplex"),
        ],
        submodules=[("scipy.spatial.distance", "cdist, pdist, squareform, metrics")],
    )


def _constructor(cls, docstring: str) -> CallableInfo:
    """``__init__`` from the class signature (cKDTree's own __init__ is ``*args, **kwargs``)."""
    params, _ = signature_parts(cls)
    inner = params[1:-1]
    return CallableInfo(
        name="__init__",
        signature=f"(self, {inner})" if inner else "(self)",
        return_type="None",
        docstring=docstring,
    )


def _tree_doc(cls, name: str, note: str) -> TypeInfo:
    return TypeInfo(
        name=name,
        base=f"scipy.spatial.{name}",
        fields=[
            FieldInfo(name="data", type="ndarray (n, m)", description="The indexed points."),
            FieldInfo(name="n", type="int", description="Number of points."),
            FieldInfo(name="m", type="int", description="Dimensions."),
        ],
        methods=[
            _constructor(cls, f"{name}(points); points: (n, m) array."),
            lib_callable(
                cls.query,
                "(dist, idx) of the k nearest points; k=1 gives arrays shaped like x[..., 0].\n"
                "Missing neighbours (distance_upper_bound) get idx == tree.n, dist == inf.",
            ),
            lib_callable(cls.query_ball_point, "Indices of points within distance r (a list)."),
            lib_callable(cls.query_pairs, "All index pairs (i < j) closer than r."),
            lib_callable(cls.query_ball_tree, "For each point of self, neighbours in other."),
            lib_callable(cls.sparse_distance_matrix, "Sparse distances up to max_distance."),
        ],
        docstring=f"k-d tree for fast nearest-neighbour queries. {note}\n"
        + _indent(SPATIAL_IDIOMS),
    )


@spec.define_doc(spatial.KDTree)
def _kdtree_doc(cls_or_instance) -> TypeInfo:
    return _tree_doc(spatial.KDTree, "KDTree", "Pure-Python wrapper of cKDTree.")


@spec.define_doc(spatial.cKDTree)
def _ckdtree_doc(cls_or_instance) -> TypeInfo:
    return _tree_doc(spatial.cKDTree, "cKDTree", "Same queries as KDTree.")


@spec.define_doc(spatial.ConvexHull)
def _convex_hull_doc(cls_or_instance) -> TypeInfo:
    return TypeInfo(
        name="ConvexHull",
        base="scipy.spatial.ConvexHull",
        fields=[
            FieldInfo(name="points", type="ndarray (n, ndim)", description="Input points."),
            FieldInfo(name="vertices", type="ndarray (k,)", description="Indices of hull points."),
            FieldInfo(
                name="simplices",
                type="ndarray (f, ndim)",
                description="Hull facets as point indices (triangles in 3-D).",
            ),
            FieldInfo(name="equations", type="ndarray (f, ndim + 1)", description="Facet planes."),
            FieldInfo(name="volume", type="float", description="Hull volume (area in 2-D)."),
            FieldInfo(name="area", type="float", description="Surface area (perimeter in 2-D)."),
        ],
        methods=[
            _constructor(spatial.ConvexHull, "Compute the hull (Qhull)."),
        ],
        docstring=(
            "Convex hull of points (Qhull). Raises QhullError for fewer than ndim + 1\n"
            "points or degenerate (e.g. coplanar 3-D) input; qhull_options='QJ' joggles.\n"
            "Facet orientation in simplices is not consistent: use equations for normals."
        ),
    )


@spec.define_doc(ndimage)
def _ndimage_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="scipy.ndimage",
        docstring="N-D image processing on arrays.\n" + _indent(NDIMAGE_IDIOMS),
        functions=_summary_infos(
            ndimage,
            [
                "label",
                "find_objects",
                "sum_labels",
                "binary_dilation",
                "binary_erosion",
                "binary_fill_holes",
                "distance_transform_edt",
                "gaussian_filter",
                "zoom",
            ],
        ),
    )


def _register_functions(module) -> None:
    for name, obj in list(vars(module).items()):
        if name.startswith("_"):
            continue
        if isinstance(obj, type) or not callable(obj) or not inspect.isroutine(obj):
            continue
        if not (getattr(obj, "__module__", "") or "").startswith("scipy"):
            continue  # re-exported numpy etc. functions keep their own adapter
        spec.define_doc(obj)(lambda func, _name=name: condensed_callable_info(func, _name))


for _module in (csgraph, spatial, spatial.distance, ndimage):
    _register_functions(_module)
