# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc adapter for NumPy: curated doc() views for numpy, ndarray and its functions.

Import to register:

    import nooa.agentdoc.adapters.numpy

Without it, ``doc(numpy)`` is ~27k lines and ``doc(np.ndarray)`` ~1.5k lines. With it:

- ``doc(numpy)``, ``doc(np.linalg)``, ``doc(np.random)``: a map of what to use.
- ``doc(np.ndarray)``, ``doc(np.random.Generator)``: key attributes and methods with
  the signatures NumPy's own docstrings give (``...`` marks NumPy's "no value" default).
- ``doc(np.zeros)`` and every other public function of numpy, numpy.linalg, numpy.fft
  and numpy.random: signature, NumPy's first summary paragraph and one line per
  parameter/return value (Examples and Notes dropped). See ``_numpydoc``.

Doc extraction never calls NumPy functions; it reads docstrings and signatures.
"""

import inspect

import numpy as np

from nooa.agentdoc import spec
from nooa.agentdoc.adapters._numpydoc import condensed_callable_info, doc_signature, summary_line
from nooa.agentdoc.ext import CallableInfo, FieldInfo, ModuleInfo, TypeInfo

# Executable idioms (tests run them); shown in doc(np.ndarray).
NDARRAY_IDIOMS = """\
a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
assert a.shape == (2, 3) and a.ndim == 2 and a.size == 6 and a.dtype == np.float32
col = a[:, 0]; row = a[1]; big = a[a > 2]       # column, row (views); mask (a copy)
b = a.reshape(-1, 1)                             # (6, 1); -1 infers the size
c = a[:, :, None] * np.ones(4)                   # broadcasting: (2, 3, 1) * (4,) -> (2, 3, 4)
s = a.sum(axis=0); m = a.mean(axis=1, keepdims=True)   # per column (3,); per row (2, 1)
i = np.argwhere(a == a.max())                    # indices of the maximum, shape (k, ndim)
v = a.astype(np.int64).tolist()                  # plain Python ints, e.g. for json.dumps
assert np.allclose(a @ a.T, np.dot(a, a.T))      # @ is the matrix product, * elementwise"""

_NDARRAY_DOC = (
    "NumPy N-dimensional array: one dtype, a fixed shape; arithmetic is elementwise and\n"
    "broadcasts. Create with np.array/np.asarray/np.zeros/np.arange (doc(numpy)).\n\n"
    "Idioms:\n"
    + "\n".join("    " + line for line in NDARRAY_IDIOMS.splitlines())
    + "\n\nPitfalls:\n"
    "- Basic slices are views: b = a[:2]; b[0] = 9 also changes a. Use .copy().\n"
    "- `a == b` is elementwise; `if a:` raises for size > 1. Use np.array_equal,\n"
    "  np.allclose, (a == b).all() or a.any().\n"
    "- Shapes (N,) and (N, 1) broadcast differently: use a[:, None] or reshape(-1, 1).\n"
    "- Integer dtypes wrap on overflow (np.uint8(255) + 1 == 0); astype(int) truncates.\n"
    "- NumPy scalars (np.int64, np.float32) are not int/float for json: use .item()/.tolist().\n"
    "- a.sort() sorts in place and returns None; np.sort(a) returns a sorted copy."
)

# (name, return type, one-line doc or None to use NumPy's own summary line)
_NDARRAY_METHODS = [
    ("reshape", "ndarray", "Same data, new shape (a view when possible); one dim may be -1."),
    ("astype", "ndarray", "Copy cast to dtype, e.g. a.astype(np.float32)."),
    ("copy", "ndarray", None),
    ("tolist", "list", "Nested Python lists of Python scalars (JSON-safe)."),
    ("item", "int | float | bool", "One element as a Python scalar; a.item() needs size 1."),
    ("view", "ndarray", "New view of the same data, optionally reinterpreted as another dtype."),
    ("fill", "None", "Fill in place with a scalar."),
    ("flatten", "ndarray", "1-D copy."),
    ("ravel", "ndarray", "1-D view when possible (else a copy)."),
    ("transpose", "ndarray", "View with axes permuted; a.T reverses all axes."),
    ("swapaxes", "ndarray", None),
    ("squeeze", "ndarray", "Remove length-1 axes."),
    ("sum", "ndarray | scalar", "Sum over axis (all elements if axis=None)."),
    ("mean", "ndarray | scalar", "Mean over axis."),
    ("std", "ndarray | scalar", "Standard deviation over axis (ddof=0: population)."),
    ("min", "ndarray | scalar", "Minimum over axis."),
    ("max", "ndarray | scalar", "Maximum over axis."),
    ("argmin", "ndarray | int", "Index of the minimum (flat index if axis=None)."),
    ("argmax", "ndarray | int", "Index of the maximum (flat index if axis=None)."),
    ("cumsum", "ndarray", None),
    ("any", "ndarray | bool", "True if any element is truthy (over axis)."),
    ("all", "ndarray | bool", "True if all elements are truthy (over axis)."),
    ("nonzero", "tuple[ndarray, ...]", "Indices of non-zero elements, one array per dimension."),
    ("sort", "None", "Sort in place (returns None); np.sort(a) returns a copy."),
    ("argsort", "ndarray", "Indices that would sort the array."),
    ("clip", "ndarray", "Limit values to [min, max]."),
    ("round", "ndarray", "Round to the given number of decimals."),
    ("dot", "ndarray", "Matrix/dot product (same as a @ other for 2-D)."),
    ("take", "ndarray", "Elements at the given indices (along axis)."),
    ("repeat", "ndarray", None),
    ("tobytes", "bytes", "Raw data bytes."),
]

_NDARRAY_FIELDS = [
    ("shape", "tuple[int, ...]", "Dimensions, e.g. (rows, cols)."),
    ("dtype", "np.dtype", "Element type, e.g. float64, int32, uint8, bool."),
    ("ndim", "int", "Number of dimensions (len(a.shape))."),
    ("size", "int", "Number of elements."),
    ("T", "ndarray", "Transposed view."),
    ("itemsize", "int", "Bytes per element."),
    ("nbytes", "int", "Total bytes of the data."),
    ("flat", "np.flatiter", "1-D iterator over all elements."),
]


def _method_signature(owner: type, name: str) -> str:
    """The signature NumPy's docstring gives for ``owner.name``, with ``self`` first."""
    method = getattr(owner, name)
    sig = doc_signature(method, name)
    if sig is None:
        sig = str(inspect.signature(method)).replace("(self, /", "(").replace("(self", "(")
    inner = sig[1:-1].replace("<no value>", "...").strip().lstrip(",").strip()
    return f"(self, {inner})" if inner else "(self)"


def _methods(owner: type, table) -> list[CallableInfo]:
    return [
        CallableInfo(
            name=name,
            signature=_method_signature(owner, name),
            return_type=return_type,
            docstring=one_line or summary_line(getattr(owner, name)),
        )
        for name, return_type, one_line in table
    ]


@spec.define_doc(np.ndarray)
def _ndarray_doc(cls_or_instance) -> TypeInfo:
    return TypeInfo(
        name="ndarray",
        base="numpy.ndarray",
        fields=[FieldInfo(name=n, type=t, description=d) for n, t, d in _NDARRAY_FIELDS],
        methods=_methods(np.ndarray, _NDARRAY_METHODS),
        docstring=_NDARRAY_DOC,
    )


_GENERATOR_METHODS = [
    ("integers", "ndarray | int", "Random ints in [low, high) (endpoint=True: [low, high])."),
    ("random", "ndarray | float", "Uniform floats in [0, 1)."),
    ("uniform", "ndarray | float", "Uniform floats in [low, high)."),
    ("normal", "ndarray | float", "Gaussian samples (loc=mean, scale=std)."),
    ("standard_normal", "ndarray | float", "Samples from N(0, 1)."),
    ("choice", "ndarray | scalar", "Random sample from a 1-D array or range(a)."),
    ("permutation", "ndarray", "Shuffled copy (or a permuted range(x) for an int)."),
    ("shuffle", "None", "Shuffle in place along axis."),
]


@spec.define_doc(np.random.Generator)
def _generator_doc(cls_or_instance) -> TypeInfo:
    return TypeInfo(
        name="Generator",
        base="numpy.random.Generator",
        fields=[],
        methods=_methods(np.random.Generator, _GENERATOR_METHODS),
        docstring=(
            "Random number generator. Create one, seeded for reproducibility:\n"
            "    rng = np.random.default_rng(0)\n"
            "    rng.integers(0, 10, size=3); rng.random((2, 2)); rng.normal(0, 1, 5)\n"
            "Prefer this over the legacy global np.random.rand/np.random.seed functions."
        ),
    )


_NUMPY_DOC = """\
NumPy: N-dimensional arrays (np.ndarray) and fast vectorized math. `import numpy as np`.
print(doc(np.ndarray)) for array attributes/methods; print(doc(np.zeros)) etc. for a
function's signature and parameters.

Create:   np.array(obj, dtype=None), np.asarray(a), np.zeros(shape), np.ones(shape),
          np.full(shape, value), np.empty(shape), np.arange(start, stop, step),
          np.linspace(start, stop, num), np.eye(n), np.zeros_like(a), np.stack/np.concatenate
Shape:    a.reshape, a.T, np.transpose, np.expand_dims, np.squeeze, np.vstack, np.hstack,
          np.split, np.broadcast_to, np.tile, np.repeat
Math:     + - * / ** @ (elementwise; @ = matmul), np.sqrt, np.exp, np.log, np.abs, np.sin,
          np.cos, np.arctan2, np.clip, np.round, np.floor, np.dot, np.cross, np.matmul
Reduce:   np.sum, np.mean, np.median, np.std, np.min, np.max, np.argmin, np.argmax,
          np.cumsum, np.prod, np.any, np.all, np.count_nonzero (all take axis=, keepdims=)
Select:   a[mask], np.where(cond, x, y), np.nonzero, np.argwhere, np.unique, np.isin,
          np.sort, np.argsort, np.searchsorted, np.take, np.isnan, np.isfinite
Compare:  np.array_equal, np.allclose, np.isclose (not ==, which is elementwise)
Linalg:   np.linalg.norm, inv, solve, det, eig, eigh, svd, lstsq, pinv (doc(np.linalg))
Random:   rng = np.random.default_rng(seed); rng.integers/random/normal/choice (doc(np.random))
I/O:      np.save/np.load (.npy), np.savez (.npz), np.loadtxt/np.savetxt/np.genfromtxt (text)
Types:    np.float32, np.float64, np.int32, np.int64, np.uint8, np.bool_; np.nan, np.inf, np.pi"""

_NUMPY_SUBMODULES = [
    ("numpy.linalg", "norm, inv, solve, det, eig/eigh, svd, lstsq, pinv, matrix_rank"),
    ("numpy.random", "default_rng() -> Generator (integers, random, normal, choice)"),
    ("numpy.fft", "fft, ifft, rfft, fft2, fftfreq, fftshift"),
    ("numpy.testing", "assert_allclose, assert_array_equal (tests)"),
    ("numpy.typing", "NDArray[np.float64], ArrayLike (type hints)"),
]


@spec.define_doc(np)
def _numpy_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="numpy",
        docstring=_NUMPY_DOC,
        functions=[],
        classes=[
            ("ndarray", "the array type: shape, dtype, reshape, astype, sum, ..."),
            ("dtype", "element type object: np.dtype('float32'), a.dtype.kind"),
        ],
        submodules=_NUMPY_SUBMODULES,
    )


def _summary_infos(module, names) -> list[CallableInfo]:
    """Signature + NumPy's first summary sentence for each function."""
    infos = []
    for name in names:
        info = condensed_callable_info(getattr(module, name), name)
        infos.append(
            CallableInfo(
                name=name,
                signature=info.signature,
                return_type=info.return_type,
                docstring=summary_line(getattr(module, name)),
            )
        )
    return infos


_LINALG_FUNCTIONS = [
    "norm", "inv", "solve", "det", "eig", "eigh", "svd", "lstsq", "pinv", "matrix_rank", "qr",
    "cholesky",
]  # fmt: skip


@spec.define_doc(np.linalg)
def _linalg_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="numpy.linalg",
        docstring=(
            "Linear algebra on arrays (batched over leading dimensions).\n"
            "Row norms of an (N, 3) array: np.linalg.norm(v, axis=1).\n"
            "Prefer solve(A, b) over inv(A) @ b. Full details: print(doc(np.linalg.svd))."
        ),
        functions=_summary_infos(np.linalg, _LINALG_FUNCTIONS),
    )


@spec.define_doc(np.random)
def _random_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="numpy.random",
        docstring=(
            "Random numbers. Use a Generator (print(doc(np.random.Generator))):\n"
            "    rng = np.random.default_rng(0)      # seed for reproducibility\n"
            "    rng.integers(0, 10, size=5); rng.random(3); rng.normal(0.0, 1.0, (2, 2))\n"
            "    rng.choice(a, size=2, replace=False); rng.permutation(n)\n"
            "Legacy global-state functions (np.random.rand, randint, seed) still work but\n"
            "share one hidden state; avoid them in new code."
        ),
        functions=_summary_infos(np.random, ["default_rng"]),
        classes=[("Generator", "rng methods: integers, random, uniform, normal, choice")],
    )


def _is_public_function(obj) -> bool:
    if isinstance(obj, type) or not callable(obj):
        return False
    return inspect.isroutine(obj) or isinstance(obj, np.ufunc) or hasattr(obj, "__wrapped__")


def _register_functions(module) -> None:
    # vars(), not dir()/getattr(): numpy's module __getattr__ would import lazy
    # submodules (numpy.testing, numpy.ma, ...) and make the first doc() call slow.
    for name, obj in list(vars(module).items()):
        if name.startswith("_"):
            continue
        if not _is_public_function(obj):
            continue
        spec.define_doc(obj)(lambda func, _name=name: condensed_callable_info(func, _name))


for _module in (np, np.linalg, np.fft, np.random):
    _register_functions(_module)
