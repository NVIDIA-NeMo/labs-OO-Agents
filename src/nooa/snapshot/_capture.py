# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The public capture entry points: ``capture`` (with its registry) and ``capture_fields``.

The top of the capture layers (``_limits`` <- ``_raw`` <- ``_visibility``
<- ``_shapes`` <- ``_walk`` <- ``_capture``). It checks the arguments, starts
one ``_State`` per call and hands the value to the ``_walk`` builder. The
builder is based on the bounded trace JSON builder of PR #342 (Severin
Klingler).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from nooa.snapshot._codec import dumps, fallback_candidates, pointer_child
from nooa.snapshot._limits import DEFAULT_LIMITS, Limits, _State, _validate_limits
from nooa.snapshot._model import Key, Mapping, Object, Snapshot, Unavailable
from nooa.snapshot._raw import _MAX_TYPE_NAME, _bounded_type_name, _builtin_name, _safe_mro
from nooa.snapshot._shapes import (
    _SHAPE_TYPES,
    _Shape,
    array_like,
    mapping_like,
    ndarray_like,
    object_like,
    text_like,
)
from nooa.snapshot._walk import _REGISTRY, _build, _Built, _fallback

__all__ = [
    "Limits",
    "array_like",
    "capture",
    "capture_fields",
    "mapping_like",
    "ndarray_like",
    "object_like",
    "text_like",
]


class _CaptureFunction:
    """``capture(value, limits=Limits()) -> Snapshot``: a bounded, detached snapshot of a value.

    Arguments:

    - ``value``: any Python object.
    - ``limits``: the shared budget (``Limits``); the default is
      ``Limits()``, #342's trace preview bounds.

    Returns a ``Snapshot`` node holding no live objects. What becomes what:

    - exact ``None``, ``bool``, ``int``, finite ``float``: a ``Scalar``;
    - exact ``str``: a chars ``Text``; exact ``bytes``: a ``Scalar`` when
      kept whole, else a bytes ``Text`` of the original bytes;
    - exact ``list`` / ``tuple``: a ``Sequence``; exact ``dict``: a
      ``Mapping``;
    - the standard types, read natively (``_std``): ``set`` / ``frozenset``
      (a ``Sequence``, head only), ``Counter`` / ``OrderedDict`` /
      ``defaultdict`` and other container subclasses that override no
      iteration method (a ``Sequence`` / ``Mapping`` / ``Text`` named after
      their class), and ``Object`` nodes with a ``python.*`` ``kind`` for
      ``range``, ``complex``, ``datetime`` / ``date`` / ``time`` /
      ``timedelta``, ``Decimal``, ``UUID``, ``pathlib`` paths, ``re.Pattern``,
      exceptions (their ``args``), enum members, functions, methods,
      built-in functions, modules and classes; a namedtuple is an
      ``Object`` of its fields;
    - a Pydantic v2 model or a dataclass: an ``Object`` of its visible
      declared fields;
    - an instance of a class registered with ``capture.register``: what its
      shape describes;
    - a plain object (no ``__getattribute__`` in its MRO, the standard
      ``__dict__`` slot or ``__slots__``): an ``Object`` of its public
      instance attributes;
    - anything else, a non-finite ``float`` included: ``Unavailable("opaque")``
      named ``module.QualName``, or with ``Limits(fallback="repr")`` its
      bounded ``repr``.

    Guarantees:

    - No user code runs on values: no ``repr``, ``str``, truthiness,
      properties, descriptors, ``__getattr__``, custom iteration, hashing or
      equality of an inspected value, its class or its metaclass. Instances
      are read from raw storage (``__dict__`` or ``__slots__``) through the
      native descriptors. The only code that runs is class metadata,
      evaluated the way Python does (``typing.get_type_hints(cls,
      include_extras=True)`` for a dataclass with string annotations; a
      success is cached by class identity, a failure is retried by the next
      ``capture``) and the accessors of registered shapes.
    - agentdoc's ``spec(max_string=..., max_length=...)`` on a field bounds
      that field's subtree; ``spec(hide_empty=True)`` (on the class or the
      field) leaves out a ``None`` or empty field.
    - Hidden fields are never read: agentdoc ``hidden`` /
      ``spec(hidden=True)`` markers and ``spec()`` overrides, Pydantic
      ``exclude=True`` / ``repr=False``, dataclass ``repr=False``,
      ``_private``, ``ClassVar`` and ``InitVar`` fields.
    - Fail closed: fields whose visibility cannot be read safely are left
      out and the object gets reason ``"unreadable"``; an accessor that
      raises becomes ``Unavailable("error")``. A value that cannot be read
      becomes a node, not an exception.
    - Bounded: the snapshot's canonical JSON is at most ``max_chars``
      characters, and the work is at most ``max_nodes`` units and
      ``max_depth`` levels (plus what registered accessors spend). A long
      string or bytes value is sliced, never copied whole.
    - Exact accounting: ``render_json(snapshot, max_chars=limits.max_chars)``
      keeps everything the snapshot holds; on #342's domain it is #342's
      text.
    - UTF-8-safe cuts: a cut bytes value starts and ends on code point
      boundaries.

    Raises ``ValueError`` when ``limits`` is out of range (see ``Limits``).

    ``capture.register(T, shape)`` and ``capture.unregister(T)`` extend it
    for other types (see ``register``).

    >>> from nooa.snapshot import Limits, capture, render_python
    >>> capture("x" * 20, Limits(max_string=8, tail_share=0.5))
    Text(head='xxxx', tail='xxxx', length=20, unit='chars', reason='max_length', full=None)
    >>> render_python(capture({"a": [1, 2], "b": object()}))
    "{'a': [1, 2], 'b': <opaque: object>}"
    """

    def __call__(self, value: object, limits: Limits = DEFAULT_LIMITS) -> Snapshot:
        """A bounded snapshot of ``value`` under ``limits``; see ``capture``."""
        _validate_limits(limits)
        state = _State(limits)
        return _build(value, limits.max_chars, 0, "", state).node

    def register(self, cls: type[Any], shape: _Shape) -> None:
        """Capture instances of ``cls`` and its subclasses as ``shape`` describes them.

        Arguments:

        - ``cls``: the class itself (so registering imports it; NOOA never
          imports third-party libraries).
        - ``shape``: what ``array_like``, ``ndarray_like``, ``mapping_like``,
          ``object_like`` or ``text_like`` returned.

        The registry is process-wide. Registering ``cls`` again replaces its
        shape. At capture time the value's raw MRO is compared with the
        registered classes by identity (``is``): the nearest registered
        class wins, also over the Pydantic and dataclass readers. No class
        is hashed or compared with ``==``, and no accessor runs during the
        lookup. Exact built-in types (``str``, ``bytes``, ``int``,
        ``float``, ``bool``, ``None``, ``list``, ``tuple``, ``dict``) are
        always captured natively, whatever is registered.

        Accessors are trusted code. They run only when capture keeps their
        result, and an ``Exception`` they raise becomes
        ``Unavailable("error")`` (see the shape constructors).

        Raises ``TypeError`` when ``cls`` is not a class or ``shape`` is not
        a shape.

        >>> from nooa.snapshot import capture, object_like, render_python
        >>> class Point:
        ...     def __init__(self, x, y):
        ...         self.x, self.y = x, y
        >>> capture.register(Point, object_like({"x": lambda p: p.x, "y": lambda p: p.y}))
        >>> render_python(capture(Point(1, 2)))
        'Point(x=1, y=2)'
        >>> capture.unregister(Point)
        """
        if not isinstance(cls, type):
            raise TypeError(f"capture.register needs a type, got {type(cls).__name__}")
        shape_type = type(shape)
        if not any(shape_type is known for known in _SHAPE_TYPES):
            raise TypeError(
                "capture.register needs a shape (array_like, ndarray_like, mapping_like,"
                f" object_like or text_like), got {_bounded_type_name(shape, _MAX_TYPE_NAME)}"
            )
        _REGISTRY[:] = [entry for entry in _REGISTRY if entry[0] is not cls]
        _REGISTRY.append((cls, shape))

    def unregister(self, cls: type[Any]) -> None:
        """Remove the registration of exactly ``cls`` (not of its bases); a no-op when there is none."""
        _REGISTRY[:] = [entry for entry in _REGISTRY if entry[0] is not cls]


capture = _CaptureFunction()


def _field_placeholder(value: Any, state: _State) -> tuple[Any, Snapshot]:
    """One framework wrapper field's minimal JSON root value, and its node when no work is left.

    An exact built-in type other than ``bytes`` reserves its smallest
    fallback (``""``, ``[]``, ``{}``, ``0``, ...), a Pydantic model ``{}``,
    anything else ``null`` (#342's ``trace_fields``).
    """
    if not state.take_node():
        raise ValueError("snapshot max_nodes cannot classify wrapper fields")
    name = _builtin_name(value)
    if name and name != "bytes":
        shape = fallback_candidates(name)[0]
        return shape, _fallback(value, "max_nodes", len(dumps(shape))).node
    mro = _safe_mro(value, state)
    if mro is not None and any(base is BaseModel for base in mro):
        name = _bounded_type_name(value, _MAX_TYPE_NAME)
        return {}, Object(name, (), "max_nodes")
    return None, Unavailable("max_nodes", "NoneType")


def capture_fields(*, limits: Limits = DEFAULT_LIMITS, **fields: object) -> Mapping:
    """Capture a framework-owned wrapper (``args=..., kwargs=...``) so that every field survives.

    Arguments:

    - ``limits``: the shared budget, as for ``capture`` (keyword-only).
    - ``**fields``: the wrapper's fields, name -> value, in order.

    Returns a ``Mapping`` (``type_name="dict"``, ``wrapper=True``) with one
    entry per field, in order. Each field first reserves the minimal value
    of its own type: ``""``, ``[]``, ``{}``, ``0``, ``0.0``, ``false`` or
    ``null`` for the exact built-in types other than ``bytes``, ``{}`` for a
    Pydantic model, ``null`` for anything else. Then the fields are captured in order, each
    with the room the earlier ones left. A field reached with no
    inspection work left keeps its minimal value (reason ``"max_nodes"``).
    ``render_json`` keeps every field of the wrapper too. This is #342's
    ``trace_fields``; the guarantees of ``capture`` hold for every field.

    Raises ``ValueError`` when ``limits`` is out of range, or when it cannot
    hold the wrapper with every field at its minimal value (``max_chars``),
    or classify every field (``max_nodes``).

    >>> from nooa.snapshot import Limits, capture_fields, render_json
    >>> wrapper = capture_fields(args=["x" * 100], kwargs={}, limits=Limits(max_chars=40))
    >>> render_json(wrapper, max_chars=40)
    '{"args": ["xxxxxxxxxxxx"], "kwargs": {}}'
    """
    _validate_limits(limits)
    state = _State(limits)
    placeholders = {name: _field_placeholder(value, state) for name, value in fields.items()}
    skeleton = dumps({name: shape for name, (shape, _node) in placeholders.items()})
    if len(skeleton) > limits.max_chars:
        raise ValueError("snapshot max_chars cannot hold required wrapper fields")

    items: list[tuple[Key, Snapshot]] = []
    total_size = len(skeleton)
    for name, value in fields.items():
        shape, starved = placeholders[name]
        reserved = len(dumps(shape))
        if state.nodes_left > 0:
            available = reserved + (limits.max_chars - total_size)
            built = _build(value, available, 1, pointer_child("", name), state)
        else:
            built = _Built(starved, reserved)
        items.append((name, built.node))
        total_size += built.size - reserved
    return Mapping("dict", len(items), tuple(items), wrapper=True)
