# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save and revive a session namespace's data, helpers and classes across restarts.

What is kept, per name, with a size cap on each and on the whole:

* **Plain data**: ``None``, ``bool``, ``int``, ``float``, ``str``, ``bytes``, and
  ``list``/``tuple``/``set``/``frozenset``/``dict`` of plain data (dict keys are
  strings or integers). Encoded with msgpack; decoding builds only these types.
* **What ``self.v`` accepts**: Pydantic models, dataclasses, ``@snapshotable``
  objects and enums of importable classes (and containers holding them) go
  through the same serializer and class allowlist as ``self.v``
  (``nooa.storage.serialization``), so a cell variable is kept under exactly
  the rule ``self.v`` already applies.
* **Helper functions defined in a cell**: kept as their source text and defined
  again on restore. Only a plain ``def``/``async def`` qualifies: no decorators,
  literal default values, no closure, defined in this namespace. Annotations are
  dropped before the definition runs, because they would be evaluated. Defining
  the function runs no user code; its body runs only when a later cell calls it.
* **Classes defined in a cell**, handled the same way as helpers: kept as source
  and defined again on restore, if defining them runs only library code. The
  base is ``BaseModel``, ``Enum``/``StrEnum``/``IntEnum`` or none; the only
  decorator is ``@dataclass``; the body holds a docstring, literal assignments,
  annotated fields with literal defaults and plain methods that meet the helper
  rules (``property``/``staticmethod``/``classmethod`` allowed; no ``__init__``,
  ``__post_init__``, ``model_post_init`` or other hooks that run on construction;
  no ``model_config``). Field annotations are limited to names, attributes,
  subscripts and unions, so evaluating them resolves types and calls nothing.
* **Instances of those classes**: kept as field data and rebuilt once the class
  is defined again (``model_validate``, the dataclass ``__init__``, the enum
  lookup, or ``__dict__`` for a plain class), so only library code runs.

Everything else is skipped and reported by name with a reason: modules (re-import
them), lambdas, decorated or closed-over functions, classes outside the rules,
objects of other types, and values over the caps. Nothing in a saved payload can
select a class to instantiate or code to run on load beyond these rules, so a
tampered or corrupt payload costs at most the names it carries.
"""

from __future__ import annotations

import ast
import base64
import dataclasses
import enum
import importlib
import json
import logging
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import msgpack
from pydantic import BaseModel

from nooa.errors.storage import DeserializationError, SerializationError
from nooa.storage.serialization import SKIP, deserialize, serialize

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
MAX_VARIABLE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
# The attribute ``ExecutionNamespace`` attaches to every function and class a cell defines.
SOURCE_ATTRIBUTE = "_generated_source"
# Marker of a rebuilt cell-class instance inside saved data; a real dict may not use it.
INSTANCE_KEY = "__cell_instance__"

_TUPLE, _SET, _FROZENSET = range(3)

# Library names a restorable class definition may use, by the dotted text in its source.
CLASS_BASES: dict[str, type] = {
    "BaseModel": BaseModel,
    "pydantic.BaseModel": BaseModel,
    "Enum": enum.Enum,
    "enum.Enum": enum.Enum,
    "StrEnum": enum.StrEnum,
    "enum.StrEnum": enum.StrEnum,
    "IntEnum": enum.IntEnum,
    "enum.IntEnum": enum.IntEnum,
}
CLASS_DECORATORS: dict[str, Any] = {
    "dataclass": dataclasses.dataclass,
    "dataclasses.dataclass": dataclasses.dataclass,
}
METHOD_DECORATORS: dict[str, Any] = {
    "property": property,
    "staticmethod": staticmethod,
    "classmethod": classmethod,
}
# Methods that run when a class is created or an instance is built or accessed.
BANNED_METHODS = frozenset(
    {
        "__init__",
        "__new__",
        "__post_init__",
        "__init_subclass__",
        "__set_name__",
        "__class_getitem__",
        "__getattr__",
        "__getattribute__",
        "__setattr__",
        "__delattr__",
        "__del__",
        "__setstate__",
        "__reduce__",
        "__reduce_ex__",
        "__enter__",
        "__exit__",
    }
)


class NotPlainData(TypeError):
    """A value that cannot be kept: not plain data."""


# ---------------------------------------------------------------- values


def _encode(obj: Any) -> Any:
    # ``strict_types=True`` sends tuples and dict/list subclasses here too.
    if type(obj) is tuple:
        return msgpack.ExtType(_TUPLE, pack_value(list(obj)))
    if type(obj) is set:
        return msgpack.ExtType(_SET, pack_value(list(obj)))
    if type(obj) is frozenset:
        return msgpack.ExtType(_FROZENSET, pack_value(list(obj)))
    raise NotPlainData(f"{type(obj).__name__} is not plain data")


def _decode_ext(code: int, data: bytes) -> Any:
    if code == _TUPLE:
        return tuple(unpack_value(data))
    if code == _SET:
        return set(unpack_value(data))
    if code == _FROZENSET:
        return frozenset(unpack_value(data))
    raise ValueError(f"unknown extension type {code}")


def pack_value(value: Any) -> bytes:
    """msgpack bytes for plain data; :class:`NotPlainData` for anything else."""
    try:
        return msgpack.packb(value, default=_encode, strict_types=True, use_bin_type=True)
    except (TypeError, ValueError, OverflowError) as exc:
        if isinstance(exc, NotPlainData):
            raise
        raise NotPlainData(str(exc)) from exc


def unpack_value(blob: bytes) -> Any:
    """The plain data in ``blob``. Only builtin container and scalar types are produced."""
    return msgpack.unpackb(blob, ext_hook=_decode_ext, raw=False, strict_map_key=False)


def _importable(fqn: str) -> bool:
    """Whether the snapshot serializer's class reference resolves to a real import."""
    module_name, _, qualname = fqn.rpartition(".")
    if not module_name or not qualname:
        return False
    try:
        target: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    except Exception:  # noqa: BLE001 - any failure means "not importable"
        return False
    return isinstance(target, type)


def serialize_like_vars(value: Any) -> tuple[Any, set[str]]:
    """The ``self.v`` serializer's JSON form of ``value`` and the classes it names.

    Raises :class:`NotPlainData` when ``self.v`` would not keep the value either,
    or when it names a class with no importable home (one defined in a cell).
    """
    try:
        blob, allowlist = serialize(value)
    except SerializationError as exc:
        raise NotPlainData(str(exc).splitlines()[0][:160]) from None
    if blob is SKIP:
        raise NotPlainData("marked nosnapshot")
    for fqn in allowlist:
        if not _importable(fqn):
            raise NotPlainData(
                f"{fqn.rpartition('.')[2]} is a class defined in a cell that is not restorable"
            )
    return blob, allowlist


# ------------------------------------------------------------- functions


def _literal(node: ast.AST) -> bool:
    try:
        ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False
    return True


def _literal_or_name(node: ast.AST) -> bool:
    """A literal, or a plain/dotted name such as ``Color.RED``: evaluating it calls nothing."""
    return _literal(node) or _dotted(node) is not None


def _strip_annotations(node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    for arg in [
        *node.args.posonlyargs,
        *node.args.args,
        *node.args.kwonlyargs,
        node.args.vararg,
        node.args.kwarg,
    ]:
        if arg is not None:
            arg.annotation = None
    node.returns = None


def _dotted(node: ast.AST) -> str | None:
    """``a.b.c`` for a Name/Attribute chain, else ``None``."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _check_function(node: ast.FunctionDef | ast.AsyncFunctionDef, *, method: bool) -> None:
    """Why defining ``node`` could run code, as :class:`ValueError`; strips annotations."""
    for decorator in node.decorator_list:
        if not method or _dotted(decorator) not in METHOD_DECORATORS:
            raise ValueError("decorators run code when the function is defined")
    if method and (node.name in BANNED_METHODS or node.name.startswith(("model_", "__pydantic"))):
        raise ValueError(f"method {node.name!r} would run when the class or an instance is built")
    for default in [*node.args.defaults, *node.args.kw_defaults]:
        if default is not None and not _literal_or_name(default):
            raise ValueError("default values must be literals or plain names")
    _strip_annotations(node)


def validate_function_source(source: str, name: str) -> ast.Module:
    """The module AST of one restorable function definition, annotations removed.

    Raises :class:`ValueError` with a reason when defining the function could
    run code: a decorator, a non-literal default value, anything but a single
    ``def``/``async def`` bound to ``name``.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"source does not parse ({exc.msg})") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], (ast.FunctionDef, ast.AsyncFunctionDef)):
        raise ValueError("source is not a single function definition")
    node = tree.body[0]
    if node.name != name:
        raise ValueError(f"source defines {node.name!r}, not {name!r}")
    _check_function(node, method=False)
    ast.fix_missing_locations(tree)
    return tree


def _function_source(value: Any, name: str, owner_globals: dict[str, Any]) -> str:
    """The restorable source of a cell-defined function, or :class:`ValueError` with why not."""
    if value.__globals__ is not owner_globals:
        raise ValueError("defined outside this session's cells")
    source = getattr(value, SOURCE_ATTRIBUTE, None)
    if not isinstance(source, str):
        raise ValueError("no source text (a lambda, or not defined in a cell)")
    if value.__closure__:
        raise ValueError("closes over another scope")
    validate_function_source(source, name)
    return source


# --------------------------------------------------------------- classes


def _annotation_ok(node: ast.AST) -> bool:
    """Names, attributes, subscripts, unions and string/None constants: no calls."""
    if isinstance(node, ast.Constant):
        return node.value is None or isinstance(node.value, str)
    if isinstance(node, ast.Name):
        return True
    if isinstance(node, ast.Attribute):
        return _dotted(node) is not None
    if isinstance(node, ast.Subscript):
        inner = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        return _annotation_ok(node.value) and all(_annotation_ok(e) for e in inner)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _annotation_ok(node.left) and _annotation_ok(node.right)
    if isinstance(node, (ast.List, ast.Tuple)):
        return all(_annotation_ok(e) for e in node.elts)
    return False


@dataclass(frozen=True)
class ClassSpec:
    kind: str  # "model" | "enum" | "dataclass" | "plain"
    needs: dict[str, Any]  # library names the definition uses -> the real objects


def validate_class_source(source: str, name: str) -> tuple[ast.Module, ClassSpec]:
    """The module AST of one restorable class definition, and what it needs to run.

    Raises :class:`ValueError` with a reason when defining the class could run
    anything but library code.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ValueError(f"source does not parse ({exc.msg})") from None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.ClassDef):
        raise ValueError("source is not a single class definition")
    node = tree.body[0]
    if node.name != name:
        raise ValueError(f"source defines {node.name!r}, not {name!r}")
    needs: dict[str, Any] = {}
    kind = "plain"
    if node.keywords:
        raise ValueError("class keywords (metaclass=...) run code")
    if len(node.bases) > 1:
        raise ValueError("more than one base class")
    for base in node.bases:
        dotted = _dotted(base)
        if dotted not in CLASS_BASES:
            raise ValueError(f"base {dotted or ast.unparse(base)!r} is not BaseModel or Enum")
        needs[dotted.split(".")[0]] = _root_object(dotted)
        kind = "model" if CLASS_BASES[dotted] is BaseModel else "enum"
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        dotted = _dotted(target)
        if dotted not in CLASS_DECORATORS or kind != "plain":
            raise ValueError("only @dataclass is allowed as a class decorator")
        if isinstance(decorator, ast.Call) and (
            decorator.args or any(not _literal(k.value) for k in decorator.keywords)
        ):
            raise ValueError("@dataclass arguments must be literal keywords")
        needs[dotted.split(".")[0]] = _root_object(dotted)
        kind = "dataclass"
    for statement in node.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue  # docstring
        if isinstance(statement, ast.Pass):
            continue
        if isinstance(statement, ast.AnnAssign):
            if not isinstance(statement.target, ast.Name) or _reserved_field(statement.target.id):
                raise ValueError("fields must be plain names")
            if not _annotation_ok(statement.annotation):
                raise ValueError(f"annotation of {statement.target.id!r} is not a plain type")
            if statement.value is not None and not _literal_or_name(statement.value):
                raise ValueError(f"default of {statement.target.id!r} must be a literal or name")
            continue
        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                if not isinstance(target, ast.Name) or _reserved_field(target.id):
                    raise ValueError("assignments must bind plain names")
            if not _literal(statement.value):
                raise ValueError("class attributes must be literals")
            continue
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _check_function(statement, method=True)
            for decorator in statement.decorator_list:
                needs[_dotted(decorator)] = METHOD_DECORATORS[_dotted(decorator)]
            continue
        raise ValueError(f"a class body may not contain {type(statement).__name__}")
    ast.fix_missing_locations(tree)
    return tree, ClassSpec(kind, needs)


def _reserved_field(name: str) -> bool:
    return name.startswith("__") or name.startswith(("model_", "_generated")) or name == "__slots__"


def _root_object(dotted: str) -> Any:
    """The object the first segment of a dotted library name must resolve to."""
    root = dotted.split(".")[0]
    if root in CLASS_BASES:
        return CLASS_BASES[root]
    if root in CLASS_DECORATORS:
        return CLASS_DECORATORS[root]
    return {"enum": enum, "pydantic": sys.modules["pydantic"], "dataclasses": dataclasses}[root]


def _class_source(value: type, name: str, owner_globals: dict[str, Any]) -> tuple[str, ClassSpec]:
    source = getattr(value, SOURCE_ATTRIBUTE, None)
    if not isinstance(source, str):
        raise ValueError("no source text (not defined in a cell)")
    if owner_globals.get(name) is not value:
        raise ValueError("not the class bound to this name")
    _tree, spec = validate_class_source(source, name)
    return source, spec


# -------------------------------------------------------------- instances


def _cell_class_of(value: Any, classes: dict[str, type]) -> str | None:
    for name, cls in classes.items():
        if type(value) is cls:
            return name
    return None


def encode_instances(value: Any, classes: dict[str, type], depth: int = 0) -> Any:
    """Replace instances of cell-defined classes inside ``value`` by field-data markers."""
    if depth > 64:
        raise NotPlainData("nested too deeply")
    cname = _cell_class_of(value, classes)
    if cname is not None:
        if isinstance(value, enum.Enum):
            data: Any = value.value
        elif isinstance(value, BaseModel):
            data = {
                f: encode_instances(getattr(value, f), classes, depth + 1)
                for f in type(value).model_fields
            }
        elif dataclasses.is_dataclass(value):
            data = {
                f.name: encode_instances(getattr(value, f.name), classes, depth + 1)
                for f in dataclasses.fields(value)
            }
        else:
            data = {
                k: encode_instances(v, classes, depth + 1)
                for k, v in vars(value).items()
                if isinstance(k, str) and not k.startswith("_")
            }
        return {INSTANCE_KEY: cname, "data": data}
    if type(value) is dict:
        if INSTANCE_KEY in value:
            raise NotPlainData(f"dict key {INSTANCE_KEY!r} is reserved")
        return {k: encode_instances(v, classes, depth + 1) for k, v in value.items()}
    if type(value) in (list, tuple, set, frozenset):
        return type(value)(encode_instances(v, classes, depth + 1) for v in value)
    return value


def decode_instances(value: Any, classes: dict[str, type], depth: int = 0) -> Any:
    """Rebuild cell-class instances from markers; only library constructors run."""
    if depth > 64:
        raise ValueError("nested too deeply")
    if type(value) is dict and isinstance(value.get(INSTANCE_KEY), str):
        cname = value[INSTANCE_KEY]
        if cname not in classes:
            raise ValueError(f"class {cname!r} was not restored")
        cls = classes[cname]
        data = decode_instances(value.get("data"), classes, depth + 1)
        if issubclass(cls, enum.Enum):
            return cls(data)
        if not isinstance(data, dict):
            raise ValueError(f"fields of {cname!r} are not a mapping")
        if issubclass(cls, BaseModel):
            return cls.model_validate(data)
        if dataclasses.is_dataclass(cls):
            return cls(**data)
        instance = cls.__new__(cls)
        instance.__dict__.update(data)
        return instance
    if type(value) is dict:
        return {k: decode_instances(v, classes, depth + 1) for k, v in value.items()}
    if type(value) in (list, tuple, set, frozenset):
        return type(value)(decode_instances(v, classes, depth + 1) for v in value)
    return value


# ---------------------------------------------------------------- capture


@dataclass
class CaptureReport:
    saved: list[str] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)
    bytes: int = 0


def _skip(report: CaptureReport, name: str, value: Any, reason: str) -> None:
    report.skipped.append({"name": name, "type": type(value).__name__[:80], "reason": reason[:200]})


def capture(
    namespace: Any,
    *,
    max_variable_bytes: int = MAX_VARIABLE_BYTES,
    max_total_bytes: int = MAX_TOTAL_BYTES,
) -> dict[str, Any]:
    """A JSON-ready snapshot of ``namespace``'s user names (an ``ExecutionNamespace``).

    Newest bindings are kept first when the total cap bites; classes are listed
    in definition order so they can be defined again in that order. The result
    carries every skipped name with its reason, so a restore notice can be honest.
    """
    report = CaptureReport()
    values: dict[str, str] = {}
    models: dict[str, Any] = {}
    allowlist: set[str] = set()
    functions: dict[str, str] = {}
    classes: dict[str, str] = {}
    instances: dict[str, str] = {}
    globals_ = namespace.globals
    bound_in = getattr(namespace, "_bound_in", {})
    names = sorted(
        (n for n in namespace.user_names if n in globals_ and not n.startswith("_")),
        key=lambda n: (-bound_in.get(n, 0), n),
    )
    # Pass 1: restorable classes, so instances of them can be recognised.
    cell_classes: dict[str, type] = {}
    for name in names:
        value = globals_[name]
        if isinstance(value, type):
            try:
                source, _spec = _class_source(value, name, globals_)
            except ValueError as exc:
                _skip(report, name, value, f"class not kept: {exc}")
                continue
            cell_classes[name] = value
            classes[name] = source
    total = 0
    for name in sorted(classes, key=lambda n: bound_in.get(n, 0)):
        payload = classes[name].encode("utf-8")
        if len(payload) > max_variable_bytes or total + len(payload) > max_total_bytes:
            _skip(report, name, globals_[name], "class source exceeds the size cap")
            del classes[name], cell_classes[name]
            continue
        total += len(payload)
        report.saved.append(name)
    classes = {n: classes[n] for n in sorted(classes, key=lambda n: bound_in.get(n, 0))}
    # Pass 2: everything else.
    for name in names:
        value = globals_[name]
        if isinstance(value, type):
            continue
        if isinstance(value, types.ModuleType):
            _skip(report, name, value, "module: import it again")
            continue
        if isinstance(value, types.FunctionType):
            try:
                source = _function_source(value, name, globals_)
            except ValueError as exc:
                _skip(report, name, value, f"function not kept: {exc}")
                continue
            payload = source.encode("utf-8")
            kind = "function"
        else:
            kind = "value"
            try:
                payload = pack_value(value)
            except RecursionError:
                _skip(report, name, value, "not plain data: nested too deeply")
                continue
            except NotPlainData:
                try:
                    # Instances of this session's own classes, by field data.
                    payload = pack_value(encode_instances(value, cell_classes))
                    kind = "instance"
                except (NotPlainData, RecursionError):
                    if hasattr(type(value), SOURCE_ATTRIBUTE):
                        _skip(
                            report,
                            name,
                            value,
                            f"instance of {type(value).__name__}, a cell class that is not restorable",
                        )
                        continue
                    # Else keep it if (and only if) ``self.v`` would.
                    try:
                        blob, needed = serialize_like_vars(value)
                    except NotPlainData as exc:
                        _skip(report, name, value, f"not kept: {exc}")
                        continue
                    payload = json.dumps(blob, sort_keys=True).encode("utf-8")
                    kind = "model"
        if len(payload) > max_variable_bytes:
            _skip(
                report,
                name,
                value,
                f"{len(payload)} bytes exceeds the per-name cap of {max_variable_bytes}",
            )
            continue
        if total + len(payload) > max_total_bytes:
            _skip(
                report,
                name,
                value,
                f"total saved state would exceed the cap of {max_total_bytes} bytes",
            )
            continue
        total += len(payload)
        if kind == "function":
            functions[name] = source
        elif kind == "model":
            models[name] = blob
            allowlist |= needed
        elif kind == "instance":
            instances[name] = base64.b64encode(payload).decode("ascii")
        else:
            values[name] = base64.b64encode(payload).decode("ascii")
        report.saved.append(name)
    report.bytes = total
    return {
        "version": FORMAT_VERSION,
        "python": sys.version.split()[0],
        "values": values,
        "models": models,
        "allowlist": sorted(allowlist),
        "classes": classes,
        "instances": instances,
        "functions": functions,
        "skipped": report.skipped,
        "bytes": total,
    }


# ---------------------------------------------------------------- restore


@dataclass
class RestoreReport:
    restored: list[str] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    # Names the capture left out, with its reasons, so the notice can list them too.
    not_saved: list[dict[str, str]] = field(default_factory=list)


@dataclass
class Staged:
    """A decoded snapshot, nothing bound yet."""

    values: dict[str, Any] = field(default_factory=dict)
    functions: dict[str, tuple[str, ast.Module]] = field(default_factory=dict)
    classes: dict[str, tuple[str, ast.Module, ClassSpec]] = field(default_factory=dict)
    instances: dict[str, Any] = field(default_factory=dict)  # still encoded: need the classes
    report: RestoreReport = field(default_factory=RestoreReport)

    @property
    def names(self) -> list[str]:
        return sorted([*self.values, *self.functions, *self.classes, *self.instances])


def _valid_name(name: Any) -> bool:
    return isinstance(name, str) and name.isidentifier() and not name.startswith("_")


def decode(
    snapshot: dict[str, Any],
    *,
    max_variable_bytes: int = MAX_VARIABLE_BYTES,
) -> Staged:
    """Stage a snapshot without touching any namespace. Every problem costs only its name."""
    staged = Staged()
    report = staged.report
    if not isinstance(snapshot, dict) or snapshot.get("version") != FORMAT_VERSION:
        report.failed.append({"name": "*", "reason": "unknown saved-state format"})
        return staged
    skipped = snapshot.get("skipped")
    if isinstance(skipped, list):
        report.not_saved = [s for s in skipped if isinstance(s, dict) and "name" in s]
    sections = {
        key: snapshot.get(key) or {}
        for key in ("values", "models", "classes", "instances", "functions")
    }
    allowlist = snapshot.get("allowlist") or []
    if not isinstance(allowlist, list) or any(not isinstance(s, dict) for s in sections.values()):
        report.failed.append({"name": "*", "reason": "corrupt saved state"})
        return staged
    allowed = {fqn for fqn in allowlist if isinstance(fqn, str)}

    def fail(name: Any, reason: str) -> None:
        report.failed.append({"name": str(name)[:80], "reason": reason[:200]})

    for name, encoded in sections["values"].items():
        if not _valid_name(name):
            fail(name, "not a valid name")
            continue
        try:
            blob = base64.b64decode(encoded, validate=True)
            if len(blob) > max_variable_bytes:
                raise ValueError("exceeds the per-name cap")
            staged.values[name] = unpack_value(blob)
        except Exception as exc:  # noqa: BLE001 - a bad record fails just its name
            fail(name, f"value could not be decoded ({exc})")
    for name, blob in sections["models"].items():
        if not _valid_name(name):
            fail(name, "not a valid name")
            continue
        try:
            if len(json.dumps(blob, sort_keys=True).encode("utf-8")) > max_variable_bytes:
                raise ValueError("exceeds the per-name cap")
            # The same restore ``self.v`` gets: classes only from the allowlist, imported by name.
            staged.values[name] = deserialize(blob, allowed)
        except (DeserializationError, SerializationError, Exception) as exc:  # noqa: BLE001
            fail(name, f"object could not be restored ({exc})")
    for name, source in sections["classes"].items():
        if not _valid_name(name):
            fail(name, "not a valid name")
            continue
        if (
            not isinstance(source, str)
            or len(source.encode("utf-8", "replace")) > max_variable_bytes
        ):
            fail(name, "class source missing or over the cap")
            continue
        try:
            tree, spec = validate_class_source(source, name)
            staged.classes[name] = (source, tree, spec)
        except ValueError as exc:
            fail(name, f"class not restored: {exc}")
    for name, encoded in sections["instances"].items():
        if not _valid_name(name):
            fail(name, "not a valid name")
            continue
        try:
            blob = base64.b64decode(encoded, validate=True)
            if len(blob) > max_variable_bytes:
                raise ValueError("exceeds the per-name cap")
            staged.instances[name] = unpack_value(blob)
        except Exception as exc:  # noqa: BLE001
            fail(name, f"instance could not be decoded ({exc})")
    for name, source in sections["functions"].items():
        if not _valid_name(name):
            fail(name, "not a valid name")
            continue
        if (
            not isinstance(source, str)
            or len(source.encode("utf-8", "replace")) > max_variable_bytes
        ):
            fail(name, "function source missing or over the cap")
            continue
        try:
            staged.functions[name] = (source, validate_function_source(source, name))
        except ValueError as exc:
            fail(name, f"function not restored: {exc}")
    return staged


def _define(namespace: Any, name: str, source: str, tree: ast.Module, needs: dict[str, Any]) -> Any:
    """Run one validated definition in the namespace's globals; return the bound object.

    Library names the definition uses must be the real library objects: a name
    bound by a cell to something else fails the definition rather than running it.
    """
    globals_ = namespace.globals
    added: list[str] = []
    for lib_name, obj in needs.items():
        if lib_name in globals_:
            if globals_[lib_name] is not obj:
                raise ValueError(f"{lib_name!r} is bound to something else in this session")
        else:
            globals_[lib_name] = obj
            added.append(lib_name)
    try:
        code = compile(tree, f"<restored {name}>", "exec", dont_inherit=True)
        exec(code, globals_)  # noqa: S102 - a validated definition; only library code runs
    finally:
        for lib_name in added:
            globals_.pop(lib_name, None)
    defined = globals_.get(name)
    try:
        setattr(defined, SOURCE_ATTRIBUTE, source)
    except Exception:  # noqa: BLE001 - metadata only
        pass
    namespace.user_names.add(name)
    return defined


def apply(namespace: Any, snapshot: dict[str, Any]) -> RestoreReport:
    """Revive a captured snapshot into ``namespace`` (an ``ExecutionNamespace``).

    Order: plain values and ``self.v``-style objects, then classes (in their
    definition order), then instances of those classes, then helper functions.
    Definitions run in the namespace's globals so they see the session's names,
    and their source is attached again so a later capture keeps them.
    """
    staged = decode(snapshot)
    report = staged.report
    globals_ = namespace.globals
    for name, value in staged.values.items():
        globals_[name] = value
        namespace.user_names.add(name)
        report.restored.append(name)
    classes: dict[str, type] = {}
    pending = dict(staged.classes)
    # A class may name another from the same cell (a field type or default), and the
    # saved order within one cell is alphabetical: retry the ones that fail on a missing
    # name until a pass defines nothing new.
    while pending:
        errors: dict[str, Exception] = {}
        for name, (source, tree, spec) in list(pending.items()):
            try:
                defined = _define(namespace, name, source, tree, spec.needs)
                if not isinstance(defined, type):
                    raise ValueError("definition did not produce a class")
                classes[name] = defined
                report.restored.append(name)
                del pending[name]
            except Exception as exc:  # noqa: BLE001 - one bad definition must not abort the rest
                globals_.pop(name, None)
                errors[name] = exc
        if len(errors) == len(pending):  # no progress: report what is left
            for name, exc in errors.items():
                report.failed.append(
                    {"name": name, "reason": f"class definition failed ({exc})"[:200]}
                )
            break
    for name, encoded in staged.instances.items():
        try:
            globals_[name] = decode_instances(encoded, classes)
            namespace.user_names.add(name)
            report.restored.append(name)
        except Exception as exc:  # noqa: BLE001 - a bad instance fails just its name
            globals_.pop(name, None)
            report.failed.append({"name": name, "reason": f"instance not rebuilt ({exc})"[:200]})
    for name, (source, tree) in staged.functions.items():
        try:
            _define(namespace, name, source, tree, {})
            report.restored.append(name)
        except Exception as exc:  # noqa: BLE001
            globals_.pop(name, None)
            report.failed.append({"name": name, "reason": f"definition failed ({exc})"[:200]})
    report.restored.sort()
    return report


def restore_notice(report: RestoreReport) -> str:
    """The one-time message telling the model what came back and what did not."""
    lines = ["[python-state-restored]"]
    if report.restored:
        lines.append(
            "Your Python cell state from the previous session was revived. These names are "
            "available again: " + ", ".join(report.restored) + "."
        )
    else:
        lines.append(
            "Your previous Python cell state could not be revived; recreate any variables, "
            "imports or helpers you need."
        )
    problems = [
        f"{entry['name']} ({entry.get('reason', '')})"
        for entry in report.failed
        if entry.get("name") != "*"
    ] + [f"{entry['name']} ({entry.get('reason', '')})" for entry in report.not_saved]
    if problems:
        lines.append("Not restored, recreate if needed: " + "; ".join(problems) + ".")
    lines.append(
        "Imports, open files, running jobs and Shells do not survive a restart; self.v and "
        "saved history remain."
    )
    return "\n".join(lines)
