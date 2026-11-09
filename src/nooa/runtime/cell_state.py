# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save and revive a session namespace's data and helper functions across restarts.

What is kept, per name, with a size cap on each and on the whole:

* **Plain data**: ``None``, ``bool``, ``int``, ``float``, ``str``, ``bytes``, and
  ``list``/``tuple``/``set``/``frozenset``/``dict`` of plain data (dict keys are
  strings or integers). Encoded with msgpack; decoding builds only these types.
* **Helper functions defined in a cell**: kept as their source text and defined
  again on restore. Only a plain ``def``/``async def`` qualifies: no decorators,
  literal default values, no closure, defined in this namespace. Annotations are
  dropped before the definition runs, because they would be evaluated. Defining
  the function runs no user code; its body runs only when a later cell calls it.

Everything else is skipped and reported by name with a reason: classes (a class
body runs code when loaded), modules (re-import them), lambdas, decorated or
closed-over functions, objects of other types, and values over the caps.
Nothing in a saved payload can select a class to instantiate or code to run on
load, so a tampered or corrupt payload costs at most the names it carries.
"""

from __future__ import annotations

import ast
import base64
import logging
import sys
import types
from dataclasses import dataclass, field
from typing import Any

import msgpack

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
MAX_VARIABLE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
# The attribute ``ExecutionNamespace`` attaches to every function a cell defines.
SOURCE_ATTRIBUTE = "_generated_source"

_TUPLE, _SET, _FROZENSET = range(3)


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


# ------------------------------------------------------------- functions


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
    if node.decorator_list:
        raise ValueError("decorators run code when the function is defined")
    for default in [*node.args.defaults, *node.args.kw_defaults]:
        if default is None:
            continue
        try:
            ast.literal_eval(default)
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            raise ValueError("default values must be literals") from None
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

    Newest bindings are kept first when the total cap bites. The result carries
    every skipped name with its reason, so a restore notice can be honest.
    """
    report = CaptureReport()
    values: dict[str, str] = {}
    functions: dict[str, str] = {}
    globals_ = namespace.globals
    bound_in = getattr(namespace, "_bound_in", {})
    names = sorted(
        (n for n in namespace.user_names if n in globals_ and not n.startswith("_")),
        key=lambda n: (-bound_in.get(n, 0), n),
    )
    total = 0
    for name in names:
        value = globals_[name]
        if isinstance(value, types.ModuleType):
            _skip(report, name, value, "module: import it again")
            continue
        if isinstance(value, type):
            _skip(report, name, value, "class definitions run code when loaded")
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
            try:
                payload = pack_value(value)
            except NotPlainData as exc:
                _skip(report, name, value, f"not plain data: {exc}")
                continue
            except RecursionError:
                _skip(report, name, value, "not plain data: nested too deeply")
                continue
            kind = "value"
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
        else:
            values[name] = base64.b64encode(payload).decode("ascii")
        report.saved.append(name)
    report.bytes = total
    return {
        "version": FORMAT_VERSION,
        "python": sys.version.split()[0],
        "values": values,
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


def decode(
    snapshot: dict[str, Any],
    *,
    max_variable_bytes: int = MAX_VARIABLE_BYTES,
) -> tuple[dict[str, Any], dict[str, tuple[str, ast.Module]], RestoreReport]:
    """Stage a snapshot without touching any namespace: values, validated functions, report.

    Every problem costs only the name it belongs to.
    """
    report = RestoreReport()
    staged_values: dict[str, Any] = {}
    staged_functions: dict[str, tuple[str, ast.Module]] = {}
    if not isinstance(snapshot, dict) or snapshot.get("version") != FORMAT_VERSION:
        report.failed.append({"name": "*", "reason": "unknown saved-state format"})
        return staged_values, staged_functions, report
    skipped = snapshot.get("skipped")
    if isinstance(skipped, list):
        report.not_saved = [s for s in skipped if isinstance(s, dict) and "name" in s]
    values = snapshot.get("values") or {}
    functions = snapshot.get("functions") or {}
    if not isinstance(values, dict) or not isinstance(functions, dict):
        report.failed.append({"name": "*", "reason": "corrupt saved state"})
        return staged_values, staged_functions, report
    for name, encoded in values.items():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            report.failed.append({"name": str(name)[:80], "reason": "not a valid name"})
            continue
        try:
            blob = base64.b64decode(encoded, validate=True)
            if len(blob) > max_variable_bytes:
                raise ValueError("exceeds the per-name cap")
            staged_values[name] = unpack_value(blob)
        except Exception as exc:  # noqa: BLE001 - a bad record fails just its name
            report.failed.append(
                {"name": name, "reason": f"value could not be decoded ({exc})"[:200]}
            )
    for name, source in functions.items():
        if not isinstance(name, str) or not name.isidentifier() or name.startswith("_"):
            report.failed.append({"name": str(name)[:80], "reason": "not a valid name"})
            continue
        if (
            not isinstance(source, str)
            or len(source.encode("utf-8", "replace")) > max_variable_bytes
        ):
            report.failed.append(
                {"name": name, "reason": "function source missing or over the cap"}
            )
            continue
        try:
            staged_functions[name] = (source, validate_function_source(source, name))
        except ValueError as exc:
            report.failed.append({"name": name, "reason": f"function not restored: {exc}"})
    return staged_values, staged_functions, report


def apply(namespace: Any, snapshot: dict[str, Any]) -> RestoreReport:
    """Revive a captured snapshot into ``namespace`` (an ``ExecutionNamespace``).

    Values are bound directly. Functions are defined by running their validated
    definition in the namespace's globals, so they see the session's names, and
    their source is attached again so a later capture keeps them.
    """
    staged_values, staged_functions, report = decode(snapshot)
    globals_ = namespace.globals
    for name, value in staged_values.items():
        globals_[name] = value
        namespace.user_names.add(name)
        report.restored.append(name)
    for name, (source, tree) in staged_functions.items():
        try:
            code = compile(tree, f"<restored {name}>", "exec", dont_inherit=True)
            exec(code, globals_)  # noqa: S102 - a validated def statement; no user code runs
            function = globals_.get(name)
            if isinstance(function, types.FunctionType):
                setattr(function, SOURCE_ATTRIBUTE, source)
            namespace.user_names.add(name)
            report.restored.append(name)
        except Exception as exc:  # noqa: BLE001 - one bad definition must not abort the rest
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
