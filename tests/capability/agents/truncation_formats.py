# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Polymorphic truncating renderers + a thin wrapper class used by the
``realfmt_*`` capability tests.

The four formats:

    today_verbose : ``[a, b, ... 90 items not shown ..., y, z]``
                    matches the current pformat shape
    xml           : ``<list len=100>[a, b, ..., y, z]</list>``
    lower         : ``list(len=100, items=[a, b, ..., y, z])``
    slice_keys    : ``list(len=100, [:5]=[a,b,c,d,e], [-5:]=[v,w,x,y,z])``

Type tags supported: list, tuple, dict, set, pydantic, dataclass, json, records.

``Wrapped`` is the bridge that lets fixtures pass real Python data to the
agent while controlling how the prefill renders that data:

    * ``__slots__`` (no ``__dict__``) → ``_pformat._is_structured_instance``
      returns False, so the framework falls back to ``repr()``.
    * ``__repr__`` → calls our renderer, producing the chosen marker shape.
    * Container pass-throughs (``__getitem__``, ``__iter__``, ``__len__``,
      ``__contains__``, plus ``__getattr__`` for ``.keys()`` / ``.items()`` /
      etc.) so CodeAct's natural code (``data[49]``, ``min(data)``) works
      unchanged on the underlying value.

``_patch_eval_pipeline_loader()`` is called at import time by the agent
modules. It monkey-patches ``eval_pipeline.config.load_tasks`` to
wrap any kwargs dict that carries the trio ``(data, type_tag, fmt)`` with
a ``Wrapped`` instance. This is the *only* injection point we need — once
``data`` is a ``Wrapped`` in the per-task kwargs, the framework's prefill
rendering picks up our ``__repr__`` for free.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# ---------------------------------------------------------------------------
# Format renderers


def _container_open_close(type_tag: str) -> tuple[str, str]:
    # ``json`` carries an inner *list* (the value of the ``"items"`` key), so
    # its visible body uses list brackets. The outer ``{"items": ...}`` shell
    # is added by the wrapping step in each renderer.
    return {
        "list": ("[", "]"),
        "tuple": ("(", ")"),
        "dict": ("{", "}"),
        "set": ("{", "}"),
        "json": ("[", "]"),
        "records": ("[", "]"),
        "pydantic": ("[", "]"),
        "dataclass": ("[", "]"),
    }[type_tag]


def _container_python_name(type_tag: str) -> str:
    return {
        "list": "list",
        "tuple": "tuple",
        "dict": "dict",
        "set": "set",
        "json": "list",
        "records": "list",
        "pydantic": "list",
        "dataclass": "list",
    }[type_tag]


def _format_item(item: Any) -> str:
    if isinstance(item, int) and not isinstance(item, bool):
        return str(item)
    return repr(item)


def _format_dict_record(d: dict) -> str:
    parts = []
    for k, v in d.items():
        kr = repr(k) if isinstance(k, str) else str(k)
        parts.append(f"{kr}: {_format_item(v)}")
    return "{" + ", ".join(parts) + "}"


def _items_strs(data: Any, type_tag: str) -> list[str]:
    # Only the genuine "dict" tag iterates as key/value pairs. "json" passes a
    # list (the ``"items"`` array); the outer ``{"items": ...}`` is added by
    # the renderer's wrapping step.
    if type_tag == "dict":
        out = []
        for k, v in data.items():
            kr = repr(k) if isinstance(k, str) else str(k)
            out.append(f"{kr}: {_format_item(v)}")
        return out
    return [_format_item(x) if not isinstance(x, dict) else _format_dict_record(x) for x in data]


def _length(data: Any, type_tag: str) -> int:
    return len(data)


def _wrap_pydantic(inner: str) -> str:
    return f"Team(name='alpha', members={inner}, status='active')"


def _wrap_dataclass(inner: str) -> str:
    return f"Project(name='alpha', tasks={inner}, owner='Bob')"


def render_today_verbose(data: Any, type_tag: str, *, head: int = 5, tail: int = 5) -> str:
    items = _items_strs(data, type_tag)
    n = _length(data, type_tag)
    open_b, close_b = _container_open_close(type_tag)
    if n <= head + tail:
        body = ", ".join(items)
    else:
        elided = n - head - tail
        body = (
            ", ".join(items[:head])
            + f", ... {elided} items not shown ..., "
            + ", ".join(items[-tail:])
        )
    inner = f"{open_b}{body}{close_b}"
    if type_tag == "pydantic":
        return _wrap_pydantic(inner)
    if type_tag == "dataclass":
        return _wrap_dataclass(inner)
    if type_tag == "json":
        return '{"items": ' + inner + "}"
    return inner


def render_xml(data: Any, type_tag: str, *, head: int = 5, tail: int = 5) -> str:
    items = _items_strs(data, type_tag)
    n = _length(data, type_tag)
    open_b, close_b = _container_open_close(type_tag)
    py_name = _container_python_name(type_tag)
    if n <= head + tail:
        body = ", ".join(items)
    else:
        body = ", ".join(items[:head]) + ", ..., " + ", ".join(items[-tail:])
    inner = f"<{py_name} len={n}>{open_b}{body}{close_b}</{py_name}>"
    if type_tag == "pydantic":
        return _wrap_pydantic(inner)
    if type_tag == "dataclass":
        return _wrap_dataclass(inner)
    if type_tag == "json":
        return '{"items": ' + inner + "}"
    return inner


def render_lower(data: Any, type_tag: str, *, head: int = 5, tail: int = 5) -> str:
    items = _items_strs(data, type_tag)
    n = _length(data, type_tag)
    open_b, close_b = _container_open_close(type_tag)
    py_name = _container_python_name(type_tag)
    if n <= head + tail:
        body = ", ".join(items)
    else:
        body = ", ".join(items[:head]) + ", ..., " + ", ".join(items[-tail:])
    inner = f"{py_name}(len={n}, items={open_b}{body}{close_b})"
    if type_tag == "pydantic":
        return _wrap_pydantic(inner)
    if type_tag == "dataclass":
        return _wrap_dataclass(inner)
    if type_tag == "json":
        return '{"items": ' + inner + "}"
    return inner


def render_slice_keys(data: Any, type_tag: str, *, head: int = 5, tail: int = 5) -> str:
    # dict/set are unordered so positional slices aren't meaningful — fall back
    # to the lower form. json carries an inner list, so slice notation works.
    if type_tag in ("dict", "set"):
        return render_lower(data, type_tag, head=head, tail=tail)
    items = _items_strs(data, type_tag)
    n = _length(data, type_tag)
    open_b, close_b = _container_open_close(type_tag)
    py_name = _container_python_name(type_tag)
    if n <= head + tail:
        body = f"items={open_b}" + ", ".join(items) + f"{close_b}"
        inner = f"{py_name}(len={n}, {body})"
    else:
        head_chunk = open_b + ", ".join(items[:head]) + close_b
        tail_chunk = open_b + ", ".join(items[-tail:]) + close_b
        inner = f"{py_name}(len={n}, [:{head}]={head_chunk}, [-{tail}:]={tail_chunk})"
    if type_tag == "pydantic":
        return _wrap_pydantic(inner)
    if type_tag == "dataclass":
        return _wrap_dataclass(inner)
    if type_tag == "json":
        return '{"items": ' + inner + "}"
    return inner


def _render_lower_unordered(
    data: Any, type_tag: str, *, head: int = 5, tail: int = 5, inner_marker: str
) -> str:
    """Targeted-ablation renderer for unordered types (dict, set).

    ``inner_marker`` controls the ellipsis between head and tail items
    (or after the head for sets):

      * "bare"   — no inner marker. ``items={a, b, c, d, e, f, g, h, i, j}``
      * "dots"   — bare ellipsis. ``items={a, b, c, d, e, ..., g, h, i, j}``
      * "plus_n" — rich-style elided count. ``items={a, ..., e, ...+90, ...}``

    For ordered types we just delegate to the canonical ``render_lower`` —
    this ablation is only about the inner-marker question for sets/dicts.
    """
    if type_tag not in ("dict", "set"):
        return render_lower(data, type_tag, head=head, tail=tail)

    items = _items_strs(data, type_tag)
    n = _length(data, type_tag)
    open_b, close_b = _container_open_close(type_tag)
    py_name = _container_python_name(type_tag)

    if n <= head + tail:
        body = ", ".join(items)
    else:
        elided = n - head - tail if type_tag == "dict" else n - head
        if type_tag == "dict":
            head_part = ", ".join(items[:head])
            tail_part = ", ".join(items[-tail:])
            sep = {
                "bare": ", ",
                "dots": ", ..., ",
                "plus_n": f", ...+{elided}, ",
            }[inner_marker]
            body = head_part + sep + tail_part
        else:
            # set: head-only (no stable order → no positional tail).
            head_part = ", ".join(items[: head + tail])
            suffix = {
                "bare": "",
                "dots": ", ...",
                "plus_n": f", ...+{elided}",
            }[inner_marker]
            body = head_part + suffix

    return f"{py_name}(len={n}, items={open_b}{body}{close_b})"


def render_lower_bare(data: Any, type_tag: str, *, head: int = 5, tail: int = 5) -> str:
    return _render_lower_unordered(data, type_tag, head=head, tail=tail, inner_marker="bare")


def render_lower_dots(data: Any, type_tag: str, *, head: int = 5, tail: int = 5) -> str:
    return _render_lower_unordered(data, type_tag, head=head, tail=tail, inner_marker="dots")


def render_lower_plus_n(data: Any, type_tag: str, *, head: int = 5, tail: int = 5) -> str:
    return _render_lower_unordered(data, type_tag, head=head, tail=tail, inner_marker="plus_n")


RENDERERS = {
    "today_verbose": render_today_verbose,
    "xml": render_xml,
    "lower": render_lower,
    "slice_keys": render_slice_keys,
    # Inner-marker ablation for unordered types (dict, set).
    "lower_bare": render_lower_bare,
    "lower_dots": render_lower_dots,
    "lower_plus_n": render_lower_plus_n,
}


def render(data: Any, type_tag: str, fmt: str, *, head: int = 5, tail: int = 5) -> str:
    return RENDERERS[fmt](data, type_tag, head=head, tail=tail)


# ---------------------------------------------------------------------------
# Wrapped: the bridge between fixture-supplied raw data and the agent's
# prefill rendering. See module docstring for design notes.


class Wrapped:
    __slots__ = ("_data", "_type_tag", "_fmt")

    def __init__(self, data: Any, type_tag: str, fmt: str) -> None:
        self._data = data
        self._type_tag = type_tag
        self._fmt = fmt

    def __repr__(self) -> str:
        # "nested" delegates to the framework's pformat — the renderer family
        # (slice-keys / items wrapper) applies recursively per container level
        # via the framework's existing dispatch, so we just let it run.
        if self._type_tag == "nested":
            from nooa.agentdoc import pformat

            return pformat(self._data, max_length=10, max_string=200, max_depth=4)
        return render(self._data, self._type_tag, self._fmt)

    # Container pass-throughs so CodeAct's natural code works on the underlying value.
    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterable[Any]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, item: Any) -> bool:
        return item in self._data

    def __getattr__(self, name: str) -> Any:
        # Delegate dict/list methods (.keys(), .values(), .items(), .index(), ...).
        # Called only when an attribute isn't found via normal lookup, so it never
        # hides our private slots.
        return getattr(self._data, name)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Lets pydantic v2 serialize Wrapped instances when they appear in
        # ``EvalTestResult.input``. Validators reject construction from JSON
        # (Wrapped only enters the call path via the loader patch).
        from pydantic_core import core_schema

        def _ser(v: Wrapped, _info):
            return {"_data": v._data, "_type_tag": v._type_tag, "_fmt": v._fmt}

        return core_schema.no_info_plain_validator_function(
            lambda v: (
                v
                if isinstance(v, cls)
                else (_ for _ in ()).throw(
                    TypeError("Wrapped must be constructed via the loader patch")
                )
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(_ser, info_arg=True),
        )


# ---------------------------------------------------------------------------
# Eval-pipeline loader monkey-patch.


_PATCHED = False


def _patch_eval_pipeline_loader() -> None:
    """Patch ``eval_pipeline.config.load_tasks`` to wrap fixture kwargs.

    Idempotent — safe to call from each agent module's import.
    Only wraps kwargs dicts that carry the trio ``(data, type_tag, fmt)``,
    so other fixtures in the same suite are unaffected.
    """
    global _PATCHED
    if _PATCHED:
        return

    # 1. Patch the loader so kwargs.data becomes a Wrapped instance.
    #    ``load_tasks`` lives in ``eval_pipeline.config`` and is looked up at
    #    call time as a module-level name, so patching the attribute is enough.
    from eval_pipeline import config as _cfg

    _orig_load = _cfg.load_tasks

    def patched_load(data_file, limit=None):
        tasks = _orig_load(data_file, limit)
        for task in tasks:
            kwargs = task.input[1] if isinstance(task.input, tuple) else None
            if (
                isinstance(kwargs, dict)
                and "data" in kwargs
                and "type_tag" in kwargs
                and "fmt" in kwargs
                and not isinstance(kwargs["data"], Wrapped)
            ):
                kwargs["data"] = Wrapped(kwargs["data"], kwargs["type_tag"], kwargs["fmt"])
        return tasks

    _cfg.load_tasks = patched_load

    # 2. Patch the experiment writer so Wrapped values in result.input get
    #    unwrapped to their underlying primitives before pydantic serialization.
    #    EvalTestResult.input is typed Any; pydantic's fallback serializer
    #    rejects unknown classes, so we replace before serializing.
    from eval_pipeline import experiment_writer as _ew

    _orig_append = _ew.ExperimentWriter.append_result

    def _unwrap(value):
        if isinstance(value, Wrapped):
            return value._data
        if isinstance(value, dict):
            return {k: _unwrap(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_unwrap(v) for v in value]
        if isinstance(value, tuple):
            return tuple(_unwrap(v) for v in value)
        return value

    def patched_append(self, result):
        # Result is an EvalTestResult or dict. Mutate in place to unwrap.
        if hasattr(result, "input"):
            result.input = _unwrap(result.input)
        elif isinstance(result, dict) and "input" in result:
            result["input"] = _unwrap(result["input"])
        return _orig_append(self, result)

    _ew.ExperimentWriter.append_result = patched_append

    _PATCHED = True
