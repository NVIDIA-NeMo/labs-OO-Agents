# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SnapshotVars — a dict-like store that only keeps snapshot-serializable values.

Agent ``vars`` (``self.v``) and per-todo ``vars`` (``t.v``) are snapshot-backed:
they must round-trip through :func:`serialize`. A single non-serializable value
used to abort the *whole* snapshot, silently losing all durable state on the
next resume.

``SnapshotVars`` moves that check to *write* time. Assigning a value that can't
be serialized logs a warning and **skips the store** — so the container only
ever holds serializable data and the snapshot can never choke on it. Because it
is a ``MutableMapping`` (not a ``dict`` subclass) decorated with
``@snapshotable``, the serializer routes it through the snapshotable branch and
captures its internal ``_data`` (already clean), instead of the strict dict
branch. Values are stored live, so one can still become unserializable after
the write (``l`` in ``v["a"] = l; l.append(l)``); the checkpoint then skips
just that key with a warning (``__snapshot_lenient_dict__``).

Pydantic models can embed ``SnapshotVars`` as a field (``Todo.vars``). Python-mode
``model_dump()`` and ``model_copy()`` keep the ``SnapshotVars`` object. JSON mode
(``model_dump(mode="json")``, ``model_dump_json()``), which is how items cross
sessions, carries plain JSON only: a value that is not JSON data (a pydantic
model, a tuple, NaN or infinity, ...) raises an error naming its key. Validation accepts a
``SnapshotVars``, a plain dict or ``None`` and never imports or constructs a
class: a payload must not be able to choose what gets instantiated. Non-JSON
values survive only the session snapshot (``nooa.storage.serialization``), whose
allowlist comes from code.
"""

import enum
import logging
import math
from collections.abc import Iterator, MutableMapping
from typing import Any

from nooa.errors.storage import SerializationError
from nooa.storage.markers import snapshotable
from nooa.storage.serialization import MAX_SNAPSHOT_DEPTH, serialize

logger = logging.getLogger(__name__)

# Key of a removed tagged JSON form that carried its own deserialization
# allowlist; payloads using it are refused instead of being restored.
_TAGGED_KEY = "__snapshot_vars__"
_TAGGED_REFUSED = (
    "tagged SnapshotVars payloads are not accepted "
    f"({{{_TAGGED_KEY!r}: {{'blob': ..., 'allowlist': ...}}}}); vars must be plain JSON"
)


def _non_json(value: Any) -> str | None:
    """Why *value* is not plain JSON data, or ``None`` when it is.

    Plain JSON is ``None``, bool, int, finite float, str, and lists and
    string-keyed dicts of those: no tuples, models or other objects, and no
    NaN or infinity (JSON has no such numbers; they would arrive as null).
    """
    if isinstance(value, enum.Enum):
        return f"a {type(value).__name__}"
    if isinstance(value, float) and not math.isfinite(value):
        return f"a non-finite float ({value!r})"
    if value is None or isinstance(value, (bool, int, float, str)):
        return None
    if isinstance(value, list):
        return next((r for item in value if (r := _non_json(item))), None)
    if isinstance(value, dict):
        for k, v in value.items():
            if not isinstance(k, str):
                return f"a dict with a {type(k).__name__} key"
            if reason := _non_json(v):
                return reason
        return None
    return f"a {type(value).__name__}"


def _is_tagged_value(value: Any) -> bool:
    """Whether *value* is the inner part of the removed tagged form: ``{"blob", "allowlist"}``."""
    return isinstance(value, dict) and set(value) == {"blob", "allowlist"}


def _is_tagged(data: Any) -> bool:
    """Whether *data* carries the removed tagged form ``{"__snapshot_vars__": {"blob", "allowlist"}}``.

    Only that shape is refused: a var named ``__snapshot_vars__`` holding
    anything else is ordinary data. Writing, JSON dumping and validation all
    apply this same test, so whatever is stored can be sent and reloaded.
    """
    return isinstance(data, dict) and _is_tagged_value(data.get(_TAGGED_KEY))


# How many containers sit above a ``SnapshotVars`` value in a real checkpoint,
# measured from the checkpoint paths (``AgentSnapshot.from_agent``):
#   agent vars:  <attr vars>._data['k']                    -> 2
#   todo vars:   <attr todo>._todos['id'].vars._data['k']  -> 5
# ``SnapshotVars`` does not know which placement it is in, so the write-time
# check uses the deepest one: a value accepted here fits within
# ``MAX_SNAPSHOT_DEPTH`` at checkpoint wherever it sits. This is conservative
# for agent vars, which reject values 3 levels shallower than they could
# checkpoint. tests/storage/test_snapshot_vars.py re-measures both placements
# and fails if this is short.
CHECKPOINT_VALUE_DEPTH = 5

# Instance attributes that are SnapshotVars' own state, not vars.
_PRIVATE_ATTRS = frozenset({"_data"})


@snapshotable
class SnapshotVars(MutableMapping):
    """Durable named state of an agent (``self.v``) or a todo (``todo.v``): a dict, also usable as attributes.

    ``v.spec = "..."`` and ``v["spec"] = "..."`` write the same entry.
    Choose the scope deliberately:

    - ``self.v`` — identity, stable preferences, environment facts, capabilities,
      and long-running coordination shared across tasks and sessions.
    - ``todo.v`` (``self.todo.get(id).v``) — plans, findings, artifacts,
      checkpoints, and verification metadata belonging to one task.
    - Cell locals — transient scratch data that does not need durable ownership.

    Examples::

        self.v.user_name = "Ada"           # create or update cross-task state
        todo.v["commit"] = "abc123"         # the same store: v["k"] is v.k
        print(self.v.user_name)              # read one
        print(dict(todo.v))                  # inspect all in one scope
        del todo.v.commit                    # remove one
        self.v.clear()                       # remove all in one scope

    Values must be snapshot-serializable (dicts, lists, strings, numbers,
    Pydantic models, ...). A write whose value can't be snapshot-serialized
    (a live client, socket, callable, ...) is **not stored**: it logs a
    warning instead and won't survive ``/exit`` + resume. This keeps the
    durable ``vars`` always snapshot-clean, so one bad value can't take down
    the whole snapshot (and with it every other var) on the next resume.

    Attribute access (``v.name``) refuses names starting with ``_`` and the
    names of this class's methods (``keys``, ``items``, ``get``, ``set``,
    ``clear``, ``pop``, ``update``, ...): read those with ``v.get(name)`` or
    ``v[name]`` and write them with ``v.set(name, value)`` or ``v[name] = value``.
    """

    _data: "dict[str, Any]"
    # Checkpoints serialize ``_data`` per key: a value that became
    # unserializable after it was stored (e.g. a list that now contains itself)
    # is skipped with a warning instead of dropping every var.
    __snapshot_lenient_dict__ = "_data"

    def __init__(self, data: "dict[str, Any] | None" = None) -> None:
        object.__setattr__(self, "_data", {})
        if data:
            for k, v in data.items():
                self[k] = v

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if not isinstance(key, str):
            logger.warning(
                "SnapshotVars: key %r (%s) is not a string and will NOT be persisted "
                "(it won't survive /exit + resume): snapshots are JSON, which requires "
                "string keys",
                key,
                type(key).__name__,
            )
            return
        if key == _TAGGED_KEY and _is_tagged_value(value):
            logger.warning("SnapshotVars: %r will NOT be stored: %s", key, _TAGGED_REFUSED)
            return
        try:
            serialize(value, base_depth=CHECKPOINT_VALUE_DEPTH)
        except (SerializationError, TypeError, ValueError, RecursionError) as exc:
            hint = ""
            if getattr(exc, "reason", None) == "depth":
                hint = (
                    f" (a vars value sits up to {CHECKPOINT_VALUE_DEPTH} levels deep in a "
                    f"checkpoint, so it may nest at most "
                    f"{MAX_SNAPSHOT_DEPTH - CHECKPOINT_VALUE_DEPTH} levels)"
                )
            logger.warning(
                "SnapshotVars: value for %r (%s) is not snapshot-serializable and "
                "will NOT be persisted (it won't survive /exit + resume): %s%s",
                key,
                type(value).__name__,
                exc,
                hint,
            )
            return
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __repr__(self) -> str:
        return repr(self._data)

    # ── attribute access: ``v.name`` is ``v["name"]`` ─────────────────────

    def __getattr__(self, key: str) -> Any:
        # Only called when normal lookup fails. Private and dunder names are
        # never vars, so probes (copy, pickle, snapshot restore) see a plain
        # AttributeError, and a missing ``_data`` cannot recurse.
        if key.startswith("_"):
            raise AttributeError(key)
        try:
            return self._data[key]
        except KeyError:
            raise AttributeError(f"No var {key!r}") from None

    def __setattr__(self, key: str, value: Any) -> None:
        if key in _PRIVATE_ATTRS:
            object.__setattr__(self, key, value)  # internal state (snapshot restore)
            return
        self._check_attr(key, "use set({key!r}, value)")
        self[key] = value

    def __delattr__(self, key: str) -> None:
        self._check_attr(key, "use del v[{key!r}]")
        try:
            del self._data[key]
        except KeyError:
            raise AttributeError(f"No var {key!r}") from None

    def _check_attr(self, key: str, fix: str) -> None:
        """Refuse ``v.key`` for ``_``-prefixed names and this class's own names."""
        if key.startswith("_") or any(key in cls.__dict__ for cls in type(self).__mro__):
            raise AttributeError(
                f"{key!r} is reserved by {type(self).__name__}; {fix.format(key=key)}"
            )

    def set(self, key: str, value: Any) -> None:
        """Store a value by name, including names reserved for attribute access."""
        self[key] = value

    # ── pydantic integration ─────────────────────

    def to_json_data(self) -> dict[str, Any]:
        """Return the plain-JSON form used by pydantic's JSON mode.

        Raises ``ValueError`` naming the first key whose value is not JSON data
        (a pydantic model, a tuple, ...): only plain JSON crosses sessions.
        """
        if _is_tagged(self._data):
            raise ValueError(_TAGGED_REFUSED)
        for key, value in self._data.items():
            if reason := _non_json(value):
                raise ValueError(
                    f"var {key!r} holds {reason}, which is not plain JSON; "
                    "Todo vars sent to another session must be JSON data "
                    "(dict, list, str, int, float, bool, None)"
                )
        return dict(self._data)

    @classmethod
    def from_data(cls, value: Any) -> "SnapshotVars":
        """Accept a ``SnapshotVars`` as-is, a plain dict or ``None``.

        Never deserializes snapshot envelopes, so validating a model that embeds
        ``SnapshotVars`` cannot import or construct arbitrary classes.
        """
        if isinstance(value, SnapshotVars):
            return value
        if value is None:
            return cls()
        if isinstance(value, dict):
            if _is_tagged(value):
                raise ValueError(_TAGGED_REFUSED)
            return cls(value)
        raise TypeError(f"expected a dict or SnapshotVars, got {type(value).__name__}")

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            cls.from_data,
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda value: value.to_json_data(), when_used="json"
            ),
        )

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: Any, handler: Any) -> dict[str, Any]:
        return {"type": "object"}
