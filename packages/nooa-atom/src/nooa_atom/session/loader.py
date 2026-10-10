# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load an agent class from a ``module:Class`` spec and build the agent for a session."""

import hashlib
import importlib
import importlib.util
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

from pydantic import BaseModel

from nooa.interactive import InteractiveAgent
from nooa.storage.manager import StorageManager
from nooa_atom.session.options import SessionOptions

logger = logging.getLogger(__name__)

AgentFactory = Callable[[SessionOptions, StorageManager], InteractiveAgent]
"""Builds the agent for a session from its options and its storage."""


ATOM_AGENT = "nooa_atom.agent:AtomAgent"
EXPERIMENTAL_ATOM_AGENT = "nooa_atom.agent:ExperimentalAtomAgent"
LEGACY_AGENT_SPECS = {
    "nooa_cli.tui.agent:TUIAgent": ATOM_AGENT,
    "nooa_cli.coding.legacy_agent:TUIAgent": ATOM_AGENT,
    "nooa_cli.tui.experimental_agent:ExperimentalTUIAgent": EXPERIMENTAL_ATOM_AGENT,
    "nooa_cli.coding.experimental_agent:ExperimentalTUIAgent": EXPERIMENTAL_ATOM_AGENT,
    # What nooa-coder, the package before nooa-atom, recorded.
    "nooa_coder.coding.agent:CodingAgent": ATOM_AGENT,
    "nooa_coder.coding.experimental_agent:ExperimentalCodingAgent": EXPERIMENTAL_ATOM_AGENT,
    # Spellings from before the agent moved out of nooa-cli.
    "nooa_cli.coding.agent:CodingAgent": ATOM_AGENT,
    "nooa_cli.coding.experimental_agent:ExperimentalCodingAgent": EXPERIMENTAL_ATOM_AGENT,
    # What the nooa-acp server and the old TUI record as a session's agent.
    "CodingAgent": ATOM_AGENT,
    "TUIAgent": ATOM_AGENT,
    "ExperimentalTUIAgent": EXPERIMENTAL_ATOM_AGENT,
}
"""Agent specs older hosts saved, and the spec that loads the class that replaced each."""

LEGACY_MODULE_PREFIXES = {
    "nooa_coder.coding.": "nooa_atom.agent.",
    "nooa_coder.": "nooa_atom.",
}
"""Module prefixes nooa-coder recorded, and the nooa-atom modules that replaced them.

The first matching prefix applies, so the more specific one comes first.
"""


def canonical_module(module_name: str) -> str:
    """The module to import for ``module_name``: its nooa-atom name if nooa-coder recorded it."""
    for old, new in LEGACY_MODULE_PREFIXES.items():
        if f"{module_name}.".startswith(old):
            return (new + f"{module_name}.".removeprefix(old)).rstrip(".")
    return module_name


def canonical_agent_spec(spec: str) -> str:
    """The spec to load for ``spec``: its replacement if an older host saved it, else itself."""
    return LEGACY_AGENT_SPECS.get(spec, spec)


class AgentSpecError(ValueError):
    """The agent spec is malformed or does not name an ``InteractiveAgent`` class."""


def load_agent_class(spec: str, *, base: str | Path | None = None) -> type[InteractiveAgent]:
    """Load the ``InteractiveAgent`` class a spec names.

    ``spec`` is ``module:Class`` (the class part may be dotted) or
    ``path/to/file.py:Class``. A file spec is one whose module part ends in
    ``.py``, contains ``/`` or starts with ``.`` or ``~``; a relative path
    resolves against ``base`` (the session's workspace) when given, else
    the process directory. Each file is imported under a module name unique
    to its resolved path and cached until the file changes, so two agent
    files never share a ``sys.modules`` entry and an unchanged file gives
    the same class every time. Spellings saved by older hosts (the
    ``nooa_cli.*`` coding agents) load the classes that replaced them.

    Raises ``AgentSpecError`` for a malformed spec, a module or file that
    cannot be imported, a missing class, or a class that is not an
    ``InteractiveAgent``.
    """
    spec = canonical_agent_spec(spec)
    module_name, _, class_path = spec.rpartition(":")
    class_path = class_path.strip()
    if not module_name or not class_path:
        raise AgentSpecError(
            f"Agent spec {spec!r} must look like 'module:Class' or 'file.py:Class'"
        )
    target: Any
    if _is_file_spec(module_name):
        file_path = Path(module_name).expanduser()
        if not file_path.is_absolute() and base is not None:
            file_path = Path(base).expanduser() / file_path
        file_path = file_path.resolve()
        if not file_path.is_file():
            raise AgentSpecError(f"Agent module file not found: {file_path}")
        try:
            target = _load_agent_file(file_path)
        except Exception as exc:
            raise AgentSpecError(f"Cannot import agent file {str(file_path)!r}: {exc}") from exc
    else:
        try:
            target = importlib.import_module(module_name)
        except ImportError as exc:
            raise AgentSpecError(f"Cannot import module {module_name!r}: {exc}") from exc
    for part in class_path.split("."):
        try:
            target = getattr(target, part)
        except AttributeError as exc:
            raise AgentSpecError(f"{module_name!r} has no attribute {class_path!r}") from exc
    if not (isinstance(target, type) and issubclass(target, InteractiveAgent)):
        raise AgentSpecError(f"{spec!r} is not an InteractiveAgent subclass")
    return target


def _is_file_spec(module_name: str) -> bool:
    return module_name.endswith(".py") or "/" in module_name or module_name.startswith((".", "~"))


# Loaded file-based agent modules by resolved path: (mtime_ns, module).
_FILE_AGENT_MODULES: dict[Path, tuple[int, ModuleType]] = {}


def _load_agent_file(file_path: Path) -> ModuleType:
    """Import an agent file under a module name unique to its resolved path.

    Each file gets its own ``sys.modules`` entry, so loading a second file
    does not replace the first one's entry (classes resolve string
    annotations through ``sys.modules[cls.__module__]``). Loading the same
    unchanged file again returns the same module, and so the same classes.
    """
    mtime = file_path.stat().st_mtime_ns
    cached = _FILE_AGENT_MODULES.get(file_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]

    digest = hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
    module_name = f"_nooa_custom_agent_{digest}"
    parent_str = str(file_path.parent)
    inserted = False
    if parent_str not in sys.path:
        sys.path.insert(0, parent_str)
        inserted = True

    try:
        mod_spec = importlib.util.spec_from_file_location(module_name, file_path)
        if mod_spec is None or mod_spec.loader is None:
            raise ImportError(f"Cannot load module from {file_path}")
        module = importlib.util.module_from_spec(mod_spec)
        # Some code (dataclasses with postponed annotations, typing.get_type_hints)
        # resolves a class's string annotations via sys.modules[cls.__module__];
        # that lookup fails unless the module is registered before exec_module()
        # runs the file's class definitions.
        previous = sys.modules.get(module_name)
        sys.modules[module_name] = module
        try:
            mod_spec.loader.exec_module(module)
        except BaseException:
            # A failed reload of an edited file keeps the last good version.
            if previous is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous
            raise
    finally:
        if inserted:
            sys.path.remove(parent_str)
    _FILE_AGENT_MODULES[file_path] = (mtime, module)
    return module


def load_typed(type_name: str | None, data: Any) -> Any:
    """Rebuild a value recorded as JSON data and the ``module:qualname`` of its class.

    Pydantic classes come back as instances of that class; anything else
    (and a class that cannot be imported or validated) stays JSON data.
    ``data`` is a JSON string or already-decoded JSON.
    """
    value = json.loads(data) if isinstance(data, str) else data
    if not type_name:
        return value
    module_name, _, qualname = type_name.partition(":")
    if module_name == "builtins" or not qualname:
        return value
    module_name = canonical_module(module_name)
    try:
        target: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError):
        logger.warning("Cannot import %s; keeping the value as JSON data", type_name)
        return value
    if not (isinstance(target, type) and issubclass(target, BaseModel)):
        return value
    if (
        target.__module__ == "nooa.interactive"
        and target.__name__ in ("NeedInput", "NeedInputForm")
        and isinstance(value, dict)
    ):
        record = migrate_input_request(value, form=target.__name__ == "NeedInputForm")
        if record is None:
            return value  # preserve obsolete typed request as data, not a fake restored class
        from nooa.interactive import NeedInput, NeedInputForm

        target = NeedInputForm if record.get("kind") == "form" else NeedInput
        value = record
    if target.__name__ == "ChildQuestion" and isinstance(value, dict):
        value = dict(value)
        presentation = value.pop("presentation", None)
        if "request_kind" not in value:
            value["request_kind"] = (
                "form"
                if presentation == "form" or value.get("answer_schema") is not None
                else "question"
            )
    try:
        return target.model_validate(value)
    except Exception:
        logger.warning("Recorded data does not validate as %s; keeping JSON data", type_name)
        return value


def migrate_input_request(value: dict[str, Any], *, form: bool = False) -> dict[str, Any] | None:
    """Migrate old untyped records; typed class/schema records stay unavailable data."""
    record = dict(value)
    record.pop("request_id", None)
    presentation = record.pop("presentation", None)
    answer_type = record.pop("answer_type", None)
    answer_schema = record.pop("answer_schema", None)
    form = form or record.get("kind") == "form" or presentation == "form"
    if answer_type is not None or answer_schema is not None:
        return None
    if not form:
        record["kind"] = "question"
        return record
    if "questions" in record:
        return record
    question = record.pop("question", "Details")
    options = record.pop("options", None)
    descriptor: dict[str, Any] = {"id": "answer", "label": question, "kind": "text"}
    if options:
        descriptor.update(kind="pick_one", choices=[{"value": c, "title": c} for c in options])
    record.update(kind="form", heading=question, questions=[descriptor])
    return record
