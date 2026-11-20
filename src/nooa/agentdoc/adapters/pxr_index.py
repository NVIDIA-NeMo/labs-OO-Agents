# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OpenUSD (``pxr``) documentation index: official C++ header docs for the Python API.

usd-core has docstrings for few names. ``scripts/agentdoc/build_pxr_index.py``
extracts the OpenUSD header comments of each release into
``nooa/agentdoc/data/pxr-<version>.json.gz``. This module:

- loads the index that matches the installed usd-core (else the nearest, with a
  one-line note), without importing pxr;
- maps live pxr classes and Boost.Python functions to index names, so ``doc()``
  shows the official text for names that have no curated page;
- adds "did you mean" hints to AttributeErrors on pxr modules, classes and
  ``Tokens`` (``UsdGeom.Tokens.materialBind`` -> ``UsdShade.Tokens.materialBind``).

``core`` imports this module lazily once ``pxr`` is imported; searching
(``doc.search``) loads only the data file.
"""

from __future__ import annotations

import difflib
import gzip
import inspect
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

from nooa.agentdoc.docindex import (
    DocEntry,
    DocIndex,
    get_doc_index,
    register_doc_index,
    register_doc_resolver,
)
from nooa.agentdoc.registry import register_attribute_hint_provider

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_FILE = re.compile(r"^pxr-(\d+)\.(\d+)\.json\.gz$")
_lock = threading.RLock()
_loaded_for: tuple[int, int] | None = None


def available_versions() -> list[tuple[int, int]]:
    """OpenUSD versions with a shipped index, e.g. [(25, 11), (26, 8)]."""
    found = []
    for path in DATA_DIR.glob("pxr-*.json.gz"):
        m = _FILE.match(path.name)
        if m:
            found.append((int(m.group(1)), int(m.group(2))))
    return sorted(found)


def installed_version() -> tuple[int, int] | None:
    """(year, month) of the usd-core in use, without importing pxr."""
    usd = sys.modules.get("pxr.Usd")
    get_version = getattr(usd, "GetVersion", None)
    if get_version is not None:
        try:
            version = get_version()
            return int(version[1]), int(version[2])
        except Exception:  # noqa: BLE001
            pass
    try:
        from importlib.metadata import version

        major, minor = version("usd-core").split(".")[:2]
        return int(major), int(re.match(r"\d+", minor).group(0))
    except Exception:  # noqa: BLE001 - not installed, or a source build
        return None


def _label(version: tuple[int, int]) -> str:
    return f"{version[0]}.{version[1]:02d}"


def choose_version(
    installed: tuple[int, int] | None, available: list[tuple[int, int]]
) -> tuple[tuple[int, int] | None, str]:
    """The index to use and the one-line note shown with it ("" when it matches)."""
    if not available:
        return None, ""
    if installed in available:
        return installed, ""
    if installed is None:
        chosen = available[-1]
        return chosen, f"Docs index: OpenUSD {_label(chosen)} (installed pxr version unknown)."

    def distance(v: tuple[int, int]) -> int:
        return abs((v[0] * 12 + v[1]) - (installed[0] * 12 + installed[1]))

    chosen = min(available, key=lambda v: (distance(v), -v[0], -v[1]))
    return chosen, (
        f"Docs index: OpenUSD {_label(chosen)}; installed usd-core is "
        f"{installed[0]}.{installed[1]}, so details may differ."
    )


def read_index_file(path: Path, note: str = "") -> DocIndex:
    data = json.loads(gzip.decompress(path.read_bytes()))
    columns = data["columns"]
    position = {name: columns.index(name) for name in columns}
    entries = []
    for row in data["entries"]:
        entries.append(
            DocEntry(
                name=row[position["name"]],
                kind=row[position["kind"]],
                signature=row[position["signature"]],
                summary=row[position["summary"]],
                text=row[position["text"]] or row[position["summary"]],
                source=row[position["source"]],
            )
        )
    return DocIndex(
        "pxr",
        entries,
        version=data["version"],
        source=f"OpenUSD {data['version']} C++ docs",
        note=note,
    )


def load_pxr_index() -> DocIndex | None:
    """The shipped index matching the installed usd-core (else the nearest one)."""
    global _loaded_for
    installed = installed_version()
    chosen, note = choose_version(installed, available_versions())
    if chosen is None:
        return None
    with _lock:
        _loaded_for = installed
    return read_index_file(DATA_DIR / f"pxr-{_label(chosen)}.json.gz", note)


# ----------------------------------------------------------------------------- objects -> names

_identity: dict[int, tuple[Any, str]] = {}
_identity_modules: frozenset[str] = frozenset()


def _pxr_modules() -> dict[str, Any]:
    return {
        name[4:]: module
        for name, module in list(sys.modules.items())
        if name.startswith("pxr.") and name.count(".") == 1 and module is not None
    }


def _walk(prefix: str, owner: Any, module_name: str, out: dict, depth: int = 0) -> None:
    for name, value in list(vars(owner).items()):
        if name.startswith("__"):
            continue
        if isinstance(value, type):
            if (
                depth < 2
                and getattr(value, "__module__", None) == module_name
                and value is not owner
            ):
                _walk(f"{prefix}{name}.", value, module_name, out, depth + 1)
            continue
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        if (
            callable(value)
            or isinstance(value, property)
            or type(value).__name__ == "StaticProperty"
        ):
            out.setdefault(id(value), (value, f"{prefix}{name}"))


def _identity_map() -> dict[int, tuple[Any, str]]:
    global _identity, _identity_modules
    modules = _pxr_modules()
    key = frozenset(modules)
    with _lock:
        if key == _identity_modules:
            return _identity
        out: dict[int, tuple[Any, str]] = {}
        for short, module in sorted(modules.items()):
            _walk(f"{short}.", module, module.__name__, out)
        _identity, _identity_modules = out, key
        return out


def pxr_name(obj: Any) -> str | None:
    """Index name of a pxr class or callable ("UsdGeom.Subset.CreateGeomSubset")."""
    if isinstance(obj, type):
        module = getattr(obj, "__module__", "") or ""
        if module.startswith("pxr."):
            return f"{module[4:]}.{obj.__qualname__}"
        return None
    target = obj.__func__ if inspect.ismethod(obj) else obj
    found = _identity_map().get(id(target))
    if found is not None and found[0] is target:
        return found[1]
    return None


def _resolve(obj: Any) -> tuple[str, str] | None:
    name = pxr_name(obj)
    return ("pxr", name) if name else None


# ----------------------------------------------------------------------------- did you mean

_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def _parts(name: str) -> list[str]:
    return [p.lower() for p in _CAMEL.findall(name)]


def _similarity(guess: str, real: str) -> float:
    a, b = guess.lower(), real.lower()
    score = difflib.SequenceMatcher(None, a, b).ratio()
    if len(a) >= 4 and a in b:
        score = max(score, 0.75)
    ga, gb = set(_parts(guess)), set(_parts(real))
    if ga and gb:
        score = max(score, 0.9 * len(ga & gb) / len(ga | gb))
    return score


def suggest(owner_name: str, members: list[str], name: str, limit: int = 4) -> list[str]:
    """Closest real names for ``owner_name.<name>``: same owner, then other pxr owners."""
    index = get_doc_index("pxr")
    leaves = index.leaves() if index is not None else {}
    owner_leaf = owner_name.rsplit(".", 1)[-1]
    owner_module = owner_name.split(".", 1)[0]
    is_module = "." not in owner_name
    out: list[str] = []

    def push(full: str) -> None:
        if full not in out and full != f"{owner_name}.{name}":
            out.append(full)

    # 1. The same name elsewhere (UsdGeom.Tokens.materialBind -> UsdShade.Tokens.materialBind).
    elsewhere = [full for full in leaves.get(name, []) if not full.startswith(owner_name + ".")]
    if not elsewhere:
        lower = name.lower()
        elsewhere = [f for leaf, fulls in leaves.items() if leaf.lower() == lower for f in fulls]

    def rank(full: str) -> tuple:
        parent = full.rsplit(".", 1)[0]
        return (
            parent.rsplit(".", 1)[-1] != owner_leaf,  # same kind of owner (Tokens)
            parent.count(".") != owner_name.count("."),  # same depth
            not parent.startswith(owner_module),
            len(full),
            full,
        )

    elsewhere.sort(key=rank)
    # 2. Close names on the owner itself (CreateSubset -> CreateGeomSubset).
    scored = sorted(
        ((_similarity(name, m), m) for m in members if not m.startswith("_")),
        key=lambda item: (-item[0], len(item[1]), item[1]),
    )
    close = [f"{owner_name}.{m}" for score, m in scored if score >= 0.6][:3]
    strong_local = bool(scored) and scored[0][0] >= 0.8
    for full in (close + elsewhere[:2]) if strong_local else (elsewhere[:2] + close):
        push(full)
    # 3. A module name glued to the class (Usd.ShadeOutput -> UsdShade.Output).
    if is_module:
        pieces = _CAMEL.findall(name)
        for k in range(1, len(pieces)):
            candidate = f"{owner_name}{''.join(pieces[:k])}.{''.join(pieces[k:])}"
            if index is not None and index.get(candidate) is not None:
                push(candidate)
    # 4. Similar names anywhere (GetExternalDependencies -> Sdf.Layer.GetExternalAssetDependencies).
    if len(out) < limit and leaves:
        for leaf in difflib.get_close_matches(name, list(leaves), n=3, cutoff=0.8):
            for full in sorted(leaves[leaf], key=rank)[:2]:
                push(full)
    if len(out) < limit and name[:1].isupper() and len(name) >= 6 and leaves:
        containing = [
            full
            for leaf, fulls in leaves.items()
            if name in leaf and leaf != name
            for full in fulls
            if full.count(".") == owner_name.count(".") + (0 if is_module else 1)
        ]
        for full in sorted(containing, key=rank)[:2]:
            push(full)
    return out[:limit]


def _attribute_hint(obj: Any, name: str, owner_key: str) -> str | None:
    if inspect.ismodule(obj):
        owner = sys.modules.get(owner_key, obj)
    else:
        owner = obj if isinstance(obj, type) else type(obj)
    owner_name = owner_key.removeprefix("pxr.")
    try:
        members = dir(owner)
    except Exception:  # noqa: BLE001
        members = []
    found = suggest(owner_name, members, name)
    if not found:
        return None
    hint = f"{owner_name} has no {name!r}. Did you mean {', '.join(found)}?"
    index = get_doc_index("pxr")
    entry = index.get(found[0]) if index is not None else None
    if entry is not None and entry.summary:
        summary = entry.summary if len(entry.summary) <= 140 else entry.summary[:139] + "…"
        hint += f"\n{found[0]}{entry.signature.split(' | ')[0]} — {summary}"
    return hint + "\ndoc(name) shows its docs; doc.search('keywords') searches the OpenUSD docs."


def install() -> None:
    """Register the resolver and the AttributeError hints (pxr is imported by now)."""
    global _loaded_for
    with _lock:
        stale = _loaded_for is not None and _loaded_for != installed_version()
    if stale:
        register_doc_index("pxr", load_pxr_index)  # loaded before pxr: pick again
    register_doc_resolver("pxr", _resolve)
    register_attribute_hint_provider("pxr", _attribute_hint)


install()
