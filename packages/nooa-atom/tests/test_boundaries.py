# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""nooa_atom stays independent of the older host packages."""

import ast
import re
from pathlib import Path

import nooa_atom

_FORBIDDEN = ("nooa_cli", "nooa_acp")


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_no_module_imports_the_cli_or_acp_packages():
    root = Path(nooa_atom.__file__).parent
    sources = sorted(root.rglob("*.py"))
    assert sources
    offenders = {
        str(path.relative_to(root)): sorted(
            name for name in _imported_modules(path) if name.split(".")[0] in _FORBIDDEN
        )
        for path in sources
    }
    assert {path: names for path, names in offenders.items() if names} == {}


def test_the_agent_modules_use_no_retired_turn_or_worker_types():
    """The Atom agent is on the Session layer: no RespondResult, no CodingWorker."""
    root = Path(nooa_atom.__file__).parent
    retired = ("RespondResult", "RespondReason", "CodingWorker")
    hits = [
        f"{path.relative_to(root)}: {name}"
        for package in ("agent", "tools", "workspace")
        for path in sorted((root / package).rglob("*.py"))
        for name in retired
        if name in path.read_text()
    ]
    assert hits == []


# Private names of the core (and of the acp library) that have public
# replacements. Each pattern is matched against the source text.
_PRIVATE_CORE_NAMES = {
    r"\._items\b": "Channel.remove()",
    r"\._on_get\b": "the ChannelItemConsumed event",
    r"\._on_discard\b": "the ChannelItemsDiscarded event",
    r"\._role\b": "EventBase.event_role",
    r"\._db_lock\b": "SQLiteStorageManager.save_snapshot_json()",
    r"\b_open_connection\b": "SQLiteStorageManager(must_exist=, journal_mode=)",
    r"\b_read_lock_owner\b": "nooa.storage.read_lock_owner",
    r"\b_acquire_session_lock\b": "a read-only lock probe (store._lock_is_held)",
    r"nooa\.tools\._\w+": "nooa.tools (BashSession, StreamEvent, StreamDone)",
    r"\._llm\b|[\"']_llm[\"']": "Agent.llm",
}


def test_no_module_uses_private_core_names_that_have_public_replacements():
    root = Path(nooa_atom.__file__).parent
    hits = [
        f"{path.relative_to(root)}:{number}: {line.strip()} (use {replacement})"
        for path in sorted(root.rglob("*.py"))
        for number, line in enumerate(path.read_text().splitlines(), start=1)
        for pattern, replacement in _PRIVATE_CORE_NAMES.items()
        if re.search(pattern, line)
    ]
    assert hits == []


def _agent_reaches(path: Path) -> list[str]:
    """Attribute reads named ``agent``/``_agent``, and ``getattr(x, "agent")``, in one file."""
    tree = ast.parse(path.read_text(), filename=str(path))
    names = {"agent", "_agent"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in names:
            hits.append(f"{path.name}:{node.lineno}: .{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in ("getattr", "setattr", "hasattr")
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in names
        ):
            hits.append(f"{path.name}:{node.lineno}: {node.func.id}(..., {node.args[1].value!r})")
    return hits


def test_the_acp_package_never_reaches_the_agent():
    """Data only across the Session boundary: the ACP adapter, router and bridge
    use the Session's methods and updates, never the agent it owns.

    Module imports such as ``acp.agent.connection`` are not attribute reads
    and do not count.
    """
    root = Path(nooa_atom.__file__).parent / "acp"
    sources = sorted(root.rglob("*.py"))
    assert sources
    assert [hit for path in sources for hit in _agent_reaches(path)] == []


async def test_the_session_keeps_its_agent_private(make_session):
    session, _ = make_session(start=False)
    assert not hasattr(session, "agent")
