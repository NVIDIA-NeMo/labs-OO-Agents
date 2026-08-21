# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct static dependency guard for the native TUI.

ACP is a peer edge adapter. Native presentation modules must not directly import
ACP protocol or runtime implementation modules. Transitive and dynamic-import
checks are outside this baseline guard; broader layering checks are added when
the host-neutral structural agent API exists.
"""

from __future__ import annotations

import ast
import tokenize
from pathlib import Path

_TUI_PACKAGE = Path(__file__).parents[2] / "src" / "nooa_cli" / "tui"
_FORBIDDEN_PREFIXES = ("nooa_acp",)


def _imports(path: Path) -> set[str]:
    with tokenize.open(path) as source:
        tree = ast.parse(source.read(), filename=str(path))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    return imported


def _is_forbidden(module: str) -> bool:
    return any(
        module == prefix or module.startswith(f"{prefix}.") for prefix in _FORBIDDEN_PREFIXES
    )


def test_native_tui_has_no_direct_static_acp_imports() -> None:
    assert _TUI_PACKAGE.is_dir(), f"TUI package not found: {_TUI_PACKAGE}"
    paths = sorted(_TUI_PACKAGE.rglob("*.py"))
    assert _TUI_PACKAGE / "tui_application.py" in paths, (
        "architecture guard scanned no TUI application"
    )

    violations: dict[Path, list[str]] = {}
    for path in paths:
        forbidden = sorted(module for module in _imports(path) if _is_forbidden(module))
        if forbidden:
            violations[path.relative_to(_TUI_PACKAGE)] = forbidden

    assert violations == {}, f"direct ACP imports found beneath {_TUI_PACKAGE}: {violations}"


def test_tui_application_does_not_own_agent_runtime_primitives() -> None:
    path = _TUI_PACKAGE / "tui_application.py"
    with tokenize.open(path) as source:
        tree = ast.parse(source.read(), filename=str(path))
    forbidden_attributes = {
        "_user_messages_in",
        "_slash_commands_in",
        "_system_messages_in",
        "queue_manager",
        "_agent_loop",
        "_agent_thread",
        "_agent_thread_future",
        "_agent_loop_task",
    }
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert attributes.isdisjoint(forbidden_attributes), (
        "renderer still owns concrete agent runtime details: "
        f"{sorted(attributes & forbidden_attributes)}"
    )
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "new_event_loop" not in calls
    assert "Thread" not in calls


def test_tui_application_has_no_concrete_agent_dependency() -> None:
    path = _TUI_PACKAGE / "tui_application.py"
    with tokenize.open(path) as source:
        tree = ast.parse(source.read(), filename=str(path))

    application = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "TUIApplication"
    )
    constructor = next(
        node
        for node in application.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    )
    parameters = {
        argument.arg for argument in (*constructor.args.args, *constructor.args.kwonlyargs)
    }
    assert "agent" in parameters, "renderer must accept the structural InteractiveAgent"
    assert "agent_runner" not in parameters, "renderer still accepts an agent runtime"
    forbidden_runtime_callbacks = {
        "run_on_host",
        "run_on_host_async",
        "job_snapshots",
        "pop_pending_input",
        "host_cancel_pending",
    }
    assert parameters.isdisjoint(forbidden_runtime_callbacks), (
        "renderer still accepts local runtime callbacks: "
        f"{sorted(parameters & forbidden_runtime_callbacks)}"
    )
    method_names = {
        node.name
        for node in application.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert method_names.isdisjoint({"agent_run", "agent_run_async"}), (
        "renderer still exposes an arbitrary host executor"
    )

    imports = _imports(path)
    assert "nooa_cli.interactive.runtime" not in imports, (
        "renderer still imports the agent runtime compatibility facade"
    )

    concrete_attributes = {
        node.attr
        for node in ast.walk(application)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    }
    assert "agent" not in concrete_attributes, "renderer still stores or reads a concrete agent"
    assert "_agent_runner" not in concrete_attributes, "renderer still retains an agent runtime"

    forbidden_attributes = {"event_manager", "_storage", "shell", "queue_manager"}
    all_attributes = {
        node.attr for node in ast.walk(application) if isinstance(node, ast.Attribute)
    }
    assert all_attributes.isdisjoint(forbidden_attributes), (
        "renderer still reaches concrete agent services: "
        f"{sorted(all_attributes & forbidden_attributes)}"
    )


def test_session_does_not_access_private_agent_channels_or_queue_manager() -> None:
    path = _TUI_PACKAGE / "session.py"
    with tokenize.open(path) as source:
        tree = ast.parse(source.read(), filename=str(path))
    forbidden = {"_user_messages_in", "_slash_commands_in", "_system_messages_in", "queue_manager"}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert attributes.isdisjoint(forbidden), (
        f"Session still bypasses LocalAgentRunner: {sorted(attributes & forbidden)}"
    )
