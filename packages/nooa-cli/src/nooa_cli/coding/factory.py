# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent construction shared by native and protocol session hosts."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any


def create_session_agent(
    *, llm: Any, storage: Any, options: Any, agent_cls: type | None = None
) -> Any:
    """Construct the configured agent with identical capabilities in each host."""
    from types import SimpleNamespace

    from nooa_cli.coding.experimental_agent import ExperimentalTUIAgent
    from nooa_cli.coding.legacy_agent import TUIAgent

    if agent_cls is None:
        if options.agent_spec and not options.legacy_agent:
            spec = options.agent_spec
            module, separator, name = spec.rpartition(":")
            if separator and (module.endswith(".py") or "/" in module):
                spec = f"{Path(options.working_dir) / Path(module).expanduser()}:{name}"
            agent_cls = load_agent_class(spec)
        else:
            agent_cls = TUIAgent if options.legacy_agent else ExperimentalTUIAgent
    parameters = inspect.signature(agent_cls).parameters
    kwargs = {"llm": llm, "storage": storage}
    if agent_cls is TUIAgent:
        kwargs.update(
            config=SimpleNamespace(
                working_dir=options.working_dir, summarization=options.summarization
            ),
            skills_dirs=options.skills_dirs,
            libs_dir=Path(options.working_dir) / ".nooa" / "libs",
        )
    else:
        for name, value in {
            "config": SimpleNamespace(
                working_dir=options.working_dir, summarization=options.summarization
            ),
            "cwd": options.working_dir,
            "skills_dirs": options.skills_dirs,
            "summarization": options.summarization,
            "libs_dir": Path(options.working_dir) / ".nooa" / "libs",
        }.items():
            if name in parameters:
                kwargs[name] = value
    return agent_cls(**kwargs)


def load_agent_class(spec: str) -> type:
    """Load an agent class from a 'module:ClassName' or './file.py:ClassName' spec.

    Args:
        spec: Agent spec in the form ``module.path:ClassName`` or
              ``./path/to/file.py:ClassName`` (absolute paths also work).

    Returns:
        The agent class (uninstantiated).

    Raises:
        ValueError: If the spec format is invalid or the class is not an Agent subclass.
        FileNotFoundError: If a file-path spec points to a missing file.
        ImportError: If the module cannot be imported.
        AttributeError: If the class name is not found in the module.
    """
    import importlib
    import importlib.util
    import sys

    if ":" not in spec:
        raise ValueError(
            f"Invalid agent spec '{spec}'. "
            "Expected 'module.path:ClassName' or './path/to/file.py:ClassName'."
        )

    module_part, class_name = spec.rsplit(":", 1)
    class_name = class_name.strip()

    # File path: ends in .py OR contains a path separator OR starts with . / ~
    is_file = module_part.endswith(".py") or "/" in module_part or module_part.startswith(".")
    if is_file:
        file_path = Path(module_part).expanduser().resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Agent module file not found: {file_path}")

        parent_str = str(file_path.parent)
        inserted = False
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
            inserted = True

        try:
            mod_spec = importlib.util.spec_from_file_location("_tui_custom_agent", file_path)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"Cannot load module from {file_path}")
            module = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(module)  # type: ignore[union-attr]
        finally:
            if inserted:
                sys.path.remove(parent_str)
    else:
        module = importlib.import_module(module_part)

    cls = getattr(module, class_name, None)
    if cls is None:
        raise AttributeError(f"Class '{class_name}' not found in '{module_part}'.")

    # Validate it's an Agent subclass
    try:
        from nooa import Agent

        if not (isinstance(cls, type) and issubclass(cls, Agent)):
            raise ValueError(
                f"'{class_name}' is not a subclass of a NOOA Agent. "
                "Make sure your class inherits from Agent."
            )
    except ImportError:
        pass  # Can't validate without nooa; proceed anyway

    return cls
