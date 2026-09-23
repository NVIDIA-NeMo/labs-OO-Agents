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

    from nooa_cli.coding.agent import CodingAgent
    from nooa_cli.coding.experimental_agent import ExperimentalCodingAgent

    if agent_cls is None:
        if options.agent_spec and not options.legacy_agent:
            spec = options.agent_spec
            module, separator, name = spec.rpartition(":")
            if separator and (module.endswith(".py") or "/" in module):
                spec = f"{Path(options.working_dir) / Path(module).expanduser()}:{name}"
            agent_cls = load_agent_class(spec)
        else:
            agent_cls = CodingAgent if options.legacy_agent else ExperimentalCodingAgent
    parameters = inspect.signature(agent_cls).parameters
    # A CodingAgent subclass overriding __init__ to forward extra kwargs (the
    # normal extension pattern: def __init__(self, llm=None, storage=None,
    # **kwargs): super().__init__(llm=llm, storage=storage, **kwargs)) has no
    # literal 'cwd'/'skills_dirs'/'summarization'/'libs_dir' parameter name to
    # match against, so a name-only check silently drops per-workspace
    # isolation for it -- cwd falls back to '.' (this process's own
    # directory) instead of the session's actual workspace. Only trust a
    # **kwargs catch-all to accept these specific names when __init__ has
    # actually been overridden on a real CodingAgent subclass (CodingAgent's
    # and ExperimentalCodingAgent's own **kwargs exists to relay unrelated
    # Agent-base keywords, not these; both already expose every one of these
    # names directly and are matched by name as before). An unrelated custom
    # Agent loaded via --agent that happens to declare **kwargs for some
    # other reason must not have them forced on it either.
    overridden_subclass = issubclass(agent_cls, CodingAgent) and (
        agent_cls.__init__ is not CodingAgent.__init__
    )
    accepts_arbitrary_kwargs = overridden_subclass and any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    kwargs = {"llm": llm, "storage": storage}
    # 'config' has no real consumer anywhere in this codebase today (no
    # class declares a literal 'config' parameter) -- kept name-matched only,
    # never forced through a **kwargs catch-all, so it stays exactly as
    # inert as it already was for every class that does not ask for it by
    # name, instead of starting to break them.
    if "config" in parameters:
        kwargs["config"] = SimpleNamespace(
            working_dir=options.working_dir, summarization=options.summarization
        )
    for name, value in {
        "cwd": options.working_dir,
        "skills_dirs": options.skills_dirs,
        "summarization": options.summarization,
        "libs_dir": Path(options.working_dir) / ".nooa" / "libs",
    }.items():
        if name in parameters or accepts_arbitrary_kwargs:
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

    from nooa_cli.coding.identity import canonical_agent_spec

    spec = canonical_agent_spec(spec)
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
            mod_spec = importlib.util.spec_from_file_location("_nooa_custom_agent", file_path)
            if mod_spec is None or mod_spec.loader is None:
                raise ImportError(f"Cannot load module from {file_path}")
            module = importlib.util.module_from_spec(mod_spec)
            # Some code (dataclasses with postponed annotations, typing.get_type_hints)
            # resolves a class's string annotations via sys.modules[cls.__module__];
            # that lookup fails unless the module is registered before exec_module()
            # runs the file's class definitions.
            sys.modules[mod_spec.name] = module
            try:
                mod_spec.loader.exec_module(module)  # type: ignore[union-attr]
            except BaseException:
                sys.modules.pop(mod_spec.name, None)
                raise
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
