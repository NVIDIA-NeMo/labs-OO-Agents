# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent construction shared by native and protocol session hosts."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

from nooa_coder.session.loader import load_agent_class


def create_session_agent(
    *, llm: Any, storage: Any, options: Any, agent_cls: type | None = None
) -> Any:
    """Construct the configured agent with identical capabilities in each host."""
    from types import SimpleNamespace

    from nooa_coder.coding.agent import CodingAgent
    from nooa_coder.coding.experimental_agent import ExperimentalCodingAgent

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
