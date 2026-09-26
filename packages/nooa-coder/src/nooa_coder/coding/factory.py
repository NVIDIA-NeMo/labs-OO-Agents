# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the agent and the model client for a session.

``create_session_agent`` is the session registry's default agent factory:
every session, root or child, is built through it from its
``SessionOptions``. ``default_llm_factory`` is the model factory a host
passes to the registry (``SessionRegistry(store, llm_factory=...)``).
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nooa.interactive import InteractiveAgent
from nooa.unifiedllm import get_llm_client
from nooa_coder.session.loader import load_agent_class
from nooa_coder.workspace.controls import behavior_commands
from nooa_coder.workspace.options import CoderOptions, configure_session_skills

if TYPE_CHECKING:
    from nooa.storage.manager import StorageManager
    from nooa_coder.session.options import SessionOptions
    from nooa_coder.session.registry import LLMFactory

logger = logging.getLogger(__name__)


def create_session_agent(options: SessionOptions, storage: StorageManager) -> InteractiveAgent:
    """Build the agent ``options.agent_spec`` names, for ``options.workspace``.

    The class is loaded with ``load_agent_class`` (``module:Class`` or a
    ``file.py:Class`` path relative to the workspace). It gets the
    session's ``storage`` and ``options.llm``. This function never builds a
    client: a registry with an ``llm_factory`` calls it for every session
    whose options carry no ``llm`` (alias ``options.model``, ``None`` for the
    default model) and passes the result here. Without an ``llm_factory``
    and without ``options.llm``, the class's own default client applies.

    A coding agent also gets the workspace's settings (``CoderOptions``):
    ``cwd``, ``skills_dirs``, ``summarization`` and a ``libs_dir`` inside
    the workspace, and then its MCP registry and configured skills
    (``configure_session_skills``) and its ``/skills`` and ``/mcp`` controls,
    before the registry restores any snapshot. Connecting remembered MCP servers is async and is left to
    the host's ``prepare`` hook.
    """
    from nooa_coder.coding.agent import CodingAgent

    agent_class = load_agent_class(options.agent_spec, base=options.workspace)
    parameters = inspect.signature(agent_class).parameters
    # A CodingAgent subclass overriding __init__ to forward extra kwargs (the
    # normal extension pattern: def __init__(self, llm=None, storage=None,
    # **kwargs): super().__init__(llm=llm, storage=storage, **kwargs)) has no
    # literal 'cwd'/'skills_dirs'/'summarization'/'libs_dir' parameter name to
    # match against, so a name-only check silently drops per-workspace
    # isolation for it -- cwd falls back to '.' (this process's own
    # directory) instead of the session's actual workspace. Only trust a
    # **kwargs catch-all to accept these names when __init__ has actually
    # been overridden on a real CodingAgent subclass (CodingAgent's own
    # **kwargs relays unrelated Agent-base keywords, and it names every one
    # of these directly). An unrelated agent that declares **kwargs for
    # some other reason must not have them forced on it.
    overridden_subclass = (
        issubclass(agent_class, CodingAgent) and agent_class.__init__ is not CodingAgent.__init__
    )
    accepts_arbitrary_kwargs = overridden_subclass and any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()
    )
    wants_workspace = accepts_arbitrary_kwargs or any(
        name in parameters for name in ("cwd", "skills_dirs", "summarization", "libs_dir")
    )
    coder_options = CoderOptions.load(options.workspace) if wants_workspace else None

    kwargs: dict[str, Any] = {"storage": storage}
    if options.llm is not None:
        kwargs["llm"] = options.llm
    if coder_options is not None:
        workspace = Path(coder_options.working_dir)
        for name, value in {
            "cwd": workspace,
            "skills_dirs": coder_options.skills_dirs,
            "summarization": coder_options.summarization,
            "libs_dir": workspace / ".nooa" / "libs",
        }.items():
            if name in parameters or accepts_arbitrary_kwargs:
                kwargs[name] = value
    agent = agent_class(**kwargs)

    if coder_options is not None and isinstance(agent, CodingAgent):
        for warning in configure_session_skills(agent, coder_options):
            logger.warning("Session in %s: %s", options.workspace, warning)
        # The /skills and /mcp controls belong to the agent, not to one host:
        # MCPApprovalRequired tells the user to run /mcp approve. set_controls()
        # also refreshes the skill commands.
        agent.slash_commands.set_controls(
            behavior_commands(
                agent,
                coder_options,
                workspace=Path(coder_options.working_dir),
                command_registry=agent.slash_commands,
            )
        )
    return agent


def default_llm_factory(*, workspace_default: str | None = None) -> LLMFactory:
    """A model factory for ``SessionRegistry(llm_factory=...)``.

    The registry calls it as ``factory(alias, workspace)`` for a session
    without a client. A named alias is built with ``get_llm_client``. No
    alias means the default model: ``workspace_default`` when given, else
    the workspace's ``CoderOptions.default_model`` (its settings files, then
    ``nooa.interactive.DEFAULT_MODEL``).
    """

    def make(alias: str | None, workspace: Path) -> Any:
        name = alias or workspace_default or CoderOptions.load(workspace).default_model
        return get_llm_client(name)

    return make
