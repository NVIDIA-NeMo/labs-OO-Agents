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

# CodingAgent subclasses already warned about parameters they do not accept.
_warned_classes: set[type] = set()


def create_session_agent(options: SessionOptions, storage: StorageManager) -> InteractiveAgent:
    """Build the agent ``options.agent_spec`` names, for ``options.workspace``.

    The class is loaded with ``load_agent_class`` (``module:Class`` or a
    ``file.py:Class`` path relative to the workspace). It gets the
    session's ``storage`` and ``options.llm``. This function never builds a
    client: a registry with an ``llm_factory`` calls it for every session
    whose options carry no ``llm`` (alias ``options.model``, ``None`` for the
    default model) and passes the result here. Without an ``llm_factory``
    and without ``options.llm``, the class's own default client applies.

    A coding agent (a ``CodingAgent`` subclass) also gets the workspace's
    settings (``CoderOptions``): ``cwd``, ``skills_dirs``, ``summarization``
    and a ``libs_dir`` inside the workspace, each one its ``__init__``
    accepts (a warning names the ones it does not), and then its MCP
    registry and configured skills (``configure_session_skills``) and its
    ``/skills`` and ``/mcp`` controls, before the registry restores any
    snapshot. Connecting remembered MCP servers is async and is left to the
    host's ``prepare`` hook. Any other agent gets only ``storage`` and ``llm``.
    """
    from nooa_coder.coding.agent import CodingAgent

    agent_class = load_agent_class(options.agent_spec, base=options.workspace)
    # Only a coding agent gets the workspace: an unrelated agent's own
    # cwd/summarization/... parameters mean something else and are left alone.
    is_coder = isinstance(agent_class, type) and issubclass(agent_class, CodingAgent)
    coder_options = CoderOptions.load(options.workspace) if is_coder else None

    kwargs: dict[str, Any] = {"storage": storage}
    if options.llm is not None:
        kwargs["llm"] = options.llm
    if coder_options is not None:
        parameters = inspect.signature(agent_class).parameters
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
        workspace = Path(coder_options.working_dir)
        missing = []
        for name, value in {
            "cwd": workspace,
            "skills_dirs": coder_options.skills_dirs,
            "summarization": coder_options.summarization,
            "libs_dir": workspace / ".nooa" / "libs",
        }.items():
            if name in parameters or accepts_any:
                kwargs[name] = value
            else:
                missing.append(name)
        if missing and agent_class not in _warned_classes:
            _warned_classes.add(agent_class)
            logger.warning(
                "%s.__init__ does not accept %s; the session's workspace settings for %s "
                "are not passed to it (add the parameters or **kwargs and forward them)",
                agent_class.__qualname__,
                ", ".join(missing),
                "them" if len(missing) > 1 else "it",
            )
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
