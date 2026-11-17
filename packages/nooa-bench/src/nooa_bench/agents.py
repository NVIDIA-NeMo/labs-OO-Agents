# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark agent registry: agent types by name.

The built-in agents (``bench``, ``rlm``) live in this package. Other packages
add agents through the ``nooa_bench.agents`` entry-point group; nooa-atom
registers ``atom`` there::

    [project.entry-points."nooa_bench.agents"]
    atom = "nooa_atom.bench_agent:AtomBenchAgent"

An agent is loaded only when it is asked for, so nooa-bench does not import
(or depend on) the packages that provide the others.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from nooa_bench import AGENT_CLASSES

ENTRY_POINT_GROUP = "nooa_bench.agents"

# Agents that other packages provide, and the package to install for each.
KNOWN_PROVIDERS: dict[str, str] = {"atom": "nooa-atom"}


class AgentNotInstalledError(ValueError):
    """No installed package provides a benchmark agent with this name."""


def _entry_points() -> dict[str, object]:
    return {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}


def available_agents() -> list[str]:
    """The names of the agents that can be loaded, sorted."""
    return sorted(set(AGENT_CLASSES) | set(_entry_points()))


def missing_agent_message(name: str) -> str:
    """The error text for an agent name nothing provides."""
    installed = ", ".join(available_agents()) or "none"
    message = f"No benchmark agent named {name!r} is installed (installed: {installed})."
    provider = KNOWN_PROVIDERS.get(name)
    if provider is not None:
        message += f" The {name!r} agent comes with {provider}: install {provider} (pip install {provider})."
    return message


def load_agent(name: str) -> type:
    """Return the agent class registered as ``name``.

    Raises ``AgentNotInstalledError`` when no built-in agent and no
    ``nooa_bench.agents`` entry point has that name.
    """
    builtin = AGENT_CLASSES.get(name)
    if builtin is not None:
        module_path, class_name = builtin.rsplit(":", 1)
        return getattr(importlib.import_module(module_path), class_name)
    entry_point = _entry_points().get(name)
    if entry_point is None:
        raise AgentNotInstalledError(missing_agent_message(name))
    return entry_point.load()


__all__ = [
    "ENTRY_POINT_GROUP",
    "AgentNotInstalledError",
    "available_agents",
    "load_agent",
    "missing_agent_message",
]
