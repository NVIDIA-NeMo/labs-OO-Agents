# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load an agent class from a ``module:Class`` spec and build the agent for a session."""

import importlib
import json
import logging
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from nooa.interactive import InteractiveAgent
from nooa.storage.manager import StorageManager
from nooa_coder.session.options import SessionOptions

logger = logging.getLogger(__name__)

AgentFactory = Callable[[SessionOptions, StorageManager], InteractiveAgent]
"""Builds the agent for a session from its options and its storage."""


class AgentSpecError(ValueError):
    """The agent spec is malformed or does not name an ``InteractiveAgent`` class."""


def load_agent_class(spec: str) -> type[InteractiveAgent]:
    """Import ``module:Class`` (the class part may be dotted) and check its type."""
    module_name, _, class_path = spec.partition(":")
    if not module_name or not class_path:
        raise AgentSpecError(f"Agent spec {spec!r} must look like 'module:Class'")
    try:
        target: Any = importlib.import_module(module_name)
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


def default_agent_factory(options: SessionOptions, storage: StorageManager) -> InteractiveAgent:
    """Instantiate ``options.agent_spec`` with the session's storage and ``options.llm``."""
    agent_class = load_agent_class(options.agent_spec)
    if options.llm is not None:
        return agent_class(llm=options.llm, storage=storage)
    return agent_class(storage=storage)


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
    try:
        target: Any = importlib.import_module(module_name)
        for part in qualname.split("."):
            target = getattr(target, part)
    except (ImportError, AttributeError):
        logger.warning("Cannot import %s; keeping the value as JSON data", type_name)
        return value
    if not (isinstance(target, type) and issubclass(target, BaseModel)):
        return value
    try:
        return target.model_validate(value)
    except Exception:
        logger.warning("Recorded data does not validate as %s; keeping JSON data", type_name)
        return value
