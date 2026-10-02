# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Headless host: one in-process session tree, no transport.

Used by benchmarks and scripts::

    async with open_tree(options) as tree:
        outcome = await tree.root.prompt(task_description)
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from nooa_coder.session.loader import AgentFactory
from nooa_coder.session.options import SessionOptions
from nooa_coder.session.registry import SessionRegistry
from nooa_coder.session.session import Session
from nooa_coder.session.store import SessionStore


@dataclass(frozen=True)
class Tree:
    """The registry of an open tree and its root session."""

    registry: SessionRegistry
    root: Session


@asynccontextmanager
async def open_tree(
    options: SessionOptions, *, agent_factory: AgentFactory | None = None
) -> AsyncIterator[Tree]:
    """Create a registry and a root session; close every session on exit.

    Sessions are closed children first, on normal exit and on an
    exception, so a crashed run leaves no live claim on any session file.
    """
    registry = SessionRegistry(SessionStore(options.sessions_dir), agent_factory=agent_factory)
    try:
        root = await registry.create(options)
        yield Tree(registry=registry, root=root)
    finally:
        await registry.close_all()
