# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Options a session is created with, and how a child inherits them."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChildMode = Literal["worker", "conversation"]
"""How a child's turns run: ``worker`` reports to its parent with a ``TaskResult``
(``handle_work``); ``conversation`` is the interactive ``handle()``."""


class SessionOptions(BaseModel):
    """How to build and run one session.

    ``agent_spec`` names the agent class as ``module:Class``. ``llm`` is an
    already-built client (tests inject a fake one); it is never
    serialised. When it is ``None`` and the registry has an ``llm_factory``,
    the registry builds a client for ``model`` (``None`` means the
    factory's default) and the session owns it. ``sessions_dir`` is one
    directory for the sessions of every workspace; ``None`` (the default)
    means ``<workspace>/.nooa/sessions`` unless ``NOOA_SESSIONS_DIR`` is set
    (see ``sessions_root``).

    A child carries two separate choices: ``child_mode`` (its turn method:
    ``worker`` runs ``handle_work``, ``conversation`` runs ``handle``) and
    ``keep`` (its lifetime: ``False`` closes it after its first result,
    ``True`` keeps it until it is closed or its tree ends).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    workspace: Path
    agent_spec: str
    model: str | None = None
    llm: Any | None = Field(default=None, exclude=True)
    turn_method: Literal["handle", "handle_batch", "handle_work"] = "handle"
    permission_mode: Literal["auto", "ask"] = "auto"
    max_depth: int = 2
    child_mode: ChildMode | None = None
    keep: bool = False
    name: str | None = None
    host: str = "headless"
    sessions_dir: Path | None = None

    def inherit(self, **overrides: Any) -> "SessionOptions":
        """Options for a child: same workspace, agent, model, mode and depth cap.

        ``name``, ``child_mode`` and ``keep`` are not inherited. An override of ``None``
        means "inherit". A different ``model`` drops the inherited client.
        """
        values = {key: value for key, value in overrides.items() if value is not None}
        llm = self.llm
        if "model" in values and values["model"] != self.model:
            llm = None
        data = self.model_dump()
        data.update(name=None, child_mode=None, keep=False)
        data.update(values)
        data["llm"] = values.get("llm", llm)
        return SessionOptions.model_validate(data)
