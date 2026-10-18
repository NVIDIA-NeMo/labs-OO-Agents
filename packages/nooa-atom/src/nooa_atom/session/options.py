# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Options a session is created with, and how a child inherits them."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

_REMOVED_OPTIONS = ("retain", "keep", "mode", "child_mode")
"""Child options that no longer exist: ``kept`` replaced all of them."""


def reject_removed_options(where: str, names: Any) -> None:
    """Raise a one-line ``TypeError`` when ``names`` holds a removed child option."""
    removed = [name for name in _REMOVED_OPTIONS if name in names]
    if removed:
        raise TypeError(
            f"{where} got {removed[0]}=, which was removed: delegate() starts a one-shot "
            "worker and spawn() a kept child (SessionOptions.kept)"
        )


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

    ``kept`` is the one choice a child carries: ``False`` is a one-shot
    worker (``delegate()``): it may not ask questions and is closed after its
    first result. ``True`` is a kept child (``spawn()``): it may ask its
    parent questions, takes follow-ups, and stays open until it is closed or
    its tree ends. Both run the worker turn. ``todo_id`` is the id of the
    Todo the child was briefed on (provenance only). Unknown fields are refused,
    so a removed option (``retain``, ``keep``, ``mode``) cannot be ignored.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    workspace: Path
    agent_spec: str
    model: str | None = None
    llm: Any | None = Field(default=None, exclude=True)
    # ``handle_orchestrate`` is an unattended root that plans, delegates and
    # verifies; its children run ``handle_work`` (SessionPort.delegate/spawn).
    turn_method: Literal["handle", "handle_batch", "handle_work", "handle_orchestrate"] = "handle"
    permission_mode: Literal["auto", "ask"] = "auto"
    max_depth: int = 2
    kept: bool = False
    todo_id: str | None = None
    name: str | None = None
    host: str = "headless"
    sessions_dir: Path | None = None

    @model_validator(mode="before")
    @classmethod
    def _no_removed_options(cls, data: Any) -> Any:
        if isinstance(data, dict):
            reject_removed_options("SessionOptions", data)
        return data

    def inherit(self, **overrides: Any) -> "SessionOptions":
        """Options for a child: same workspace, agent, model, mode and depth cap.

        ``name``, ``kept`` and ``todo_id`` are not inherited. An override of ``None``
        means "inherit". A different ``model`` drops the inherited client.
        """
        reject_removed_options("inherit()", overrides)
        values = {key: value for key, value in overrides.items() if value is not None}
        llm = self.llm
        if "model" in values and values["model"] != self.model:
            llm = None
        data = self.model_dump()
        data.update(name=None, kept=False, todo_id=None)
        data.update(values)
        data["llm"] = values.get("llm", llm)
        return SessionOptions.model_validate(data)
