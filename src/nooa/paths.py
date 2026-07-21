# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Well-known filesystem paths for NVIDIA OO Agents.

Two root directories:

- **User dir** — ``~/.config/nooa/`` on every platform (honors
  ``XDG_CONFIG_HOME`` if set). A single, predictable location so the
  installer and the runtime always agree. Override the whole base with
  ``NEMO_OO_USER_DIR``.

- **Project dir** (``<project-root>/.nooa/``) — local to the current
  project; config, optional trace files, library code. Override with
  ``NEMO_OO_PROJECT_DIR``.

Usage::

    from nooa.paths import get_user_dir, get_project_dir

    sessions = get_user_dir("sessions")          # ~/.config/nooa/sessions
    config   = get_project_dir("settings.yaml")  # <root>/.nooa/settings.yaml
"""

import os
from pathlib import Path

#: Directory name used for the project-local directory.
DIR_NAME = ".nooa"


def find_project_root() -> Path:
    """Walk up from this file to find the project root (where pyproject.toml lives).

    Falls back to ``Path.cwd()`` if no ``pyproject.toml`` is found (e.g. when
    the package is installed into site-packages rather than run from source).
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def get_user_dir(*parts: str) -> Path:
    """Return the user-global NVIDIA OO Agents directory, optionally joined with *parts*.

    Location is ``~/.config/nooa/`` on every platform (honoring
    ``XDG_CONFIG_HOME`` when set), so the path is predictable. Override
    the entire base with ``NEMO_OO_USER_DIR``.

    Examples::

        get_user_dir()              # ~/.config/nooa/
        get_user_dir("sessions")    # ~/.config/nooa/sessions/
        get_user_dir("traces.db")   # ~/.config/nooa/traces.db
    """
    override = os.environ.get("NEMO_OO_USER_DIR")
    if override:
        base = Path(override)
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        config_home = Path(xdg) if xdg else Path.home() / ".config"
        base = config_home / "nooa"
    return base.joinpath(*parts) if parts else base


def get_project_dir(*parts: str) -> Path:
    """Return the project-local NVIDIA OO Agents directory, optionally joined with *parts*.

    Defaults to ``<project-root>/.nooa/`` where the project root is
    the nearest ancestor directory containing a ``pyproject.toml``.  Override
    with the ``NEMO_OO_PROJECT_DIR`` environment variable.

    Examples::

        get_project_dir()                   # <root>/.nooa/
        get_project_dir("settings.yaml")    # <root>/.nooa/settings.yaml
        get_project_dir("traces")           # <root>/.nooa/traces/
    """
    base_str = os.environ.get("NEMO_OO_PROJECT_DIR")
    base = Path(base_str) if base_str else find_project_root() / DIR_NAME
    return base.joinpath(*parts) if parts else base
