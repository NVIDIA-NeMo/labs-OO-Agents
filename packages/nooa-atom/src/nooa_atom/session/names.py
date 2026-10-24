# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Names and paths of a session's background work: its children and jobs.

A session has one namespace of live names, shared by its running jobs and
its open children. A name follows Python identifier rules
(``name.isidentifier()``, not a keyword). The top session is ``/root``; a
child's or job's path is its parent's path, ``/``, and its name:
``/root/tests``, ``/root/review_pr_462/verifier``.

A name may not be one ``Waiting(on=[...])`` reads as a channel or as the
person (``RESERVED_NAMES``), nor the name of a channel the agent's queue
manager has: ``Waiting(on=["jobs"])`` must mean the jobs channel, never a
child or job called ``jobs``.
"""

import keyword
from collections.abc import Iterable
from typing import Any

ROOT_PATH = "/root"

RESERVED_NAMES = frozenset(
    {
        "jobs",
        "delegates",
        "user",
        "user_messages",
        "system_messages",
        "slash_commands",
        "person",
        "human",
    }
)
"""Names ``Waiting.on`` reads as a channel or the person; no child or job may use them."""


def is_valid_name(name: Any) -> bool:
    """Whether ``name`` can name a child or a job: an identifier that is not a keyword."""
    return isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name)


def check_name(name: Any, what: str, channels: Iterable[str] = ()) -> str:
    """Return ``name``, or raise ``ValueError`` saying what a valid ``what`` name looks like.

    ``channels``: the queue-manager channel names in use; like
    ``RESERVED_NAMES`` they are refused, since ``Waiting.on`` reads them as channels.
    """
    if not is_valid_name(name):
        raise ValueError(
            f"{what} name {name!r} is not valid: use a Python identifier such as "
            "'tests' or 'review_pr_462' (letters, digits and _, not starting with a "
            "digit, not a keyword)"
        )
    if name in RESERVED_NAMES or name in set(channels):
        raise ValueError(
            f"{what} name {name!r} is reserved: Waiting(on=[{name!r}]) means the "
            f"{name!r} channel, not a {what}; pick a name that says what it does, "
            "such as 'tests' or 'review_pr_462'"
        )
    return name


def child_path(parent_path: str, name: str) -> str:
    """The path of ``name`` under ``parent_path``."""
    return f"{parent_path}/{name}"
