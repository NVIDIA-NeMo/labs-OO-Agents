# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the live (dynamic) context reaches the model.

The cached renderer splits an agent's system blocks into a static, cacheable
prefix and the *live context*: blocks re-evaluated every turn (``state``,
``todo_status``, ``context_usage``, anything from ``set_dynamic``).

A :data:`LiveContextPolicy` is one of three kinds, each carrying only the
parameters it reads:

:class:`DynamicLiveContext` (``kind="dynamic"``)
    A cache boundary, then the fresh ``<context>`` message, regenerated on
    every request and never stored in the history. The behaviour before
    ``append`` became the default.

:class:`NoLiveContext` (``kind="off"``)
    No live context at all: static blocks and history, then the boundary.

:class:`AppendedLiveContext` (``kind="append"``, the default)
    The live context becomes part of the append-only history. A fresh
    snapshot, opened by a one-line note (:func:`snapshot_preface`), is sent
    before the first request and then whenever ``every_n_turns`` or more
    visible responses are counted from the last snapshot-carrying one
    (inclusive): at least every N turns of the visible history, exactly every N
    when nothing is archived. The runtime records the text each response saw in
    ``LLMResponse.dynamic_context``; the renderer replays that text, byte for
    byte, where the request carried it, so every request is a strict
    prefix-extension of the previous one and provider prompt caching covers
    the whole history.

Wherever a policy is accepted, its dict form (``{"kind": "append",
"every_n_turns": 5}``) and a bare kind name (``"off"``; YAML's bare ``off``,
read as ``false``, too) validate through the same :data:`LiveContextPolicy`
annotation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, TypeAdapter


class _Kinded(BaseModel):
    """Base of the policy classes: ``kind`` always counts as set, so a dump with
    ``exclude_unset=True`` (settings layering) keeps the discriminator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    def model_post_init(self, _context: Any, /) -> None:
        self.__pydantic_fields_set__.add("kind")


class DynamicLiveContext(_Kinded):
    """Regenerate the live context as a trailing ``<context>`` message on every request."""

    kind: Literal["dynamic"] = "dynamic"


class NoLiveContext(_Kinded):
    """Send no live context; static prefix and history only."""

    kind: Literal["off"] = "off"


class AppendedLiveContext(_Kinded):
    """Append a prefaced snapshot of the live context to the history.

    A snapshot is sent before the first request and then whenever
    ``every_n_turns`` or more visible responses are counted from the last
    snapshot-carrying one (inclusive): at least every N turns of the visible
    history, exactly every N when nothing is archived.
    """

    kind: Literal["append"] = "append"
    every_n_turns: int = Field(
        default=5,
        ge=1,
        description=(
            "Refresh the snapshot at least every N turns of the visible history "
            "(exactly every N when nothing is archived)"
        ),
    )


def _kind_name_to_dict(value: Any) -> Any:
    """A bare kind name (or YAML's bare ``off``, read as ``false``) is ``{"kind": name}``."""
    if value is False:
        return {"kind": "off"}
    if isinstance(value, str):
        return {"kind": value}
    if isinstance(value, dict) and value.get("kind") is False:
        return {**value, "kind": "off"}
    return value


LiveContextPolicy = Annotated[
    DynamicLiveContext | NoLiveContext | AppendedLiveContext,
    Field(discriminator="kind"),
    BeforeValidator(_kind_name_to_dict),
]
"""One of the three live-context kinds, discriminated by ``kind``."""

_POLICY = TypeAdapter(LiveContextPolicy)


def coerce_live_context(value: Any) -> DynamicLiveContext | NoLiveContext | AppendedLiveContext:
    """Validate a policy object, its dict form or a bare kind name into a policy."""
    return _POLICY.validate_python(value)


def snapshot_due(events: Iterable[Any], every_n: int) -> bool:
    """Whether the next request records a fresh live-context snapshot.

    Due when the visible history holds no :class:`LiveContextSnapshot` (the first
    request, or every earlier snapshot was archived), or when at least
    ``every_n`` model responses follow the last one. With nothing archived this
    is exactly requests 0, N, 2N, ...; archiving a response delays the next
    snapshot, archiving the last snapshot brings it forward.
    """
    from nooa.context_blocks.events import LiveContextSnapshot
    from nooa.llm_types import LLMResponse

    since: int | None = None
    for event in events:
        if isinstance(event, LiveContextSnapshot):
            since = 0
        elif since is not None and isinstance(event, LLMResponse):
            since += 1
    return since is None or since >= every_n


def snapshot_preface(every_n: int) -> str:
    """The one-line note that opens every live-context snapshot in ``append`` mode."""
    return (
        f"[live context snapshot; refreshed at least every {every_n} turns of the visible "
        "history so the conversation stays append-only; values may be stale, a later "
        "snapshot supersedes it]"
    )


__all__ = [
    "AppendedLiveContext",
    "DynamicLiveContext",
    "LiveContextPolicy",
    "NoLiveContext",
    "coerce_live_context",
    "snapshot_due",
    "snapshot_preface",
]
