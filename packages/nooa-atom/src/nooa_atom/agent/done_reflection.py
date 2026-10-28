# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in, post-delivery Done maintenance using existing history/context APIs.

One structured prediction (2048 output tokens), no tools, a 15-second cooperative
wall deadline, and no detached cleanup job. Only completed-call event ranges
and one reflection-owned notes block are editable. Other context/instructions
are preserved verbatim. Namespace cleanup is an independent conservative plan.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field

from nooa.context_blocks.scoped import ScopedContext
from nooa.runtime.event_query import EventQuery

logger = logging.getLogger(__name__)
REFLECTION_TIMEOUT = 15.0
EVIDENCE_CHARS = 12_000
NOTES_KEY = "done_reflection_notes"


class DoneReflectionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(default="", max_length=4000)
    useful_findings: str = Field(default="", max_length=1200)
    decisions: str = Field(default="", max_length=1200)
    outstanding_work: str = Field(default="", max_length=1200)
    drop_names: list[Annotated[str, Field(max_length=80, pattern=r"^[A-Za-z_]\w*$")]] = Field(
        default_factory=list,
        max_length=20,
    )


def _completed_range(agent: Any, call_id: str, start_tag: str | None) -> list[tuple[str, Any]]:
    """One bounded contiguous active run; never cross other input/work events.

    Whole active tags are retained as boundaries. Existing summary overlap
    cannot expand this run because its active slice is revalidated on apply.
    """
    selected: list[tuple[str, Any]] = []
    tags = agent.events.keys()
    try:
        start = tags.index(start_tag)
    except ValueError:
        return []
    # Fetch at most 200 event objects, not every earlier turn's stored payload.
    for tag in tags[start : start + 200]:
        event = agent.events[tag]
        if event.metadata.get("call_id") == call_id:
            selected.append((tag, event))
            if len(selected) == 200:
                break
        elif selected:
            break
    return selected


@dataclass(frozen=True)
class CapturedEvent:
    tag: str
    id: str
    call_id: str
    event_type: str
    content: str


def _bounded_evidence(
    selected: list[tuple[str, Any]],
) -> tuple[list[CapturedEvent], list[dict[str, str]]]:
    """A bounded prediction input, not another history store.

    Only a complete serializable prefix is offered for compaction. Stop before
    an oversized or opaque event rather than silently summarizing unseen data.
    The selected originals remain in the existing EventManager throughout.
    """
    kept, evidence = [], []
    remaining = EVIDENCE_CHARS
    for tag, event in selected:
        try:
            text = event.model_dump_json(
                exclude_none=True,
                exclude={
                    "id",
                    "metadata",
                    "status",
                    "timestamp",
                    "reasoning",
                    "raw_response",
                },
            )
        except Exception:
            break
        cost = len(text) + len(tag) + len(event.event_type) + 50
        if cost > remaining:
            break
        # Only immutable scalar provenance enters the proposal snapshot. In a
        # memory backend, EventManager.update mutates the original Event object.
        event_id, call_id = event.id, event.metadata.get("call_id")
        if type(event_id) is not str or type(call_id) is not str:
            break
        kept.append(CapturedEvent(tag, event_id, call_id, event.event_type, text))
        evidence.append({"tag": tag, "type": event.event_type, "content": text})
        remaining -= cost
    return kept, evidence


def _range_still_current(agent: Any, selected: list[CapturedEvent]) -> bool:
    if not selected:
        return True  # empty evidence is valid for notes/cleanup, never archival
    tags = agent.events.keys()
    wanted = [item.tag for item in selected]
    try:
        start = tags.index(wanted[0])
    except ValueError:
        return False
    if tags[start : start + len(wanted)] != wanted:
        return False
    current, _ = _bounded_evidence([(tag, agent.events[tag]) for tag in wanted])
    return current == selected  # frozen id, call ownership, type AND content


async def reflect_accepted_done(agent: Any, outcome: Any) -> None:
    candidate, agent._reflection_candidate = agent._reflection_candidate, None
    if not agent._done_reflection or candidate is None or candidate[0] is not outcome:
        return
    # No recursive trigger: only AtomCodeAct.handle records a candidate, while
    # _reflect_done_plan is Predict and is never a session turn/settlement.
    deadline = asyncio.get_running_loop().time() + REFLECTION_TIMEOUT
    call_id = candidate[1]
    selected, evidence = _bounded_evidence(_completed_range(agent, call_id, candidate[2]))
    namespace = agent._handle_namespace
    snapshot = namespace.cleanup_snapshot() if namespace is not None else None
    candidates = namespace.cleanup_candidates(snapshot) if snapshot is not None else []
    if agent.jobs.running():
        candidates = []
    old_notes = agent.context.get(NOTES_KEY)
    if old_notes is not None and type(old_notes) is not str:
        return  # someone replaced the reflection-owned block with an expression/object
    request = {
        "accepted_outcome": outcome.explanation[:4000],
        "completed_event_tags": [item.tag for item in selected],
        "completed_evidence": evidence,
        "previous_notes": (old_notes or "")[:4000],
        "namespace_inventory": namespace.inventory() if namespace is not None else None,
        "eligible_drop_names": candidates,
        "active_work": bool(agent.jobs.running()),
    }
    try:
        if asyncio.get_running_loop().time() >= deadline:
            return
        async with asyncio.timeout_at(deadline):
            # The request explicitly contains bounded completed evidence. Do not
            # rely on default history visibility, nor replay unbounded history.
            # Governing/context blocks remain managed by the existing renderer.
            with ScopedContext(events=EventQuery.current_call()):
                plan = await agent._reflect_done_plan(request)
        # A stale/overlapping event proposal or changed notes is rejected as a
        # whole before editing anything. Appended input after the watermark is
        # fine: it is not part of the captured range and cannot be collapsed.
        if not any(
            (
                plan.summary,
                plan.drop_names,
                plan.useful_findings,
                plan.decisions,
                plan.outstanding_work,
            )
        ):
            return  # genuine no-op; no mutation or observer invocation
        if agent.context.get(NOTES_KEY) is not old_notes:
            return
        if not _range_still_current(agent, selected):
            return  # every effect, not just archival, depends on captured evidence
        if plan.summary and not selected:
            return
        if not set(plan.drop_names) <= set(candidates):
            return
        # Namespace changes can be rejected independently: preserve the
        # original result and context compaction even if active work appeared.
        if namespace is not None and snapshot is not None:
            namespace.apply_cleanup(
                snapshot, plan.drop_names, active_work=bool(agent.jobs.running())
            )
        if plan.summary:
            agent.events.collapse(selected[0].tag, selected[-1].tag, plan.summary)
        notes = "\n".join(
            text
            for text in (
                "Useful findings: " + plan.useful_findings if plan.useful_findings else "",
                "Decisions: " + plan.decisions if plan.decisions else "",
                "Outstanding work: " + plan.outstanding_work if plan.outstanding_work else "",
            )
            if text
        )
        # collapse synchronously emits Summary to observers. They may have
        # installed a newer note; archival does not authorize overwriting it.
        if notes and agent.context.get(NOTES_KEY) is old_notes:
            agent.context[NOTES_KEY] = notes
    except asyncio.CancelledError:
        raise  # session/loop close owns and awaits this maintenance task
    except Exception:
        # Error/timeout/budget rejection never changes the already-delivered
        # accepted result and never emits a second user reply.
        logger.warning("Done reflection skipped after error/timeout", exc_info=True)
