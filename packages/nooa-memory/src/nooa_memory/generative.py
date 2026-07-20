# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LLM-backed reflection steps — the production reconciler and reasoner.

Reflection's deterministic ops are conservative hygiene (merge near-verbatim
duplicates, link, rescore, prune). These factories build the two *generative*
steps the engine accepts:

* **reconciler** — given a cluster of related memories (cosine >= the recon
  threshold, oldest first), decide whether they are redundant/stale versions of
  one fact; if so, return one consolidated CURRENT memory + the ids to archive
  as superseded. This is what folds paraphrase duplicates and outdated values.
* **reasoner** — given episode memories, distill durable insights as new
  ``reflection``-type memories (the engine links them ``DERIVED_FROM`` the
  episodes it showed).

Both take a ``get_llm`` **getter**, not a client — the model is resolved per
call, so a host's ``/model`` switch applies to the next reflection
automatically. Both raise on malformed LLM output; the engine's per-cluster /
per-run containment turns that into "skip this item", never a corrupted store.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from nooa_memory.schema import Memory, MemoryType

_MAX_INSIGHTS = 5  # reasoner output cap per reflection run
_CONTENT_CLIP = 700  # per-memory text shown to the model

_RECONCILER_PROMPT = """\
You maintain an agent's long-term memory. Below is a cluster of memory records
that are semantically related (listed oldest first). Decide whether they are
redundant phrasings and/or stale versions of ONE underlying fact.

- If they are: write ONE consolidated, current, self-contained record that
  preserves every distinct detail, and list the ids that it supersedes.
- If they record genuinely different facts: they are not redundant.

Records:
{items}

Reply with ONLY a JSON object:
{{"redundant": true/false,
  "consolidated": "the merged record text, or null",
  "supersede": ["id", ...]}}
"""

_REASONER_PROMPT = """\
You maintain an agent's long-term memory. Below are episode records (what
happened in past tasks). Distill at most {max_insights} durable, reusable
insights worth keeping long-term — conventions, causal lessons, reusable
procedures. Skip anything one-off or already obvious from a single episode.

Episodes:
{items}

Reply with ONLY a JSON object:
{{"insights": [{{"title": "short title", "content": "one self-contained insight",
                "tags": ["tag", ...]}}]}}
"""


def _render_items(memories: list[Memory]) -> str:
    lines = []
    for m in memories:
        age_h = max(0.0, (time.time() - m.created_at) / 3600.0)
        content = m.content.replace("\n", " ").strip()[:_CONTENT_CLIP]
        lines.append(f"- id={m.id} age={age_h:.1f}h type={m.type.value}: {content}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """The object between the first '{' and the last '}' — raise otherwise."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"no JSON object in LLM reply: {text[:120]!r}")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM reply is not a JSON object")
    return data


def _complete(get_llm: Callable[[], object], prompt: str) -> str:
    llm = get_llm()
    response = llm.call(messages=[{"role": "user", "content": prompt}])
    return response.content or ""


_EPISODE_PROMPT = """\
You maintain an agent's long-term memory. Below are the agent's recent session
events (newest last). Write ONE short episode record — what was worked on, what
was decided or produced, and how it ended — as durable, self-contained prose a
future session can learn from. If nothing noteworthy happened (small talk,
trivial lookups), the episode is not worth keeping.

Recent events:
{items}

Reply with ONLY a JSON object:
{{"noteworthy": true/false, "episode": "the episode text, or null"}}
"""

_MAX_EVENTS = 30  # recent-event window rendered for the episode writer
_EVENT_CLIP = 400  # per-event text clip


def render_recent_events(agent: object, n: int = _MAX_EVENTS) -> str:
    """The recent-events transcript the episode writer summarizes.

    Reuses retrieval's event access (same package); call on the agent's
    thread/loop — the event manager is not a cross-thread surface.
    """
    from nooa_memory.retrieval import _event_text, _recent_events

    lines: list[str] = []
    for event in reversed(_recent_events(agent, n)):  # oldest first
        text = _event_text(event)
        if text:
            lines.append(f"- [{type(event).__name__}] {text[:_EVENT_CLIP]}")
    return "\n".join(lines)


def llm_episode_writer(get_llm: Callable[[], object]) -> Callable[[str], str | None]:
    """Build the episode writer: recent-events transcript -> episode text.

    Returns None when the model judges the window not noteworthy (or the
    transcript is empty). Used by the TUI idle runner as the FIRST phase of a
    generative reflection: write the episode, then consolidate — which also
    feeds the reasoner, whose input is episode memories.
    """

    def write_episode(events_text: str) -> str | None:
        if not events_text.strip():
            return None
        prompt = _EPISODE_PROMPT.format(items=events_text)
        data = _extract_json(_complete(get_llm, prompt))
        episode = data.get("episode")
        if not data.get("noteworthy") or not isinstance(episode, str) or not episode.strip():
            return None
        return episode.strip()

    return write_episode


def llm_reconciler(
    get_llm: Callable[[], object],
) -> Callable[[list[Memory]], tuple[Memory | None, list[str]]]:
    """Build the engine-facing reconciler callable (see module docstring)."""

    def reconcile(cluster: list[Memory]) -> tuple[Memory | None, list[str]]:
        prompt = _RECONCILER_PROMPT.format(items=_render_items(cluster))
        data = _extract_json(_complete(get_llm, prompt))
        if not data.get("redundant"):
            return None, []
        supersede = [i for i in data.get("supersede") or [] if isinstance(i, str)]
        consolidated: Memory | None = None
        text = data.get("consolidated")
        if isinstance(text, str) and text.strip():
            # Inherit the cluster's identity: dominant type, max importance,
            # union of tags. The engine stamps owner and REFINES provenance.
            types = [m.type for m in cluster]
            dominant = max(set(types), key=types.count)
            tags = sorted({t for m in cluster for t in m.tags})
            consolidated = Memory(
                content=text.strip(),
                type=dominant if dominant is not MemoryType.EPISODE else MemoryType.INFO,
                importance=max(m.importance for m in cluster),
                tags=tags,
            )
        return consolidated, supersede

    return reconcile


def llm_reasoner(get_llm: Callable[[], object]) -> Callable[[list[Memory]], list[Memory]]:
    """Build the engine-facing reasoner callable (see module docstring)."""

    def reason(episodes: list[Memory]) -> list[Memory]:
        prompt = _REASONER_PROMPT.format(items=_render_items(episodes), max_insights=_MAX_INSIGHTS)
        data = _extract_json(_complete(get_llm, prompt))
        out: list[Memory] = []
        for item in (data.get("insights") or [])[:_MAX_INSIGHTS]:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, str) or not content.strip():
                continue
            title = item.get("title")
            tags = [t for t in (item.get("tags") or []) if isinstance(t, str)]
            out.append(
                Memory(
                    content=content.strip(),
                    type=MemoryType.REFLECTION,
                    title=title if isinstance(title, str) else None,
                    tags=tags,
                    importance=6.0,  # insights start above the default band
                )
            )
        return out

    return reason
