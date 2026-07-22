# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Memory record schema — intentionally *loose*.

Only ``id``, ``type``, ``content`` and ``created_at`` are required; every
descriptor, metadata field and edge is optional with a sane default, so a
memory can be a one-line fact or a fully-annotated, graph-linked episode.

The type taxonomy follows Tulving/Squire (episodic/semantic/procedural +
prospective + working); descriptors (importance/salience/confidence/mood) follow the
poignancy/emotional-tagging literature; ``access_log``/``strength`` feed the
ACT-R base-level activation and the Ebbinghaus spaced-repetition decay.
"""

from __future__ import annotations

import re
import time
import uuid
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

from nooa_memory.descriptors import TODO_STATUSES, to_label


class MemoryType(StrEnum):
    """Taxonomy of memory kinds (extends the user's ``skill`` + ``info``)."""

    INFO = "info"  # semantic: facts, prefs, domain rules, conventions
    SKILL = "skill"  # procedural: reusable verified procedures + applicability
    EPISODE = "episode"  # episodic: a specific task run (goal/actions/outcome)
    INTENT = "intent"  # prospective: trigger-based reminder ("when X happens, do Y")
    TODO = "todo"  # prospective: durable commitment with an open/done lifecycle
    REFLECTION = "reflection"  # schema/gist: insight distilled from episodes
    SCRATCH = "scratch"  # working memory: transient, never durably consolidated


class EdgeType(StrEnum):
    """Typed, directed edges of the memory graph."""

    DERIVED_FROM = "derived_from"  # CAUSAL provenance: this came from <target>
    CREATED_BY = "created_by"  # CAUSAL: task/event that encoded this
    SUPPORTS = "supports"  # corroborating evidence
    CONTRADICTS = "contradicts"  # conflicting belief
    REFINES = "refines"  # this updates/sharpens a prior memory
    RELATED = "related"  # generic associative/semantic link
    CAUSES = "causes"  # domain causal: A -> outcome B
    PRECEDES = "precedes"  # temporal ordering of episodes
    PART_OF = "part_of"  # composition (skill composed of sub-skills)
    TRIGGERS = "triggers"  # intent -> action (prospective)


# Edge types whose causal/structural meaning warrants a higher default traversal
# weight than a generic ``related`` link.
CAUSAL_EDGE_TYPES: frozenset[EdgeType] = frozenset(
    {EdgeType.DERIVED_FROM, EdgeType.CREATED_BY, EdgeType.CAUSES, EdgeType.REFINES}
)


# Max length of a memory's access_log (bounds row size; older accesses drop off).
_ACCESS_LOG_CAP = 64

# Every way a memory can reach the agent (or be consulted by the system).
# "created" seeds ACT-R at write time and marks coerced pre-v2 entries.
ACCESS_CHANNELS: tuple[str, ...] = (
    "created",
    "recalled",
    "searched",
    "injected",
    "reinforced",
    "deref",
    "reflected",
)
# Channels whose entries feed ACT-R base-level activation. "injected" is
# deliberately absent: spontaneous injection is LOGGED but must never
# self-reinforce activation (v1 invariant, manager.py touch=False semantics).
ACTR_CHANNELS: frozenset[str] = frozenset(
    {"created", "recalled", "searched", "reinforced", "deref"}
)
# Channels that update last_accessed/access_count (and by default strength).
TOUCHING_CHANNELS: frozenset[str] = frozenset({"recalled", "searched", "reinforced", "deref"})
# channel -> uncapped per-channel counter field on Memory.
_CHANNEL_COUNTERS: dict[str, str] = {
    "recalled": "recalled_count",
    "searched": "searched_count",
    "injected": "injected_count",
    "reinforced": "reinforced_count",
    "deref": "deref_count",
}


def _now() -> float:
    return time.time()


def role_of(owner: str) -> str:
    """The role part of a hierarchical owner (``role[@instance]``)."""
    return owner.partition("@")[0]


def owner_matches(row_owner: str, scope: str | None) -> bool:
    """Row visibility under an owner scope (design-memory-scope-and-owner §5b).

    ``None`` matches everything; unowned rows match every scope; a scope
    carrying an instance (``role@inst``) matches that exact instance; a bare
    role matches every instance of the role (incl. bare-role legacy rows).
    """
    if scope is None or row_owner == "":
        return True
    if "@" in scope:
        return row_owner == scope
    return role_of(row_owner) == scope


def _new_id() -> str:
    return uuid.uuid4().hex


class Edge(BaseModel):
    """A directed edge from one memory to another."""

    target_id: str
    type: EdgeType = EdgeType.RELATED
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=_now)


# Reference kinds a memory may point at (see memory/references.py for resolution).
REF_KINDS: tuple[str, ...] = ("var", "context", "file", "todo", "memory")


class MemoryRef(BaseModel):
    """A typed pointer from a memory to live agent state (pass-by-reference).

    ``kind:key`` at the agent-facing boundary — e.g. ``var:plan``,
    ``file:docs/spec.md``. Resolution is strict name lookup (never eval);
    ``preview`` is a truncated snapshot captured at write time, rendered when
    the target no longer resolves (DANGLING).
    """

    kind: str
    key: str
    preview: str | None = None
    captured_at: float = Field(default_factory=_now)

    def model_post_init(self, __context: object) -> None:  # noqa: D105
        if self.kind not in REF_KINDS:
            raise ValueError(f"unknown reference kind {self.kind!r}; expected one of {REF_KINDS}")

    def spec(self) -> str:
        """The compact agent-facing form, ``kind:key``."""
        return f"{self.kind}:{self.key}"


class AccessRecord(BaseModel):
    """One structured access to a memory — self-contained observability.

    Lives ON the record (capped ring in ``Memory.access_log``): copy or export
    one row and its usage story travels with it. ``trace_ref`` is the active
    span id when tracing is on — the viewer deep-links it to the exact trace
    moment ("last fetch point").
    """

    ts: float
    channel: str = "recalled"
    reader_owner: str = ""  # who fetched (cross-owner analysis)
    session_ref: str | None = None
    trace_ref: str | None = None
    query: str | None = None  # truncated; None for reinforced/created
    score: float | None = None
    rank: int | None = None
    components: dict | None = None  # {rel, rec, imp, spread} at fetch time

    def model_post_init(self, __context: object) -> None:  # noqa: D105
        if self.channel not in ACCESS_CHANNELS:
            raise ValueError(
                f"unknown access channel {self.channel!r}; expected one of {ACCESS_CHANNELS}"
            )


class Memory(BaseModel):
    """A single long-term memory record."""

    # --- identity (required: id/type/content) ---
    id: str = Field(default_factory=_new_id)
    type: MemoryType = MemoryType.INFO
    title: str | None = None
    content: str
    # Writer's stable agent identifier; "" = unowned/shared (pre-v2 rows and
    # deliberately common memories). Promoted to a SQL column for filtering.
    owner: str = ""

    # --- structural (auto-derived from content if left at defaults) ---
    size_chars: int = 0
    token_len: int = 0
    sentence_count: int = 0

    # --- descriptors ---
    importance: float = Field(default=5.0, ge=0.0, le=10.0)  # LLM "poignancy" 1-10
    salience: float = Field(default=0.0, ge=0.0, le=1.0)  # outcome/surprise/novelty tag
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)  # belief certainty
    mood: str | None = None
    strength: int = 1  # spaced-repetition counter (+1 on recall)
    reinforcement_count: int = 0  # merge/replay reinforcement tally

    # --- metadata ---
    created_at: float = Field(default_factory=_now)
    last_accessed_at: float = Field(default_factory=_now)
    # Capped ring of structured accesses (see AccessRecord); pre-v2 bare
    # timestamps coerce to channel="created" entries on read.
    access_log: list[AccessRecord] = Field(default_factory=list)
    access_count: int = 0  # touching accesses only (v1 semantics preserved)
    # Uncapped per-channel totals — survive access_log rotation.
    recalled_count: int = 0
    searched_count: int = 0
    injected_count: int = 0
    reinforced_count: int = 0
    deref_count: int = 0
    source_task_ref: str | None = None
    related_files: list[str] = Field(default_factory=list)
    chat_turn_ref: str | None = None
    valid_from: float | None = None
    valid_to: float | None = None  # bi-temporal: invalidate-don't-delete
    trigger: dict | None = None  # INTENT only: {kind, cue, fire_at}
    # Typed pointers to live agent state (pass-by-reference; re-read at recall).
    references: list[MemoryRef] = Field(default_factory=list)

    # --- encoding context cue (for encoding-specificity overlap) ---
    tags: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    place_or_task: str | None = None

    # --- graph + lifecycle ---
    edges: list[Edge] = Field(default_factory=list)
    archived: bool = False  # soft-delete tombstone (forgotten but recoverable)
    # TODO lifecycle state (open/done/dropped); None for every other type.
    # Promoted to a SQL column for filtering. Semantics enforced from v2 on.
    status: str | None = None

    @field_validator("access_log", mode="before")
    @classmethod
    def _coerce_legacy_access_log(cls, value: object) -> object:
        """Pre-v2 logs were bare timestamps; coerce them to 'created' entries.

        Migration-by-validator: no SQL involved. Coerced entries stay in the
        ACT-R set (v1 entries all fed activation) without inflating the new
        per-channel counters.
        """
        if isinstance(value, list):
            return [
                {"ts": item, "channel": "created"} if isinstance(item, (int, float)) else item
                for item in value
            ]
        return value

    def model_post_init(self, __context: object) -> None:  # noqa: D105
        # Derive structural fields when not explicitly provided.
        if not self.size_chars:
            self.size_chars = len(self.content)
        if not self.token_len:
            self.token_len = max(1, len(self.content) // 4)  # ~4 chars/token
        if not self.sentence_count:
            self.sentence_count = max(1, len(re.findall(r"[.!?]+", self.content)) or 1)
        if not self.access_log:
            self.access_log = [AccessRecord(ts=self.created_at, channel="created")]
        # TODO lifecycle: todos always carry a status; no other type may.
        if self.type is MemoryType.TODO:
            if self.status is None:
                self.status = "open"
            elif self.status not in TODO_STATUSES:
                raise ValueError(
                    f"invalid todo status {self.status!r}; expected one of {TODO_STATUSES}"
                )
        elif self.status is not None:
            raise ValueError("status is only valid on todo memories")

    # ------------------------------------------------------------------
    # behaviour
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        # Compact by design: agents print Memory objects inside REPL cells and
        # the full-record default (~22KB with edges/access log) ballooned LLM
        # prompts to 180k tokens. Full data stays available via model_dump().
        head = (self.content or "").replace("\n", " ")[:80]
        suffix = "…" if len(self.content or "") > 80 else ""
        title = f" title={self.title!r}" if self.title else ""
        return f"<Memory {self.id[:8]} {self.type.value}{title} {head!r}{suffix}>"

    def __str__(self) -> str:
        return self.__repr__()

    def embedding_text(self) -> str:
        """Text to embed: title + content + cue tags/entities (boosts recall)."""
        parts: list[str] = []
        if self.title:
            parts.append(self.title)
        parts.append(self.content)
        if self.tags:
            parts.append(" ".join(self.tags))
        if self.entities:
            parts.append(" ".join(self.entities))
        return "\n".join(parts)

    def log_access(
        self,
        record: AccessRecord,
        *,
        reinforce: bool | None = None,
        cap: int = _ACCESS_LOG_CAP,
    ) -> None:
        """Record one structured access on the memory itself.

        Every channel appends to the (capped) log and bumps its uncapped
        counter; only TOUCHING channels update ``last_accessed_at`` /
        ``access_count`` and (by default) ``strength`` — so an ``injected``
        entry is *logged without touching* and ACT-R activation is unchanged
        by spontaneous injection (the v1 invariant). ``reinforce=False``
        suppresses the strength bump for touching channels (the update path).
        """
        self.access_log.append(record)
        if len(self.access_log) > cap:
            self.access_log = self.access_log[-cap:]
        counter = _CHANNEL_COUNTERS.get(record.channel)
        if counter is not None:
            setattr(self, counter, getattr(self, counter) + 1)
        if record.channel in TOUCHING_CHANNELS:
            self.last_accessed_at = record.ts
            self.access_count += 1
            if reinforce is None or reinforce:
                self.strength += 1

    def touch(self, *, when: float | None = None, reinforce: bool = True) -> None:
        """Record a plain deliberate access (compat wrapper over ``log_access``).

        This is the online half of memory dynamics — retrieval strengthens a
        memory (anti-forgetting) and refreshes its recency.
        """
        ts = _now() if when is None else when
        self.log_access(AccessRecord(ts=ts, channel="recalled"), reinforce=reinforce)

    def add_edge(
        self, target_id: str, type: EdgeType = EdgeType.RELATED, weight: float = 1.0
    ) -> None:
        """Add (or update the weight of) a directed edge to ``target_id``."""
        for e in self.edges:
            if e.target_id == target_id and e.type == type:
                e.weight = max(e.weight, weight)
                return
        self.edges.append(Edge(target_id=target_id, type=type, weight=weight))

    def cue_set(self) -> set[str]:
        """Lower-cased tag + entity + place cues, for Jaccard context overlap."""
        cues = {t.lower() for t in self.tags} | {e.lower() for e in self.entities}
        if self.place_or_task:
            cues.add(self.place_or_task.lower())
        return cues

    # ------------------------------------------------------------------
    # verbal descriptor rendering (numeric -> ALL-CAPS band, for the agent)
    # ------------------------------------------------------------------
    def importance_label(self) -> str:
        """This memory's importance as a verbal band (CRITICAL..TRIVIAL)."""
        return to_label("importance", self.importance)

    def salience_label(self) -> str:
        """This memory's salience as a verbal band (PIVOTAL..NONE)."""
        return to_label("salience", self.salience)

    def confidence_label(self) -> str:
        """This memory's confidence as a verbal band (CERTAIN..UNCERTAIN)."""
        return to_label("confidence", self.confidence)
