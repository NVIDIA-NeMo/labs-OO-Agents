# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the loose memory record schema."""

import pytest
from nooa_memory.descriptors import ladder, to_numeric
from nooa_memory.schema import _ACCESS_LOG_CAP, Edge, EdgeType, Memory, MemoryType


def test_required_minimal_and_derived_structural_fields():
    m = Memory(content="Hello world. This is the second sentence! And a third?")
    assert m.id and m.type == MemoryType.INFO
    assert m.size_chars == len(m.content)
    assert m.token_len >= 1
    assert m.sentence_count == 3  # . ! ?
    assert [(a.ts, a.channel) for a in m.access_log] == [(m.created_at, "created")]  # seeded
    assert m.archived is False


def test_explicit_structural_fields_are_not_overwritten():
    m = Memory(content="x", size_chars=999, token_len=5, sentence_count=7)
    assert (m.size_chars, m.token_len, m.sentence_count) == (999, 5, 7)


def test_touch_bumps_recency_frequency_and_strength():
    m = Memory(content="fact", strength=1, access_count=0)
    before = m.strength
    m.touch(when=m.created_at + 10.0)
    assert m.access_count == 1
    assert m.strength == before + 1
    assert m.last_accessed_at == m.created_at + 10.0
    assert m.access_log[-1].ts == m.created_at + 10.0


def test_touch_without_reinforce_keeps_strength():
    m = Memory(content="fact", strength=3)
    m.touch(reinforce=False)
    assert m.strength == 3
    assert m.access_count == 1


def test_access_log_is_capped():
    m = Memory(content="fact")
    for i in range(200):
        m.touch(when=float(i))
    assert len(m.access_log) <= _ACCESS_LOG_CAP


def test_add_edge_dedups_and_keeps_max_weight():
    m = Memory(content="a")
    m.add_edge("t1", EdgeType.RELATED, 0.4)
    m.add_edge("t1", EdgeType.RELATED, 0.9)  # same target+type -> max weight
    m.add_edge("t1", EdgeType.CAUSES, 0.5)  # different type -> new edge
    assert len(m.edges) == 2
    related = next(e for e in m.edges if e.type == EdgeType.RELATED)
    assert related.weight == 0.9


def test_embedding_text_includes_title_content_and_cues():
    m = Memory(content="deploy", title="how-to", tags=["ops", "ci"], entities=["make"])
    txt = m.embedding_text()
    assert "how-to" in txt and "deploy" in txt and "ops" in txt and "make" in txt


def test_cue_set_is_lowercased_union():
    m = Memory(content="x", tags=["Ops"], entities=["Make"], place_or_task="Deploy")
    assert m.cue_set() == {"ops", "make", "deploy"}


def test_edge_weight_bounds():
    e = Edge(target_id="t", type=EdgeType.SUPPORTS, weight=1.0)
    assert 0.0 <= e.weight <= 1.0


# --- verbal ordered descriptors -------------------------------------------------
def test_to_numeric_maps_labels_and_raises_on_unknown():
    assert to_numeric("importance", "CRITICAL") == 10.0
    assert to_numeric("importance", "MEDIUM") == 5.0  # == schema default (round-trips)
    assert to_numeric("salience", "NONE") == 0.0
    assert to_numeric("confidence", "TENTATIVE") == 0.5
    assert to_numeric("edge_weight", "STRONG") == 1.0
    with pytest.raises(ValueError):  # no fallback / dual-accept
        to_numeric("importance", "VERY-HIGH")


def test_defaults_round_trip_to_their_band():
    m = Memory(content="x")  # schema defaults
    assert m.importance_label() == "MEDIUM"  # 5.0
    assert m.salience_label() == "NONE"  # 0.0
    assert m.confidence_label() == "TENTATIVE"  # 0.5


def test_importance_label_is_monotonic_and_ordered_from_config():
    # bands are the single source of truth, highest-first
    assert ladder("importance") == ("CRITICAL", "HIGH", "MEDIUM", "LOW", "TRIVIAL")
    rank = {lbl: i for i, lbl in enumerate(reversed(ladder("importance")))}  # TRIVIAL=0..CRITICAL=4
    prev = -1
    for v in [x / 2 for x in range(0, 21)]:  # sweep 0.0..10.0
        r = rank[Memory(content="x", importance=v).importance_label()]
        assert r >= prev  # non-decreasing as importance rises
        prev = r
    assert Memory(content="x", importance=1.0).importance_label() == "TRIVIAL"
    assert Memory(content="x", importance=9.5).importance_label() == "CRITICAL"


def test_label_accessors_are_pure():
    m = Memory(content="x", importance=8.0)
    before = m.model_dump()
    assert m.importance_label() == "HIGH"
    assert m.model_dump() == before  # no mutation
