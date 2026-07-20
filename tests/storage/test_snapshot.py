# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for AgentSnapshot edge cases.

Covers:
- Line 101: ``_agentdoc_``-prefixed fields are skipped in from_agent()
- Line 105: callable attributes are skipped in from_agent() removed dynamic method restoration from AgentSnapshot — methods live on
the class and are never reattached via a snapshot. Tests for the old
``snap.methods`` / ``restore()`` method-warning paths have been removed.
"""

from nooa import Agent
from nooa.storage.snapshot import AgentSnapshot
from nooa.unifiedllm import FakeLLMClient


class _SimpleAgent(Agent, llm=FakeLLMClient()):
    value: int = 0


class TestAgentSnapshotFromAgent:
    """Tests for AgentSnapshot.from_agent() field filtering."""

    def test_agentdoc_prefix_fields_are_skipped(self):
        """from_agent() skips attributes whose name starts with '_agentdoc_' (line 101)."""
        agent = _SimpleAgent()
        agent.__dict__["_agentdoc_hidden"] = "should be excluded"
        agent.value = 5

        snap = AgentSnapshot.from_agent(agent)

        assert "_agentdoc_hidden" not in snap.attributes
        assert snap.attributes["value"] == 5

    def test_callable_attributes_are_skipped(self):
        """from_agent() skips callable attributes set on the instance (line 105).: direct ``agent.my_fn = lambda: None`` is now blocked by the
        Agent guard; route the callable straight into ``__dict__`` so we still
        exercise the from_agent() filter.
        """
        agent = _SimpleAgent()
        agent.__dict__["my_fn"] = lambda: None
        agent.value = 7

        snap = AgentSnapshot.from_agent(agent)

        assert "my_fn" not in snap.attributes
        assert snap.attributes["value"] == 7


class _UnsupportedThing:
    """A plain class with no serialization support."""

    pass


class TestFromAgentSkipsUnserializableAttributes:
    """A non-serializable agent attribute must be skipped (and warned), not abort the snapshot."""

    def test_unserializable_attribute_is_skipped_not_raised(self):
        """from_agent() skips an unserializable attribute and still captures the rest."""
        agent = _SimpleAgent()
        agent.bad_attr = _UnsupportedThing()  # type: ignore[attr-defined]
        agent.value = 11

        snap = AgentSnapshot.from_agent(agent)

        assert "bad_attr" not in snap.attributes
        assert snap.attributes["value"] == 11

    def test_unserializable_attribute_emits_warning(self, caplog):
        """Skipping an unserializable attribute logs a warning naming the attribute."""
        import logging

        agent = _SimpleAgent()
        agent.bad_attr = _UnsupportedThing()  # type: ignore[attr-defined]

        with caplog.at_level(logging.WARNING):
            AgentSnapshot.from_agent(agent)

        assert any("bad_attr" in rec.message for rec in caplog.records)


class TestSnapshotRoundTripWithUnserializableAttr:
    """An unserializable attribute must not prevent durable state (vars) from round-tripping."""

    def test_vars_survive_when_sibling_attr_is_unserializable(self):
        """Snapshot/restore preserves user vars even if another attr can't serialize."""
        from nooa.storage.json_snapshot import snapshot_from_dict, snapshot_to_dict

        agent = _SimpleAgent()
        agent.value = 7
        agent.__dict__["_live_only"] = _UnsupportedThing()

        data = snapshot_to_dict(AgentSnapshot.from_agent(agent))
        assert "_live_only" not in data["attributes"]
        assert data["attributes"]["value"] == 7

        restored = _SimpleAgent()
        snapshot_from_dict(data).restore(restored)
        assert restored.value == 7
