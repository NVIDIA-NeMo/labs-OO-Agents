# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for agent snapshot serialization."""

import json
from typing import Annotated

import pytest
from pydantic import BaseModel

from nooa import Agent, Context
from nooa.context_blocks import ExpressionContextBlock, LiteralContextBlock
from nooa.errors.storage import SerializationError
from nooa.storage.json_snapshot import (
    snapshot_from_dict,
    snapshot_from_json,
    snapshot_to_dict,
    snapshot_to_json,
)
from nooa.storage.markers import nosnapshot
from nooa.storage.snapshot import SNAPSHOT_VERSION, AgentSnapshot
from nooa.unifiedllm import FakeLLMClient

fake_llm = FakeLLMClient()


class SimpleAgent(Agent, llm=fake_llm):
    pass


class SnapshotResult(BaseModel):
    """A typed result for integration testing typed attribute snapshots."""

    status: str
    score: int


@pytest.fixture
def agent():
    """Create a fresh SimpleAgent for each test."""
    return SimpleAgent()


class TestSnapshotRoundtrip:
    """Tests for snapshot_to_json and snapshot_from_json roundtripping."""

    def test_static_context_roundtrip(self, agent):
        """Static context blocks survive a snapshot roundtrip."""
        agent.context_manager["notes"] = "some notes"
        agent.context_manager["count"] = 42

        snap = snapshot_to_json(agent)
        agent2 = SimpleAgent()
        snapshot_from_json(snap, agent2)

        assert agent2.context_manager["notes"] == "some notes"
        assert agent2.context_manager["count"] == 42

    def test_dynamic_context_roundtrip(self, agent):
        """Dynamic context blocks are stored as DynamicContext markers, not resolved values."""
        agent.context_manager.set_dynamic("status", "self.__class__.__name__")

        snap = snapshot_to_json(agent)
        agent2 = SimpleAgent()
        snapshot_from_json(snap, agent2)

        raw = dict(agent2.context_manager._raw_items())
        assert isinstance(raw["status"], ExpressionContextBlock)
        assert raw["status"].expr == "self.__class__.__name__"

    def test_content_kind_and_placement_roundtrip_independently(self, agent):
        """All literal/expression and prefix/suffix combinations retain both axes."""
        agent.context_manager["prefix_literal"] = Context("stable", prefix=True)
        agent.context_manager["suffix_literal"] = Context("volatile")
        agent.context_manager["prefix_expression"] = Context(expr="'stable'", prefix=True)
        agent.context_manager["suffix_expression"] = Context(expr="'volatile'")

        snapshot = snapshot_to_json(agent)
        restored = SimpleAgent()
        snapshot_from_json(snapshot, restored)

        blocks = dict(restored.context_manager._raw_items())
        assert isinstance(blocks["prefix_literal"], LiteralContextBlock)
        assert blocks["prefix_literal"].prefix is True
        assert isinstance(blocks["suffix_literal"], LiteralContextBlock)
        assert blocks["suffix_literal"].prefix is False
        assert isinstance(blocks["prefix_expression"], ExpressionContextBlock)
        assert blocks["prefix_expression"].prefix is True
        assert isinstance(blocks["suffix_expression"], ExpressionContextBlock)
        assert blocks["suffix_expression"].prefix is False

    def test_expression_display_expr_roundtrip(self, agent):
        block = ExpressionContextBlock(
            key="status",
            expr="self.__class__.__name__",
            display_expr="self.context['status']",
            prefix=True,
        )
        agent.context_manager.restore_block(block)

        restored = SimpleAgent()
        snapshot_from_json(snapshot_to_json(agent), restored)

        restored_block = dict(restored.context_manager._raw_items())["status"]
        assert isinstance(restored_block, ExpressionContextBlock)
        assert restored_block.display_expr == "self.context['status']"
        assert restored_block.prefix is True

    def test_event_manager_state_not_in_snapshot(self, agent):
        """Snapshots no longer carry next_tag_num; allocation lives on the backend."""
        snap_dict = snapshot_to_dict(AgentSnapshot.from_agent(agent))
        assert "event_manager" not in snap_dict

    def test_user_attributes_roundtrip(self, agent):
        """Public, JSON-serializable user attributes survive a snapshot roundtrip."""
        agent.score = 99
        agent.name = "test"
        agent.data = [1, 2, 3]

        snap = snapshot_to_json(agent)
        agent2 = SimpleAgent()
        snapshot_from_json(snap, agent2)

        assert agent2.score == 99
        assert agent2.name == "test"
        assert agent2.data == [1, 2, 3]

    def test_nosnapshot_attrs_skipped(self):
        """Fields annotated with nosnapshot are excluded from the snapshot."""

        class AgentWithTransient(Agent, llm=fake_llm):
            cache: Annotated[dict, nosnapshot]

        agent = AgentWithTransient()
        agent.cache = {"big": "data"}
        agent.score = 42

        snap = snapshot_to_json(agent)
        assert "score" in snap["attributes"]
        assert "cache" not in snap["attributes"]

    def test_non_serializable_attr_is_skipped(self, agent):
        """Non-serializable user attributes are skipped, not fatal to the snapshot."""
        agent.bad = object()
        agent.score = 42

        snap = snapshot_to_json(agent)

        assert "bad" not in snap["attributes"]
        assert snap["attributes"]["score"] == 42

    def test_non_serializable_context_raises(self, agent):
        """Non-serializable static context values raise SerializationError."""
        agent.context_manager["bad"] = object()

        with pytest.raises(SerializationError, match="not serializable"):
            snapshot_to_json(agent)

    def test_version_mismatch_raises(self, agent):
        """Restoring a snapshot with a mismatched version raises SerializationError."""
        snap = snapshot_to_json(agent)
        snap["version"] = 999

        agent2 = SimpleAgent()
        with pytest.raises(SerializationError, match="version mismatch"):
            snapshot_from_json(snap, agent2)

    def test_snapshot_has_version(self, agent):
        """Snapshot dict includes the current SNAPSHOT_VERSION."""
        snap = snapshot_to_json(agent)
        assert snap["version"] == SNAPSHOT_VERSION

    def test_empty_agent_roundtrip(self, agent):
        """An agent with no user state should roundtrip cleanly."""
        snap = snapshot_to_json(agent)
        agent2 = SimpleAgent()
        snapshot_from_json(snap, agent2)

        assert snap["version"] == SNAPSHOT_VERSION
        assert snap["attributes"] == {}


class TestAgentSnapshot:
    """Tests for the AgentSnapshot intermediate representation."""

    def test_from_agent_produces_model(self, agent):
        """from_agent returns an AgentSnapshot with correct version and typed context."""
        agent.context_manager["key"] = "value"
        snap = AgentSnapshot.from_agent(agent)

        assert isinstance(snap, AgentSnapshot)
        assert snap.version == SNAPSHOT_VERSION
        assert len(snap.context) == 1
        assert isinstance(snap.context[0], LiteralContextBlock)
        assert snap.context[0].key == "key"
        assert snap.context[0].value == "value"

    def test_model_dump_roundtrip(self, agent):
        """AgentSnapshot survives a model_dump/model_validate roundtrip."""
        agent.context_manager["notes"] = "hello"
        agent.context_manager.set_dynamic("status", "self.__class__.__name__")
        agent.score = 42

        original = AgentSnapshot.from_agent(agent)
        data = snapshot_to_dict(original)
        restored = snapshot_from_dict(data)

        assert restored.version == original.version
        assert len(restored.context) == len(original.context)
        assert restored.attributes == {"score": 42}

    @pytest.mark.parametrize(
        ("legacy_type", "payload_field", "payload_value", "expected_type"),
        [
            ("static", "value", {"nested": [1, 2]}, LiteralContextBlock),
            ("dynamic", "expr", "'value'", ExpressionContextBlock),
        ],
    )
    def test_v2_snapshot_migrates_old_tags(
        self, legacy_type, payload_field, payload_value, expected_type
    ):
        legacy = {
            "version": 2,
            "context": [{"key": "legacy", "type": legacy_type, payload_field: payload_value}],
        }

        migrated = snapshot_from_dict(legacy)

        assert migrated.version == SNAPSHOT_VERSION
        assert isinstance(migrated.context[0], expected_type)
        assert migrated.context[0].prefix is False

    def test_v2_snapshot_preserves_compatibility_fix_placement(self):
        legacy = {
            "version": 2,
            "context": [
                {
                    "key": "python_cell_tools",
                    "type": "dynamic",
                    "expr": "'tools'",
                    "prefix": True,
                }
            ],
        }

        migrated = snapshot_from_dict(legacy)

        block = migrated.context[0]
        assert isinstance(block, ExpressionContextBlock)
        assert block.prefix is True
        assert block.display_expr is None

    def test_restore_via_model(self, agent):
        """AgentSnapshot.restore mutates agent correctly."""
        agent.context_manager["key"] = "value"

        snap = AgentSnapshot.from_agent(agent)
        agent2 = SimpleAgent()
        snap.restore(agent2)

        assert agent2.context_manager["key"] == "value"

    def test_restore_version_mismatch(self):
        """restore raises if snapshot version was tampered with."""
        snap = AgentSnapshot(version=999)
        agent = SimpleAgent()
        with pytest.raises(SerializationError, match="version mismatch"):
            snap.restore(agent)

    def test_restore_is_additive(self, agent):
        """Restoring onto an agent with existing state doesn't clear pre-existing entries."""
        agent.context_manager["existing"] = "stays"
        agent.existing_attr = "also stays"

        # Snapshot a different agent with different state
        other = SimpleAgent()
        other.context_manager["new"] = "added"
        other.new_attr = "also added"
        snap = AgentSnapshot.from_agent(other)

        snap.restore(agent)

        # Snapshot state was applied
        assert agent.context_manager["new"] == "added"
        assert agent.new_attr == "also added"
        # Pre-existing state survives (additive, not replacement)
        assert agent.context_manager["existing"] == "stays"
        assert agent.existing_attr == "also stays"

    def test_sequential_snapshots_are_independent(self, agent):
        """Multiple snapshots capture state at their point in time, independently."""
        agent.score = 1
        snap1 = snapshot_to_json(agent)

        agent.score = 2
        agent.context_manager["added_later"] = "yes"
        snap2 = snapshot_to_json(agent)

        # snap1 should reflect original state
        assert snap1["attributes"] == {"score": 1}
        assert len(snap1["context"]) == 0

        # snap2 should reflect modified state
        assert snap2["attributes"] == {"score": 2}
        assert len(snap2["context"]) == 1

        # Restoring snap1 gives original state
        agent3 = SimpleAgent()
        snapshot_from_json(snap1, agent3)
        assert agent3.score == 1
        assert "added_later" not in agent3.context_manager


class TestSnapshotWithTypedAttributes:
    """Integration tests for typed attributes through the full snapshot pipeline."""

    def test_snapshot_with_pydantic_attribute_json_roundtrip(self):
        """Typed attributes survive snapshot -> JSON string -> snapshot -> restore.

        Exercises the full pipeline including type_allowlist persistence.
        """
        agent = SimpleAgent()
        agent.result = SnapshotResult(status="done", score=99)
        agent.context_manager["summary"] = "all good"

        # 1. Agent -> snapshot dict (exercises serialize + model_dump)
        snap_dict = snapshot_to_json(agent)

        # 2. Snapshot dict -> JSON string -> back to dict
        #    (proves the blob is truly JSON-serializable)
        json_str = json.dumps(snap_dict)
        restored_dict = json.loads(json_str)

        # 3. Verify type_allowlist survived JSON roundtrip
        assert len(restored_dict["type_allowlist"]) > 0
        result_fqn = f"{SnapshotResult.__module__}.SnapshotResult"
        assert result_fqn in restored_dict["type_allowlist"]

        # 4. Dict -> agent restore
        agent2 = SimpleAgent()
        snapshot_from_json(restored_dict, agent2)

        assert isinstance(agent2.result, SnapshotResult)
        assert agent2.result.status == "done"
        assert agent2.result.score == 99
        assert agent2.context_manager["summary"] == "all good"
