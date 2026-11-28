# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The v2 core's protocol helpers and the v1 shim's conversions."""

from typing import Any

import pytest
from acp.schema import AgentMessageChunk, ToolCallProgress, ToolCallStart
from nooa_atom.acp import v2
from nooa_atom.acp.v1_shim import V1Peer, config_option_to_v1, tool_content_to_v1

V2_INIT = {"protocolVersion": 2, "info": {"name": "roy", "version": "1"}}


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        (V2_INIT, 2),
        ({**V2_INIT, "protocolVersion": 3}, 2),
        # T3 Code asks for 2 in v1 shape and reads v1.
        ({"protocolVersion": 2, "clientCapabilities": {}, "clientInfo": {"name": "t3"}}, 1),
        ({**V2_INIT, "clientCapabilities": {}}, 1),
        ({"protocolVersion": 2}, 1),
        ({"protocolVersion": 1, "clientCapabilities": {}}, 1),
        ({"protocolVersion": 0}, 1),
        ({"protocolVersion": "2"}, None),
        ({"protocolVersion": True}, None),
        ({}, None),
    ],
)
def test_negotiation_speaks_v2_only_to_v2_shaped_requests(params, expected):
    assert v2.negotiate(params) == expected


def test_the_v2_answer_and_the_probe():
    answer = v2.initialize_result({"name": "roy", "version": "1"})
    assert answer["protocolVersion"] == 2
    assert answer["info"]["name"] == "nooa-atom"
    assert answer["capabilities"]["session"]["delete"] == {}
    assert v2.is_probe({"protocolVersion": 1, "_meta": {v2.META_KEY: {"probe": True}}})
    assert not v2.is_probe({"protocolVersion": 1, "_meta": {v2.META_KEY: {}}})
    probe = v2.probe_answer()
    assert probe["protocolVersions"] == [1, 2]
    assert probe["initialize"]["protocolVersion"] == 2


def test_a_long_message_is_chunks_of_one_message_id():
    chunks = v2.message("agent_message", "x" * (v2.TEXT_LIMIT + 5))
    assert [len(chunk["content"]["text"]) for chunk in chunks] == [v2.TEXT_LIMIT, 5]
    assert {chunk["sessionUpdate"] for chunk in chunks} == {"agent_message_chunk"}
    assert len({chunk["messageId"] for chunk in chunks}) == 1


def test_tool_calls_leave_out_unknown_fields():
    assert v2.tool_call("t1", title="Run", raw_input=None, status="in_progress") == {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "t1",
        "title": "Run",
        "status": "in_progress",
    }


@pytest.mark.parametrize(
    ("old", "new", "start_line"),
    [
        (None, "a\nb\n", None),
        ("a\nb\n", "a\nc\n", None),
        ("x\n", "y\nz\n", 10),
        ("", "new\n", None),
    ],
)
def test_a_git_patch_gives_back_its_texts(old, new, start_line):
    patch = v2.git_patch("/w/f.py", old, new, start_line=start_line)
    assert patch.startswith("diff --git")
    assert v2.patch_texts(patch) == (old or None if old is None else old, new)


def test_a_shifted_hunk_names_its_line():
    patch = v2.git_patch("/w/f.py", "x\n", "y\n", start_line=10)
    assert "@@ -10" in patch and "+10" in patch


def test_v1_gets_a_diff_from_a_git_patch():
    item = v2.diff("/w/f.py", "a\n", "b\n")
    converted = tool_content_to_v1(item)
    assert converted["type"] == "diff"
    assert (converted["path"], converted["oldText"], converted["newText"]) == (
        "/w/f.py",
        "a\n",
        "b\n",
    )


def test_a_v2_config_option_converts_to_v1():
    option = v2.select_option("model", "Model", "fast", [("fast", "Fast"), ("smart", "Smart")])
    converted = config_option_to_v1(option)
    assert converted.id == "model"
    assert converted.current_value == "fast"


class _Client:
    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))


async def test_v1_sees_tool_call_then_progress_and_no_v2_only_updates():
    client = _Client()
    peer = V1Peer(client)  # type: ignore[arg-type]
    await peer.update("s", v2.state_update("running"))
    await peer.update("s", v2.tool_call("t1", title="Run", status="in_progress"))
    await peer.update("s", v2.tool_call("t1", status="completed"))
    await peer.update("s", v2.tool_call("job:build", title="build", status="in_progress"))
    await peer.update("s", {"sessionUpdate": "replay_complete"})
    kinds = [type(update) for _, update in client.updates]
    assert kinds == [ToolCallStart, ToolCallProgress]


async def test_v1_shows_a_mirrored_child_card_in_the_parent_under_a_prefixed_id():
    client = _Client()
    peer = V1Peer(client)  # type: ignore[arg-type]
    await peer.update(
        "p", {"sessionUpdate": "subagent_update", "sessionId": "c", "title": "helper"}
    )
    await peer.update("c", v2.tool_call("t1", title="Run", status="in_progress"), mirrored=True)
    await peer.update("c", v2.message("agent_message", "hidden")[0], mirrored=True)
    # A child the client loaded has its own bridge: its updates are its own.
    await peer.update("c", v2.message("agent_message", "shown")[0])
    [(parent, card), (child, chunk)] = client.updates
    assert parent == "p" and card.tool_call_id == "c:t1"
    assert child == "c" and isinstance(chunk, AgentMessageChunk)
    assert chunk.content.text == "shown"
