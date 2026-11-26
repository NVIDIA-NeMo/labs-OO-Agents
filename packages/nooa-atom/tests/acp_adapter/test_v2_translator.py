# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The ACP v2 translator on its own: v2 client frames in, v1 agent frames out, and back."""

from typing import Any

import pytest
from nooa_atom.acp.v2 import (
    META_KEY,
    V1,
    V2,
    V2Translator,
    git_patch,
    initialize_result_to_v2,
    negotiate,
    tool_call_to_v2,
)


class Wire:
    def __init__(self) -> None:
        self.client: list[dict[str, Any]] = []
        self.agent: list[dict[str, Any]] = []
        self.t = V2Translator(self.client.append, self.agent.append)

    def updates(self, session_id: str | None = None) -> list[dict[str, Any]]:
        return [
            m["params"]["update"]
            for m in self.client
            if m.get("method") == "session/update"
            and (session_id is None or m["params"]["sessionId"] == session_id)
        ]

    def states(self, session_id: str) -> list[tuple[str, str | None]]:
        return [
            (u["state"], u.get("stopReason"))
            for u in self.updates(session_id)
            if u["sessionUpdate"] == "state_update"
        ]

    def agent_update(self, session_id: str, update: dict[str, Any]) -> None:
        self.t.from_agent(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {"sessionId": session_id, "update": update},
            }
        )

    def hint(self, session_id: str, hint: dict[str, Any]) -> None:
        self.agent_update(
            session_id, {"sessionUpdate": "session_info_update", "_meta": {META_KEY: hint}}
        )

    def prompt(self, request_id: int, session_id: str = "s", **params: Any) -> str:
        self.t.from_client(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "session/prompt",
                "params": {
                    "sessionId": session_id,
                    "prompt": [{"type": "text", "text": "hi"}],
                    **params,
                },
            }
        )
        forwarded = self.agent[-1]
        assert forwarded["method"] == "session/prompt"
        return forwarded["id"]

    def ack(self, internal: str, message_id: str) -> None:
        self.t.from_agent(
            {
                "jsonrpc": "2.0",
                "method": "_nooa/v2/accepted",
                "params": {"ack": internal, "messageId": message_id},
            }
        )


@pytest.mark.parametrize(
    ("requested", "version"), [(1, V1), (0, V1), (2, V2), (7, V2), ("2", None), (None, None)]
)
def test_negotiation(requested, version):
    assert negotiate({"protocolVersion": requested}) == version


def test_initialize_result_shape():
    result = initialize_result_to_v2(
        {
            "protocolVersion": 1,
            "agentCapabilities": {
                "loadSession": True,
                "mcpCapabilities": {"http": True, "sse": True},
                "sessionCapabilities": {"list": {}, "close": {}},
                "_meta": {"dev.nooa/inject": {}},
            },
            "authMethods": [],
            "agentInfo": {"name": "nooa-atom", "version": "1.0"},
        }
    )
    assert result == {
        "protocolVersion": 2,
        "info": {"name": "nooa-atom", "version": "1.0"},
        "capabilities": {
            "session": {"mcp": {"stdio": {}, "http": {}}, "delete": {}},
            "_meta": {"dev.nooa/inject": {}},
        },
    }


def test_prompt_lifecycle_ack_then_idle():
    wire = Wire()
    internal = wire.prompt(1, idempotencyKey="k")
    assert wire.agent[-1]["params"]["_meta"][META_KEY] == {"ack": internal}
    wire.ack(internal, "item-1")
    assert wire.client[0] == {"jsonrpc": "2.0", "id": 1, "result": {"messageId": "item-1"}}
    wire.hint("s", {"turn": "started"})
    wire.agent_update(
        "s", {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "a"}}
    )
    wire.agent_update(
        "s", {"sessionUpdate": "agent_message_chunk", "content": {"type": "text", "text": "b"}}
    )
    wire.hint("s", {"turn": "ended", "outcome": "done"})
    assert wire.states("s") == [("running", None)]  # the prompt's answer carries the stop reason
    wire.t.from_agent({"jsonrpc": "2.0", "id": internal, "result": {"stopReason": "max_tokens"}})
    assert wire.states("s") == [("running", None), ("idle", "max_tokens")]
    chunks = [u for u in wire.updates("s") if u["sessionUpdate"] == "agent_message_chunk"]
    assert len({chunk["messageId"] for chunk in chunks}) == 1
    # No hint reaches the client, and the prompt's answer is not forwarded twice.
    assert all(u["sessionUpdate"] != "session_info_update" for u in wire.updates())
    assert [m for m in wire.client if m.get("id") == 1] == [wire.client[0]]
    # A retry with the same key gets the same message, without a new prompt.
    count = len(wire.agent)
    wire.t.from_client(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "session/prompt",
            "params": {
                "sessionId": "s",
                "prompt": [{"type": "text", "text": "hi"}],
                "idempotencyKey": "k",
            },
        }
    )
    assert len(wire.agent) == count
    assert wire.client[-1] == {"jsonrpc": "2.0", "id": 2, "result": {"messageId": "item-1"}}


def test_errors_before_and_after_insertion():
    wire = Wire()
    internal = wire.prompt(1)
    error = {"code": -32602, "message": "Invalid params"}
    wire.t.from_agent({"jsonrpc": "2.0", "id": internal, "error": error})
    assert wire.client == [{"jsonrpc": "2.0", "id": 1, "error": error}]

    internal = wire.prompt(2)
    wire.ack(internal, "m2")
    wire.hint("s", {"turn": "started"})
    failure = {"code": -32603, "message": "boom"}
    wire.t.from_agent({"jsonrpc": "2.0", "id": internal, "error": failure})
    assert wire.updates("s")[-1] == {
        "sessionUpdate": "state_update",
        "state": "idle",
        "stopReason": "error",
        "error": failure,
    }


def test_local_command_gets_an_id_and_a_running_idle_pair():
    wire = Wire()
    internal = wire.prompt(1)
    wire.t.from_agent({"jsonrpc": "2.0", "id": internal, "result": {"stopReason": "end_turn"}})
    assert "messageId" in wire.client[0]["result"]
    assert wire.states("s") == [("running", None), ("idle", "end_turn")]


def test_cancel_request_for_an_inserted_prompt_is_dropped():
    wire = Wire()
    internal = wire.prompt(1)
    wire.t.from_client({"jsonrpc": "2.0", "method": "$/cancel_request", "params": {"requestId": 1}})
    assert wire.agent[-1]["params"] == {"requestId": internal}
    wire.ack(internal, "m")
    count = len(wire.agent)
    wire.t.from_client({"jsonrpc": "2.0", "method": "$/cancel_request", "params": {"requestId": 1}})
    assert len(wire.agent) == count


def test_close_ends_foreground_work_first():
    wire = Wire()
    internal = wire.prompt(1)
    wire.ack(internal, "m")
    wire.hint("s", {"turn": "started"})
    wire.t.from_client(
        {"jsonrpc": "2.0", "id": 9, "method": "session/close", "params": {"sessionId": "s"}}
    )
    wire.t.from_agent({"jsonrpc": "2.0", "id": 9, "result": {}})
    assert wire.client[-2]["params"]["update"] == {
        "sessionUpdate": "state_update",
        "state": "idle",
        "stopReason": "cancelled",
    }
    assert wire.client[-1] == {"jsonrpc": "2.0", "id": 9, "result": {}}
    # The v1 prompt's late answer is not reported again.
    wire.t.from_agent({"jsonrpc": "2.0", "id": internal, "result": {"stopReason": "cancelled"}})
    assert wire.client[-1] == {"jsonrpc": "2.0", "id": 9, "result": {}}


def test_permission_and_elicitation_require_action():
    wire = Wire()
    internal = wire.prompt(1)
    wire.ack(internal, "m")
    wire.hint("s", {"turn": "started"})
    wire.t.from_agent(
        {
            "jsonrpc": "2.0",
            "id": 1 << 32,
            "method": "session/request_permission",
            "params": {
                "sessionId": "s",
                "toolCall": {
                    "toolCallId": "t",
                    "title": "Run it",
                    "kind": "execute",
                    "status": None,
                },
                "options": [{"optionId": "a", "name": "Allow", "kind": "allow_once"}],
            },
        }
    )
    request = wire.client[-1]
    assert request["params"]["title"] == "Run it"
    assert request["params"]["subject"] == {
        "type": "tool_call",
        "toolCall": {"toolCallId": "t", "title": "Run it", "kind": "execute"},
    }
    assert wire.states("s")[-1] == ("requires_action", None)
    wire.t.from_client(
        {"jsonrpc": "2.0", "id": 1 << 32, "result": {"outcome": {"outcome": "cancelled"}}}
    )
    assert wire.states("s")[-1] == ("running", None)
    wire.t.from_agent(
        {
            "jsonrpc": "2.0",
            "id": (1 << 32) + 1,
            "method": "elicitation/create",
            "params": {
                "sessionId": "s",
                "mode": "form",
                "message": "Pick",
                "requestedSchema": {
                    "type": "object",
                    "properties": {"v": {"type": "string", "pattern": "^a$"}},
                },
            },
        }
    )
    assert wire.client[-1]["params"]["requestedSchema"]["properties"]["v"] == {"type": "string"}
    assert wire.states("s")[-1] == ("requires_action", None)


def test_updates_take_their_v2_shapes():
    wire = Wire()
    wire.t.from_client(
        {"jsonrpc": "2.0", "id": 1, "method": "session/new", "params": {"cwd": "/w"}}
    )
    assert wire.agent[-1]["params"] == {"cwd": "/w", "mcpServers": []}
    wire.t.from_agent(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "sessionId": "s",
                "modes": {"currentModeId": "auto", "availableModes": []},
                "configOptions": [
                    {
                        "id": "model",
                        "name": "Model",
                        "type": "select",
                        "currentValue": "a",
                        "options": [
                            {"group": "g", "name": "G", "options": [{"value": "a", "name": "a"}]}
                        ],
                    }
                ],
                "_meta": {
                    META_KEY: {
                        "availableCommands": [
                            {"name": "x", "description": "d", "input": {"hint": "h"}}
                        ]
                    }
                },
            },
        }
    )
    result = wire.client[-1]["result"]
    assert result == {
        "sessionId": "s",
        "configOptions": [
            {
                "configId": "model",
                "name": "Model",
                "type": "select",
                "currentValue": "a",
                "options": [
                    {"groupId": "g", "name": "G", "options": [{"value": "a", "name": "a"}]}
                ],
            }
        ],
        "availableCommands": [
            {
                "type": "command",
                "name": "x",
                "description": "d",
                "input": {"type": "text", "hint": "h"},
            }
        ],
    }
    wire.agent_update(
        "s",
        {
            "sessionUpdate": "plan",
            "entries": [{"content": "c", "priority": "high", "status": "pending"}],
        },
    )
    wire.agent_update("s", {"sessionUpdate": "current_mode_update", "currentModeId": "auto"})
    wire.agent_update(
        "s",
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "t",
            "title": "Edited a",
            "kind": "edit",
            "status": "completed",
            "locations": [{"path": "a", "line": 0}],
            "content": [{"type": "diff", "path": "a", "oldText": "x\n", "newText": "y\n"}],
        },
    )
    plan, tool = wire.updates("s")
    assert plan == {
        "sessionUpdate": "plan_update",
        "plan": {
            "type": "items",
            "planId": "plan",
            "entries": [{"content": "c", "priority": "high", "status": "pending"}],
        },
    }
    assert tool["sessionUpdate"] == "tool_call_update"
    assert tool["locations"] == [{"path": "/w/a", "line": 1}]
    [diff] = tool["content"]
    assert diff["changes"] == [{"operation": "modify", "path": "/w/a", "fileType": "text"}]
    assert diff["patch"]["text"] == git_patch("/w/a", "x\n", "y\n")


def test_git_patch():
    assert git_patch("/w/a", None, "one") == (
        "diff --git /w/a /w/a\nnew file mode 100644\n--- /dev/null\n+++ /w/a\n"
        "@@ -0,0 +1 @@\n+one\n\\ No newline at end of file\n"
    )
    assert tool_call_to_v2({"toolCallId": "t", "title": None}, None) == {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "t",
    }


def test_subagent_announcement_state_and_rehoming():
    wire = Wire()
    wire.hint("p", {"child": {"sessionId": "c", "parentId": "p", "title": "helper", "cwd": "/w"}})
    assert wire.updates("p") == [
        {
            "sessionUpdate": "subagent_update",
            "sessionId": "c",
            "capabilities": {"cancel": {}},
            "title": "helper",
            "cwd": "/w",
        }
    ]
    wire.hint("p", {"childMessage": {"sessionId": "c", "messageId": "i1", "text": "Do it"}})
    wire.hint("p", {"childTurn": {"sessionId": "c", "state": "running"}})
    wire.agent_update(
        "p",
        {
            "sessionUpdate": "tool_call",
            "toolCallId": "c:t1",
            "title": "Ran",
            "status": "in_progress",
        },
    )
    wire.agent_update(
        "p",
        {
            "sessionUpdate": "agent_message_chunk",
            "content": {"type": "text", "text": "x"},
            "_meta": {META_KEY: {"session": "c"}},
        },
    )
    wire.hint("p", {"childTurn": {"sessionId": "c", "state": "idle", "outcome": "done"}})
    child = wire.updates("c")
    assert [u["sessionUpdate"] for u in child] == [
        "session_message",
        "state_update",
        "tool_call_update",
        "agent_message_chunk",
        "state_update",
    ]
    assert child[0]["senderSessionId"] == "p" and child[0]["content"][0]["text"] == "Do it"
    assert child[2]["toolCallId"] == "t1"
    assert "_meta" not in child[3]
    parent = wire.updates("p")
    assert {"sessionUpdate": "subagent_update", "sessionId": "c", "description": "Do it"} in parent
    assert parent[-1] == {
        "sessionUpdate": "subagent_update",
        "sessionId": "c",
        "state": {"state": "idle", "stopReason": "end_turn"},
    }
    # The client may not prompt or close a child.
    wire.t.from_client(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "session/prompt",
            "params": {"sessionId": "c", "prompt": [{"type": "text", "text": "x"}]},
        }
    )
    assert wire.client[-1]["error"]["code"] == -32602
    wire.t.from_client({"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": "c"}})
    assert wire.agent[-1]["method"] == "session/cancel"


def test_waiting_turn_reports_background_work():
    wire = Wire()
    internal = wire.prompt(1)
    wire.ack(internal, "m")
    wire.hint("s", {"turn": "started"})
    wire.hint("s", {"turn": "ended", "outcome": "waiting", "waitingOn": ["build"]})
    updates = wire.updates("s")
    assert {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "job:build",
        "title": "Waiting on build",
        "kind": "other",
        "status": "in_progress",
    } in updates
    assert updates[-1] == {
        "sessionUpdate": "state_update",
        "state": "idle",
        "stopReason": "end_turn",
        "background": [{"type": "tool_call", "toolCallId": "job:build"}],
    }
    wire.hint("s", {"turn": "started"})
    assert wire.updates("s")[-1]["background"] == [{"type": "tool_call", "toolCallId": "job:build"}]
    wire.hint("s", {"turn": "ended", "outcome": "done"})
    assert {
        "sessionUpdate": "tool_call_update",
        "toolCallId": "job:build",
        "status": "completed",
    } in wire.updates("s")
    wire.t.from_agent({"jsonrpc": "2.0", "id": internal, "result": {"stopReason": "end_turn"}})
    assert wire.updates("s")[-1] == {
        "sessionUpdate": "state_update",
        "state": "idle",
        "stopReason": "end_turn",
        "background": [],
    }


def test_resume_maps_to_load_and_synchronizes_state():
    wire = Wire()
    wire.t.from_client(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "session/resume",
            "params": {"sessionId": "s", "cwd": "/w", "replayFrom": {"type": "start"}},
        }
    )
    assert wire.agent[-1] == {
        "jsonrpc": "2.0",
        "id": 3,
        "method": "session/load",
        "params": {"sessionId": "s", "cwd": "/w", "mcpServers": []},
    }
    wire.agent_update(
        "s",
        {
            "sessionUpdate": "user_message_chunk",
            "messageId": "i1",
            "content": {"type": "text", "text": "hi\n"},
        },
    )
    wire.t.from_agent({"jsonrpc": "2.0", "id": 3, "result": {"modes": {}, "configOptions": []}})
    kinds = [u["sessionUpdate"] for u in wire.updates("s")]
    assert kinds == ["user_message", "user_message_chunk", "replay_complete", "state_update"]
    assert wire.client[-1] == {"jsonrpc": "2.0", "id": 3, "result": {"configOptions": []}}
    wire.t.from_client(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "session/resume",
            "params": {"sessionId": "s", "cwd": "/w", "replayFrom": {"type": "later"}},
        }
    )
    assert wire.client[-1]["error"]["code"] == -32602
