# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for ``_build_journal_payload``.

The skeleton + content-addressed blocks the journal streams to the viewer
have to mirror what actually went on the wire to the LLM. Two concrete
invariants the viewer relies on:

1. Roles the provider formatter strips (``RUNTIME_EVENT``, ``METADATA``)
   never go to the LLM, so they must not appear in the skeleton either —
   otherwise the viewer's reconstructed message list contains rows the
   LLM never saw.

2. Assistant tool calls must be serialized as ``tool_calls`` (a
   single-entry list in OpenAI wire shape: ``[{"id", "type", "function":
   {"name", "arguments"}}]``). The viewer's ``_flatten_msg_to_attrs``
   reads that key to emit ``llm.input_messages.N.message.tool_calls.0.*``
   attributes; an internal-shape ``tool_call`` dict is silently dropped.
"""

from __future__ import annotations

import json

from nooa.context_blocks.models import RenderedMessage, Role, ToolCallInfo
from nooa.tracing._journal_builder import (
    build_journal_payload as _build_journal_payload,
)


def _text_msg(role: Role, content: str) -> RenderedMessage:
    return RenderedMessage(role=role, content=content)


def _assistant_tool_call(call_id: str, name: str, arguments: dict) -> RenderedMessage:
    return RenderedMessage(
        role=Role.ASSISTANT,
        tool_call=ToolCallInfo(id=call_id, name=name, arguments=arguments),
    )


class TestRoleFiltering:
    def test_metadata_and_runtime_event_are_stripped(self):
        messages = [
            _text_msg(Role.SYSTEM, "system"),
            _text_msg(Role.METADATA, "should not leak"),
            _text_msg(Role.USER, "hi"),
            _text_msg(Role.RUNTIME_EVENT, "should not leak"),
            _text_msg(Role.ASSISTANT, "bye"),
        ]
        payload = _build_journal_payload(messages)
        assert payload is not None
        roles = [m["role"] for m in payload.skeleton]
        assert roles == ["system", "user", "assistant"]
        # And the stripped content isn't hiding inside any surviving entry
        # or its content-addressed blocks.
        for m in payload.skeleton:
            assert "should not leak" not in (m.get("content") or "")
        for content in payload.blocks.values():
            assert "should not leak" not in content

    def test_all_non_stripped_roles_pass_through(self):
        messages = [
            _text_msg(Role.SYSTEM, "s"),
            _text_msg(Role.USER, "u"),
            _text_msg(Role.ASSISTANT, "a"),
            RenderedMessage(role=Role.TOOL, content="t", tool_call_id="tc_1"),
        ]
        payload = _build_journal_payload(messages)
        assert payload is not None
        assert [m["role"] for m in payload.skeleton] == ["system", "user", "assistant", "tool"]


class TestAssistantToolCallWireShape:
    def test_tool_call_arguments_hashed_into_blocks(self):
        """The assistant tool_call's ``arguments`` JSON — the dominant
        bulk in a CodeAct loop's per-call journal payload — is now
        content-addressed into the blocks dict. The skeleton entry
        carries ``arguments_hash`` instead of ``arguments`` so long
        conversation histories don't re-ship each prior cell's code
        body inline on every new LLM call.
        """
        args_dict = {"code": "print('hi')"}
        msg = _assistant_tool_call("call_abc", "execute_python", args_dict)
        payload = _build_journal_payload([msg])
        assert payload is not None
        (entry,) = payload.skeleton
        assert entry["role"] == "assistant"
        # The singular ``tool_call`` field (our internal shape) must NOT
        # appear — the viewer ignores it.
        assert "tool_call" not in entry

        (tc,) = entry["tool_calls"]
        assert tc["id"] == "call_abc"
        assert tc["type"] == "function"
        assert tc["function"]["name"] == "execute_python"
        # ``arguments`` is now a hash ref; actual bytes live in blocks.
        assert "arguments" not in tc["function"]
        ah = tc["function"]["arguments_hash"]
        assert payload.blocks[ah] == json.dumps(args_dict)

    def test_flattener_recovers_tool_call_arguments_via_reconstruction(self):
        """Round-trip: the stored skeleton plus blocks rebuild the
        original message shape (including ``arguments``) when read
        back via ``_resolve_message``, so the viewer's flattener
        emits ``llm.input_messages.N.message.tool_calls.0.*`` attrs
        with the real code body — not an empty string.
        """
        from nooa.viewer.otlp_store import (
            _flatten_msg_to_attrs,
            _resolve_message,
        )

        msg = _assistant_tool_call("call_xyz", "search", {"q": "bedrock"})
        payload = _build_journal_payload([msg])
        assert payload is not None
        entry = payload.skeleton[0]
        rebuilt = _resolve_message(entry, payload.blocks)

        attrs = _flatten_msg_to_attrs(rebuilt, "llm.input_messages.0.message")
        by_key = {a["key"]: a["value"]["stringValue"] for a in attrs}

        prefix = "llm.input_messages.0.message.tool_calls.0.tool_call"
        assert by_key[f"{prefix}.id"] == "call_xyz"
        assert by_key[f"{prefix}.function.name"] == "search"
        assert by_key[f"{prefix}.function.arguments"] == json.dumps({"q": "bedrock"})

    def test_identical_tool_call_arguments_across_messages_dedup(self):
        """Two assistant messages calling the same function with the
        same arguments produce a single blocks entry for ``arguments``.
        This is where the big win lives: a multi-turn CodeAct session
        re-references the same hash for every prior cell's code body
        instead of re-shipping it.
        """
        a = _assistant_tool_call("call_1", "execute_python", {"code": "x = 1"})
        b = _assistant_tool_call("call_2", "execute_python", {"code": "x = 1"})
        payload = _build_journal_payload([a, b])
        assert payload is not None
        hashes = [m["tool_calls"][0]["function"]["arguments_hash"] for m in payload.skeleton]
        assert hashes[0] == hashes[1]
        # One ``arguments`` block (its bytes), not two.
        arg_blocks = [
            h for h in payload.blocks if payload.blocks[h] == json.dumps({"code": "x = 1"})
        ]
        assert len(arg_blocks) == 1

    def test_tool_result_message_carries_tool_call_id(self):
        """Tool-role messages keep ``tool_call_id`` (singular). Their content
        now lives in the blocks dict behind a single-part block-ref, so the
        viewer's ``_resolve_message`` reconstructs the original
        ``content`` verbatim on read."""
        from nooa.viewer.otlp_store import _resolve_message

        msg = RenderedMessage(role=Role.TOOL, content="status: complete", tool_call_id="call_abc")
        payload = _build_journal_payload([msg])
        assert payload is not None
        (entry,) = payload.skeleton
        assert entry["role"] == "tool"
        assert entry["tool_call_id"] == "call_abc"
        # Content is in the blocks dict, not inline.
        assert "content" not in entry
        (part,) = entry["parts"]
        assert payload.blocks[part["block_hash"]] == "status: complete"

        # Round-trip via the viewer reconstruction path: tool_call_id
        # survives and content comes back verbatim.
        rendered = _resolve_message(entry, payload.blocks)
        assert rendered["role"] == "tool"
        assert rendered["tool_call_id"] == "call_abc"
        assert rendered["content"] == "status: complete"


class TestPartsAndBlocks:
    def test_plain_content_message_hashes_into_blocks(self):
        """Regression: plain-content messages (user / assistant / tool)
        used to embed ``content`` inline in the skeleton, so every LLM
        call's journal record re-shipped the entire N-message history
        verbatim. Now they go through the same block dedup path as
        system-prompt blocks, so the per-call skeleton is ``O(1)`` in
        the repeated-turn case and the /v1/journal/calls payloads stop
        growing with the conversation.
        """
        from nooa.viewer.otlp_store import _resolve_message

        payload = _build_journal_payload([_text_msg(Role.USER, "hello")])
        assert payload is not None
        (entry,) = payload.skeleton

        # Content no longer inline on the skeleton.
        assert "content" not in entry
        assert entry["role"] == "user"
        (part,) = entry["parts"]
        h = part["block_hash"]
        assert payload.blocks[h] == "hello"

        # Round-trips through the viewer's reconstruction untouched.
        rendered = _resolve_message(entry, payload.blocks)
        assert rendered == {"role": "user", "content": "hello"}

    def test_images_hashed_into_blocks_and_round_trip(self):
        """Multimodal images — LiteLLM-shape dicts wrapping multi-MB
        base64 strings — get content-addressed into the blocks dict
        so they ship at most once per session. The viewer's
        ``_resolve_message`` restores the original ``images`` list
        (including dict shape) byte-for-byte.
        """
        from nooa.viewer.otlp_store import _resolve_message

        def _image(tag: str, body_char: str) -> dict:
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{tag}{body_char * 4096}"},
            }

        shared = _image("AAAA", "a")
        unique_a = _image("BBBB", "b")
        unique_b = _image("CCCC", "c")
        msg_a = RenderedMessage(role=Role.USER, content="look at these", images=[shared, unique_a])
        msg_b = RenderedMessage(role=Role.USER, content="and these", images=[shared, unique_b])

        payload = _build_journal_payload([msg_a, msg_b])
        assert payload is not None
        (entry_a, entry_b) = payload.skeleton

        # Images live under ``image_hashes`` (plural), not inline ``images``.
        assert "images" not in entry_a
        assert "images" not in entry_b
        assert entry_a["image_hashes"][0] == entry_b["image_hashes"][0]  # shared
        assert entry_a["image_hashes"][1] != entry_b["image_hashes"][1]  # unique

        # Block dedup: 3 distinct image bodies, not 4.
        all_image_hashes = set(entry_a["image_hashes"]) | set(entry_b["image_hashes"])
        assert len(all_image_hashes) == 3

        # Round-trip via the viewer path rebuilds the original dict shape.
        rebuilt_a = _resolve_message(entry_a, payload.blocks)
        rebuilt_b = _resolve_message(entry_b, payload.blocks)
        assert rebuilt_a["images"] == [shared, unique_a]
        assert rebuilt_b["images"] == [shared, unique_b]
        assert "image_hashes" not in rebuilt_a
        assert "image_hashes" not in rebuilt_b

    def test_identical_content_across_messages_deduped_in_blocks(self):
        """Two messages with the same body produce one blocks entry.

        This is the mechanism that collapses an N-turn conversation's
        per-call journal payload from ``O(total bytes)`` to ``O(delta
        bytes)`` — ``MessageJournalCallback._send_new_blocks`` only ships
        block hashes it hasn't already sent for this session, so the
        Nth turn's /v1/journal/calls record is small even though the
        prompt passed to the LLM contained all N-1 prior messages.
        """
        messages = [
            _text_msg(Role.USER, "repeated"),
            _text_msg(Role.ASSISTANT, "one-off"),
            _text_msg(Role.USER, "repeated"),  # same bytes, same hash
        ]
        payload = _build_journal_payload(messages)
        assert payload is not None
        # Two distinct block contents total, not three.
        assert len(payload.blocks) == 2
        # Both "repeated" messages point at the same block.
        hashes = [m["parts"][0]["block_hash"] for m in payload.skeleton]
        assert hashes[0] == hashes[2] != hashes[1]

    def test_parts_are_hashed_into_blocks(self):
        """Block-aware formatters emit ``parts``; the payload replaces block
        content with ``{block_hash, key}`` refs and accumulates the blocks
        dict for dedup across turns."""
        from nooa.context_blocks.models import BlockPart, TextPart

        msg = RenderedMessage(
            role=Role.SYSTEM,
            content="<a>A</a>\n\n<b>B</b>",
            parts=[
                BlockPart(key="a", content="<a>A</a>"),
                TextPart(text="\n\n"),
                BlockPart(key="b", content="<b>B</b>"),
            ],
        )
        payload = _build_journal_payload([msg])
        assert payload is not None
        (entry,) = payload.skeleton
        assert entry["role"] == "system"
        assert "content" not in entry  # replaced by parts
        assert len(entry["parts"]) == 3

        block_refs = [p for p in entry["parts"] if "block_hash" in p]
        assert {p["key"] for p in block_refs} == {"a", "b"}
        # Hashes in the skeleton correspond to blocks in the payload.
        for ref in block_refs:
            assert ref["block_hash"] in payload.blocks
