# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire-protocol invariants the journal sideband must hold.

Scope:
- Output-block sideband: large assistant replies (with multi-KB
  ``tool_call.arguments``) must round-trip through ``/v1/journal/blocks``
  and resolve back on the read side; the call record itself stays small.
- Per-session, per-destination dedup: across N back-to-back calls
  carrying overlapping content, only the *new* blocks are reposted.
- ContextVar isolation: concurrent ``asyncio.gather`` LLM calls each
  see their own journal payload and end up with distinct call records.
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse

import pytest

from nooa.tracing._context_sideband import (
    JournalPayload,
    set_journal_payload,
)
from nooa.tracing._litellm_journal import MessageJournalCallback


def _capture_posts():
    """Return ``(posts_list, fake_post_fn)`` for use with ``patch``ing
    ``_post_json``.  ``posts_list`` is appended on each call."""
    posts: list[tuple[str, object, str]] = []

    def fake_post(url, payload, *, session_id="", timeout=15.0, on_success=None):
        posts.append((url, payload, session_id))
        if on_success is not None:
            on_success()

    return posts, fake_post


class _MsgDict(dict):
    """Dict-like response message that ``_extract_output_msgs`` accepts.

    ``_msg_to_dict`` returns dicts as-is, which is what we want here --
    a SimpleNamespace falls through to the ``str(msg)`` fallback and
    the test no longer exercises the real shape.
    """


def _fake_response(content: str = "out") -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=_MsgDict(role="assistant", content=content, tool_calls=None))
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
    )


def _fake_response_with_tool(arguments: str) -> SimpleNamespace:
    """Response with a tool_call carrying a (potentially large) arguments string."""
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=_MsgDict(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        {
                            "id": "tc1",
                            "type": "function",
                            "function": {
                                "name": "execute_python",
                                "arguments": arguments,
                            },
                        }
                    ],
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=10),
    )


# ---------------------------------------------------------------------------
# Output-block sideband
# ---------------------------------------------------------------------------


def test_large_tool_call_arguments_routed_through_blocks():
    """Big assistant tool_call arguments must travel via
    ``/v1/journal/blocks``, not be embedded inline in the call record.

    Reason: the call record is one row in the receiver's ``llm_calls``
    table.  Embedding multi-MB tool_call argument bodies inline blows
    out per-row size, defeats the per-session dedup, and causes the
    same body to be re-shipped on every retry/turn.
    """
    posts, fake_post = _capture_posts()
    cb = MessageJournalCallback("http://example.invalid")
    big = "X" * (200 * 1024)  # 200 KB of code

    from nooa.tracing import set_session

    set_session("output-blocks-1")

    with patch("nooa.tracing._litellm_journal._post_json", side_effect=fake_post):
        cb.log_pre_api_call(
            model="gpt-x",
            messages=[{"role": "user", "content": "run this"}],
            kwargs={"litellm_call_id": "c1"},
        )
        cb.log_success_event(
            kwargs={"litellm_call_id": "c1", "model": "gpt-x"},
            response_obj=_fake_response_with_tool(arguments=big),
            start_time=0.0,
            end_time=1.0,
        )

    # The blocks POST must contain the full big string.
    block_posts = [p for p in posts if urlparse(p[0]).path == "/v1/journal/blocks"]
    assert block_posts, f"no /v1/journal/blocks POST: {posts!r}"
    flat_blocks = [b for _u, payload, _s in block_posts for b in payload]
    big_blocks = [b for b in flat_blocks if b["content"] == big]
    assert big_blocks, "the big tool_call.arguments string was not routed through blocks"

    # The call record itself must NOT contain the raw arguments inline --
    # only the hash reference.
    call_posts = [p for p in posts if urlparse(p[0]).path == "/v1/journal/calls"]
    assert len(call_posts) == 1, f"expected one call record, got {call_posts!r}"
    record = call_posts[0][1]
    record_json = json.dumps(record)
    assert big not in record_json, (
        "the raw 200KB arguments string ended up inline in the /v1/journal/calls "
        "record -- the output-block sideband isn't routing it through blocks. "
        "Per-call payloads should stay small."
    )
    # The arguments_hash reference must be present.
    out_msgs = record["output_messages"]
    tc = out_msgs[0]["tool_calls"][0]
    assert "arguments_hash" in tc["function"]
    assert "arguments" not in tc["function"]


# ---------------------------------------------------------------------------
# Per-session, per-destination dedup across calls
# ---------------------------------------------------------------------------


def test_per_session_dedup_is_o_delta_across_calls():
    """N back-to-back calls in the same session, each with one new block
    on top of accumulated history -- only the deltas should ship.

    Operationalises "per-call wire is O(delta)" from the MR description.
    Without this, an N-turn agentic loop reposts the entire conversation
    history on every turn.
    """
    posts, fake_post = _capture_posts()
    cb = MessageJournalCallback("http://example.invalid")

    from nooa.tracing import set_session

    set_session("dedup-session")

    base_blocks = {f"sha256:hash{i}": f"block{i}" for i in range(3)}

    def drive_call(call_id: str, blocks: dict[str, str]) -> None:
        set_journal_payload(JournalPayload(skeleton=[], blocks=blocks))
        cb.log_pre_api_call(model="m", messages=[], kwargs={"litellm_call_id": call_id})
        cb.log_success_event(
            kwargs={"litellm_call_id": call_id, "model": "m"},
            response_obj=_fake_response(),
            start_time=0.0,
            end_time=1.0,
        )

    with patch("nooa.tracing._litellm_journal._post_json", side_effect=fake_post):
        # Call 1: ships 3 blocks.
        drive_call("c1", dict(base_blocks))
        # Call 2: same 3 blocks + 1 new -- only the new one should ship.
        drive_call(
            "c2",
            {**base_blocks, "sha256:hashnew": "blocknew"},
        )
        # Call 3: same 4 -- nothing new.
        drive_call(
            "c3",
            {**base_blocks, "sha256:hashnew": "blocknew"},
        )

    block_posts = [
        payload for url, payload, _s in posts if urlparse(url).path == "/v1/journal/blocks"
    ]
    # Concatenate all hashes shipped in any block POST.
    all_shipped_hashes = [b["hash"] for batch in block_posts for b in batch]
    # Each known hash must ship at most once -- the dedup invariant the
    # MR description calls out as the wire-compression win.
    seen: set[str] = set()
    duplicates: list[str] = []
    for h in all_shipped_hashes:
        if h in seen:
            duplicates.append(h)
        seen.add(h)
    assert not duplicates, (
        f"per-session dedup broken: hashes shipped more than once: {duplicates!r}\n"
        f"all shipped: {all_shipped_hashes!r}"
    )

    # Every input block we explicitly set must have shipped.  The output
    # path also generates blocks for the fake response's "out" content;
    # we don't enumerate those here -- the invariant is "no duplicates",
    # not "exactly these hashes".
    input_hashes = set(base_blocks) | {"sha256:hashnew"}
    assert input_hashes <= seen, (
        f"missing input block hashes from shipped set: {input_hashes - seen!r}"
    )


# ---------------------------------------------------------------------------
# ContextVar isolation across asyncio tasks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_tasks_each_get_own_journal_payload():
    """Two parallel ``asyncio.gather`` LLM calls must each carry their own
    journal payload through their respective ``log_pre_api_call`` /
    ``log_success_event`` -- the sideband ``ContextVar`` must not bleed
    between tasks.

    The Async-boundary note in ``_context_sideband.py`` warns that
    ``ContextVar.set`` is task-local; this regression-tests that the
    invariant actually holds end-to-end.
    """
    posts, fake_post = _capture_posts()
    cb = MessageJournalCallback("http://example.invalid")

    from nooa.tracing import set_session

    async def one_call(session_id: str, content: str, call_id: str) -> None:
        # Each task gets its own context: set_session + set_journal_payload
        # mutate ContextVars that are task-local.
        set_session(session_id)
        set_journal_payload(JournalPayload(skeleton=[], blocks={f"sha256:{call_id}": content}))
        cb.log_pre_api_call(model="m", messages=[], kwargs={"litellm_call_id": call_id})
        # Yield to let other tasks run between pre and success.
        await asyncio.sleep(0)
        cb.log_success_event(
            kwargs={"litellm_call_id": call_id, "model": "m"},
            response_obj=_fake_response(content=content),
            start_time=0.0,
            end_time=1.0,
        )

    with patch("nooa.tracing._litellm_journal._post_json", side_effect=fake_post):
        await asyncio.gather(
            one_call("sess-A", "alpha", "ca"),
            one_call("sess-B", "beta", "cb"),
        )

    # Bucket posts by session.
    by_session: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for url, payload, sid in posts:
        by_session[sid].append((url, payload))

    assert "sess-A" in by_session and "sess-B" in by_session, (
        f"each task's posts should be tagged with its own session, got: {sorted(by_session)!r}"
    )

    # Session A's blocks must contain "alpha" and not "beta".
    a_blocks_payloads = [
        p for url, p in by_session["sess-A"] if urlparse(url).path == "/v1/journal/blocks"
    ]
    a_block_contents = [b["content"] for batch in a_blocks_payloads for b in batch]
    assert "alpha" in a_block_contents
    assert "beta" not in a_block_contents

    b_blocks_payloads = [
        p for url, p in by_session["sess-B"] if urlparse(url).path == "/v1/journal/blocks"
    ]
    b_block_contents = [b["content"] for batch in b_blocks_payloads for b in batch]
    assert "beta" in b_block_contents
    assert "alpha" not in b_block_contents

    # Each task's call record must be tagged with its own session.
    a_calls = [p for url, p in by_session["sess-A"] if urlparse(url).path == "/v1/journal/calls"]
    b_calls = [p for url, p in by_session["sess-B"] if urlparse(url).path == "/v1/journal/calls"]
    assert len(a_calls) == 1 and a_calls[0]["call_id"] == "ca"
    assert len(b_calls) == 1 and b_calls[0]["call_id"] == "cb"
    assert a_calls[0]["session_id"] == "sess-A"
    assert b_calls[0]["session_id"] == "sess-B"
