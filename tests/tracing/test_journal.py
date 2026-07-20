# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the skeleton + content-addressed blocks journal protocol.

The callback consumes a :class:`JournalPayload` from the sideband
ContextVar (populated by the nooa actor) and posts:

1. New blocks to ``/v1/journal/blocks`` (dedup'd per session by hash).
2. A call record (skeleton + output messages) to ``/v1/journal/calls``.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nooa.tracing._context_sideband import (
    JournalPayload,
    set_journal_payload,
)
from nooa.tracing._litellm_journal import (
    MessageJournalCallback,
)


def _posts():
    """Return a list that captures every _post_json call's (url, payload) tuple."""
    calls: list[tuple[str, object]] = []

    def fake_post(url, payload, *, session_id="", timeout=15.0, on_success=None):
        calls.append((url, payload))
        if on_success is not None:
            on_success()

    return calls, fake_post


@pytest.fixture
def session_ctx():
    """Install a known session for the callback to read via get_session()."""
    from nooa.tracing._session import set_session

    set_session("test-session")
    yield "test-session"
    set_session(None)


def _fake_response(content: str = ""):
    """Minimal stand-in for a litellm completion response."""
    choice = SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            model_dump=lambda exclude_unset=True: {"role": "assistant", "content": content},
        )
    )
    return SimpleNamespace(choices=[choice], usage=None)


class TestJournalCallbackPreApiCall:
    def test_skeleton_consumed_from_sideband(self, session_ctx):
        cb = MessageJournalCallback("http://localhost:5001")
        calls, fake_post = _posts()

        payload = JournalPayload(
            skeleton=[
                {"role": "system", "parts": [{"block_hash": "h1", "key": "k"}]},
                {"role": "user", "content": "hi"},
            ],
            blocks={"h1": "<k>content</k>"},
        )
        set_journal_payload(payload)

        with patch(
            "nooa.tracing._litellm_journal._post_json",
            side_effect=fake_post,
        ):
            cb.log_pre_api_call("model-x", [], {"litellm_call_id": "cid1"})

        # Blocks were posted.
        block_posts = [c for c in calls if c[0].endswith("/v1/journal/blocks")]
        assert len(block_posts) == 1
        assert block_posts[0][1] == [{"hash": "h1", "content": "<k>content</k>"}]

        # Skeleton stashed on the callback for the forthcoming success event.
        assert "cid1" in cb._call_inputs
        stored_skeleton, _span = cb._call_inputs["cid1"]
        assert stored_skeleton == payload.skeleton

    def test_block_dedup_per_session(self, session_ctx):
        """Second call with the same block hash must not re-post it."""
        cb = MessageJournalCallback("http://localhost:5001")
        calls, fake_post = _posts()

        payload = JournalPayload(skeleton=[], blocks={"h1": "<b/>"})
        with patch(
            "nooa.tracing._litellm_journal._post_json",
            side_effect=fake_post,
        ):
            set_journal_payload(payload)
            cb.log_pre_api_call("m", [], {"litellm_call_id": "c1"})
            set_journal_payload(payload)
            cb.log_pre_api_call("m", [], {"litellm_call_id": "c2"})

        block_posts = [c for c in calls if c[0].endswith("/v1/journal/blocks")]
        # First call posts the block; second call skips it (already sent).
        assert len(block_posts) == 1
        assert block_posts[0][1] == [{"hash": "h1", "content": "<b/>"}]

    def test_no_sideband_falls_back_to_raw_messages(self, session_ctx):
        cb = MessageJournalCallback("http://localhost:5001")
        calls, fake_post = _posts()

        set_journal_payload(None)  # explicitly empty
        messages = [{"role": "user", "content": "hello"}]
        with patch(
            "nooa.tracing._litellm_journal._post_json",
            side_effect=fake_post,
        ):
            cb.log_pre_api_call("m", messages, {"litellm_call_id": "cid"})

        # No block POSTs (nothing to post).
        assert not any(c[0].endswith("/v1/journal/blocks") for c in calls)
        # Skeleton = raw messages as-is.
        skeleton, _ = cb._call_inputs["cid"]
        assert skeleton == [{"role": "user", "content": "hello"}]


class TestJournalCallbackSuccessEvent:
    def test_success_posts_call_record(self, session_ctx):
        cb = MessageJournalCallback("http://localhost:5001")
        calls, fake_post = _posts()

        set_journal_payload(
            JournalPayload(
                skeleton=[{"role": "user", "content": "q"}],
                blocks={},
            )
        )
        with patch(
            "nooa.tracing._litellm_journal._post_json",
            side_effect=fake_post,
        ):
            cb.log_pre_api_call("m", [], {"litellm_call_id": "cid"})
            cb.log_success_event(
                {"litellm_call_id": "cid", "model": "m"},
                _fake_response("answer"),
                1.0,
                2.0,
            )

        call_posts = [c for c in calls if c[0].endswith("/v1/journal/calls")]
        assert len(call_posts) == 1
        record = call_posts[0][1]
        assert record["call_id"] == "cid"
        assert record["session_id"] == "test-session"
        assert record["input_skeleton"] == [{"role": "user", "content": "q"}]
        # output_messages are content-addressed: content → parts[{block_hash}]
        (out_msg,) = record["output_messages"]
        assert out_msg["role"] == "assistant"
        assert "content" not in out_msg
        (part,) = out_msg["parts"]
        answer_hash = part["block_hash"]
        # The block content should have been POSTed to /v1/journal/blocks
        block_posts = [c for c in calls if c[0].endswith("/v1/journal/blocks")]
        assert len(block_posts) >= 1
        all_blocks = {e["hash"]: e["content"] for post in block_posts for e in post[1]}
        assert all_blocks[answer_hash] == "answer"


class TestSentBlocksBounding:
    """Tests for single-session tracking and deferred hash marking."""

    def test_session_switch_drops_old_hashes(self, session_ctx):
        """Switching sessions forgets the old session's hashes, so blocks are re-posted."""
        from nooa.tracing._session import set_session

        cb = MessageJournalCallback("http://localhost:5001")
        calls, fake_post = _posts()

        payload = JournalPayload(skeleton=[], blocks={"h1": "block1"})
        with patch(
            "nooa.tracing._litellm_journal._post_json",
            side_effect=fake_post,
        ):
            # Post block under session A
            set_session("session-A")
            set_journal_payload(payload)
            cb.log_pre_api_call("m", [], {"litellm_call_id": "c1"})

            # Switch to session B — same block hash should be re-posted
            set_session("session-B")
            set_journal_payload(payload)
            cb.log_pre_api_call("m", [], {"litellm_call_id": "c2"})

        block_posts = [c for c in calls if c[0].endswith("/v1/journal/blocks")]
        assert len(block_posts) == 2, "Block should be posted once per session"

    def test_failed_post_allows_retry(self, session_ctx):
        """When POST fails (on_success not called), hashes stay unmarked for retry."""
        cb = MessageJournalCallback("http://localhost:5001")
        calls: list[tuple[str, object]] = []

        def failing_post(url, payload, *, session_id="", timeout=15.0, on_success=None):
            calls.append((url, payload))
            # Simulate POST failure: do NOT call on_success

        payload = JournalPayload(skeleton=[], blocks={"h1": "block1"})
        with patch(
            "nooa.tracing._litellm_journal._post_json",
            side_effect=failing_post,
        ):
            set_journal_payload(payload)
            cb.log_pre_api_call("m", [], {"litellm_call_id": "c1"})

            # Same block again — should be re-posted since prior POST "failed"
            set_journal_payload(payload)
            cb.log_pre_api_call("m", [], {"litellm_call_id": "c2"})

        block_posts = [c for c in calls if c[0].endswith("/v1/journal/blocks")]
        assert len(block_posts) == 2, "Failed POST should not mark hashes as sent"
