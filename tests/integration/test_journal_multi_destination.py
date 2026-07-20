# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T4: every journal-exporter destination receives every event for every call.

The eval pipeline registers two journal exporters by design: one to its
in-process headless backend (per-run temp DB) and one to the user's running
viewer (long-lived, cross-run).  Both must receive *all* journal data for
every LLM call, not just the first-installed one.

Today this fails: ``MessageJournalCallback`` is registered as a separate
``litellm.callbacks`` entry per exporter, and litellm fires
``log_pre_api_call`` on both instances but ``log_success_event`` on only
*one*.  Result: the second-registered destination (the viewer in real eval
runs) sees ``/v1/journal/messages`` POSTs but never ``/v1/journal/calls``,
so its ``llm_calls`` table stays empty.

Phase 2 fix: a single callback with a list of destinations, dispatched
internally.  This sidesteps litellm's same-class callback dedup entirely.

This test pins the post-fix contract: when N destinations are configured,
each one must receive every ``/v1/journal/*`` POST.  It drives the
callback's hooks directly (``log_pre_api_call`` + ``log_success_event``)
because litellm's ``mock_response`` shortcut bypasses its callback chain
entirely, so we can't observe dispatch through ``acompletion`` without a
live or stubbed HTTP backend.  T2 and T6 cover the live path.
"""

from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import patch
from urllib.parse import urlparse


def _drive_one_call(
    callback, *, session_id: str, call_id: str = "call-1"
) -> dict[str, list[tuple[str, object]]]:
    """Drive a callback through one fake LLM call and return ``{base_url:
    [(/v1/journal/* path, payload), …]}``.

    Captures every ``_post_json`` invocation from inside the callback,
    bucketed by base URL so the test can assert *each* destination got
    *each* POST kind *and* compare the actual payloads -- a fan-out bug
    that sent *different* records per destination would slip past a
    "POST landed on both" assertion.
    """
    posts: dict[str, list[tuple[str, object]]] = defaultdict(list)

    def fake_post(url, payload, *, session_id="", timeout=15.0, on_success=None):
        u = urlparse(url)
        posts[f"{u.scheme}://{u.netloc}"].append((u.path, payload))
        if on_success is not None:
            on_success()

    from nooa.tracing import set_session

    set_session(session_id)

    fake_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    role="assistant",
                    content="hello",
                    tool_calls=None,
                )
            )
        ],
        usage=SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )

    with patch(
        "nooa.tracing._litellm_journal._post_json",
        side_effect=fake_post,
    ):
        callback.log_pre_api_call(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "hi"}],
            kwargs={"litellm_call_id": call_id},
        )
        callback.log_success_event(
            kwargs={"litellm_call_id": call_id, "model": "gpt-3.5-turbo"},
            response_obj=fake_response,
            start_time=0.0,
            end_time=1.0,
        )

    return posts


def test_callback_fans_out_call_record_to_every_destination():
    """The callback must be configurable with multiple destination base URLs
    and POST ``/v1/journal/calls`` to each of them on every ``log_success_event``."""
    from nooa.tracing._litellm_journal import MessageJournalCallback

    base_a = "http://a.invalid"
    base_b = "http://b.invalid"

    # Phase-2 contract: one callback, list of destinations.
    # The current constructor takes a single string; this test drives the
    # post-fix shape and will fail today.
    cb = MessageJournalCallback([base_a, base_b])

    posts = _drive_one_call(cb, session_id="t4-fanout")

    calls_a = [body for path, body in posts[base_a] if path == "/v1/journal/calls"]
    calls_b = [body for path, body in posts[base_b] if path == "/v1/journal/calls"]

    assert calls_a, (
        f"first destination ({base_a}) never got a /v1/journal/calls POST.\nposts: {dict(posts)!r}"
    )
    assert calls_b, (
        f"second destination ({base_b}) never got a /v1/journal/calls POST. "
        f"This is the fan-out bug: log_success_event only flushes to a single "
        f"destination instead of all configured ones.\n"
        f"posts: {dict(posts)!r}"
    )

    # Both destinations must receive *equivalent* records, not just any
    # record.  A fan-out bug that sends a different call_id / skeleton /
    # output to each destination would slip past a non-empty check.
    assert len(calls_a) == len(calls_b), (
        f"unequal POST count: A got {len(calls_a)}, B got {len(calls_b)}"
    )
    for a, b in zip(calls_a, calls_b, strict=True):
        assert a["call_id"] == b["call_id"], (
            f"call_ids diverged across destinations: {a['call_id']!r} vs {b['call_id']!r}"
        )
        assert a["session_id"] == b["session_id"]
        assert a["input_skeleton"] == b["input_skeleton"]
        assert a["output_messages"] == b["output_messages"]
        assert a.get("span_id") == b.get("span_id")

    # Same for /v1/journal/blocks: a fan-out that sent different blocks
    # to each destination would corrupt reconstruction.
    blocks_a = [body for path, body in posts[base_a] if path == "/v1/journal/blocks"]
    blocks_b = [body for path, body in posts[base_b] if path == "/v1/journal/blocks"]
    assert len(blocks_a) == len(blocks_b)
    for a, b in zip(blocks_a, blocks_b, strict=True):
        # The wire payload is a list of {hash, content}; order shouldn't
        # matter for set-based dedup so compare by hash.
        a_by_hash = {item["hash"]: item["content"] for item in a}
        b_by_hash = {item["hash"]: item["content"] for item in b}
        assert a_by_hash == b_by_hash


def test_two_callbacks_each_fan_out_independently():
    """Backwards-compat: two separately-constructed callbacks (the way the
    eval pipeline does it today) must each post to their own destination
    when both are driven, even if they share state through ``litellm.callbacks``.

    This test is the one that maps directly onto the eval pipeline's
    ``_start_tracing`` shape: ``exporters.journal(headless) +
    exporters.journal(viewer)`` -> two callbacks installed.  After Phase 2
    they may collapse into one shared callback with two destinations; either
    way, each destination must end up with the call record.
    """
    from nooa.tracing._litellm_journal import MessageJournalCallback

    base_a = "http://a.invalid"
    base_b = "http://b.invalid"

    cb_a = MessageJournalCallback(base_a)
    cb_b = MessageJournalCallback(base_b)

    # Drive each callback independently.  The bug-today is that real
    # litellm only delivers log_success_event to one of two same-class
    # callbacks; here we verify each callback works in isolation.
    posts_a = _drive_one_call(cb_a, session_id="t4-cb-a", call_id="c-a")
    posts_b = _drive_one_call(cb_b, session_id="t4-cb-b", call_id="c-b")

    assert any(path == "/v1/journal/calls" for path, _body in posts_a[base_a]), (
        f"callback A failed to post to its destination: {dict(posts_a)!r}"
    )
    assert any(path == "/v1/journal/calls" for path, _body in posts_b[base_b]), (
        f"callback B failed to post to its destination: {dict(posts_b)!r}"
    )
