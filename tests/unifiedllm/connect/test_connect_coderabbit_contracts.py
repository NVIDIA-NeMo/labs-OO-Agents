# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Approved CodeRabbit regressions: skip accounting and Responses file projection."""

import base64
import copy
import json
from dataclasses import replace

import httpx
import pytest

from nooa.unifiedllm import ResponsesClient, connect
from nooa.unifiedllm.connect import _session
from nooa.unifiedllm.connect._records import text_baseline_passed
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def proposal(style, **kwargs):
    """Build a detached plan with a known reply cap and catalogue prices."""
    return connect.plan(
        "local",
        "model",
        style,
        "https://api.test/v1",
        "",
        reply_tokens=1000,
        catalogue={"pricing": {"prompt": "0.001", "completion": "0.002"}},
        **kwargs,
    )


@pytest.mark.parametrize("style", ["responses", "anthropic"])
def test_local_skips_have_no_aggregate_cost_but_keep_safe_reservations(style):
    """Only runnable probes contribute tokens/price, including after cap edits."""
    baseline = proposal(style)
    media = proposal(style, test_media_inputs=["audio", "video"])
    assert media.token_estimate == baseline.token_estimate
    assert media.price_estimate == baseline.price_estimate
    skips = tuple(p for p in media.probes if p.skip_reason)
    assert [p.name for p in skips] == ["media:audio", "media:video"]
    assert all(p.input_reservation == 4096 and p.token_estimate == 5096 for p in skips)
    only_skips = connect.refresh_plan(replace(media, probes=skips))
    assert only_skips.token_estimate == 0
    # Refresh still approves the full per-probe cap, even for local skips.
    cap_key = "max_output_tokens" if style == "responses" else "max_tokens"
    entry = copy.deepcopy(media.entry)
    entry["max_tokens"] = 2000
    refreshed = connect.refresh_plan(replace(media, entry=entry))
    assert refreshed.token_estimate == 2 * (512 + 2000)
    assert all(
        p.body[cap_key] == 2000 and p.token_estimate == 6096
        for p in refreshed.probes
        if p.skip_reason
    )


@pytest.mark.parametrize("style", ["responses", "anthropic"])
def test_session_remaining_budget_excludes_local_skips(monkeypatch, style):
    """Planning and refresh pass real remaining budget to session reservation."""
    remaining = []

    def reply_budget(entry, tokens):
        """Expose the remaining budget passed by each planning path."""
        remaining.append(tokens)
        return 1000, 1000, 17

    monkeypatch.setattr(_session, "reply_budget", reply_budget)
    media = proposal(
        style, budget_tokens=4000, session_checks=True, test_media_inputs=["audio", "video"]
    )
    assert remaining == [4000 - 3024]
    assert media.token_estimate == 3024 + 3 * 17 and media.price_estimate is None
    remaining.clear()
    refreshed = connect.refresh_plan(media)
    assert remaining == [4000 - 3024]
    assert refreshed.token_estimate == media.token_estimate


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("delta", [-1, 0])
async def test_runnable_boundary_and_local_skips_remain_visible(monkeypatch, style, delta):
    """A runnable image still requires its full reservation at the exact boundary."""
    sent = []

    def handle(request):
        """Return native typed responses through the actual SDK transport."""
        sent.append(json.loads(request.content))
        return httpx.Response(
            200, json=response_body(style, '{"value":323}' if len(sent) == 1 else '{"value":true}')
        )

    mock_http(monkeypatch, handle)
    media = proposal(
        style, test_media_inputs=["image", "audio", "video"], budget_tokens=1512 + 5096 + delta
    )
    # Stage excludes the tools probe; routing + image are the exact boundary.
    result = await connect.check_stage(media, "media", api_key="fake")
    records = result.entry["provenance"]["probes"]
    assert text_baseline_passed(records["routing"])
    assert len(sent) == (1 if delta < 0 else 2)
    image = records["media:image"]
    assert image.get("reason") == "budget exhausted" if delta < 0 else image["outcome"] == "passed"
    assert result.entry["provenance"]["tokens_charged_to_budget"] == (1512 if delta < 0 else 6608)
    for kind in ("audio", "video"):
        record = records["media:" + kind]
        assert record["outcome"] == "not_probed"
        if style != "chat":
            skip = next(p for p in media.probes if p.name == "media:" + kind)
            assert record["reason"] == skip.skip_reason
        else:
            assert record["reason"] == "budget exhausted"


@pytest.mark.parametrize("refresh", [False, True])
@pytest.mark.parametrize("delta", [-1, 0])
async def test_removed_skip_restores_full_execution_budget(monkeypatch, refresh, delta):
    """Removing a skip cannot inherit a zero reservation or bypass cap safety."""
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(request) or httpx.Response(200, json=response_body("responses"))
        ),
    )
    media = proposal("responses", test_media_inputs=["audio"])
    skip = next(p for p in media.probes if p.skip_reason)
    # Turn this into a runnable planned check; no media transport support is implied.
    runnable = replace(
        skip, name="reintroduced", skip_reason=None, body=copy.deepcopy(media.probes[0].body)
    )
    changed = replace(media, probes=(runnable,), budget_tokens=5096 + delta)
    if refresh:
        changed = connect.refresh_plan(changed)
        assert changed.token_estimate == 5096
    result = await connect.run(changed, approved="all", api_key="fake")
    record = result.entry["provenance"]["probes"]["reintroduced"]
    assert len(sent) == (0 if delta < 0 else 1)
    if delta < 0:
        assert record["reason"] == "budget exhausted"
    sent.clear()
    unsafe = replace(runnable, body={**runnable.body, "max_output_tokens": 2000})
    result = await connect.run(
        replace(changed, probes=(unsafe,), budget_tokens=10000), approved="all", api_key="fake"
    )
    assert not sent
    assert "cap" in result.entry["provenance"]["probes"]["reintroduced"]["reason"]


@pytest.mark.parametrize(
    "block",
    [
        {"type": "file"},
        {"type": "file", "file": None},
        {"type": "file", "file": "private-payload"},
        {"type": "file", "file": 7},
        {"type": "file", "file": []},
        {"type": "file", "file": {}},
        {"type": "file", "file_data": "data:application/pdf;base64,eA=="},
        {"type": "file", "file": {"filename": "private.pdf", "unknown": "private-payload"}},
    ],
)
@pytest.mark.parametrize("mode", ["projection", "sync", "async"])
async def test_malformed_file_wrapper_fails_locally_without_history_mutation(
    monkeypatch, block, mode
):
    """Malformed/flat type:file wrappers fail clearly before SDK dispatch."""
    sent = []
    mock_http(
        monkeypatch,
        lambda request: (
            sent.append(request) or httpx.Response(200, json=response_body("responses"))
        ),
    )
    # Known PDF metadata admits the wrapper so sync/async exercise projection,
    # rather than the intentional generic-file policy notice before dispatch.
    admitted = {"mime_type": "application/pdf", **copy.deepcopy(block)}
    history = [{"role": "user", "content": [admitted]}]
    original = copy.deepcopy(history)
    client = ResponsesClient(
        "openai/model", api_key="fake", api_base="https://api.test/v1", media_inputs=["pdf"]
    )
    try:
        with pytest.raises(ValueError, match="Responses file.*(?:mapping|requires)") as exc:
            if mode == "projection":
                client._transform_messages(history)
            elif mode == "sync":
                client.call(history)
            else:
                await client.acall(history)
        assert "private-payload" not in str(exc.value) and "private.pdf" not in str(exc.value)
    finally:
        await client.aclose()
    assert not sent and history == original


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("mode", ["projection", "sync", "async"])
async def test_valid_file_projection_is_owned_and_preserves_pdf_bytes(monkeypatch, native, mode):
    """Nested supported keys and admitted native input_file survive SDK dispatch."""
    payload = b"%PDF-1.4 original bytes"
    data = "data:application/pdf;base64," + base64.b64encode(payload).decode()
    fields = {"file_data": data, "filename": "original.pdf"}
    block = (
        {"type": "input_file", **fields}
        if native
        else {
            "type": "file",
            "file": {
                **fields,
                "format": "application/pdf",
                "unknown": {"file_url": "https://not-forwarded.invalid"},
            },
        }
    )
    messages = [{"role": "user", "content": [block]}]
    original = copy.deepcopy(messages)
    sent = []

    def handle(request):
        """Capture the projected payload below the SDK without a live call."""
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=response_body("responses"))

    mock_http(monkeypatch, handle)
    sync_init = httpx.Client.__init__

    def init(client, **kwargs):
        """Give the sync SDK the same mock-only transport."""
        kwargs["transport"] = httpx.MockTransport(handle)
        sync_init(client, **kwargs)

    monkeypatch.setattr(httpx.Client, "__init__", init)
    client = ResponsesClient(
        "openai/model", api_key="fake", api_base="https://api.test/v1", media_inputs=["pdf"]
    )
    try:
        projected, _ = client._transform_messages(messages, media_inputs=("pdf",))
        expected = {"type": "input_file", **fields}
        assert projected[0]["content"] == [expected]
        projected[0]["content"][0]["filename"] = "edited.pdf"
        assert messages == original
        if mode == "sync":
            client.call(messages)
        elif mode == "async":
            await client.acall(messages)
        if mode != "projection":
            assert len(sent) == 1 and sent[0]["input"][0]["content"] == [expected]
        denied, _ = client._transform_messages(
            client._prepare_media_messages(messages, {"media_inputs": []}), media_inputs=()
        )
        assert denied[0]["content"][0]["type"] == "input_text"
        assert "was removed here" in denied[0]["content"][0]["text"]
    finally:
        await client.aclose()
    assert messages == original
