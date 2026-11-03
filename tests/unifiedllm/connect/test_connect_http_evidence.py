# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Direct HTTP evidence outranks incidental SDK connection-error chains."""

import httpx
import pytest
from litellm import InternalServerError

from nooa.unifiedllm.connect._media import exception_reason


@pytest.mark.parametrize("observed_status", [None, 422, 500])
def test_observed_http_status_distinguished_from_synthetic_transport_status(observed_status):
    exc = InternalServerError(message="private", model="model", llm_provider="openai")
    exc.__cause__ = httpx.ConnectError("private connection details")
    if observed_status is not None:
        exc._connect_http_status = observed_status
    record = {"status_code": observed_status or 500, "error": "InternalServerError"}
    cause = (
        "Could not connect to the service"
        if observed_status is None
        else f"Service returned HTTP {observed_status}"
    )
    assert exception_reason(record, "image", exc) == f"{cause}; image support was not confirmed"


def test_accepted_response_parse_failure_retains_final_answer_diagnostic():
    exc = ValueError("private")
    exc._connect_http_status = 200
    record = {"request_accepted": True, "finish_reason": "stop", "final_answer_observed": False}
    assert (
        exception_reason(record, "pdf", exc)
        == "No final answer returned; pdf support was not confirmed"
    )


async def test_mock_http_rejects_cached_unmocked_client_before_io(monkeypatch):
    from tests.unifiedllm.connect.connect_http import mock_http

    cached = httpx.AsyncClient(trust_env=False)
    try:
        mock_http(monkeypatch, lambda request: httpx.Response(200))
        with pytest.raises(AssertionError, match="mocked SDK transport"):
            await cached.get("https://api.test/v1")
        async with httpx.AsyncClient() as mocked:
            assert (await mocked.get("https://api.test/v1")).status_code == 200
    finally:
        await cached.aclose()


@pytest.mark.parametrize("kind", ["image", "video"])
async def test_captured_422_survives_sdk_connection_error_chain(monkeypatch, kind):
    from nooa.unifiedllm import connect
    from tests.unifiedllm.connect.connect_http import mock_http, response_body

    sent = []

    def handle(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(200, json=response_body("chat"))
        return httpx.Response(422, json={"error": {"message": "request rejected"}})

    mock_http(monkeypatch, handle)
    original = connect._run_probe

    async def chained(*args):
        try:
            return await original(*args)
        except Exception as exc:
            assert exc._connect_http_status == 422  # Actual response-hook evidence.
            exc.__cause__ = httpx.ConnectError("private incidental cause")
            raise

    monkeypatch.setattr(connect, "_run_probe", chained)
    proposal = connect.plan(
        "local",
        "model",
        "chat",
        "https://api.test/v1",
        "",
        reply_tokens=1000,
        test_media_inputs=[kind],
    )
    result = await connect.check_stage(proposal, "media", api_key="fake")
    record = result.entry["provenance"]["probes"]["media:" + kind]
    assert len(sent) == 2
    assert record["status_code"] == 422 and record["outcome"] == "inconclusive"
    assert record["reason"] == f"Service returned HTTP 422; {kind} support was not confirmed"
    assert "private" not in repr(result)
