# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""_describe_exception must surface the real HTTP error, not mask it as ResponseNotRead.

Regression: a streamable-http MCP transport returns a streaming httpx response.
`_describe_exception` accessed `response.text` directly, which raises
`httpx.ResponseNotRead` on an unread streaming body — masking the actual error
(e.g. a 401 "Authentication required") behind a confusing message.
"""

import httpx

from nooa.mcp.tool import _describe_exception, _describe_exceptions


def _streaming_401() -> httpx.HTTPStatusError:
    """Build an HTTPStatusError whose response body has NOT been read (streaming)."""
    request = httpx.Request("POST", "https://example.test/mcp")
    # stream= makes .text raise ResponseNotRead until .read() is called.
    response = httpx.Response(
        401,
        request=request,
        headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        stream=httpx.ByteStream(
            b'{"error": "invalid_token", "error_description": "Authentication required"}'
        ),
    )
    return httpx.HTTPStatusError("401", request=request, response=response)


def test_describe_exception_reads_streaming_body_instead_of_masking():
    exc = _streaming_401()
    desc = _describe_exception(exc)
    # Must report the real status, not "Attempted to access streaming response content".
    assert "401" in desc
    assert "ResponseNotRead" not in desc
    assert "invalid_token" in desc


def test_describe_exceptions_flattens_and_surfaces_real_status():
    exc = _streaming_401()
    group = ExceptionGroup("connect failed", [exc])
    desc = _describe_exceptions([group])
    assert "401" in desc
    assert "ResponseNotRead" not in desc


def test_describe_exception_already_read_body_still_works():
    request = httpx.Request("POST", "https://example.test/mcp")
    response = httpx.Response(500, request=request, content=b"boom")
    response.read()
    exc = httpx.HTTPStatusError("500", request=request, response=response)
    desc = _describe_exception(exc)
    assert "500" in desc
    assert "boom" in desc
