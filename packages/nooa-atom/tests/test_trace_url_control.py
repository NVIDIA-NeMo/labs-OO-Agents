# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""/trace-url: the viewer URL of the current trace session."""

import sys

import pytest
from nooa_atom.workspace.controls import TraceUrlControl

import nooa.tracing


def _control():
    return TraceUrlControl(None, None)


@pytest.mark.parametrize(
    ("endpoint", "base"),
    [
        ("https://viewer.example:5443/v1/traces", "https://viewer.example:5443"),
        ("http://host:9000/v1/", "http://host:9000"),
        ("http://host:9000", "http://host:9000"),
    ],
)
async def test_the_url_names_the_trace_session_on_the_viewer(monkeypatch, endpoint, base):
    monkeypatch.setattr(nooa.tracing, "get_session", lambda: "run 1/a")
    monkeypatch.setenv("OTLP_ENDPOINT", endpoint)
    result = await _control().invoke("")
    assert result.success
    assert str(result) == f"Trace viewer:\n```text\n{base}/traces/view?session_id=run%201/a\n```"


async def test_the_default_viewer_is_local(monkeypatch):
    monkeypatch.setattr(nooa.tracing, "get_session", lambda: "abc")
    monkeypatch.delenv("OTLP_ENDPOINT", raising=False)
    result = await _control().invoke("")
    assert (
        str(result)
        == "Trace viewer:\n```text\nhttp://localhost:5001/traces/view?session_id=abc\n```"
    )


async def test_without_a_trace_session_it_says_so(monkeypatch):
    monkeypatch.setattr(nooa.tracing, "get_session", lambda: None)
    result = await _control().invoke("")
    assert not result.success
    assert str(result) == "No active trace session."


async def test_without_the_tracing_package_it_says_so(monkeypatch):
    monkeypatch.setitem(sys.modules, "nooa.tracing", None)
    result = await _control().invoke("")
    assert not result.success
    assert str(result) == "Tracing package not installed."


async def test_it_takes_no_arguments():
    result = await _control().invoke("extra")
    assert not result.success
    assert "Usage: /trace-url" in str(result)
