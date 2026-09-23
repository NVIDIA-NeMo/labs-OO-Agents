# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Targeted regressions for MCP approval fingerprint scope, type normalization,
and the approval store's atomic-write descriptor handling.
"""

import pytest
from nooa_coder.interactive.mcp_approval import MCPApprovalStore, build_approval_request


def test_fingerprint_binds_the_workspace_scope():
    """An approval for one workspace must not silently approve the identical
    definition in another: the stdio client has no separate cwd, so a
    relative command resolves against whatever workspace launches it, and
    saved servers can auto-connect on startup.
    """
    servers = {"tool": {"command": "run-tool"}}
    a = build_approval_request("tool", mcp_file=None, servers=servers, scope="/workspace/a")
    b = build_approval_request("tool", mcp_file=None, servers=servers, scope="/workspace/b")
    assert a.fingerprint != b.fingerprint
    assert not a.accepts_confirmation(b.confirmation)


def test_claude_code_style_type_field_maps_to_transport():
    request = build_approval_request(
        "tool",
        mcp_file=None,
        servers={"tool": {"type": "http", "url": "https://example.com/mcp"}},
        scope="/workspace",
    )
    assert request.transport == "streamable-http"


def test_conflicting_type_and_transport_is_rejected():
    with pytest.raises(ValueError, match="conflicting"):
        build_approval_request(
            "tool",
            mcp_file=None,
            servers={
                "tool": {"type": "http", "transport": "stdio", "command": "run-tool"},
            },
            scope="/workspace",
        )


def test_approval_store_write_failure_does_not_double_close_fd(tmp_path, monkeypatch):
    """A failure after os.fdopen() takes ownership of fd must not also close
    it directly -- the process runs other threads (agent loop, to_thread
    workers) that can reuse that descriptor number in between, so a second
    os.close would close an unrelated open file or socket.
    """
    store = MCPApprovalStore(tmp_path / "approvals.json")
    closed: list[int] = []
    original_close = __import__("os").close

    def tracking_close(fd, *a, **kw):
        closed.append(fd)
        return original_close(fd, *a, **kw)

    monkeypatch.setattr("nooa_coder.interactive.mcp_approval.os.close", tracking_close)
    monkeypatch.setattr(
        "nooa_coder.interactive.mcp_approval.os.replace",
        lambda *a, **kw: (_ for _ in ()).throw(OSError("replace failed")),
    )

    with pytest.raises(OSError, match="replace failed"):
        store._write({"version": 1, "approvals": {}})

    assert len(closed) == len(set(closed)), f"fd closed more than once: {closed}"
