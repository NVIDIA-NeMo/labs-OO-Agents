# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LSP document synchronization."""

import asyncio

import pytest

from nooa.lsp.client import LSPClient, LSPClientStatus
from nooa.lsp.skill import LSPSkill


def _skill_with_client(tmp_path):
    skill = LSPSkill(root_uri=tmp_path.as_uri())
    config = skill.registry.get_server_for_extension(".py")
    client = LSPClient(command=config.command, root_uri=tmp_path.as_uri())
    client.status = LSPClientStatus.COMPLETE
    skill._clients[tuple(config.command)] = client
    return skill, client


async def test_for_file_opens_once_then_sends_full_text_changes(
    tmp_path, monkeypatch
):
    skill, client = _skill_with_client(tmp_path)
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    sent = []

    async def record_notification(method, params=None):
        sent.append((method, params))

    monkeypatch.setattr(client, "send_notification", record_notification)

    first_facade = await skill.for_file(str(source))
    await skill.for_file(str(source))
    source.write_text("value = 2\n", encoding="utf-8")
    second_facade = await skill.for_file(str(source))

    uri = source.as_uri()
    assert first_facade is not None and second_facade is not None
    assert [method for method, _ in sent] == [
        "textDocument/didOpen",
        "textDocument/didChange",
    ]
    assert sent[0][1]["textDocument"] == {
        "uri": uri,
        "languageId": "python",
        "version": 1,
        "text": "value = 1\n",
    }
    assert sent[1][1] == {
        "textDocument": {"uri": uri, "version": 2},
        "contentChanges": [{"text": "value = 2\n"}],
    }
    assert skill._opened_documents[uri] == (2, "value = 2\n")


async def test_for_file_updates_cached_state_only_after_notification(
    tmp_path, monkeypatch
):
    skill, client = _skill_with_client(tmp_path)
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    fail = False

    async def send_notification(method, params=None):
        if fail:
            raise OSError("notification write failed")

    monkeypatch.setattr(client, "send_notification", send_notification)
    await skill.for_file(str(source))
    uri = source.as_uri()
    assert skill._opened_documents[uri] == (1, "value = 1\n")

    source.write_text("value = 2\n", encoding="utf-8")
    fail = True
    with pytest.raises(OSError, match="notification write failed"):
        await skill.for_file(str(source))

    assert skill._opened_documents[uri] == (1, "value = 1\n")


async def test_for_file_replaces_failed_cached_client(tmp_path, monkeypatch):
    skill, failed_client = _skill_with_client(tmp_path)
    failed_client.status = LSPClientStatus.FAILED
    source = tmp_path / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    uri = source.as_uri()
    stale_uri = (tmp_path / "stale.py").as_uri()
    healthy_uri = (tmp_path / "healthy.js").as_uri()
    failed_key = tuple(failed_client.command)
    healthy_key = ("healthy-server",)
    skill._opened_documents[uri] = (3, "stale content")
    skill._opened_documents[stale_uri] = (2, "stale content")
    skill._opened_documents[healthy_uri] = (4, "healthy content")
    skill._opened_document_clients[uri] = failed_key
    skill._opened_document_clients[stale_uri] = failed_key
    skill._opened_document_clients[healthy_uri] = healthy_key
    started = []
    sent = []
    stopped = []

    async def start(client):
        client.status = LSPClientStatus.COMPLETE
        started.append(client)

    async def send_notification(client, method, params=None):
        sent.append((method, params))

    async def stop(client):
        stopped.append(client)

    monkeypatch.setattr(LSPClient, "start", start)
    monkeypatch.setattr(LSPClient, "send_notification", send_notification)
    monkeypatch.setattr(LSPClient, "stop", stop)

    await skill.for_file(str(source))

    replacement = skill._clients[tuple(failed_client.command)]
    assert replacement is not failed_client
    assert replacement.status == LSPClientStatus.COMPLETE
    assert started == [replacement]
    assert stopped == [failed_client]
    assert [method for method, _ in sent] == ["textDocument/didOpen"]
    assert sent[0][1]["textDocument"]["version"] == 1
    assert skill._opened_documents[uri] == (1, "value = 1\n")
    assert stale_uri not in skill._opened_documents
    assert skill._opened_documents[healthy_uri] == (4, "healthy content")


async def test_concurrent_for_file_calls_start_one_client(tmp_path, monkeypatch):
    skill = LSPSkill(root_uri=tmp_path.as_uri())
    first_source = tmp_path / "first.py"
    second_source = tmp_path / "second.py"
    first_source.write_text("first = 1\n", encoding="utf-8")
    second_source.write_text("second = 2\n", encoding="utf-8")
    start_calls = 0

    async def start(client):
        nonlocal start_calls
        start_calls += 1
        await asyncio.sleep(0)
        client.status = LSPClientStatus.COMPLETE

    async def send_notification(client, method, params=None):
        pass

    monkeypatch.setattr(LSPClient, "start", start)
    monkeypatch.setattr(LSPClient, "send_notification", send_notification)

    first, second = await asyncio.gather(
        skill.for_file(str(first_source)),
        skill.for_file(str(second_source)),
    )

    assert first is not None and second is not None
    assert start_calls == 1
    assert first._client is second._client


async def test_for_file_propagates_file_read_failures(tmp_path):
    skill, _ = _skill_with_client(tmp_path)
    missing_source = tmp_path / "missing.py"

    with pytest.raises(FileNotFoundError):
        await skill.for_file(str(missing_source))

    assert missing_source.as_uri() not in skill._opened_documents