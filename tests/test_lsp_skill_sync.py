# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LSP document synchronization."""

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
    started = []

    async def start(client):
        client.status = LSPClientStatus.COMPLETE
        started.append(client)

    async def send_notification(client, method, params=None):
        pass

    monkeypatch.setattr(LSPClient, "start", start)
    monkeypatch.setattr(LSPClient, "send_notification", send_notification)

    await skill.for_file(str(source))

    replacement = skill._clients[tuple(failed_client.command)]
    assert replacement is not failed_client
    assert replacement.status == LSPClientStatus.COMPLETE
    assert started == [replacement]


async def test_for_file_propagates_file_read_failures(tmp_path):
    skill, _ = _skill_with_client(tmp_path)
    missing_source = tmp_path / "missing.py"

    with pytest.raises(FileNotFoundError):
        await skill.for_file(str(missing_source))

    assert missing_source.as_uri() not in skill._opened_documents