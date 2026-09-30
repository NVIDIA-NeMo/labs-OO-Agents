# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consistent file encodings across ShellTools reads and edits, without Bash."""

import json
from unittest.mock import AsyncMock

import pytest

from nooa.tools.shell_tools import ShellTools


async def test_utf8_file_operations_with_legacy_default(tmp_path, monkeypatch):
    monkeypatch.setattr("io.text_encoding", lambda encoding, stacklevel=2: encoding or "cp1252")
    shell = ShellTools(cwd=str(tmp_path))
    try:
        text = "coffee ☕\n中文\n"
        await shell.write_file("note.txt", text)
        assert (tmp_path / "note.txt").read_text(encoding="utf-8") == text
        assert (await shell.read("note.txt")).text == text
        await shell.replace("note.txt", "☕", "🚀")
        region = await shell.read("note.txt", lines=(2, 2))
        await shell.replace(region, "café €\n")
        expected = "coffee 🚀\ncafé €\n"
        assert (tmp_path / "note.txt").read_text(encoding="utf-8") == expected
        assert (await shell.read("note.txt")).text == expected
    finally:
        await shell.close()


async def test_read_existing_utf8_file_with_legacy_default(tmp_path, monkeypatch):
    text = "café €\n"
    (tmp_path / "note.txt").write_text(text, encoding="utf-8")
    monkeypatch.setattr("io.text_encoding", lambda encoding, stacklevel=2: encoding or "cp1252")
    shell = ShellTools(cwd=str(tmp_path))
    try:
        assert (await shell.read("note.txt")).text == text
    finally:
        await shell.close()


async def test_search_anchors_use_file_encoding(tmp_path, monkeypatch):
    text = "coffee café\n"
    (tmp_path / "note.txt").write_text(text, encoding="utf-8")
    monkeypatch.setattr("io.text_encoding", lambda encoding, stacklevel=2: encoding or "cp1252")
    shell = ShellTools(cwd=str(tmp_path))
    session = AsyncMock()
    session.run_with_timeout_flag.return_value = (
        json.dumps({"type": "match", "data": {"path": {"text": "note.txt"}, "line_number": 1}}),
        "",
        0,
        False,
    )
    monkeypatch.setattr(shell, "_get_session", AsyncMock(return_value=session))
    try:
        matches = await shell._harvest_matches("rg -n coffee note.txt", "1:" + text)
        assert matches is not None
        assert len(matches) == 1
        assert matches[0].text == text
        await shell.replace(matches[0], "tea 中文\n")
        assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "tea 中文\n"
    finally:
        await shell.close()


@pytest.mark.parametrize("encoding", ["cp1252", None])
async def test_explicit_legacy_encoding_preserves_existing_files(tmp_path, monkeypatch, encoding):
    path = tmp_path / "legacy.txt"
    path.write_bytes("café €\n".encode("cp1252"))
    monkeypatch.setattr("io.text_encoding", lambda encoding, stacklevel=2: encoding or "cp1252")
    shell = ShellTools(cwd=str(tmp_path), encoding=encoding)
    try:
        assert (await shell.read("legacy.txt")).text == "café €\n"
        await shell.replace("legacy.txt", "café", "thé")
        assert path.read_text(encoding="cp1252") == "thé €\n"
    finally:
        await shell.close()
