# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Redirected output on native Windows is cp1252 by default; the CLI switches it to UTF-8."""

import io
import sys

import nooa_cli


def _cp1252() -> io.TextIOWrapper:
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252")


def test_redirected_output_switches_to_utf8_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    out, err = _cp1252(), _cp1252()
    monkeypatch.setattr(sys, "stdout", out)
    monkeypatch.setattr(sys, "stderr", err)

    nooa_cli._use_utf8_output()
    out.write("\u2713 \u2192 \u21bb\n")
    out.flush()

    assert out.buffer.getvalue() == "\u2713 \u2192 \u21bb\n".encode()
    assert err.encoding == "utf-8"


def test_output_untouched_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    out = _cp1252()
    monkeypatch.setattr(sys, "stdout", out)

    nooa_cli._use_utf8_output()

    assert out.encoding == "cp1252"


def test_piped_stdin_reads_utf8_on_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    stdin = io.TextIOWrapper(io.BytesIO("caf\u00e9\n".encode()), encoding="cp1252")
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", _cp1252())
    monkeypatch.setattr(sys, "stderr", _cp1252())

    nooa_cli._use_utf8_output()

    assert stdin.read() == "caf\u00e9\n"


def test_missing_streams_are_skipped(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    monkeypatch.setattr(sys, "stdin", None)

    nooa_cli._use_utf8_output()
