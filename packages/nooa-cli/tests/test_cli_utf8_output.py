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
