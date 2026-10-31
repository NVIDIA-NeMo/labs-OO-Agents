# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CaptureConfig.file_dir / file_notice reach the file-backed capture buffers."""

import json

from nooa.agentdoc import FileBackedTruncatingStringIO, TruncatingStringIO
from nooa.config import CaptureConfig
from nooa.runtime.actor import _make_capture_buffer


def test_defaults_unchanged():
    config = CaptureConfig()
    assert config.file_backed is False and config.file_dir is None and config.file_notice is None
    buf = _make_capture_buffer(limit=10, tail_chars=None, file_backed=False)
    assert type(buf) is TruncatingStringIO


def test_file_dir_is_created_and_used_and_notice_phrases_the_path(tmp_path):
    target = tmp_path / "session" / "cells"
    buf = _make_capture_buffer(
        limit=10,
        tail_chars=None,
        file_backed=True,
        file_dir=str(target),
        file_notice="Full output: read({quoted_path})",
    )
    assert isinstance(buf, FileBackedTruncatingStringIO)
    buf.write("x" * 100)
    buf.close()
    assert buf.path is not None and buf.path.startswith(str(target))
    assert f"Full output: read({json.dumps(buf.path)})\n" in buf.getvalue()


def test_unusable_file_dir_falls_back_to_the_temp_dir(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("")
    buf = _make_capture_buffer(
        limit=10, tail_chars=None, file_backed=True, file_dir=str(blocker / "d")
    )
    try:
        assert buf.path is not None and not buf.path.startswith(str(blocker))
    finally:
        buf.cleanup()
