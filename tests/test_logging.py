# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa logging helpers."""

from __future__ import annotations

import io
import logging
import uuid

import pytest

from nooa._logging import enable_logging


@pytest.fixture()
def fresh_logger():
    """Yield a unique logger and clean up all handlers added during the test."""
    name = f"nooa.test_logging.{uuid.uuid4().hex[:8]}"
    logger = logging.getLogger(name)
    original_handlers = list(logger.handlers)

    yield logger, name

    # Teardown: remove any handlers that were added during the test.
    for h in logger.handlers[:]:
        if h not in original_handlers:
            logger.removeHandler(h)
    logger.setLevel(logging.NOTSET)


# ── 1. NullHandler attached on import ────────────────────────────────────────


def test_nullhandler_attached():
    root = logging.getLogger("nooa")
    assert any(isinstance(h, logging.NullHandler) for h in root.handlers)


# ── 2. enable_logging importable from public API ─────────────────────────────


def test_enable_logging_importable():
    from nooa import enable_logging as el

    assert callable(el)


# ── 3. enable_logging adds a StreamHandler and sets DEBUG ─────────────────────


def test_enable_logging_adds_handler(fresh_logger):
    logger, name = fresh_logger

    enable_logging(name=name)

    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(stream_handlers) == 1
    assert logger.level == logging.DEBUG


# ── 4. Custom level ──────────────────────────────────────────────────────────


def test_enable_logging_custom_level(fresh_logger):
    logger, name = fresh_logger

    enable_logging(name=name, level=logging.WARNING)

    assert logger.level == logging.WARNING


# ── 5. Idempotent — same stream ─────────────────────────────────────────────


def test_enable_logging_idempotent(fresh_logger):
    logger, name = fresh_logger
    stream = io.StringIO()

    enable_logging(name=name, stream=stream)
    enable_logging(name=name, stream=stream)

    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(stream_handlers) == 1


# ── 6. Idempotent but updates level ─────────────────────────────────────────


def test_enable_logging_idempotent_updates_level(fresh_logger):
    logger, name = fresh_logger
    stream = io.StringIO()

    enable_logging(name=name, level=logging.DEBUG, stream=stream)
    enable_logging(name=name, level=logging.ERROR, stream=stream)

    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(stream_handlers) == 1
    assert logger.level == logging.ERROR


# ── 7. Targets subtree only ─────────────────────────────────────────────────


def test_enable_logging_targets_subtree(fresh_logger):
    _, name = fresh_logger
    child_name = f"{name}.strategies"
    child_logger = logging.getLogger(child_name)

    try:
        enable_logging(name=child_name)

        parent_logger = logging.getLogger(name)
        parent_stream = [h for h in parent_logger.handlers if type(h) is logging.StreamHandler]
        child_stream = [h for h in child_logger.handlers if type(h) is logging.StreamHandler]

        assert len(child_stream) == 1
        assert len(parent_stream) == 0
    finally:
        for h in child_logger.handlers[:]:
            if type(h) is logging.StreamHandler:
                child_logger.removeHandler(h)
        child_logger.setLevel(logging.NOTSET)


# ── 8. Custom stream receives output ────────────────────────────────────────


def test_enable_logging_custom_stream(fresh_logger):
    logger, name = fresh_logger
    buf = io.StringIO()

    enable_logging(name=name, stream=buf)
    logger.debug("hello from test")

    output = buf.getvalue()
    assert "hello from test" in output


# ── 9. Different streams = different handlers ────────────────────────────────


def test_enable_logging_different_streams(fresh_logger):
    logger, name = fresh_logger
    stream_a = io.StringIO()
    stream_b = io.StringIO()

    enable_logging(name=name, stream=stream_a)
    enable_logging(name=name, stream=stream_b)

    stream_handlers = [h for h in logger.handlers if type(h) is logging.StreamHandler]
    assert len(stream_handlers) == 2
