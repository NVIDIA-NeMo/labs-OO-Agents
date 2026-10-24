# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for media capture and object transfer from brokered skill calls."""

from __future__ import annotations

import asyncio
import contextlib
import multiprocessing as mp
import threading
import types
from typing import Any

import pytest

from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.media import Image
from nooa.runtime.media_capture import show
from nooa.runtime.sandbox import wire
from nooa.runtime.sandbox.cell_core import run_cell_source
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.runtime.sandbox.serialization import (
    dto_from_wire,
    dto_to_result,
    dto_to_wire,
    result_to_dto,
)
from nooa.runtime.sandbox.worker import ChildBroker, ParentAgentProxy, build_namespace
from nooa.skill import Skill
from nooa.strategies.codeact import _ReturnResultSignal


class _MultimodalSkill(Skill):
    def from_file(self, path: str) -> Image:
        return Image.from_bytes(f"bytes-for-{path}".encode(), media_type="image/png")

    def show(self, obj: Any) -> None:
        if isinstance(obj, str):
            obj = self.from_file(obj)
        show(obj)


class _MediaAgent:
    def __init__(self) -> None:
        self.multimodal = _MultimodalSkill()


def _setup_sandbox_environment(agent: Any | None = None, max_media: int = 100):
    agent = agent or _MediaAgent()
    parent_conn, child_conn = mp.Pipe(duplex=True)

    ex = object.__new__(SandboxedExecutor)
    ex._conn = parent_conn
    ex._proc = types.SimpleNamespace(is_alive=lambda: True)
    ex._config = SandboxConfig(require=False)
    ex._cell_timeout = None
    ex._max_error = DEFAULT_TRUNCATION_CONFIG.capture.max_error
    ex._max_media = max_media
    ex._codec = wire.Codec(wire.agent_types(agent))
    ex._agent = agent

    broker = ChildBroker(child_conn, threading.RLock())
    proxy = ParentAgentProxy(broker, agent)
    namespace = build_namespace(agent, {}, proxy, None)

    return ex, broker, namespace, parent_conn, child_conn


def _run_broker_listener(parent_conn, ex, stop_event, parent_loop):
    while not stop_event.is_set():
        if parent_conn.poll(0.01):
            try:
                raw = parent_conn.recv_bytes()
                msg = ex._codec.loads(raw)
                if msg.get("type") == "tool_call":
                    ex._service_tool_call(msg, parent_loop)
            except (EOFError, OSError):
                break


@contextlib.contextmanager
def _broker_service(parent_conn, ex):
    parent_loop = asyncio.new_event_loop()
    parent_thread = threading.Thread(target=parent_loop.run_forever, daemon=True)
    parent_thread.start()

    stop_event = threading.Event()
    listener = threading.Thread(
        target=_run_broker_listener,
        args=(parent_conn, ex, stop_event, parent_loop),
        daemon=True,
    )
    listener.start()
    try:
        yield
    finally:
        stop_event.set()
        listener.join(timeout=2.0)
        parent_loop.call_soon_threadsafe(parent_loop.stop)
        parent_thread.join(timeout=2.0)
        parent_loop.close()


@pytest.mark.asyncio
async def test_brokered_media_skill_reproduction():
    """Verify both reproduced gaps are resolved:
    1. Media capture from brokered calls: self.multimodal.show('probe.png') contributes attachments.
    2. Media object transfer: image = self.multimodal.from_file(...) followed by
       self.multimodal.show(image) transfers Image back to parent without TypeError.
    """
    ex, broker, namespace, parent_conn, child_conn = _setup_sandbox_environment()

    code = """
self.multimodal.show("probe.png")
image = self.multimodal.from_file("probe.png")
self.multimodal.show(image)
"""
    try:
        with _broker_service(parent_conn, ex):
            result = await run_cell_source(code, namespace, execution_count=1)
            assert result.error is None, f"Cell execution failed: {result.error}"
            assert len(result.images) == 2, f"Expected 2 images, got {len(result.images)}"
            assert result.images[0]["type"] == "image_url"
            assert result.images[1]["type"] == "image_url"
            assert result.images[0]["image_url"]["url"].startswith("data:image/png;base64,")
            assert result.images[1]["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        parent_conn.close()
        child_conn.close()


@pytest.mark.asyncio
async def test_worker_local_show_with_brokered_image():
    """Verify an Image returned from parent can be shown locally via worker show()."""
    ex, broker, namespace, parent_conn, child_conn = _setup_sandbox_environment()

    code = """
image = self.multimodal.from_file("probe.png")
assert image.media_type == "image/png"
show(image)
"""
    try:
        with _broker_service(parent_conn, ex):
            result = await run_cell_source(code, namespace, execution_count=1)
            assert result.error is None, f"Cell execution failed: {result.error}"
            assert len(result.images) == 1
            assert result.images[0]["type"] == "image_url"
    finally:
        parent_conn.close()
        child_conn.close()


@pytest.mark.asyncio
async def test_brokered_media_attachment_cap():
    """Verify max_attachments limit is enforced when brokered calls produce media."""
    ex, broker, namespace, parent_conn, child_conn = _setup_sandbox_environment(max_media=1)

    code = """
self.multimodal.show("img1.png")
self.multimodal.show("img2.png")
"""
    try:
        with _broker_service(parent_conn, ex):
            result = await run_cell_source(code, namespace, execution_count=1, max_attachments=1)
            assert result.error is None
            assert len(result.images) == 1
            assert "[show() limit reached (1), attachment not added]" in result.stdout
    finally:
        parent_conn.close()
        child_conn.close()


@pytest.mark.asyncio
async def test_cell_return_media_object():
    """Verify returning a Media object from a cell serializes through ResultDTO to parent."""
    ex, broker, namespace, parent_conn, child_conn = _setup_sandbox_environment()

    code = """
image = self.multimodal.from_file("probe.png")
image
"""
    try:
        with _broker_service(parent_conn, ex):
            result = await run_cell_source(code, namespace, execution_count=1)
            assert result.error is None
            assert isinstance(result.returned_value, Image)

            # Round trip worker -> parent through serialization boundary
            dto = result_to_dto(result)
            wire_dict = dto_to_wire(dto)
            loaded_dto = dto_from_wire(wire_dict)
            parent_result = dto_to_result(loaded_dto, codec=ex._codec)

            assert isinstance(parent_result.returned_value, Image)
            assert parent_result.returned_value.media_type == "image/png"
            assert parent_result.returned_value.data_url == result.returned_value.data_url
    finally:
        parent_conn.close()
        child_conn.close()


@pytest.mark.asyncio
async def test_cell_return_result_signal_media():
    """Verify return_result(image) control-flow signal transfers Image safely."""

    def return_result(arg):
        raise _ReturnResultSignal(result=arg)

    ex, broker, namespace, parent_conn, child_conn = _setup_sandbox_environment()
    namespace["return_result"] = return_result

    code = """
image = self.multimodal.from_file("probe.png")
return_result(image)
"""
    try:
        with _broker_service(parent_conn, ex):
            result = await run_cell_source(code, namespace, execution_count=1)
            assert result.error is None
            assert isinstance(result.signal, _ReturnResultSignal)
            assert isinstance(result.signal.result, Image)

            # Round trip worker -> parent through serialization boundary
            dto = result_to_dto(result)
            wire_dict = dto_to_wire(dto)
            loaded_dto = dto_from_wire(wire_dict)
            parent_result = dto_to_result(
                loaded_dto,
                signal_factory=lambda payload: _ReturnResultSignal(result=payload),
                codec=ex._codec,
            )

            assert isinstance(parent_result.signal, _ReturnResultSignal)
            assert isinstance(parent_result.signal.result, Image)
            assert parent_result.signal.result.media_type == "image/png"
    finally:
        parent_conn.close()
        child_conn.close()


@pytest.mark.asyncio
async def test_cell_setattr_media_object():
    """Verify assigning a Media object to self.<attr> reaches the parent agent."""

    class _TargetAgent:
        def __init__(self) -> None:
            self.multimodal = _MultimodalSkill()
            self.saved_image = None

    agent = _TargetAgent()
    ex, broker, namespace, parent_conn, child_conn = _setup_sandbox_environment(agent=agent)

    code = """
image = self.multimodal.from_file("probe.png")
self.saved_image = image
"""
    try:
        with _broker_service(parent_conn, ex):
            result = await run_cell_source(code, namespace, execution_count=1)
            assert result.error is None
            assert isinstance(agent.saved_image, Image)
            assert agent.saved_image.media_type == "image/png"
    finally:
        parent_conn.close()
        child_conn.close()
