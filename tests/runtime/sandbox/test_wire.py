# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker->parent pipe is msgpack: nothing the worker sends is ever unpickled."""

from __future__ import annotations

import asyncio
import collections
import datetime
import decimal
import enum
import os
import pathlib
import threading
import types
import uuid
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import msgpack
import pytest
from pydantic import BaseModel

from nooa import Agent, hidden
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG
from nooa.events import _NO_RETURN
from nooa.runtime.sandbox import wire
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.errors import CellSerializationError, WorkerDiedError
from nooa.runtime.sandbox.executor import SandboxedExecutor
from nooa.runtime.sandbox.serialization import (
    ErrorDTO,
    ResultDTO,
    SignalDTO,
    dto_from_wire,
    dto_to_result,
    dto_to_wire,
)
from nooa.unifiedllm.fake import FakeLLMClient

CODEC = wire.Codec()


class Kind(enum.Enum):
    INFO = "info"
    WARN = "warn"


class Detail(BaseModel):
    n: int


class Line(BaseModel):
    text: str
    detail: Detail | None = None


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Tally:
    n: int
    total: int = field(init=False, default=0)


with hidden:
    # Hidden from module globals: reachable only through ``Report.nested``.
    class Nested(BaseModel):
        value: int


class Report(BaseModel):
    lines: list[Line]
    kind: Kind = Kind.INFO
    origin: Point | None = None
    nested: dict[str, Nested] = {}


class Pair(NamedTuple):
    a: int
    b: str


class Plain:
    def __init__(self, v: int = 0) -> None:
        self.v = v


class _ReportAgent(Agent, llm=FakeLLMClient()):
    def accept(self, report: Report) -> str:
        return report.kind.value


def _forged(code: int, payload: Any) -> bytes:
    return msgpack.packb(msgpack.ExtType(code, CODEC.dumps(payload)))


# --- codec --------------------------------------------------------------------
@pytest.mark.parametrize(
    "value",
    [
        [1, 2.5, "s", b"b", None, True, (1, 2), {"a": [1]}, {1: "int-key", b"k": 2}],
        {1, 2},
        frozenset({1}),
        (1, (2, {3})),
        {(1, 2): "tuple-key"},
        collections.OrderedDict(a=1),
        collections.Counter("aab"),
        datetime.datetime(2026, 1, 1, 12, 30, tzinfo=datetime.UTC),
        datetime.datetime(2026, 1, 1, 12, 30),
        datetime.date(2026, 1, 1),
        datetime.timedelta(days=1, seconds=5, microseconds=7),
        decimal.Decimal("1.5"),
        pathlib.Path("/tmp/x"),
        uuid.UUID(int=7),
    ],
    ids=lambda v: type(v).__name__,
)
def test_value_types_round_trip(value):
    assert CODEC.loads(CODEC.dumps(value)) == value


def test_numpy_round_trips():
    np = pytest.importorskip("numpy")
    arrays = [
        np.arange(6, dtype=np.float64).reshape(2, 3),
        np.array([1, 2, 3]),
        np.array([True, False]),
        np.array(["a", "b"]),
        np.array([[1, 2], "a", None], dtype=object),
        np.zeros(2, dtype=[("a", "i4"), ("b", "f8")]),
        np.arange(4)[::2],
    ]
    for array in arrays:
        back = CODEC.loads(CODEC.dumps(array))
        assert back.dtype == array.dtype and back.shape == array.shape
        assert np.array_equal(back, array)
    assert CODEC.loads(CODEC.dumps(np.float64(1.5))) == 1.5
    assert CODEC.loads(CODEC.dumps(np.int64(3))) == 3


@pytest.mark.parametrize(
    "value",
    [lambda: 1, object(), types.SimpleNamespace(a=1), Plain(), collections.deque([1]), 2**70],
    ids=["function", "object", "SimpleNamespace", "user-class", "deque", "bigint"],
)
def test_unsupported_values_fail_at_encode(value):
    with pytest.raises((TypeError, OverflowError)):
        CODEC.dumps(value)


def test_reduce_is_never_consulted(monkeypatch):
    marker = "NOOA_TEST_SANDBOX_ESCAPE"
    monkeypatch.delenv(marker, raising=False)

    class Bomb:
        def __reduce__(self):
            return (eval, (f"__import__('os').environ.__setitem__({marker!r}, 'pwned')",))

    with pytest.raises(TypeError):
        CODEC.dumps(Bomb())
    assert marker not in os.environ


# --- agent data types -----------------------------------------------------------
def test_agent_types_are_declared_data_types_and_their_fields():
    found = wire.agent_types(_ReportAgent())
    keys = {k.rsplit(":", 1)[1] for k in found if k.startswith(__name__ + ":")}
    assert {"Report", "Line", "Detail", "Kind", "Point", "Pair", "Nested"} <= keys
    assert not {"Plain", "_ReportAgent"} & keys
    assert found[wire.type_key(Nested)] is Nested


def test_declared_types_round_trip_and_are_validated():
    codec = wire.Codec(wire.agent_types(_ReportAgent()))
    report = Report(
        lines=[Line(text="a", detail=Detail(n=1))],
        kind=Kind.WARN,
        origin=Point(1, 2),
        nested={"k": Nested(value=3)},
    )
    assert codec.loads(codec.dumps(report)) == report
    assert codec.loads(codec.dumps(Pair(1, "b"))) == Pair(1, "b")
    assert codec.loads(codec.dumps([Kind.WARN, Point(0, 0)])) == [Kind.WARN, Point(0, 0)]
    tally = Tally(n=2)
    tally.total = 5  # init=False state survives the round trip
    assert codec.loads(codec.dumps(tally)) == tally
    # A forged model payload still goes through pydantic validation.
    with pytest.raises(CellSerializationError, match="malformed"):
        codec.loads(_forged(wire._MODEL, [wire.type_key(Detail), {"n": "not an int"}]))


@pytest.mark.parametrize(
    ("code", "payload"),
    [
        (wire._MODEL, [wire.type_key(Report), {"lines": []}]),
        (wire._DATACLASS, ["os:system", {"command": "true"}]),
        (wire._ENUM, ["builtins:eval", "1"]),
        (wire._NAMEDTUPLE, ["subprocess:Popen", [["true"]]]),
    ],
    ids=["undeclared-model", "os.system", "eval", "Popen"],
)
def test_undeclared_class_keys_are_refused_by_name(code, payload):
    with pytest.raises(CellSerializationError, match="not a data type declared by the agent"):
        CODEC.loads(_forged(code, payload))


@pytest.mark.parametrize("data", [b"\xc1", b"\x93\x01", msgpack.packb(msgpack.ExtType(99, b""))])
def test_corrupt_or_unknown_streams_are_refused(data):
    with pytest.raises(CellSerializationError):
        CODEC.loads(data)


# --- ResultDTO ------------------------------------------------------------------
def test_result_dto_round_trips_through_the_pipe_encoding():
    returned = ResultDTO(
        stdout="out",
        stderr="err",
        returned_value=b"ret",
        has_return=True,
        explicit_return=True,
        images=[{"data_url": "data:image/png;base64,AA=="}],
        wrapper_line_offset=3,
        defined_method_names=["helper"],
    )
    failed = ResultDTO(error=ErrorDTO("ValueError", "msg", "diag"))
    signaled = ResultDTO(signal=SignalDTO(result=b"sig"))
    for dto in (returned, failed, signaled):
        assert dto_from_wire(CODEC.loads(CODEC.dumps(dto_to_wire(dto)))) == dto


@pytest.mark.parametrize(
    "data",
    [
        "nope",
        {"stdout": 1},
        {"returned_value": "text"},
        {"error": {"type_name": 1}},
        {"bogus": 1},
        {"error": {"type_name": "E", "message": "m"}, "has_return": True},
        {"explicit_return": True},
    ],
    ids=[
        "not-a-dict",
        "stdout-not-str",
        "returned_value-not-bytes",
        "error-field",
        "unknown",
        "error-and-return",
        "explicit-without-return",
    ],
)
def test_result_dto_rejects_malformed_shapes(data):
    with pytest.raises(CellSerializationError, match="malformed result"):
        dto_from_wire(data)


def test_undeclared_return_value_becomes_serialization_error():
    dto = ResultDTO(returned_value=CODEC.dumps(Detail(n=1)), has_return=True)
    refused = dto_to_result(dto)
    assert isinstance(refused.error, CellSerializationError)
    assert "Detail" in str(refused.error)
    assert refused.returned_value is _NO_RETURN

    accepted = dto_to_result(dto, codec=wire.Codec(wire.agent_types(_ReportAgent())))
    assert accepted.error is None
    assert accepted.returned_value == Detail(n=1)


# --- executor, parent side (no worker forked) -------------------------------------
class _FakeConn:
    def __init__(self, raw: bytes = b"") -> None:
        self._raw = raw
        self.sent: list[dict[str, Any]] = []

    def poll(self, timeout: float | None = None) -> bool:
        return True

    def recv_bytes(self) -> bytes:
        return self._raw

    def send(self, obj: dict[str, Any]) -> None:
        self.sent.append(obj)


class _Target:
    value = 41

    def add_one(self, n: int) -> int:
        return n + 1


def _bare_executor(conn: _FakeConn) -> SandboxedExecutor:
    ex = object.__new__(SandboxedExecutor)
    ex._conn = conn  # type: ignore[assignment]
    ex._proc = types.SimpleNamespace(is_alive=lambda: True)  # type: ignore[assignment]
    ex._config = SandboxConfig(require=False)
    ex._cell_timeout = None
    ex._max_error = DEFAULT_TRUNCATION_CONFIG.capture.max_error
    ex._codec = wire.Codec()
    ex._agent = _Target()
    return ex


@pytest.mark.parametrize("raw", [b"\xc1", CODEC.dumps([1, 2])], ids=["corrupt", "non-dict"])
def test_recv_retires_worker_on_malformed_message(raw):
    ex = _bare_executor(_FakeConn(raw))
    loop = asyncio.new_event_loop()
    try:
        with pytest.raises(WorkerDiedError, match="malformed message"):
            ex._recv_until_result(1, None, loop)
    finally:
        loop.close()


def test_refused_payload_is_answered_as_tool_error_without_dispatch():
    conn = _FakeConn()
    ex = _bare_executor(conn)
    msg = {
        "type": "tool_call",
        "kind": "setattr",
        "path": ["value"],
        "tool_call_id": 7,
        "payload": CODEC.dumps(Detail(n=1)),
    }
    ex._service_tool_call(msg, asyncio.new_event_loop())
    (response,) = conn.sent
    assert response["type"] == "tool_result" and response["tool_call_id"] == 7
    assert response["ok"] is False
    assert response["error_type"] == "CellSerializationError"
    assert "Detail" in response["error"] and "sandbox boundary" in response["error"]
    assert ex._agent.value == 41


@pytest.mark.parametrize(
    "msg",
    [
        {"kind": "call", "path": ["add_one"]},
        {"kind": "call", "path": [1], "payload": b""},
        {"kind": "call", "path": ["add_one"], "payload": CODEC.dumps([1])},
        {"kind": "call", "path": ["add_one"], "payload": CODEC.dumps((1, {}))},
        {"kind": "bogus", "path": ["add_one"]},
    ],
    ids=["payload-missing", "path-not-str", "not-args-kwargs", "args-not-tuple", "unknown-kind"],
)
def test_malformed_broker_requests_are_answered_as_tool_errors(msg):
    conn = _FakeConn()
    ex = _bare_executor(conn)
    ex._service_tool_call({"type": "tool_call", "tool_call_id": 1, **msg}, asyncio.new_event_loop())
    assert conn.sent[0]["ok"] is False
    assert "malformed" in conn.sent[0]["error"]


def test_decoded_call_payload_reaches_the_agent():
    conn = _FakeConn()
    ex = _bare_executor(conn)
    msg = {
        "type": "tool_call",
        "kind": "call",
        "path": ["add_one"],
        "tool_call_id": 2,
        "payload": CODEC.dumps(((41,), {})),
    }
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    try:
        ex._service_tool_call(msg, loop)
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()
    assert conn.sent[0]["ok"] is True and conn.sent[0]["result"] == 42
