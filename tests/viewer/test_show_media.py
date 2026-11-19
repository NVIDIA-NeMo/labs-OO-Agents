# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""show() media reach the viewer: LLM spans reference them and /api/trace/media serves them.

show() blocks travel the message journal as ``image_hashes`` + content-addressed
blocks. The UI path (``/api/trace``) must reference them by hash instead of
inlining the payload; the export path inlines the original data URL; the media
endpoint serves only safe types inline.
"""

from __future__ import annotations

import base64
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from nooa.runtime.media_capture import media_to_content_block
from nooa.tracing._hooks_impl import OpenInferenceHooks
from nooa.tracing._journal_builder import _encode_image
from nooa.tracing._media_parts import describe_media_block, media_block_hash, media_summary
from nooa.viewer import main as main_module
from nooa.viewer import otlp_store
from nooa.viewer.main import app

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_BLOCK = {
    "type": "image_url",
    "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(PNG).decode(),
        "format": "image/png",
    },
}
PDF = b"%PDF-1.4\n%fake\n"
PDF_BLOCK = {
    "type": "file",
    "file": {
        "file_data": "data:application/pdf;base64," + base64.b64encode(PDF).decode(),
        "filename": "attachment",
        "format": "application/pdf",
    },
}
SVG_BLOCK = {
    "type": "image_url",
    "image_url": {"url": "data:image/svg+xml;base64," + base64.b64encode(b"<svg/>").decode()},
}
SID = "sess-media-1"


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(otlp_store, "DB_PATH", tmp_path / "test.db")
    monkeypatch.setattr(otlp_store, "_db", None)
    monkeypatch.setattr(
        main_module,
        "_write_executor",
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="sqlite-writer-test"),
    )
    for tls in (otlp_store._read_tls, otlp_store._write_tls):
        if hasattr(tls, "conn"):
            del tls.conn
    with TestClient(app) as c:
        db = otlp_store._db
        otlp_store._read_tls.conn = db
        otlp_store._write_tls.conn = db
        yield c
    for tls in (otlp_store._read_tls, otlp_store._write_tls):
        if hasattr(tls, "conn"):
            del tls.conn


def _seed(db: sqlite3.Connection, blocks: list[dict]) -> list[str]:
    """One session with one LLM span whose input message carries *blocks*."""
    db.execute(
        "INSERT INTO sessions (session_id, experiment, span_count, modified) "
        "VALUES (?, 'default', 0, 0)",
        (SID,),
    )
    attrs = [{"key": "openinference.span.kind", "value": {"stringValue": "LLM"}}]
    db.execute(
        "INSERT INTO spans (session_id, trace_id, span_id, parent_span_id, name, kind, "
        "start_time_ns, end_time_ns, status_code, status_message, attributes, resource, events) "
        "VALUES (?, ?, 'span1', NULL, 'acompletion', 3, 1, 2, NULL, NULL, ?, '{}', NULL)",
        (SID, "0" * 32, json.dumps(attrs)),
    )
    db.commit()
    text = "In [1]: [shown: Image]"
    hashes = [media_block_hash(b) for b in blocks]
    items = [{"hash": otlp_store_hash(text), "content": text}]
    items += [{"hash": h, "content": _encode_image(b)} for h, b in zip(hashes, blocks, strict=True)]
    otlp_store.ingest_journal_blocks(SID, items)
    otlp_store.ingest_journal_call(
        {
            "call_id": "c1",
            "session_id": SID,
            "span_id": "span1",
            "model": "m",
            "ts_start": 1.0,
            "ts_end": 2.0,
            "input_skeleton": [
                {
                    "role": "user",
                    "parts": [{"block_hash": otlp_store_hash(text)}],
                    "image_hashes": hashes,
                }
            ],
            "output_messages": [],
        }
    )
    return hashes


def otlp_store_hash(s: str) -> str:
    from nooa.tracing._journal_builder import _hash

    return _hash(s)


def _attrs(span: dict) -> dict[str, str]:
    return {a["key"]: a["value"].get("stringValue") for a in span["attributes"]}


def test_describe_blocks_from_media_capture():
    from nooa.media import Image

    block = media_to_content_block(Image.from_bytes(PNG, media_type="image/png"))
    desc = describe_media_block(block)
    assert desc is not None and desc["kind"] == "image" and desc["mime_type"] == "image/png"
    assert base64.b64decode(desc["base64"]) == PNG
    assert describe_media_block(PDF_BLOCK)["mime_type"] == "application/pdf"
    audio = describe_media_block(
        {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mp3"}}
    )
    assert audio == {"kind": "audio", "mime_type": "audio/mpeg", "base64": "AAAA"}
    remote = describe_media_block({"type": "image_url", "image_url": {"url": "https://x/y.png"}})
    assert remote["url"] == "https://x/y.png"
    assert describe_media_block({"type": "text", "text": "hi"}) is None


def test_ui_spans_reference_media_by_hash_without_payload(client):
    hashes = _seed(otlp_store._db, [PNG_BLOCK, PDF_BLOCK])
    resp = client.get("/api/trace", params={"session_id": SID})
    assert resp.status_code == 200
    assert "base64," not in resp.text  # payload stays out of the span listing
    attrs = _attrs(resp.json()["events"][0])
    p = "llm.input_messages.0.message.contents"
    assert attrs[f"{p}.0.message_content.type"] == "image"
    assert attrs[f"{p}.0.message_content.image.image.url"] == otlp_store.media_ref_url(
        SID, hashes[0]
    )
    assert attrs[f"{p}.1.message_content.type"] == "file"
    assert attrs[f"{p}.1.message_content.media.mime_type"] == "application/pdf"
    assert attrs["llm.input_messages.0.message.content"] == "In [1]: [shown: Image]"

    img = client.get(attrs[f"{p}.0.message_content.image.image.url"])
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert img.headers["x-content-type-options"] == "nosniff"
    assert img.content == PNG
    assert img.headers["cache-control"] == "no-store"
    pdf = client.get(attrs[f"{p}.1.message_content.media.url"])
    assert pdf.status_code == 200 and pdf.content == PDF
    assert pdf.headers["content-type"] == "application/pdf"


def test_export_inlines_original_data_url(client):
    _seed(otlp_store._db, [PNG_BLOCK])
    spans = otlp_store.reconstruct_full_spans(SID)
    attrs = _attrs(spans[0])
    url = attrs["llm.input_messages.0.message.contents.0.message_content.image.image.url"]
    assert url == PNG_BLOCK["image_url"]["url"]


def test_unsafe_types_download_and_unknown_hash_404(client):
    hashes = _seed(otlp_store._db, [SVG_BLOCK])
    r = client.get(otlp_store.media_ref_url(SID, hashes[0]))
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/octet-stream"
    assert r.headers["content-disposition"] == "attachment"
    assert "sandbox" in r.headers["content-security-policy"]
    assert r.headers["cache-control"] == "no-store"
    assert client.get(otlp_store.media_ref_url(SID, "sha256:nope")).status_code == 404
    assert client.get(otlp_store.media_ref_url("other-session", hashes[0])).status_code == 404


def test_code_execution_output_lists_media_refs_without_payload():
    result = SimpleNamespace(
        stdout="[shown: x]", stderr="", returned_value=None, images=[PNG_BLOCK]
    )
    data = json.loads(OpenInferenceHooks._safe_serialize_execution_result(result))
    assert data["media"] == [media_summary(PNG_BLOCK)]
    assert data["media"][0]["hash"] == media_block_hash(PNG_BLOCK)
    assert "base64" not in json.dumps(data["media"])
    no_media = SimpleNamespace(stdout="", stderr="", returned_value=None, images=[])
    assert "media" not in json.loads(OpenInferenceHooks._safe_serialize_execution_result(no_media))


def test_code_execution_media_refs_beyond_cap_are_counted():
    blocks = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{i:04d}AAAA"}}
        for i in range(65)
    ]
    result = SimpleNamespace(stdout="", stderr="", returned_value=None, images=blocks)
    data = json.loads(OpenInferenceHooks._safe_serialize_execution_result(result))
    assert len(data["media"]) == 64
    assert data["media"][-1]["hash"] == media_block_hash(blocks[63])
    assert data["media_omitted"] == 1
    at_cap = SimpleNamespace(stdout="", stderr="", returned_value=None, images=blocks[:64])
    data = json.loads(OpenInferenceHooks._safe_serialize_execution_result(at_cap))
    assert len(data["media"]) == 64 and "media_omitted" not in data
