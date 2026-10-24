# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny generated fixtures for attachment access checks, never user files.

These checks describe one request, not permanent model capability. No encoder,
TTS, file upload, network fetch or additional dependency is required.
"""

import base64
import io
import math
import re
import struct
import wave
import zlib

from ._video_fixture import video_fixture


def _data_url(mime, data):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def png_fixture():
    """A valid 32x32 RGB PNG, entirely red."""

    def chunk(kind, data):
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff\0\0" * 32) * 32))
        + chunk(b"IEND", b"")
    )


def wav_fixture():
    """Two separate 440Hz tones, with silence before, between and after."""
    rate = 8000
    samples = []
    for tone, duration in [(False, 0.2), (True, 0.3), (False, 0.3), (True, 0.3), (False, 0.2)]:
        samples.extend(
            int(12000 * math.sin(2 * math.pi * 440 * i / rate)) if tone else 0
            for i in range(round(rate * duration))
        )
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return buffer.getvalue()


def pdf_fixture():
    """One valid PDF page with a single word, MAPLE, in Helvetica."""
    stream = b"BT /F1 36 Tf 40 100 Td (MAPLE) Tj ET\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 300 200] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"endstream",
    ]
    result = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for index, obj in enumerate(objects, 1):
        offsets.append(len(result))
        result += f"{index} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(result)
    result += f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode()
    result += b"".join(f"{offset:010} 00000 n \n".encode() for offset in offsets[1:])
    return (
        result
        + f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode()
    )


def media_content(kind, style):
    """Return planned attachment content or an honest local skip reason."""
    from nooa.media_policy import implemented_input_kinds, unsupported_input_reason

    if kind not in implemented_input_kinds(style):
        return None, None, unsupported_input_reason(style, [kind])
    if kind == "image":
        block = {"type": "image_url", "image_url": {"url": _data_url("image/png", png_fixture())}}
    elif kind == "video":
        block = {"type": "video_url", "video_url": {"url": _data_url("video/mp4", video_fixture())}}
    elif kind == "pdf":
        if style == "anthropic":
            block = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.b64encode(pdf_fixture()).decode("ascii"),
                },
            }
        else:
            block = {
                "type": "file",
                "file": {
                    "filename": "probe.pdf",
                    "file_data": _data_url("application/pdf", pdf_fixture()),
                },
            }
    else:
        block = {
            "type": "input_audio",
            "input_audio": {
                "data": base64.b64encode(wav_fixture()).decode("ascii"),
                "format": "wav",
            },
        }
    noun = "document" if kind == "pdf" else kind
    prompt = (
        f"Can you access and interpret the attached {noun}? "
        "Return false if you cannot access or interpret its contents."
    )
    return [{"type": "text", "text": prompt}, block], None, None


def exception_reason(record, kind, exc):
    """Use only status/type evidence, never credential-bearing exception text."""
    connection_error = False
    current = exc
    for _ in range(6):
        if current is None:
            break
        connection_error |= type(current).__name__ in {"APIConnectionError", "ConnectError"}
        current = current.__cause__ or current.__context__
    if record.get("timeout_kind"):
        cause = "Timed out; server receipt and completion are unknown"
    elif isinstance(observed_status := getattr(exc, "_connect_http_status", None), int) and not (
        200 <= observed_status < 300
    ):
        # Response-hook evidence is authoritative even if SDK translation chains
        # a connection error. status_code alone may be a synthetic transport 500.
        cause = f"Service returned HTTP {observed_status}"
    elif connection_error:
        # SDKs can assign synthetic HTTP 500 to transport failures.
        cause = "Could not connect to the service"
    elif record.get("request_accepted") is True:
        cause = {
            "length": "Reply reached the token limit",
            "error": "Reply ended with an error",
            "tool_calls": "Reply was not a completed final answer",
        }.get(record.get("finish_reason"), "Could not read the answer")
        if record.get("finish_reason") == "stop" and record.get("final_answer_observed") is False:
            cause = "No final answer returned"
    elif isinstance(record.get("status_code"), int):
        cause = f"Service returned HTTP {record['status_code']}"
    else:
        error = record.get("error", "")
        cause = (
            "Reply not understood by the runtime"
            if error == "ReasoningReplayError"
            else "Request failed"
        )
        # Exception class identifiers are safe; exception messages/URLs are not.
        if cause == "Request failed" and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,79}", error):
            cause += f" ({error})"
    return f"{cause}; {kind} support was not confirmed"


def reply_failure_reason(kind, record):
    """Explain missing evidence without retaining reply text or scoring content."""
    finish = record.get("finish_reason")
    if finish == "length":
        cause = "Reply reached the token limit"
    elif finish == "content_filter":
        cause = "Reply was content-filtered"
    elif finish == "error":
        cause = "Reply ended with an error"
    elif finish != "stop":
        cause = "Reply was not a completed final answer"
    elif record.get("final_answer_observed") is not True:
        cause = "No final answer returned"
    elif record.get("typed_result_validated") is not True:
        cause = "Could not read the answer"
    elif record.get("settings_sent") is None:
        cause = "Actual request was not observed; media and output settings could not be verified"
    else:
        cause = "Generated attachment or output settings were missing from the observed request"
    return f"{cause}; {kind} support was not confirmed"


def explicitly_rejected(exc, kind):
    """Only a media-specific provider rejection, not a generic HTTP 400."""
    status = getattr(exc, "_connect_http_status", getattr(exc, "status_code", None))
    if status not in {400, 415, 422}:
        return False
    body = getattr(exc, "body", None)
    error = body.get("error", body) if isinstance(body, dict) else {}
    param = error.get("param") if isinstance(error, dict) else None
    text = (
        str(error.get("message", str(exc))).lower() if isinstance(error, dict) else str(exc).lower()
    )
    if isinstance(param, str) and param.split(".")[0] in {
        "model",
        "include",
        "store",
        "max_tokens",
        "max_output_tokens",
        "api_key",
    }:
        return False
    # Match a statement about the media/input itself. A media-looking model
    # name plus an unrelated 'invalid' parameter is not explicit rejection.
    names = {
        "image": r"images?|vision inputs?",
        "audio": r"audio|wav",
        "video": r"videos?",
        "pdf": r"pdfs?|documents?|files?",
    }[kind]
    noun = rf"(?:{names})(?: (?:input|content|format|type|data|url|attachment|processing)s?)?"
    # Text-only models: the whole request is refused because it has media.
    if re.search(r"\bnot a multimodal model\b|\bdoes(?: not|n't) support multimodal", text):
        return True
    # "Unsupported content part type: 'file'" names the wire part for this format.
    part = re.search(
        r"unsupported (?:chat )?(?:content(?: part)?|media) type:?\s*[\\'\"]*([a-z_]+)", text
    )
    parts = {
        "image": {"image_url", "input_image"},
        "audio": {"input_audio", "audio_url"},
        "video": {"video_url"},
        "pdf": {"file", "input_file", "document"},
    }[kind]
    if part and part.group(1) in parts:
        return True
    # A strict server whose message schema has no variant for this media part.
    if "did not match any variant of untagged enum" in text and "messagecontent" in text:
        return True
    return bool(
        re.search(
            rf"\b{noun}\b (?:is |are |was |were )?(?:not supported|unsupported|not allowed|rejected|invalid)\b",
            text,
        )
        or re.search(rf"\b(?:unsupported|invalid|rejected) {noun}\b(?![-\w])", text)
        or re.search(rf"\b(?:does not|doesn't|cannot) (?:support|accept|process) {noun}\b", text)
    )


def media_on_wire(kind, probe_body, wire):
    """Verify the generated attachment survived projection, not just its answer.

    Inspect public message-content blocks only, never request text/tool arguments.
    Adapters may translate wrappers (image_url -> source/input_image, file ->
    document/input_file); the exact generated base64 payload must still be there.
    """
    from nooa.unifiedllm.media_inputs import attachment_kind

    def blocks(body):
        messages = body.get("messages", body.get("input", []))
        if not isinstance(messages, list):
            return
        for message in messages:
            content = message.get("content", []) if isinstance(message, dict) else []
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and attachment_kind(block) == kind:
                        yield block

    def payloads(block):
        # Only documented attachment slots; matching in a text notice is not
        # proof that a provider received media.
        nodes = [block]
        for key in ("image_url", "video_url", "input_audio", "source", "file"):
            if isinstance(block.get(key), dict):
                nodes.append(block[key])
        for node in nodes:
            for key in ("url", "data", "image_url", "file_data"):
                value = node.get(key)
                if isinstance(value, str):
                    if value.startswith("data:") and ";base64," in value:
                        value = value.split(";base64,", 1)[1]
                    yield value

    expected_blocks = list(blocks(probe_body))
    actual_blocks = list(blocks(wire))
    if len(expected_blocks) != 1 or len(actual_blocks) != 1:
        return False
    expected = set(payloads(expected_blocks[0]))
    actual = set(payloads(actual_blocks[0]))
    return bool(expected and expected == actual)
