# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media test calls: tiny generated attachments and how to read their results.

Connect checks whether a model accepts an image, audio clip, video or PDF by
sending a small attachment generated here; user files are never used. A
passing check says that one request worked, not that the model will always
handle that media. Generating the fixtures needs no encoder, text-to-speech,
upload, network access or extra dependency.

The module also turns failed media test calls into short, safe explanations
(``exception_reason``, ``reply_failure_reason``, ``explicitly_rejected``) and
checks that the attachment really reached the request that was sent
(``media_on_wire``). "Wire" means the request body as actually sent to the
provider, after the client adapter has translated it.
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
    """Return ``data`` as a base64 ``data:`` URL with the given MIME type."""
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
    """Build the message content for a media test call of ``kind`` on ``style``.

    Returns ``(content, None, None)`` where ``content`` is a text question plus
    one attachment block in the format expected by that API style. If NOOA
    cannot send this kind of media on ``style``, returns ``(None, None,
    reason)`` with the reason the check is skipped, without sending anything.
    """
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
    """Explain why a media test call that raised ``exc`` did not confirm support.

    Uses only the probe ``record`` (timeout, status code, finish reason), the
    HTTP status seen by Connect's response hook, and exception class names.
    Exception messages are never included, because they may contain URLs or
    credentials. Returns a sentence ending in "<kind> support was not
    confirmed".
    """
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
        # The status seen by Connect's response hook is trusted even if the SDK
        # wrapped the error as a connection error. The exception's status_code
        # alone is not, because some SDKs report transport failures as HTTP 500.
        cause = f"Service returned HTTP {observed_status}"
    elif connection_error:
        # Checked before status_code because SDKs can report transport failures
        # as HTTP 500.
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
        # Class names are safe to show; exception messages and URLs are not.
        if cause == "Request failed" and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,79}", error):
            cause += f" ({error})"
    return f"{cause}; {kind} support was not confirmed"


def reply_failure_reason(kind, record):
    """Explain why a completed media test call did not confirm support.

    Based only on fields of the probe ``record`` (finish reason, whether a
    final answer was seen, whether the typed result validated, whether the
    attachment and settings were seen in the sent request). The reply text is
    neither stored nor judged here.
    """
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
    """Return True if ``exc`` is the provider refusing this kind of media.

    Requires HTTP 400/415/422 and an error message that is specifically about
    the media (or the content part used to send it), for example "image input
    is not supported" or "Unsupported content part type: 'file'". A generic
    400, or an error about another parameter (model, max_tokens, ...), does
    not count.
    """
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
    # Only match statements about the media input itself: a model name that
    # contains e.g. "vision" next to an unrelated "invalid" parameter must not
    # count as a rejection.
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
    """Return True if the generated attachment is present in the request sent.

    ``probe_body`` is the request Connect built and ``wire`` the request body
    actually sent. A correct answer is not enough: the attachment must have
    reached the provider. Both bodies must contain exactly one attachment
    block of ``kind`` in their message content, and both must carry the same
    base64 data. Client adapters may rename the wrapper (``image_url`` ->
    ``source``/``input_image``, ``file`` -> ``document``/``input_file``), so only
    the data is compared. Message text and tool arguments are not searched.
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
        # Only look in the documented attachment fields: the data appearing in
        # some text would not prove the provider received it as media.
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
