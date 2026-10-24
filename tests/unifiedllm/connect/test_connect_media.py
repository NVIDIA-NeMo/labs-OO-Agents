# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated samples, fake HTTP only: no live calls or private configuration."""

import base64
import hashlib
import io
import json
import shutil
import struct
import subprocess
import wave
import zlib
from contextlib import aclosing
from dataclasses import replace

import httpx
import pytest

from nooa.media_policy import unsupported_input_reason
from nooa.unifiedllm import connect
from nooa.unifiedllm.connect._media import pdf_fixture, png_fixture, video_fixture, wav_fixture
from tests.unifiedllm.connect.connect_http import mock_http, response_body


def plan(style="chat", **kwargs):
    return connect.plan(
        "local", "model", style, "https://api.test/v1", "", reply_tokens=1000, **kwargs
    )


def test_generated_fixtures_have_valid_structure_and_independent_answers():
    png = png_fixture()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    offset = 8
    chunks = {}
    while offset < len(png):
        length = struct.unpack(">I", png[offset : offset + 4])[0]
        kind, data = png[offset + 4 : offset + 8], png[offset + 8 : offset + 8 + length]
        assert (
            zlib.crc32(kind + data)
            == struct.unpack(">I", png[offset + 8 + length : offset + 12 + length])[0]
        )
        chunks[kind] = data
        offset += length + 12
    assert struct.unpack(">II", chunks[b"IHDR"][:8]) == (32, 32)
    assert zlib.decompress(chunks[b"IDAT"]) == (b"\0" + b"\xff\0\0" * 32) * 32
    with wave.open(io.BytesIO(wav_fixture()), "rb") as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 8000)
        frames = audio.readframes(audio.getnframes())
    samples = struct.unpack(f"<{len(frames) // 2}h", frames)
    # Count independent sound windows, not the fixture generator's expected value.
    windows = [any(samples[i : i + 800]) for i in range(0, len(samples), 800)]
    assert (
        sum(on and not previous for previous, on in zip([False, *windows], windows, strict=False))
        == 2
    )
    pdf = pdf_fixture()
    assert pdf.startswith(b"%PDF-1.4") and pdf.endswith(b"%%EOF\n")
    assert b"(MAPLE) Tj" in pdf and b"/BaseFont /Helvetica" in pdf
    start = int(pdf.split(b"startxref\n")[1].splitlines()[0])
    assert pdf[start:].startswith(b"xref\n0 6\n")
    for index, line in enumerate(pdf[start:].splitlines()[3:8], 1):
        assert pdf[int(line[:10]) :].startswith(f"{index} 0 obj".encode())


@pytest.mark.parametrize(
    "style,kind,answer",
    [
        (style, kind, answer)
        for style in ["chat", "responses", "anthropic"]
        for kind, answer in [("image", '{"value":true}'), ("pdf", '{"value":true}')]
    ]
    + [("chat", "audio", '{"value":true}'), ("chat", "video", '{"value":true}')],
)
async def test_media_reaches_runtime_wire_and_only_validated_access_passes(
    monkeypatch, style, kind, answer
):
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        text = '{"value":323}' if len(bodies) == 1 else answer
        return httpx.Response(200, json=response_body(style, text))

    mock_http(monkeypatch, handle)
    proposal = plan(style, test_media_inputs=[kind])
    assert proposal.entry["media_inputs"] == []
    assert proposal.token_estimate == 2 * 1512 + 5096
    result = await connect.check_stage(proposal, "media", api_key="fake")
    assert len(bodies) == 2
    messages = bodies[-1].get("messages", bodies[-1].get("input"))
    content = next(m["content"] for m in messages if isinstance(m.get("content"), list))
    assert any(block.get("type") != "text" for block in content)
    assert "was removed here" not in json.dumps(content).lower()
    assert bodies[-1].get("max_tokens", bodies[-1].get("max_output_tokens")) == 1000
    assert "media_inputs" not in bodies[-1]
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert record["outcome"] == "passed", record
    assert record.get("access_reported") and record["settings_sent"]
    assert result.entry["media_inputs"] == []  # Facts do not mutate declarations in library.
    assert proposal.entry["media_inputs"] == []
    assert "base64" not in json.dumps(connect.public_record(record))
    if kind == "video":
        block = content[-1]
        assert block["type"] == "video_url"
        url = block["video_url"]["url"]
        assert url.startswith("data:video/mp4;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == video_fixture()
        assert "was removed here" not in content[0]["text"]
    if kind == "audio":
        assert base64.b64decode(content[-1]["input_audio"]["data"]).startswith(b"RIFF")


@pytest.mark.parametrize(
    "text,finish,cause",
    [
        ("", "stop", "No final answer returned"),
        ('{"value":1}', "stop", "Could not read the answer"),
        ('{"value":true}', "length", "Reply reached the token limit"),
        ('{"value":true}', "content_filter", "Reply was content-filtered"),
    ],
)
@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
async def test_invalid_empty_truncated_media_is_inconclusive(
    monkeypatch, kind, text, finish, cause
):
    calls = []

    def handle(request):
        calls.append(request)
        body = response_body("chat", '{"value":323}' if len(calls) == 1 else text)
        if len(calls) > 1:
            body["choices"][0]["finish_reason"] = finish
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    assert len(calls) == 2
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert record["outcome"] == "inconclusive"
    assert record["reason"] == f"{cause}; {kind} support was not confirmed"
    assert result.entry["media_inputs"] == []


@pytest.mark.parametrize(
    "status,message,outcome",
    [
        (400, "invalid image input: not supported", "rejected"),
        (400, "invalid model", "inconclusive"),
        (401, "invalid image key", "inconclusive"),
        (503, "private network error", "inconclusive"),
        (422, "request rejected", "inconclusive"),
    ],
)
@pytest.mark.parametrize("kind", ["image", "video"])
async def test_media_rejection_requires_explicit_media_provider_evidence(
    monkeypatch, kind, status, message, outcome
):
    message = message.replace("image", kind)
    calls = []

    def handle(request):
        calls.append(request)
        return (
            httpx.Response(200, json=response_body("chat"))
            if len(calls) == 1
            else httpx.Response(
                status, json={"error": {"message": message, "type": "invalid_request_error"}}
            )
        )

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert len(calls) == 2
    assert record["outcome"] == outcome, record
    assert message not in repr(result)
    assert record["reason"] == (
        "Provider explicitly rejected this media request"
        if outcome == "rejected"
        else f"Service returned HTTP {status}; {kind} support was not confirmed"
    )


@pytest.mark.parametrize("baseline", ["", "wrong"])
@pytest.mark.parametrize("kind", ["image", "video"])
async def test_empty_or_wrong_text_baseline_never_retries_or_tests_media(
    monkeypatch, kind, baseline
):
    calls = []
    mock_http(
        monkeypatch,
        lambda request: (
            calls.append(request) or httpx.Response(200, json=response_body("chat", baseline))
        ),
    )
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    assert len(calls) == 1
    assert result.entry["provenance"]["probes"][f"media:{kind}"]["outcome"] == "not_probed"


@pytest.mark.parametrize(
    "approval,budget,count",
    [
        ("none", 10000, 0),
        ("minimal", 10000, 1),
        ("all", 1512, 1),
        ("all", 6607, 1),
        ("all", 6608, 2),
    ],
)
@pytest.mark.parametrize("kind", ["image", "video"])
async def test_media_shares_approval_and_exact_budget(monkeypatch, kind, approval, budget, count):
    calls = []
    mock_http(
        monkeypatch,
        lambda request: (
            calls.append(request)
            or httpx.Response(
                200,
                json=response_body(
                    "chat",
                    '{"value":323}' if len(calls) == 1 else '{"value":true}',
                ),
            )
        ),
    )
    proposal = plan(test_media_inputs=[kind], budget_tokens=budget)
    proposal = replace(proposal, probes=tuple(p for p in proposal.probes if p.name != "tools"))
    result = await connect.run(proposal, approved=approval, api_key="fake")
    assert len(calls) == count
    assert result.entry["media_inputs"] == []


@pytest.mark.parametrize("budget,count", [(1, 0), (12000, 3)])
async def test_all_media_formats_keep_budget_skips_and_saved_reply_cap(monkeypatch, budget, count):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json=response_body("chat", '{"value":323}' if len(calls) <= 2 else '{"value":true}'),
        )

    mock_http(monkeypatch, handle)
    proposal = plan(
        test_media_inputs=["image", "audio", "video", "pdf"],
        budget_tokens=budget,
        session_checks=False,
    )
    result = await connect.run(proposal, approved="all", api_key="fake")
    assert len(calls) == count
    assert proposal.budget_tokens == budget
    assert result.entry["media_inputs"] == []
    assert all(body["max_tokens"] == 1000 for body in calls)
    records = result.entry["provenance"]["probes"]
    assert records["media:video"] == {
        "outcome": "not_probed",
        "reason": "budget exhausted"
        if budget == 12000
        else "Text baseline was not tested: approved budget exhausted; media not tested",
    }
    if budget == 12000:
        assert records["media:image"]["outcome"] == "passed"
        for kind in ["audio", "video", "pdf"]:
            assert records[f"media:{kind}"] == {
                "outcome": "not_probed",
                "reason": "budget exhausted",
            }


@pytest.mark.parametrize(
    "style,kind",
    [
        ("responses", "video"),
        ("anthropic", "video"),
        ("responses", "audio"),
        ("anthropic", "audio"),
    ],
)
async def test_unsupported_portable_probe_is_honestly_not_tested(monkeypatch, style, kind):
    calls = []
    mock_http(
        monkeypatch,
        lambda request: calls.append(request) or httpx.Response(200, json=response_body(style)),
    )
    result = await connect.check_stage(
        plan(style, test_media_inputs=[kind]), "media", api_key="fake"
    )
    assert len(calls) == 1
    assert result.entry["provenance"]["probes"][f"media:{kind}"] == {
        "outcome": "not_probed",
        "reason": unsupported_input_reason(style, [kind]),
        "implementation_supported": False,
    }


async def test_abort_before_media_send_closes_iterator_without_more_requests(monkeypatch):
    calls = []
    mock_http(
        monkeypatch,
        lambda request: calls.append(request) or httpx.Response(200, json=response_body("chat")),
    )
    proposal = plan(test_media_inputs=["image"])
    proposal = replace(proposal, probes=tuple(p for p in proposal.probes if p.name != "tools"))
    async with aclosing(connect.run_steps(proposal, approved="all", api_key="fake")) as events:
        async for event in events:
            if (
                isinstance(event, connect.ProbeUpdate)
                and event.name == "media:image"
                and event.outcome["outcome"] == "running"
            ):
                break
    assert len(calls) == 1


@pytest.mark.parametrize(
    "kind,message,param",
    [
        ("pdf", "Invalid model: document-reader does not exist", "model"),
        ("image", "Invalid model: vision-model does not exist", None),
        ("pdf", "Invalid API parameter include; file input is supported", "include"),
        ("pdf", "Invalid API parameter include; file input is supported", None),
    ],
)
def test_unrelated_provider_errors_with_media_nouns_are_inconclusive(kind, message, param):
    from nooa.unifiedllm.connect._media import explicitly_rejected

    class Error(Exception):
        status_code = 400
        body = {"error": {"message": message, "param": param}}

    assert not explicitly_rejected(Error(message), kind)


def test_media_wire_evidence_rejects_dropped_attachment_and_text_echo():
    from nooa.unifiedllm.connect._media import media_on_wire

    proposal = plan(test_media_inputs=["image"])
    probe = proposal.probes[-1]
    url = probe.body["messages"][0]["content"][-1]["image_url"]["url"]
    assert media_on_wire("image", probe.body, probe.body)
    assert not media_on_wire(
        "image",
        probe.body,
        {"messages": [{"role": "user", "content": [{"type": "text", "text": url}]}]},
    )
    assert not media_on_wire(
        "image",
        probe.body,
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,different"},
                        }
                    ],
                }
            ]
        },
    )


@pytest.mark.parametrize("kind", ["image", "video"])
async def test_adapter_dropping_media_even_with_expected_answer_is_inconclusive(monkeypatch, kind):
    from nooa.unifiedllm.unifiedllm import CompletionClient

    original = CompletionClient._prepare_media_messages

    def drop(self, messages, config):
        result = original(self, messages, config)
        return [
            {
                **message,
                "content": [block for block in message["content"] if block.get("type") == "text"],
            }
            if isinstance(message.get("content"), list)
            else message
            for message in result
        ]

    monkeypatch.setattr(CompletionClient, "_prepare_media_messages", drop)
    calls = []
    mock_http(
        monkeypatch,
        lambda request: (
            calls.append(request)
            or httpx.Response(
                200,
                json=response_body(
                    "chat",
                    '{"value":323}' if len(calls) == 1 else '{"value":true}',
                ),
            )
        ),
    )
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    assert len(calls) == 2
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert record.get("access_reported") and not record["settings_sent"]
    assert record["outcome"] == "inconclusive"
    assert record["reason"] == (
        "Generated attachment or output settings were missing from the observed request; "
        f"{kind} support was not confirmed"
    )


async def test_reported_usage_excess_skips_next_media_without_budget_expansion(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        body = response_body("chat", '{"value":323}' if len(calls) == 1 else '{"value":true}')
        if len(calls) == 2:
            body["usage"] = {
                "prompt_tokens": 10000,
                "completion_tokens": 9000,
                "total_tokens": 19000,
            }
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    proposal = plan(test_media_inputs=["image", "pdf"], budget_tokens=12000)
    result = await connect.check_stage(proposal, "media", api_key="fake")
    assert len(calls) == 2
    assert proposal.budget_tokens == 12000
    assert result.entry["provenance"]["probes"]["media:pdf"]["reason"] == "budget exhausted"
    assert result.entry["provenance"]["tokens_charged_to_budget"] == 1512 + 19000


async def test_network_failure_media_inconclusive_no_retry(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json=response_body("chat"))
        raise httpx.ConnectError("fake private connection detail", request=request)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=["image"]), "media", api_key="fake")
    assert len(calls) == 2
    assert result.entry["provenance"]["probes"]["media:image"]["outcome"] == "inconclusive"
    assert "fake private connection detail" not in repr(result)


@pytest.mark.parametrize("error_type", [httpx.ConnectError, httpx.ReadTimeout])
async def test_media_errors_keep_safe_reasons_and_unknown_support(monkeypatch, error_type):
    calls = []
    secret = "sk-private-key"
    endpoint = "https://private.test/v1?api_key=sk-private-key"

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(200, json=response_body("chat"))
        raise error_type(f"Failure at {endpoint} with {secret}", request=request)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=["image"]), "media", api_key=secret)
    record = result.entry["provenance"]["probes"]["media:image"]
    assert len(calls) == 2
    assert record["outcome"] == "inconclusive"
    assert result.entry["media_inputs"] == []
    cause = (
        "Timed out; server receipt and completion are unknown"
        if error_type is httpx.ReadTimeout
        else "Could not connect to the service"
    )
    assert record["reason"] == f"{cause}; image support was not confirmed", record
    public = json.dumps(connect.public_record(record))
    assert secret not in public and endpoint not in public and "https://" not in public


@pytest.mark.parametrize("status", [400, 401, 429, 503])
async def test_media_http_reason_never_exposes_provider_body(monkeypatch, status):
    calls = []
    secret = "sk-private-key"
    endpoint = "https://private.test/v1?api_key=sk-private-key"

    def handle(request):
        calls.append(request)
        return (
            httpx.Response(200, json=response_body("chat"))
            if len(calls) == 1
            else httpx.Response(
                status, json={"error": {"message": f"invalid key {secret}: {endpoint}"}}
            )
        )

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=["image"]), "media", api_key=secret)
    record = result.entry["provenance"]["probes"]["media:image"]
    assert len(calls) == 2
    assert record["outcome"] == "inconclusive"
    assert record["reason"] == f"Service returned HTTP {status}; image support was not confirmed"
    public = json.dumps(connect.public_record(record))
    assert secret not in public and endpoint not in public and "https://" not in public


@pytest.mark.parametrize("kind", ["image", "video"])
async def test_media_unobserved_request_retains_unknown_classification(monkeypatch, kind):
    original = connect._run_probe

    async def unobserved(alias, entry, probe, api_key):
        response, settings_sent, transport = await original(alias, entry, probe, api_key)
        return response, None if probe.name.startswith("media:") else settings_sent, transport

    monkeypatch.setattr(connect, "_run_probe", unobserved)
    calls = []
    mock_http(
        monkeypatch,
        lambda request: (
            calls.append(request)
            or httpx.Response(
                200,
                json=response_body(
                    "chat",
                    '{"value":323}' if len(calls) == 1 else '{"value":true}',
                ),
            )
        ),
    )
    result = await connect.check_stage(plan(test_media_inputs=[kind]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert len(calls) == 2
    assert record["outcome"] == "inconclusive"
    assert record.get("access_reported") and record["settings_sent"] is None
    assert record["reason"] == (
        "Actual request was not observed; media and output settings could not be verified; "
        f"{kind} support was not confirmed"
    )


@pytest.mark.parametrize(
    "content,finish,correct,settings,cause",
    [
        ("", "length", False, None, "Reply reached the token limit"),
        ("RED", "content_filter", True, True, "Reply was content-filtered"),
        ("", "error", False, True, "Reply ended with an error"),
    ],
)
def test_media_reason_uses_runtime_finish_reason_before_answer(
    content, finish, correct, settings, cause
):
    from nooa.unifiedllm.connect._media import reply_failure_reason

    assert reply_failure_reason(
        "image",
        {
            "finish_reason": finish,
            "final_answer_observed": bool(content),
            "typed_result_validated": correct,
            "settings_sent": settings,
        },
    ) == (f"{cause}; image support was not confirmed")


def test_bundled_video_container_and_integrity():
    data = video_fixture()
    assert len(data) == 1675 < 50000
    assert hashlib.sha256(data).hexdigest() == (
        "dc8c43009ecf3887c4d9d4b5dd8be8adfefb0517e85d12b4aeeb3d0e59eb8c90"
    )
    boxes = {}
    offset = 0
    while offset < len(data):
        length, kind = struct.unpack(">I4s", data[offset : offset + 8])
        assert length >= 8 and offset + length <= len(data)
        boxes[kind] = data[offset + 8 : offset + length]
        offset += length
    assert offset == len(data)
    assert boxes[b"ftyp"].startswith(b"isom")
    assert boxes[b"mdat"] and b"avc1" in boxes[b"moov"]
    assert data.index(b"moov") < data.index(b"mdat")  # faststart


def test_video_independently_decodes_ordered_frames(tmp_path):
    # Development-only validation: normal Connect needs neither tool.
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("Independent H.264 decode requires development FFmpeg/ffprobe")
    path = tmp_path / "sample.mp4"
    path.write_bytes(video_fixture())

    def run(args):
        return subprocess.run(args, check=True, capture_output=True, timeout=20).stdout

    metadata = json.loads(
        run(
            [
                ffprobe,
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ]
        )
    )
    assert len(metadata["streams"]) == 1  # no audio or second visual stream
    stream = metadata["streams"][0]
    assert (stream["codec_name"], stream["pix_fmt"], stream["width"], stream["height"]) == (
        "h264",
        "yuv420p",
        64,
        64,
    )
    assert stream["r_frame_rate"] == "4/1"
    assert int(stream["nb_read_frames"]) == 4
    assert float(stream["duration"]) == 1.0
    assert float(metadata["format"]["duration"]) == 1.0
    raw = run(
        [
            ffmpeg,
            "-v",
            "error",
            "-i",
            str(path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "pipe:1",
        ]
    )
    assert len(raw) == 4 * 64 * 64 * 3
    colors = []
    for index in range(4):
        frame = raw[index * 12288 : (index + 1) * 12288]
        means = [sum(frame[channel::3]) / 4096 for channel in range(3)]
        dominant = max(range(3), key=lambda channel: means[channel])
        assert means[dominant] > 230
        assert all(means[c] < 20 for c in range(3) if c != dominant)
        colors.append(["RED", "GREEN", "BLUE"][dominant])
    assert colors == ["RED", "RED", "BLUE", "BLUE"]
    assert raw[:12288] != raw[-12288:]  # genuinely distinct frames, not still images


def test_video_wire_evidence_requires_video_slot_not_text_or_images():
    from nooa.unifiedllm.connect._media import media_on_wire

    probe = plan(test_media_inputs=["video"]).probes[-1]
    url = probe.body["messages"][0]["content"][-1]["video_url"]["url"]
    assert media_on_wire("video", probe.body, probe.body)
    for block in [
        {"type": "text", "text": url},
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,different"}},
    ]:
        wire = {"messages": [{"role": "user", "content": [block]}]}
        assert not media_on_wire("video", probe.body, wire)


@pytest.mark.parametrize("unfinished_at", ["baseline", "video"])
async def test_unfinished_tool_turn_cannot_authorize_or_pass_video(monkeypatch, unfinished_at):
    calls = []

    def handle(request):
        calls.append(json.loads(request.content))
        body = response_body("chat", '{"value":323}' if len(calls) == 1 else '{"value":true}')
        if len(calls) == (1 if unfinished_at == "baseline" else 2):
            body["choices"][0]["finish_reason"] = "tool_calls"
            body["choices"][0]["message"]["tool_calls"] = [
                {
                    "id": "pending",
                    "type": "function",
                    "function": {"name": "unfinished", "arguments": "{}"},
                }
            ]
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    result = await connect.check_stage(plan(test_media_inputs=["video"]), "media", api_key="fake")
    record = result.entry["provenance"]["probes"]["media:video"]
    if unfinished_at == "baseline":
        assert len(calls) == 1
        assert record == {
            "outcome": "not_probed",
            "reason": "Text answer check failed: no validated integer response was available; media not tested",
        }
    else:
        assert len(calls) == 2
        assert not record["typed_result_validated"] and record["settings_sent"]
        assert record["outcome"] == "inconclusive"
        assert (
            record["reason"]
            == "Reply was not a completed final answer; video support was not confirmed"
        )


@pytest.mark.parametrize("style", ["chat", "responses", "anthropic"])
@pytest.mark.parametrize("kind", ["image", "pdf"])
@pytest.mark.parametrize("value", [True, False, "true"])
async def test_native_access_adapter_contract_and_parse_failure_usage(
    monkeypatch, style, kind, value
):
    from nooa.unifiedllm.connect._media import media_on_wire

    sent = []

    def handle(request):
        sent.append(json.loads(request.content))
        text = json.dumps({"value": 323 if len(sent) == 1 else value})
        body = response_body(style, text)
        if len(sent) == 2:
            body["usage"] = (
                {"prompt_tokens": 123, "completion_tokens": 7, "total_tokens": 130}
                if style == "chat"
                else {"input_tokens": 123, "output_tokens": 7}
            )
        return httpx.Response(200, json=body)

    mock_http(monkeypatch, handle)
    proposal = plan(style, test_media_inputs=[kind])
    result = await connect.check_stage(proposal, "media", api_key="fake")
    record = result.entry["provenance"]["probes"][f"media:{kind}"]
    assert len(sent) == 2
    assert record["request_accepted"] is True
    assert record["attachment_observed"] is True and record["output_settings_observed"] is True
    assert record["finish_reason"] == "stop" and record["final_answer_observed"] is True
    assert record["reported_tokens"] == 130 and record["output_tokens"] == 7
    assert record["outcome"] == (
        "inconclusive" if type(value) is not bool else "passed" if value else "unavailable"
    )
    assert record["typed_result_validated"] is (type(value) is bool)
    assert record.get("access_reported") is (value if type(value) is bool else None)
    assert media_on_wire(kind, proposal.probes[-1].body, sent[-1])
    messages = sent[-1].get("messages", sent[-1].get("input"))
    blocks = [
        b
        for msg in messages
        if isinstance(msg.get("content"), list)
        for b in msg["content"]
        if b.get("type") not in {"text", "input_text"}
    ]
    assert len(blocks) == 1
    block = blocks[0]
    if style == "anthropic":
        payload = block["source"]["data"]
        assert block["source"]["media_type"] == (
            "image/png" if kind == "image" else "application/pdf"
        )
    elif style == "responses":
        payload = block["image_url" if kind == "image" else "file_data"].split(",", 1)[1]
    else:
        payload = (
            block["image_url"]["url"] if kind == "image" else block["file"]["file_data"]
        ).split(",", 1)[1]
        if kind == "pdf":
            assert (
                block["file"]["filename"] == "attachment"
            )  # native collector, not plan's probe.pdf
    assert base64.b64decode(payload) == (png_fixture() if kind == "image" else pdf_fixture())
    public = json.dumps(connect.public_record(record))
    assert "base64" not in public and "answer_correct" not in public


@pytest.mark.parametrize("kind", ["image", "audio", "video", "pdf"])
def test_wire_evidence_rejects_duplicate_original_attachments(kind):
    from copy import deepcopy

    from nooa.unifiedllm.connect._media import media_on_wire

    body = plan(test_media_inputs=[kind]).probes[-1].body
    duplicate = deepcopy(body)
    duplicate["messages"][0]["content"].append(deepcopy(body["messages"][0]["content"][-1]))
    assert media_on_wire(kind, body, body)
    assert not media_on_wire(kind, body, duplicate)


@pytest.mark.parametrize(
    "message,kind",
    [
        ("nvcf/llama-70b-instruct is not a multimodal model", "image"),
        ("nvcf/llama-70b-instruct is not a multimodal model", "audio"),
        ("Unsupported chat content part type: 'file'. Supported types: image_url, text", "pdf"),
        ("Unsupported chat content part type: 'video_url'.", "video"),
    ],
)
def test_text_only_server_refusals_count_as_rejected(message, kind):
    """Hub/vLLM refusals for media on text-only models are explicit, not inconclusive."""
    from nooa.unifiedllm.connect._media import explicitly_rejected

    class Error(Exception):
        status_code = 400
        body = {"error": {"message": message}}

    assert explicitly_rejected(Error(message), kind)


def test_unrelated_content_part_rejection_does_not_reject_other_formats():
    from nooa.unifiedllm.connect._media import explicitly_rejected

    class Error(Exception):
        status_code = 400
        body = {"error": {"message": "Unsupported chat content part type: 'file'."}}

    assert not explicitly_rejected(Error("x"), "image")


@pytest.mark.parametrize(
    "message,kind",
    [
        ("Unsupported media type: audio_url. Supported types: image_url", "audio"),
        ('messages[1]: unsupported content part type \\"input_audio\\"', "audio"),
        ("model deepseek does not support video inputs", "video"),
        (
            "Failed to deserialize the JSON body into the target type: data did not match any "
            "variant of untagged enum ChatCompletionRequestUserMessageContent at line 1",
            "pdf",
        ),
    ],
)
def test_more_hub_media_refusal_phrasings(message, kind):
    from nooa.unifiedllm.connect._media import explicitly_rejected

    class Error(Exception):
        status_code = 400
        body = {"error": {"message": message}}

    assert explicitly_rejected(Error(message), kind)


def test_internal_errors_stay_inconclusive():
    from nooa.unifiedllm.connect._media import explicitly_rejected

    class Error(Exception):
        status_code = 400
        body = {"error": {"message": "an internal error occurred"}}

    assert not explicitly_rejected(Error("x"), "image")
