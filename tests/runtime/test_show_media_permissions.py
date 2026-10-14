# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit current-model permissions at show(), before capture/acknowledgement."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.events import PythonOutput, ResultStatus
from nooa.media import Audio, File, Image, Video
from nooa.runtime.actor import _current_llm_var
from nooa.runtime.media_capture import _media_buffer_var, _MediaBuffer, show
from nooa.runtime.sandbox.cell_core import run_cell_source
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from nooa.unifiedllm.media_inputs import attachment_kind

ALL = ("image", "audio", "video", "pdf")
MEDIA = [
    Image.from_bytes(b"image", media_type="image/png"),
    Audio.from_bytes(b"audio", media_type="audio/wav"),
    Video.from_bytes(b"video", media_type="video/mp4"),
    File.from_bytes(b"pdf", media_type="application/pdf"),
]


class MediaAgent(Agent, llm=FakeLLMClient()):
    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=4)))
    async def inspect(self) -> str:
        """Inspect self.attachment using show()."""
        ...


def response(code):
    return LLMResponse(
        content="",
        finish_reason="tool_calls",
        tool_calls=[
            ToolCall(id="cell", name="execute_python", arguments=json.dumps({"code": code}))
        ],
    )


def blocks(messages):
    return [
        block
        for msg in messages
        if isinstance(msg.get("content"), (list, tuple))
        for block in msg["content"]
        if attachment_kind(block) is not None
    ]


@pytest.mark.parametrize("kind,obj", list(zip(ALL, MEDIA, strict=True)))
@pytest.mark.parametrize("enabled", [False, True])
async def test_codeact_show_next_request_and_events(kind, obj, enabled):
    client = FakeLLMClient(
        [response("show(self.attachment)\nprint('after-show')"), response("return_result('done')")],
        strict_exhaustion=True,
        media_inputs=[kind] if enabled else [],
    )
    agent = MediaAgent(llm=client)
    agent.attachment = obj
    before = (obj.data_url, obj.media_type, dict(obj.vendor_metadata))
    assert await agent.inspect() == "done"
    outputs = [e for e in agent.event_manager.values() if isinstance(e, PythonOutput)]
    output = next(e for e in outputs if e.tool_call_id == "cell")
    outgoing = blocks(client.calls[1].messages)
    if enabled:
        assert output.execution_status is ResultStatus.COMPLETE
        assert "[shown:" in output.stdout and "after-show" in output.stdout
        assert [attachment_kind(b) for b in outgoing] == [kind]
    else:
        assert output.execution_status is ResultStatus.ERROR
        assert f"Cannot show {kind}: {kind} input is not enabled for the current model." in str(
            output
        )
        assert "[shown:" not in output.stdout and "after-show" not in output.stdout
        assert not outgoing
        assert f"media_inputs: [{kind}]" in output.error
        assert f'media_inputs=["{kind}"]' in output.error
        assert "llm_config.yaml" in output.error and "nooa connect" in output.error
        assert "include any other inputs you want to keep" in output.error
        assert "Only enable types your endpoint supports." in output.error
    assert agent.attachment is obj
    assert (obj.data_url, obj.media_type, obj.vendor_metadata) == before
    assert _media_buffer_var.get() is None


@pytest.mark.parametrize("kind,obj", list(zip(ALL, MEDIA, strict=True)))
@pytest.mark.parametrize(
    "client",
    [
        SimpleNamespace(),
        SimpleNamespace(media_inputs=None),
        FakeLLMClient(),
        FakeLLMClient(media_inputs=[]),
    ],
)
async def test_absent_and_empty_current_client_is_text_only(kind, obj, client):
    agent = MediaAgent(llm=FakeLLMClient(media_inputs=ALL))
    agent.attachment = obj
    token = _current_llm_var.set(client)
    try:
        result = await agent.runtime.execute_code("show(self.attachment)")
        assert isinstance(result.error, ValueError)
        assert kind in str(result.error)
        assert not result.images and "[shown:" not in result.stdout
    finally:
        _current_llm_var.reset(token)
    assert _media_buffer_var.get() is None


@pytest.mark.parametrize(
    "obj,kind,allowed",
    [
        (File.from_url("https://example.test/report.pdf"), "pdf", True),
        (File.from_url("https://example.test/report.pdf", media_type=""), "file", False),
        (File.from_bytes(b"x", media_type="application/octet-stream"), "file", False),
        (File.from_bytes(b"x", media_type="application/pdf; charset=utf-8"), "file", False),
        (File.from_bytes(b"x", media_type="application/x-pdf"), "file", False),
        (Image.from_bytes(b"x", media_type="application/pdf"), "file", False),
    ],
)
async def test_exact_pdf_and_unknown_files(obj, kind, allowed):
    agent = MediaAgent(llm=FakeLLMClient(media_inputs=ALL))
    agent.attachment = obj
    result = await agent.runtime.execute_code("show(self.attachment)")
    assert (result.error is None) == allowed
    assert bool(result.images) == allowed
    if allowed:
        assert attachment_kind(result.images[0]) == kind
    else:
        assert isinstance(result.error, ValueError)
        if isinstance(obj, Image):
            assert "known MIME metadata does not match" in str(result.error)
        else:
            assert "file input" in str(result.error)
            assert "application/pdf" in str(result.error)
            assert "media_inputs: [pdf]" in str(result.error)
        assert "media_inputs: [file]" not in str(result.error)
        assert "[shown:" not in result.stdout


@pytest.mark.parametrize("factory", ["pil", "matplotlib"])
async def test_inferred_image(factory):
    if factory == "pil":
        obj = pytest.importorskip("PIL.Image").new("RGB", (2, 2))
    else:
        obj = pytest.importorskip("matplotlib.figure").Figure()
    agent = MediaAgent()
    agent.attachment = obj
    disabled = await agent.runtime.execute_code("show(self.attachment)")
    assert isinstance(disabled.error, ValueError) and not disabled.images
    agent._llm = FakeLLMClient(media_inputs=["image"])
    enabled = await agent.runtime.execute_code("show(self.attachment)")
    assert enabled.error is None and attachment_kind(enabled.images[0]) == "image"


async def test_nested_concurrent_and_failed_cell_cleanup():
    outer = _MediaBuffer(5, media_inputs=("image",))
    token = _media_buffer_var.set(outer)
    agent = MediaAgent(llm=FakeLLMClient(media_inputs=["image"]))
    agent.attachment = MEDIA[0]

    async def run(enabled):
        client_token = _current_llm_var.set(FakeLLMClient(media_inputs=enabled))
        try:
            result = await agent.runtime.execute_code(
                "await asyncio.sleep(0)\nshow(self.attachment)"
            )
            assert _media_buffer_var.get() is outer
            return result
        finally:
            _current_llm_var.reset(client_token)

    try:
        allowed, denied = await asyncio.gather(run(["image"]), run([]))
        assert allowed.error is None and len(allowed.images) == 1
        assert isinstance(denied.error, ValueError) and not denied.images
        assert not outer.blocks
        show(MEDIA[0])
        assert len(outer.blocks) == 1
        recovered = await agent.runtime.execute_code("show(self.attachment)")
        assert recovered.error is None and len(recovered.images) == 1
    finally:
        _media_buffer_var.reset(token)
    assert _media_buffer_var.get() is None


async def test_invalid_object_typeerror_even_textonly_and_cap_does_not_hide_denial():
    agent = MediaAgent()
    invalid = await agent.runtime.execute_code("show(42)")
    assert isinstance(invalid.error, TypeError) and "got int" in str(invalid.error)
    buf = _MediaBuffer(0, media_inputs=())
    token = _media_buffer_var.set(buf)
    try:
        with pytest.raises(ValueError, match="video input"):
            show(MEDIA[2])
        assert not buf.blocks
    finally:
        _media_buffer_var.reset(token)


async def test_switch_model_next_turn_and_retained_history():
    first = FakeLLMClient(
        [response("show(self.attachment)"), response("return_result('done')")],
        media_inputs=["image"],
        strict_exhaustion=True,
    )
    agent = MediaAgent(llm=first)
    agent.attachment = MEDIA[0]
    assert await agent.inspect() == "done"
    history = [(e, e.model_dump()) for e in agent.event_manager.values()]
    second = FakeLLMClient(
        [response("show(self.attachment)"), response("return_result('done')")],
        strict_exhaustion=True,
    )
    agent._llm = second
    assert await agent.inspect() == "done"
    assert not blocks(second.calls[1].messages)
    for event, snapshot in history:
        assert event.model_dump() == snapshot
    assert "[shown:" in str(first.calls[1].messages)


async def test_sandbox_cell_core_permissions_and_cap():
    namespace = {"show": show, "attachment": MEDIA[2]}
    denied = await run_cell_source("show(attachment)", namespace)
    assert isinstance(denied.error, ValueError)
    assert not denied.images and "[shown:" not in denied.stdout
    allowed = await run_cell_source(
        "show(attachment)\nshow(attachment)", namespace, media_inputs=("video",), max_attachments=1
    )
    assert allowed.error is None and len(allowed.images) == 1
    assert "limit reached (1)" in allowed.stdout
    assert _media_buffer_var.get() is None


async def test_call_site_and_decorator_client_permissions():
    default = FakeLLMClient(media_inputs=ALL)
    decorator = FakeLLMClient(
        [response("show(self.attachment)"), response("return_result('done')")],
        strict_exhaustion=True,
    )

    class RoutedAgent(Agent, llm=default):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=4)), llm=decorator)
        async def inspect(self) -> str:
            """Inspect self.attachment."""
            ...

    agent = RoutedAgent()
    agent.attachment = MEDIA[2]
    assert await agent.inspect() == "done"
    assert not blocks(decorator.calls[1].messages)
    override = FakeLLMClient(
        [response("show(self.attachment)"), response("return_result('done')")],
        strict_exhaustion=True,
        media_inputs=["video"],
    )
    assert await agent.inspect(llm=override) == "done"
    assert [attachment_kind(b) for b in blocks(override.calls[1].messages)] == ["video"]
    assert default.call_count == 0 and decorator.call_count == 2
    assert agent._llm is default


@pytest.mark.parametrize("enabled", [False, True])
async def test_prefill_auto_show_obeys_policy(enabled):
    client = FakeLLMClient(
        [response("return_result('done')")],
        strict_exhaustion=True,
        media_inputs=["image"] if enabled else [],
    )

    class PrefillAgent(Agent, llm=client):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2)))
        async def inspect(self, attachment: Image) -> str:
            """Inspect attachment."""
            ...

    agent = PrefillAgent()
    assert await agent.inspect(attachment=MEDIA[0]) == "done"
    outputs = [e for e in agent.event_manager.values() if isinstance(e, PythonOutput)]
    prefill = outputs[0]
    assert bool(prefill.images) == enabled
    assert prefill.execution_status is ResultStatus.COMPLETE
    assert bool(blocks(client.calls[0].messages)) == enabled
    if not enabled:
        assert (
            "Cannot show image: image input is not enabled for the current model." in prefill.stdout
        )
        assert "[shown:" not in prefill.stdout


@pytest.mark.parametrize("enabled", [False, True])
async def test_predict_implicit_attachment_keeps_filter_semantics(enabled):
    from nooa.strategies.predict import PredictStrategy

    client = FakeLLMClient(
        [LLMResponse(content='{"value": "done"}')],
        strict_exhaustion=True,
        media_inputs=["image"] if enabled else [],
    )

    class PredictAgent(Agent, llm=client):
        @strategy(PredictStrategy())
        async def inspect(self, attachment: Image) -> str:
            """Inspect attachment."""
            ...

    agent = PredictAgent()
    assert await agent.inspect(MEDIA[0]) == "done"
    assert bool(blocks(client.calls[0].messages)) == enabled
    # Predict doesn't call show(): implicit attachments remain archived.
    from nooa.events import Task

    task = next(e for e in agent.event_manager.values() if isinstance(e, Task))
    assert len(task.images) == 1


@pytest.mark.sandbox
@pytest.mark.parametrize("enabled", [False, True])
async def test_real_sandbox_raw_show_current_policy(enabled):
    from nooa.runtime.sandbox.config import SandboxConfig

    client = FakeLLMClient(
        [
            response(
                "show(Video.from_bytes(b'clip', media_type='video/mp4'))\nprint('after-show')"
            ),
            response("return_result('done')"),
        ],
        strict_exhaustion=True,
        media_inputs=["video"] if enabled else [],
    )

    class SandboxAgent(Agent, llm=client):
        @strategy(
            CodeActStrategy(
                config=CodeActConfig(
                    execution_backend="sandbox",
                    cell_timeout=15,
                    max_iterations=4,
                    sandbox=SandboxConfig(require=False, network=True, filesystem=False),
                )
            )
        )
        async def inspect(self) -> str:
            """Show a video."""
            ...

    agent = SandboxAgent()
    assert await agent.inspect() == "done"
    output = next(
        e
        for e in agent.event_manager.values()
        if isinstance(e, PythonOutput) and e.tool_call_id == "cell"
    )
    assert bool(output.images) == enabled
    assert (output.execution_status is ResultStatus.COMPLETE) == enabled
    assert bool(blocks(client.calls[1].messages)) == enabled
    if not enabled:
        assert (
            "Cannot show video: video input is not enabled for the current model." in output.error
        )
        assert "[shown:" not in output.stdout and "after-show" not in output.stdout


@pytest.mark.sandbox
async def test_sandbox_executor_per_cell_policy_switch_and_cap():
    from nooa.runtime.sandbox.config import SandboxConfig
    from nooa.runtime.sandbox.executor import SandboxedExecutor

    agent = MediaAgent(llm=FakeLLMClient(media_inputs=ALL))
    executor = SandboxedExecutor(
        agent, SandboxConfig(require=False, network=True, filesystem=False), cell_timeout=15
    )
    try:
        denied = await executor.run_cell(
            "clip = Video.from_bytes(b'clip', media_type='video/mp4')\nshow(clip)"
        )
        assert denied.error is not None and not denied.images
        assert "Cannot show video:" in str(denied.error)
        allowed = await executor.run_cell(
            "show(clip)\nshow(clip)", media_inputs=("video",), max_attachments=1
        )
        assert allowed.error is None and len(allowed.images) == 1
        assert "limit reached (1)" in allowed.stdout
        reset = await executor.run_cell("show(clip)")
        assert reset.error is not None and not reset.images
        assert "[shown:" not in reset.stdout
    finally:
        await executor.aclose()


@pytest.mark.parametrize("backend", [None, pytest.param("sandbox", marks=pytest.mark.sandbox)])
async def test_prefill_denial_keeps_later_text_enabled_media_and_return_doc(backend):
    from pydantic import BaseModel

    class InspectionResult(BaseModel):
        summary: str

    client = FakeLLMClient(
        [response("return_result({'summary': 'done'})")],
        strict_exhaustion=True,
        media_inputs=["audio"],
    )

    from nooa.runtime.sandbox.config import SandboxConfig

    config = CodeActConfig(max_iterations=2)
    if backend:
        config = CodeActConfig(
            max_iterations=2,
            execution_backend=backend,
            sandbox=SandboxConfig(require=False, network=True, filesystem=False),
        )

    class InspectAgent(Agent, llm=client):
        @strategy(CodeActStrategy(config=config))
        async def inspect(self, image: Image, text: str, audio: Audio) -> InspectionResult:
            """Inspect inputs."""
            ...

    agent = InspectAgent()
    result = await agent.inspect(MEDIA[0], "IMPORTANT_TEXT_AFTER_DISABLED_MEDIA", MEDIA[1])
    assert result.summary == "done"
    prefill = next(e for e in agent.event_manager.values() if isinstance(e, PythonOutput))
    assert prefill.execution_status is ResultStatus.COMPLETE and not prefill.error
    assert "IMPORTANT_TEXT_AFTER_DISABLED_MEDIA" in prefill.stdout
    assert "Return type:" in prefill.stdout and "summary" in prefill.stdout
    assert [attachment_kind(b) for b in prefill.images] == ["audio"]
    assert "IMPORTANT_TEXT_AFTER_DISABLED_MEDIA" in str(client.calls[0].messages)
    assert [attachment_kind(b) for b in blocks(client.calls[0].messages)] == ["audio"]
    # Automatic recovery must not change explicit user-cell behavior.
    agent.attachment = MEDIA[0]
    explicit = await agent.runtime.execute_code("show(self.attachment)")
    assert isinstance(explicit.error, ValueError) and not explicit.images


@pytest.mark.parametrize(
    "cls,mime",
    [
        (Image, "application/pdf"),
        (Audio, "application/pdf"),
        (Video, "application/pdf"),
        (Audio, "video/mp4"),
        (Video, "audio/wav"),
    ],
)
@pytest.mark.parametrize("source", ["bytes", "url", "conflicting_data_url"])
async def test_known_python_mime_conflicts_fail_show_and_predict(cls, mime, source):
    from nooa.runtime.media_capture import MediaPermissionError, media_to_content_block
    from nooa.strategies.predict import PredictStrategy

    if source == "bytes":
        obj = cls.from_bytes(b"not-sniffed", media_type=mime)
    elif source == "url":
        obj = cls.from_url("https://example.invalid/media", media_type=mime)
    else:
        obj = cls("data:application/pdf;base64,eA==", f"{cls._modality}/valid")
    with pytest.raises(MediaPermissionError, match="known MIME metadata"):
        media_to_content_block(obj)
    client = FakeLLMClient(media_inputs=ALL)
    agent = MediaAgent(llm=client)
    agent.attachment = obj
    result = await agent.runtime.execute_code("show(self.attachment)")
    assert isinstance(result.error, MediaPermissionError) and not result.images

    class PredictAgent(Agent, llm=client):
        @strategy(PredictStrategy())
        async def inspect(self, attachment) -> str:
            """Inspect attachment."""
            ...

    with pytest.raises(MediaPermissionError):
        await PredictAgent().inspect(obj)
    assert client.call_count == 0


@pytest.mark.parametrize("obj,kind", list(zip(MEDIA, ALL, strict=True)))
async def test_matching_python_mime_predict_show_symmetry(obj, kind):
    from nooa.strategies.predict import PredictStrategy

    client = FakeLLMClient([LLMResponse(content='{"value":"done"}')], media_inputs=[kind])
    agent = MediaAgent(llm=client)
    agent.attachment = obj
    captured = await agent.runtime.execute_code("show(self.attachment)")
    assert captured.error is None and attachment_kind(captured.images[0]) == kind

    class PredictAgent(Agent, llm=client):
        @strategy(PredictStrategy())
        async def inspect(self, attachment) -> str:
            """Inspect attachment."""
            ...

    assert await PredictAgent().inspect(obj) == "done"
    assert [attachment_kind(b) for b in blocks(client.calls[0].messages)] == [kind]


async def test_unrelated_auto_show_error_not_swallowed(monkeypatch):
    from nooa.runtime import media_capture

    def broken(media):
        raise RuntimeError("unrelated conversion failure")

    monkeypatch.setattr(media_capture, "media_to_content_block", broken)
    client = FakeLLMClient([response("return_result('done')")], media_inputs=["image"])

    class InspectAgent(Agent, llm=client):
        @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=2)))
        async def inspect(self, image: Image, text: str) -> str:
            """Inspect inputs."""
            ...

    agent = InspectAgent()
    assert await agent.inspect(MEDIA[0], "LATER_TEXT") == "done"
    prefill = next(e for e in agent.event_manager.values() if isinstance(e, PythonOutput))
    assert prefill.execution_status is ResultStatus.ERROR
    assert "unrelated conversion failure" in prefill.error
    assert "LATER_TEXT" not in prefill.stdout


@pytest.mark.sandbox
async def test_worker_rejects_known_mime_conflict_before_capture():
    from nooa.runtime.sandbox.config import SandboxConfig
    from nooa.runtime.sandbox.executor import SandboxedExecutor

    agent = MediaAgent(llm=FakeLLMClient(media_inputs=ALL))
    executor = SandboxedExecutor(
        agent, SandboxConfig(require=False, network=True, filesystem=False), cell_timeout=15
    )
    try:
        for source in (
            "Audio.from_bytes(b'pdf', media_type='application/pdf')",
            "Video.from_url('https://example.invalid/video', media_type='application/pdf')",
        ):
            result = await executor.run_cell(f"show({source})", media_inputs=ALL)
            assert result.error is not None and not result.images
            assert "known MIME metadata does not match" in str(result.error)
            assert "[shown:" not in result.stdout
    finally:
        await executor.aclose()
