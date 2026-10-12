# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shipped guidance discovery and synthetic media through Atom's Python cells."""

import base64
import io
import os
import re
from importlib.metadata import entry_points
from inspect import cleandoc

import pytest
from nooa_atom.agent.agent import AtomAgent
from nooa_atom.skills.multimodal import Multimodal
from test_experimental_agent import python_cell

from nooa import strategy
from nooa.config import CodeActConfig
from nooa.events import PythonOutput, ResultStatus
from nooa.interactive import Done
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.runtime.sandbox.guards import probe_capabilities
from nooa.strategies import CodeActV2
from nooa.unifiedllm import FakeLLMClient


@pytest.fixture(autouse=True)
def isolated_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    # Avoid framework auto-tracing endpoint discovery in these local-only tests.
    monkeypatch.setattr("nooa.tracing._default_exporters", lambda: None)


async def test_installed_skill_is_discoverable_and_guidance_is_on_demand(tmp_path):
    [point] = [ep for ep in entry_points(group="nooa.skills") if ep.name == "nemo.multimodal"]
    assert point.value == "nooa_atom.skills.multimodal:Multimodal"
    assert point.load() is Multimodal
    llm = FakeLLMClient(
        [python_cell("return_result(Done(explanation='done'))", "done")], strict_exhaustion=True
    )
    agent = AtomAgent(llm=llm, cwd=tmp_path, libs_dir=tmp_path / "libs")
    try:
        assert "nemo.multimodal" in agent.skills.loaded()
        assert "nemo.multimodal" not in agent.skills.activated()
        for query in ("multimodal", "screenshots", "charts", "audio", "video", "files"):
            assert "multimodal (code, available)" in agent.skills.search(query)
        await agent.handle({"user_messages": ["finish without media"]})
        guide_opening = cleandoc(Multimodal.__doc__).splitlines()[0]
        assert guide_opening not in str(llm.calls[0].messages)
        assert await agent.skills.activate(["multimodal"]) == (
            "multimodal: active; use self.multimodal"
        )
        guide = agent.skills.doc("multimodal")
        assert guide_opening in guide
        assert agent.multimodal.context_block is None
        assert "multimodal" in agent.skills.status().splitlines()[0]
        assert await agent.skills.deactivate(["multimodal"]) == "multimodal: inactive"
    finally:
        await agent.aclose()


def test_guidance_matches_runtime_contract_and_limits():
    guide = cleandoc(Multimodal.__doc__)
    # Bound prompt cost without pinning exact wording or line wrapping.
    assert len(guide.splitlines()) <= 80
    assert len(guide.split()) <= 650
    assert guide.startswith("Use model-native multimodal capability to inspect")
    assert "self.skills." not in guide  # Already-reading models need no activation recipe.
    assert "self.multimodal.from_file(" in guide
    assert "self.multimodal.show(" in guide

    # Check actionable processing and safety contracts, not implementation jargon.
    normalized = " ".join(guide.split()).lower()
    for contract in (
        r"screenshots:.*crop.*preserve readable text",
        r"charts:.*underlying data.*exact values",
        r"audio:.*short relevant excerpt",
        r"video:.*relevant segment or timestamped frames",
        r"pdfs:.*relevant pages",
        r"model/provider must support the format",
        r"do not transcribe, extract, capture screens, browse or upload",
        r"inspect.*next model step.*do not complete.*return_result.*same cell",
        r"in-process nooa python cell.*sandbox.*raise before file access/conversion",
        r"relative paths.*current shell cwd",
        r"ambiguous/unknown extensions.*explicit media_type mime",
        r"10 mib cap.*max_bytes applies only to paths",
        r"existing-object size is the caller's responsibility",
        r"configured model service.*events/traces",
        r"untrusted content",
        r"not user/pool delivery",
    ):
        assert re.search(contract, normalized), contract


class SandboxedMediaAtom(AtomAgent):
    @strategy(
        CodeActV2(
            config=CodeActConfig(
                execution_backend="sandbox",
                cell_timeout=30,
                # Confine network; filesystem confinement is exercised by framework tests.
                sandbox=SandboxConfig(filesystem=False, network=False),
            )
        )
    )
    async def handle(self, notification: dict) -> Done:
        """Execute the scripted synthetic-media inspection."""
        ...


@pytest.mark.parametrize(
    "agent_type",
    [AtomAgent, pytest.param(SandboxedMediaAtom, marks=pytest.mark.sandbox)],
)
async def test_show_and_chart_examples_reach_next_model_context(tmp_path, agent_type):
    if agent_type is SandboxedMediaAtom and not probe_capabilities().seccomp:
        pytest.skip("network confinement requires seccomp")
    # Real, tiny PNG/WAV; no user files, remote fetches or inference.
    code = r"""
from nooa import Image, Audio, Video, File
from nooa.runtime.media_capture import show
from PIL import Image as PILImage
from matplotlib.figure import Figure
import io
import wave
import os
print("cell_pid", os.getpid())
png = io.BytesIO()
PILImage.new("RGB", (8, 6), "red").save(png, format="PNG")
show(Image.from_bytes(png.getvalue(), media_type="image/png"))
show(PILImage.new("RGB", (8, 6), "blue"))
fig = Figure(figsize=(1, 1), dpi=50)
fig.subplots().plot([0, 1, 2], [0, 1, 4])
show(fig)
wav = io.BytesIO()
with wave.open(wav, "wb") as clip:
    clip.setnchannels(1)
    clip.setsampwidth(2)
    clip.setframerate(8000)
    clip.writeframes(b"\x00\x00" * 80)
show(Audio.from_bytes(wav.getvalue(), media_type="audio/wav"))
show(File.from_bytes(b"%PDF-1.4\n%%EOF", media_type="application/pdf"))
"""
    llm = FakeLLMClient(
        [
            python_cell(code, "media"),
            python_cell(
                "show(Video.from_bytes(b'synthetic-container', media_type='video/mp4'))",
                "video",
            ),
            python_cell("return_result(Done(explanation='captured'))", "done"),
        ],
        strict_exhaustion=True,
    )
    agent = agent_type(llm=llm, cwd=tmp_path, libs_dir=tmp_path / "libs")
    try:
        await agent.skills.activate(["multimodal"])
        assert await agent.handle({"user_messages": ["inspect synthetic media"]}) == Done(
            explanation="captured"
        )
        output = next(
            event
            for event in agent.event_manager.values()
            if isinstance(event, PythonOutput) and event.tool_call_id == "media"
        )
        assert output.execution_status == ResultStatus.COMPLETE, output.error
        assert [block["type"] for block in output.images] == [
            "image_url",
            "image_url",
            "image_url",
            "input_audio",
            "file",
        ]
        from PIL import Image as PILImage

        for block in output.images[:3]:
            url = block["image_url"]["url"]
            assert url.startswith("data:image/png;base64,")
            with PILImage.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))) as image:
                image.verify()
        # Capture must reach the next provider request, not merely stdout or a buffer.
        content_parts = [
            part
            for message in llm.calls[1].messages
            if isinstance(message.get("content"), (list, tuple))
            for part in message["content"]
        ]
        for block in output.images:
            assert block in content_parts
        assert "not captured" not in output.stdout
        pid_line = next(line for line in output.stdout.splitlines() if line.startswith("cell_pid "))
        cell_pid = int(pid_line.split()[1])
        assert (cell_pid != os.getpid()) is (agent_type is SandboxedMediaAtom)
        video_output = next(
            event
            for event in agent.event_manager.values()
            if isinstance(event, PythonOutput) and event.tool_call_id == "video"
        )
        assert video_output.execution_status == ResultStatus.COMPLETE, video_output.error
        assert video_output.images == [
            {
                "type": "video_url",
                "video_url": {"url": "data:video/mp4;base64,c3ludGhldGljLWNvbnRhaW5lcg=="},
            }
        ]
        assert video_output.images[0] in [
            part
            for message in llm.calls[2].messages
            if isinstance(message.get("content"), (list, tuple))
            for part in message["content"]
        ]
    finally:
        await agent.aclose()


def test_helpers_outside_execution_fail_before_file_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported execution must not touch files or convert media")

    monkeypatch.setattr("pathlib.Path.stat", forbidden)
    monkeypatch.setattr("pathlib.Path.open", forbidden)
    monkeypatch.setattr("nooa.runtime.media_capture.show", forbidden)
    skill = Multimodal()
    for call in (
        lambda: skill.from_file("missing.png"),
        lambda: skill.show("missing.png"),
        lambda: skill.show(object()),
    ):
        with pytest.raises(RuntimeError, match="does not work in NOOA sandbox-mediated"):
            call()


@pytest.mark.parametrize("override_sandbox_method", [False, True])
async def test_callable_helpers_capture_and_follow_fresh_shell_cwd(
    tmp_path, override_sandbox_method
):
    (tmp_path / "sub").mkdir()
    for name, data in (
        ("tiny.png", b"synthetic-png"),
        ("unknown", b"synthetic-wav"),
        ("clip.mp4", b"synthetic-video"),
        ("page.pdf", b"synthetic-pdf"),
        ("too-big.png", b"12345"),
    ):
        (tmp_path / "sub" / name).write_bytes(data)
    code = """
await self.shell.run("cd sub")
media = self.multimodal.from_file("tiny.png", max_bytes=13)
assert type(media).__name__ == "Image"
assert media.media_type == "image/png"
self.multimodal.show(media)
self.multimodal.show("unknown", media_type="audio/wav")
self.multimodal.show("clip.mp4")
self.multimodal.show("page.pdf")
for path, kwargs, message in (
    ("unknown", {}, "Unknown media type"),
    ("unknown", {"media_type": "application/octet-stream"}, "Unknown media type"),
    ("tiny.png", {"media_type": "not-a-mime"}, "Unknown media type"),
    ("tiny.png", {"max_bytes": 0}, "positive integer"),
    ("tiny.png", {"max_bytes": True}, "positive integer"),
    ("too-big.png", {"max_bytes": 4}, "exceeds max_bytes"),
    (".", {"media_type": "application/pdf"}, "regular local file"),
):
    try:
        self.multimodal.from_file(path, **kwargs)
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError("Expected bounded loader error")
print("loader checks passed")
"""
    llm = FakeLLMClient(
        [
            python_cell(code, "callable-media"),
            python_cell("return_result(Done(explanation='captured'))", "done"),
        ],
        strict_exhaustion=True,
    )
    agent_type = SandboxedMediaAtom if override_sandbox_method else AtomAgent
    agent = agent_type(llm=llm, cwd=tmp_path, libs_dir=tmp_path / "libs")
    try:
        await agent.skills.activate(["multimodal"])
        kwargs = (
            {"_strategy": CodeActV2(config=CodeActConfig(execution_backend="inprocess"))}
            if override_sandbox_method
            else {}
        )
        await agent.handle({"user_messages": ["inspect bounded local files"]}, **kwargs)
        output = next(
            e
            for e in agent.event_manager.values()
            if isinstance(e, PythonOutput) and e.tool_call_id == "callable-media"
        )
        assert output.execution_status == ResultStatus.COMPLETE, output.error
        assert "loader checks passed" in output.stdout
        assert [b["type"] for b in output.images] == [
            "image_url",
            "input_audio",
            "video_url",
            "file",
        ]
        assert output.images[0]["image_url"]["url"] == (
            "data:image/png;base64," + base64.b64encode(b"synthetic-png").decode()
        )
        next_parts = [
            part
            for msg in llm.calls[1].messages
            if isinstance(msg.get("content"), (list, tuple))
            for part in msg["content"]
        ]
        assert all(block in next_parts for block in output.images)
        assert "not captured" not in output.stdout
    finally:
        await agent.aclose()


class NestedMediaAtom(SandboxedMediaAtom):
    @strategy(CodeActV2(config=CodeActConfig(execution_backend="inprocess")))
    async def outer(self) -> Done:
        """Run the nested media method."""
        ...


@pytest.mark.sandbox
@pytest.mark.parametrize("nested", [False, True])
async def test_sandbox_callable_helpers_fail_before_read_or_conversion(
    tmp_path, monkeypatch, nested
):
    if not probe_capabilities().seccomp:
        pytest.skip("network confinement requires seccomp")
    reads = []
    conversions = []
    original_open = type(tmp_path).open
    original_stat = type(tmp_path).stat
    from nooa.runtime.media_capture import show as original_show

    def watched_open(path, *args, **kwargs):
        if path.name == "must-not-read.png":
            reads.append("open")
            pytest.fail("Sandbox helper opened a parent file")
        return original_open(path, *args, **kwargs)

    def watched_stat(path, *args, **kwargs):
        if path.name == "must-not-read.png":
            reads.append("stat")
            pytest.fail("Sandbox helper probed a parent file")
        return original_stat(path, *args, **kwargs)

    def watched_show(obj):
        conversions.append(type(obj).__name__)
        return original_show(obj)

    monkeypatch.setattr(type(tmp_path), "open", watched_open)
    monkeypatch.setattr(type(tmp_path), "stat", watched_stat)
    monkeypatch.setattr("nooa.runtime.media_capture.show", watched_show)
    code = """
for operation in (
    lambda: self.multimodal.from_file("must-not-read.png"),
    lambda: self.multimodal.show("must-not-read.png"),
    lambda: self.multimodal.show(b"not-media"),
):
    try:
        operation()
    except Exception as exc:
        assert "active in-process NOOA Python cell" in str(exc), str(exc)
        assert "does not work in NOOA sandbox-mediated" in str(exc), str(exc)
        print("expected actionable error", str(exc))
    else:
        raise AssertionError("Sandbox helper unexpectedly returned")
# Worker Media cannot cross the broker at all. Keep that restriction unchanged.
try:
    self.multimodal.show(Image.from_bytes(b"synthetic", media_type="image/png"))
except Exception as exc:
    assert "cannot cross the sandbox boundary" in str(exc), str(exc)
    print("expected unchanged serialization error")
else:
    raise AssertionError("Worker Image unexpectedly crossed the broker")
"""
    responses = []
    if nested:
        responses.append(
            python_cell("result = await self.handle({})\nreturn_result(result)", "outer")
        )
    responses.extend(
        [
            python_cell(code, "guard"),
            python_cell("return_result(Done(explanation='rejected'))", "done"),
        ]
    )
    llm = FakeLLMClient(responses, strict_exhaustion=True)
    agent = NestedMediaAtom(llm=llm, cwd=tmp_path, libs_dir=tmp_path / "libs")
    try:
        await agent.skills.activate(["multimodal"])
        result = await agent.outer() if nested else await agent.handle({})
        assert result == Done(explanation="rejected")
        output = next(
            e
            for e in agent.event_manager.values()
            if isinstance(e, PythonOutput) and e.tool_call_id == "guard"
        )
        assert output.execution_status == ResultStatus.COMPLETE, output.error
        assert output.stdout.count("expected actionable error") == 3
        assert "not captured" not in output.stdout
        assert not output.images
        assert not reads
        assert not conversions
        assert all(
            not e.images for e in agent.event_manager.values() if isinstance(e, PythonOutput)
        )
    finally:
        await agent.aclose()
