# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for multimodal media capture via show() in CodeAct execution."""

import base64
from pathlib import Path

import pytest

from nooa.media import Audio, File, Image, Media, Video
from nooa.runtime.media_capture import (
    _media_buffer_var,
    _MediaBuffer,
    _try_auto_convert,
    media_to_content_block,
    show,
)


def _write_fake_mp4(tmp_path: Path) -> Path:
    """Write minimal .mp4 bytes; mimetypes guesses video/mp4 from the extension."""
    mp4 = tmp_path / "clip.mp4"
    mp4.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)
    return mp4


# ---------------------------------------------------------------------------
# Media class hierarchy
# ---------------------------------------------------------------------------


class TestMediaHierarchy:
    def test_image_is_media(self):
        img = Image.from_bytes(b"fake", media_type="image/png")
        assert isinstance(img, Media)
        assert img.modality == "image"

    def test_audio_is_media(self):
        audio = Audio.from_bytes(b"fake", media_type="audio/wav")
        assert isinstance(audio, Media)
        assert audio.modality == "audio"

    def test_file_is_media(self):
        f = File.from_url("https://example.com/report.pdf")
        assert isinstance(f, Media)
        assert f.modality == "file"

    def test_video_is_media(self):
        v = Video.from_url("https://example.com/clip.mp4")
        assert isinstance(v, Media)
        assert v.modality == "video"

    def test_no_llm_specific_methods(self):
        """Media should NOT have LLM-specific methods."""
        img = Image.from_bytes(b"fake", media_type="image/png")
        assert not hasattr(img, "to_content_block")


class TestImage:
    def test_from_bytes(self):
        img = Image.from_bytes(b"\x89PNG" + b"\x00" * 100, media_type="image/png")
        r = repr(img)
        assert r.startswith("Image(image/png, ")
        assert "bytes)" in r

    def test_data_url_property(self):
        img = Image.from_bytes(b"fake", media_type="image/png")
        assert img.data_url.startswith("data:image/png;base64,")

    def test_from_url_default_media_type(self):
        img = Image.from_url("https://example.com/photo.jpg")
        assert img.media_type == "image/jpeg"


class TestAudio:
    def test_from_bytes(self):
        audio = Audio.from_bytes(b"RIFF" + b"\x00" * 100, media_type="audio/wav")
        r = repr(audio)
        assert r.startswith("Audio(audio/wav, ")
        assert "bytes)" in r

    def test_from_url_default_media_type(self):
        audio = Audio.from_url("https://example.com/clip.wav")
        assert audio.media_type == "audio/wav"

    def test_base64_data(self):
        data = b"test audio data"
        audio = Audio.from_bytes(data, media_type="audio/wav")
        decoded = base64.b64decode(audio._base64_data())
        assert decoded == data


class TestFile:
    def test_from_url(self):
        f = File.from_url("https://example.com/report.pdf")
        assert f.media_type == "application/pdf"
        assert f.data_url == "https://example.com/report.pdf"


class TestVideo:
    def test_from_file(self, tmp_path: Path):
        v = Video.from_file(_write_fake_mp4(tmp_path))
        assert v.media_type == "video/mp4"
        assert v.data_url.startswith("data:video/mp4;base64,")
        r = repr(v)
        assert r.startswith("Video(video/mp4, ")
        assert "bytes)" in r

    def test_from_url_default_media_type(self):
        v = Video.from_url("https://example.com/clip.mp4")
        assert v.media_type == "video/mp4"
        assert v.data_url == "https://example.com/clip.mp4"

    def test_from_bytes_explicit_media_type(self):
        v = Video.from_bytes(b"webm-bytes", media_type="video/webm")
        assert v.media_type == "video/webm"
        decoded = base64.b64decode(v._base64_data())
        assert decoded == b"webm-bytes"


# ---------------------------------------------------------------------------
# Content block conversion
# ---------------------------------------------------------------------------


class TestMediaToContentBlock:
    def test_image_block(self):
        img = Image.from_bytes(b"fake", media_type="image/png")
        block = media_to_content_block(img)
        assert block["type"] == "image_url"
        assert block["image_url"]["url"].startswith("data:image/png;base64,")
        assert block["image_url"]["format"] == "image/png"

    def test_image_url_block(self):
        img = Image.from_url("https://example.com/photo.jpg")
        block = media_to_content_block(img)
        assert block["image_url"]["url"] == "https://example.com/photo.jpg"

    def test_audio_block(self):
        audio = Audio.from_bytes(b"wav-data", media_type="audio/wav")
        block = media_to_content_block(audio)
        assert block["type"] == "input_audio"
        assert block["input_audio"]["format"] == "wav"
        decoded = base64.b64decode(block["input_audio"]["data"])
        assert decoded == b"wav-data"

    def test_file_block(self):
        f = File.from_url("https://example.com/report.pdf")
        block = media_to_content_block(f)
        assert block["type"] == "file"
        assert block["file"]["file_data"] == "https://example.com/report.pdf"

    def test_video_block_from_file(self, tmp_path: Path):
        v = Video.from_file(_write_fake_mp4(tmp_path))
        block = media_to_content_block(v)
        assert block["type"] == "video_url"
        assert block["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_video_block_url_preserves_url(self):
        v = Video.from_url("https://example.com/clip.mp4")
        block = media_to_content_block(v)
        assert block == {
            "type": "video_url",
            "video_url": {"url": "https://example.com/clip.mp4"},
        }

    def test_video_vendor_metadata_merged_into_block(self):
        v = Video.from_url("https://example.com/clip.mp4", fps=1, max_frames=32)
        block = media_to_content_block(v)
        assert block["type"] == "video_url"
        assert block["video_url"]["url"] == "https://example.com/clip.mp4"
        assert block["video_url"]["fps"] == 1
        assert block["video_url"]["max_frames"] == 32

    def test_raises_on_non_media(self):
        with pytest.raises(TypeError, match="Expected Media"):
            media_to_content_block("not media")

    def test_image_base64_roundtrips(self):
        original = b"test image data"
        img = Image.from_bytes(original, media_type="image/png")
        block = media_to_content_block(img)
        _, b64_data = block["image_url"]["url"].split(",", 1)
        assert base64.b64decode(b64_data) == original


# ---------------------------------------------------------------------------
# show() function
# ---------------------------------------------------------------------------


class TestShow:
    def test_show_collects_image(self, capsys):
        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            img = Image.from_bytes(b"test", media_type="image/png")
            show(img)
            assert len(buf.blocks) == 1
            assert buf.blocks[0]["type"] == "image_url"
        finally:
            _media_buffer_var.reset(token)

    def test_show_collects_audio(self, capsys):
        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            audio = Audio.from_bytes(b"test", media_type="audio/wav")
            show(audio)
            assert len(buf.blocks) == 1
            assert buf.blocks[0]["type"] == "input_audio"
        finally:
            _media_buffer_var.reset(token)

    def test_show_collects_file(self, capsys):
        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            f = File.from_url("https://example.com/doc.pdf")
            show(f)
            assert len(buf.blocks) == 1
            assert buf.blocks[0]["type"] == "file"
        finally:
            _media_buffer_var.reset(token)

    def test_show_collects_video(self, capsys, tmp_path: Path):
        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            video = Video.from_file(_write_fake_mp4(tmp_path))
            show(video)
            assert len(buf.blocks) == 1
            assert buf.blocks[0]["type"] == "video_url"
            assert buf.blocks[0]["video_url"]["url"].startswith("data:video/mp4;base64,")
        finally:
            _media_buffer_var.reset(token)

    def test_show_prints_acknowledgment(self, capsys):
        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            show(Image.from_bytes(b"test", media_type="image/png"))
            captured = capsys.readouterr()
            assert "[shown:" in captured.out
        finally:
            _media_buffer_var.reset(token)

    def test_show_raises_on_non_media(self):
        buf = _MediaBuffer(max_attachments=10)
        token = _media_buffer_var.set(buf)
        try:
            with pytest.raises(TypeError, match="got int"):
                show(42)
        finally:
            _media_buffer_var.reset(token)

    def test_show_limit_enforced_by_buffer(self, capsys):
        buf = _MediaBuffer(max_attachments=2)
        token = _media_buffer_var.set(buf)
        try:
            for i in range(4):
                show(Image.from_bytes(f"img{i}".encode(), media_type="image/png"))
            assert len(buf.blocks) == 2
            captured = capsys.readouterr()
            assert "limit reached (2)" in captured.out
        finally:
            _media_buffer_var.reset(token)

    def test_show_outside_context_warns(self, capsys):
        show(Image.from_bytes(b"test", media_type="image/png"))
        captured = capsys.readouterr()
        assert "outside execution context" in captured.out


class TestAutoConvert:
    def test_unknown_type_returns_none(self):
        assert _try_auto_convert("not an image") is None
        assert _try_auto_convert(42) is None

    def test_pil_image_conversion(self):
        try:
            from PIL import Image as PILImage
        except ImportError:
            pytest.skip("Pillow not installed")
        pil_img = PILImage.new("RGB", (10, 10), color="red")
        block = _try_auto_convert(pil_img)
        assert block is not None
        assert block["type"] == "image_url"
        assert block["image_url"]["format"] == "image/png"
