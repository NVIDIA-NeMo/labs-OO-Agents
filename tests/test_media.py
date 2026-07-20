# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for nooa.media — classmethods and _base64_data edge cases.

Covers:
- Image.from_file(path) via tmp_path fixture (lines 64-68)
- Image.from_url(url, media_type="") — empty media_type (line 82)
- Image._base64_data() when data_url does NOT start with 'data:' (line 88)
"""

import base64
from pathlib import Path

from nooa.media import Image, Video


class TestImageFromFile:
    """Tests for Image.from_file() class method."""

    def test_from_file_reads_bytes_and_returns_image(self, tmp_path: Path):
        """Image.from_file() loads a file and returns an Image instance."""
        png_file = tmp_path / "test.png"
        # Minimal valid-looking bytes; mimetypes will guess image/png from extension
        png_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        img = Image.from_file(png_file)

        assert isinstance(img, Image)
        assert img.media_type == "image/png"
        assert img.data_url.startswith("data:image/png;base64,")
        assert img.modality == "image"

    def test_from_file_unknown_extension_falls_back_to_octet_stream(self, tmp_path: Path):
        """from_file() uses application/octet-stream for unrecognised extensions."""
        unknown_file = tmp_path / "data.xyzzy"
        unknown_file.write_bytes(b"\x00\x01\x02\x03")

        img = Image.from_file(unknown_file)

        assert isinstance(img, Image)
        assert img.media_type == "application/octet-stream"


class TestImageFromUrl:
    """Tests for Image.from_url() class method."""

    def test_from_url_with_empty_media_type(self):
        """Image.from_url() accepts an empty string as media_type."""
        url = "https://example.com/photo.jpg"
        img = Image.from_url(url, media_type="")

        assert isinstance(img, Image)
        assert img.data_url == url
        assert img.media_type == ""

    def test_from_url_default_media_type_is_image_jpeg(self):
        """Image.from_url() defaults to 'image/jpeg' when no media_type is given."""
        img = Image.from_url("https://example.com/photo.jpg")

        assert img.media_type == "image/jpeg"


class TestMediaFromUrl:
    """Tests for the base Media.from_url() class method."""

    def test_base_media_from_url_stores_url_and_media_type(self):
        """Media.from_url() stores the URL and media type without downloading."""
        from nooa.media import Media

        url = "https://example.com/file.bin"
        m = Media.from_url(url, media_type="application/octet-stream")

        assert m.data_url == url
        assert m.media_type == "application/octet-stream"
        assert m.size_bytes is None  # URL reference, no local data


class TestBaseBase64Data:
    """Tests for the _base64_data() method on media objects."""

    def test_data_url_starting_with_data_colon_extracts_base64(self):
        """_base64_data() strips the data URL prefix and returns the base64 portion."""
        img = Image.from_bytes(b"hello", media_type="image/png")
        b64 = img._base64_data()

        # Should not contain the data URL prefix
        assert not b64.startswith("data:")
        # Verify it is pure base64
        decoded = base64.b64decode(b64)
        assert decoded == b"hello"

    def test_data_url_not_starting_with_data_returns_url_as_is(self):
        """_base64_data() returns the URL verbatim when it doesn't start with 'data:'."""
        url = "https://example.com/image.png"
        img = Image.from_url(url)

        result = img._base64_data()

        assert result == url


class TestVideo:
    """Tests for the Video media subclass — mirrors Image's surface."""

    def test_from_file_reads_bytes_and_returns_video(self, tmp_path: Path):
        """Video.from_file() loads an .mp4 and produces a data:video/mp4 URL."""
        mp4_file = tmp_path / "clip.mp4"
        # Minimal bytes; mimetypes will guess video/mp4 from the extension
        mp4_file.write_bytes(b"\x00\x00\x00\x20ftypisom" + b"\x00" * 32)

        video = Video.from_file(mp4_file)

        assert isinstance(video, Video)
        assert video.media_type == "video/mp4"
        assert video.modality == "video"
        assert video.data_url.startswith("data:video/mp4;base64,")

    def test_from_url_default_media_type_is_video_mp4(self):
        """Video.from_url() defaults to 'video/mp4' when no media_type is given."""
        url = "https://example.com/clip.mp4"
        video = Video.from_url(url)

        assert isinstance(video, Video)
        assert video.media_type == "video/mp4"
        # URL is preserved verbatim — no base64 wrapping
        assert video.data_url == url

    def test_from_bytes_honors_explicit_media_type(self):
        """Video.from_bytes() respects an explicit media_type like video/webm."""
        video = Video.from_bytes(b"webm-fake-bytes", media_type="video/webm")

        assert isinstance(video, Video)
        assert video.media_type == "video/webm"
        assert video.data_url.startswith("data:video/webm;base64,")
        decoded = base64.b64decode(video._base64_data())
        assert decoded == b"webm-fake-bytes"

    def test_vendor_metadata_round_trips_via_from_url(self):
        """from_url() vendor_metadata kwargs are stored on the instance."""
        video = Video.from_url("https://example.com/clip.mp4", fps=1, max_frames=32)
        assert video.vendor_metadata == {"fps": 1, "max_frames": 32}
