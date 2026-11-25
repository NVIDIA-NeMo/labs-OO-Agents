# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597 item 3: images of unknown type are sniffed, and the note says what they are."""

import base64

import pytest

from nooa.media import Image, sniff_image_type
from nooa.runtime.media_capture import media_to_content_block
from nooa.unifiedllm.media_inputs import filter_media_messages

ALL = ("image", "audio", "video", "pdf")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 16
GIF = b"GIF89a" + b"\x00" * 16
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 8
SVG = b'<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"/>'


@pytest.mark.parametrize(
    "data,expected",
    [
        (PNG, "image/png"),
        (JPEG, "image/jpeg"),
        (GIF, "image/gif"),
        (WEBP, "image/webp"),
        (SVG, "image/svg+xml"),
        (b"\xef\xbb\xbf  <svg/>", "image/svg+xml"),
        (b"<?xml version='1.0'?><note/>", ""),
        (b"plain text", ""),
    ],
)
def test_sniff_image_type(data, expected):
    assert sniff_image_type(data) == expected


def test_from_file_without_extension_sniffs_png(tmp_path):
    path = tmp_path / "screenshot"
    path.write_bytes(PNG)
    image = Image.from_file(path)
    assert image.media_type == "image/png"
    block = media_to_content_block(image)
    assert block["image_url"]["format"] == "image/png"
    assert block["image_url"]["url"].startswith("data:image/png;base64,")
    sent = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert sent == [block]


def test_svg_bytes_without_extension_get_the_svg_note(tmp_path):
    path = tmp_path / "chart"
    path.write_bytes(SVG)
    block = media_to_content_block(Image.from_file(path))
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert "image/svg+xml" in sent["text"]


def test_an_obvious_header_mismatch_in_a_file_uses_the_bytes(tmp_path):
    path = tmp_path / "photo.png"
    path.write_bytes(JPEG)
    assert Image.from_file(path).media_type == "image/jpeg"
    path.write_bytes(b"\x00\x01")  # unknown bytes keep the extension's type
    assert Image.from_file(path).media_type == "image/png"
    # An explicit type given to from_bytes is the caller's: kept as given.
    assert Image.from_bytes(JPEG, media_type="image/png").media_type == "image/png"


def test_unknown_image_note_says_image_not_only_pdf():
    image = Image.from_bytes(b"\x00\x01\x02", media_type="application/octet-stream")
    block = media_to_content_block(image)
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert "image of unknown type" in sent["text"]
    assert "application/octet-stream" in sent["text"]
    assert "Only PDF" not in sent["text"]


def test_a_real_file_block_keeps_the_pdf_note():
    data = base64.b64encode(b"PK\x03\x04").decode()
    block = {"type": "file", "file": {"file_data": f"data:application/zip;base64,{data}"}}
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert "Only PDF files" in sent["text"]
