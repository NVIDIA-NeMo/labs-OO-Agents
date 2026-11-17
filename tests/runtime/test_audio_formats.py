# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597 item 2 and review missed case: audio format names and URL audio."""

import pytest

from nooa.media import Audio
from nooa.media_inputs import audio_format_name
from nooa.runtime.media_capture import media_to_content_block
from nooa.unifiedllm.media_inputs import filter_media_messages, media_occurrences

ALL = ("image", "audio", "video", "pdf")


@pytest.mark.parametrize(
    "media_type,expected",
    [
        ("audio/mpeg", "mp3"),
        ("audio/mp3", "mp3"),
        ("audio/x-wav", "wav"),
        ("audio/wave", "wav"),
        ("audio/vnd.wave", "wav"),
        ("audio/wav", "wav"),
        ("audio/mp4", "mp4"),  # never relabelled as mp3: it is not a codec conversion
        ("audio/ogg", "ogg"),
    ],
)
def test_audio_mime_subtypes_map_to_provider_format_names(media_type, expected):
    assert audio_format_name(media_type) == expected
    block = media_to_content_block(Audio.from_bytes(b"RIFF", media_type=media_type))
    assert block["input_audio"]["format"] == expected


def test_mp3_and_wav_from_file_are_sent(tmp_path):
    for name in ("clip.mp3", "clip.wav"):
        path = tmp_path / name
        path.write_bytes(b"\x00" * 8)
        block = media_to_content_block(Audio.from_file(path))
        (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
        assert sent == block and sent["input_audio"]["format"] in {"mp3", "wav"}


def test_an_old_history_alias_is_sent_as_the_provider_name():
    block = {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "mpeg"}}
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert sent["input_audio"]["format"] == "mp3"
    assert block["input_audio"]["format"] == "mpeg"  # history unchanged


def test_m4a_is_replaced_by_a_visible_note():
    block = media_to_content_block(Audio.from_bytes(b"\x00", media_type="audio/mp4"))
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert sent["type"] == "text"
    assert "mp4 audio cannot be sent" in sent["text"] and "WAV and MP3" in sent["text"]
    assert media_occurrences([{"role": "user", "content": [block]}], ALL) == []


def test_audio_from_url_never_sends_the_url_as_base64():
    audio = Audio.from_url("https://example.com/clip.mp3")
    assert audio.media_type == "audio/mpeg"
    block = media_to_content_block(audio)
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert sent["type"] == "text" and "URL cannot be sent" in sent["text"]
    assert "https://example.com" not in str(sent)


def test_audio_from_a_data_url_is_sent():
    audio = Audio.from_url("data:audio/wav;base64,UklGRg==")
    block = media_to_content_block(audio)
    assert block["input_audio"] == {"data": "UklGRg==", "format": "wav"}
    (sent,) = filter_media_messages([{"role": "user", "content": [block]}], ALL)[0]["content"]
    assert sent == block


@pytest.mark.parametrize("media_type", ["audio/ogg", "audio/flac"])
def test_inline_audio_formats_are_route_specific(media_type):
    from nooa.unifiedllm import CompletionClient

    block = media_to_content_block(Audio.from_bytes(b"OggS", media_type=media_type))
    url_block = media_to_content_block(Audio.from_url("https://example.com/clip.ogg"))
    messages = [{"role": "user", "content": [block, url_block]}]
    for model, restricted in (
        ("gemini/gemini-2.5-flash", False),
        ("vertex_ai/gemini-2.5-flash", False),
        ("openai/gpt-4o-audio-preview", True),
        ("azure/gpt-4o-audio", True),
        ("hosted_vllm/qwen-omni", True),
    ):
        with CompletionClient(model, media_inputs=["audio"], cache_breakpoint=None) as client:
            sent, sent_url = client._prepare_media_messages(
                messages, client._prepare_call_config({})
            )[0]["content"]
        # A URL is never sent as audio data, on any route.
        assert sent_url["type"] == "text" and "URL cannot be sent" in sent_url["text"]
        if restricted:  # OpenAI input_audio: WAV and MP3 only
            assert sent["type"] == "text" and "WAV and MP3" in sent["text"], model
        else:  # Gemini and other routes: the adapter receives the block unchanged
            assert sent == block, model
