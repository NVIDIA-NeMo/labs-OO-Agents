# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""#597 items 6 and 9: show() warns about unsendable media; the filter's replacements are counted."""

from unittest.mock import patch

import litellm

from nooa.media import Audio, Image
from nooa.runtime.media_capture import _MediaBuffer, capture_media, show
from nooa.unifiedllm import CompletionClient

SVG = b'<svg xmlns="http://www.w3.org/2000/svg"/>'
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8


def test_show_says_an_svg_cannot_be_sent_but_still_attaches_it(capsys):
    buffer = _MediaBuffer(max_attachments=5)
    with capture_media(buffer):
        show(Image.from_bytes(SVG, media_type="image/svg+xml"))
    out = capsys.readouterr().out
    assert "[attached:" in out and "SVG images cannot be sent" in out and "PNG" in out
    assert "[shown:" not in out
    assert len(buffer.blocks) == 1  # kept in history; the filter replaces it on send


def test_show_warns_about_url_audio_and_keeps_png_quiet(capsys):
    with capture_media(_MediaBuffer(max_attachments=5)):
        show(Audio.from_url("https://example.com/a.mp3"))
        show(Image.from_bytes(PNG, media_type="image/png"))
    first, second = capsys.readouterr().out.strip().splitlines()
    assert "audio given as a URL cannot be sent" in first
    assert second.startswith("[shown:")


def test_filter_replacements_are_counted_once_per_occurrence():
    from nooa.runtime.actor import _make_llm_metrics_bridge
    from nooa.runtime.harness_metrics import HarnessMetrics
    from nooa.runtime.media_capture import media_to_content_block
    from nooa.unifiedllm.unifiedllm import _llm_metrics_callback

    svg = media_to_content_block(Image.from_bytes(SVG, media_type="image/svg+xml"))
    messages = [{"role": "user", "content": [svg]}]
    metrics = HarnessMetrics()
    token = _llm_metrics_callback.set(_make_llm_metrics_bridge(metrics))
    try:
        with CompletionClient("openai/test", media_inputs=["image"], cache_breakpoint=None) as c:
            with patch("litellm.completion") as wire:
                wire.return_value = litellm.ModelResponse(choices=[{"message": {"content": "x"}}])
                c.call(messages)
                c.call(messages)  # the same history again: not counted twice
                c.call([*messages, {"role": "user", "content": [svg]}])  # a second occurrence
    finally:
        _llm_metrics_callback.reset(token)
    attrs = metrics.to_span_attributes()
    assert attrs["harness.media.unsendable_replaced.count"] == 2
    assert attrs["harness.media.unsendable_replaced.reasons"] == ["image:image/svg+xml"] * 2
