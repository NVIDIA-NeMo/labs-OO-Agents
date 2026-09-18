# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests: PredictStrategy must attach each media input exactly once.

CurrentCall keeps positional args in ``call.args`` and also mirrors them into
``call.kwargs`` by parameter name for template expansion, so the old
``list(call.args) + list(call.kwargs.values())`` collection counted a positional
Image/Audio/File twice. Each effective input must be attached exactly once.
"""

from pydantic import BaseModel

from nooa.media import Image, Media
from nooa.runtime.media_capture import media_to_content_block
from nooa.strategies.current_call import CurrentCall


class Result(BaseModel):
    ok: bool


def _media_blocks(call: CurrentCall) -> list:
    """Mirror PredictStrategy's media collection."""
    values = []
    for name, value in call.bound_parameters().items():
        if name == call.var_positional_param_name and isinstance(value, tuple):
            values.extend(value)
        elif name == call.var_keyword_param_name and isinstance(value, dict):
            values.extend(value.values())
        else:
            values.append(value)
    return [media_to_content_block(value) for value in values if isinstance(value, Media)]


def test_positional_media_attached_once():
    """The issue's minimal repro: positional Image must produce one media block."""

    def analyze(self, image: Image) -> Result:
        """Analyze {image}."""
        ...

    img = Image.from_bytes(b"abc", media_type="image/png")
    call = CurrentCall.from_method(analyze, args=(img,), kwargs={})

    # Sanity: from_method creates the overlap that caused the bug.
    assert img in call.args
    assert call.kwargs == {"image": img}

    # The old buggy expression would have produced two blocks.
    buggy = [v for v in list(call.args) + list(call.kwargs.values()) if isinstance(v, Media)]
    assert len(buggy) == 2  # guards against silently reverting the fix

    # The fixed collection produces exactly one.
    assert len(_media_blocks(call)) == 1


def test_keyword_media_attached_once():
    """Keyword-passed media is also attached exactly once (unchanged behavior)."""

    def analyze(self, image: Image) -> Result:
        """Analyze {image}."""
        ...

    img = Image.from_bytes(b"abc", media_type="image/png")
    call = CurrentCall.from_method(analyze, args=(), kwargs={"image": img})

    assert len(_media_blocks(call)) == 1


def test_var_positional_media_each_attached_once():
    """*imgs: two distinct positional images attach two blocks, no duplication."""

    def analyze(self, *imgs: Image) -> Result:
        """Analyze the images."""
        ...

    img1 = Image.from_bytes(b"one", media_type="image/png")
    img2 = Image.from_bytes(b"two", media_type="image/png")
    call = CurrentCall.from_method(analyze, args=(img1, img2), kwargs={})

    assert len(_media_blocks(call)) == 2


def test_colliding_variadic_media_each_attached_once():
    """Same-named *args and **kwargs media are both retained without duplication."""

    def analyze(self, *imgs: Image, **metadata: Image) -> Result:
        """Analyze the images."""
        ...

    positional = Image.from_bytes(b"positional", media_type="image/png")
    keyword = Image.from_bytes(b"keyword", media_type="image/png")
    call = CurrentCall.from_method(analyze, args=(positional,), kwargs={"imgs": keyword})

    assert call.bound_parameters() == {
        "imgs": (positional,),
        "metadata": {"imgs": keyword},
    }
    assert len(_media_blocks(call)) == 2
