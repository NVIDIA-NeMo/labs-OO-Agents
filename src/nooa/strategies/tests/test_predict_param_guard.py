# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for PredictStrategy parameter size guard.

PredictStrategy is single-shot — a silently truncated input produces wrong output.
Oversized params must raise ValueError with a clear message rather than truncating.
"""

import pytest

from nooa.config.strategy_config import PredictConfig
from nooa.strategies.current_call import CurrentCall
from nooa.strategies.predict import PredictStrategy
from tests.helpers.signature_utils import param_names_from_signature


def _make_call(args=(), kwargs=None, signature=None, param_names=None):
    # Real CurrentCalls carry param_names captured from the live signature
    # (from_method / actor). Derive them here so the guard reports real parameter
    # names rather than arg_<i>.
    if param_names is None and signature:
        param_names = param_names_from_signature(signature)
    return CurrentCall(
        id="test-id",
        method_name="test",
        decorator="agent",
        signature=signature,
        args=args,
        kwargs=kwargs or {},
        param_names=param_names,
    )


class TestPredictParamGuard:
    """_assert_param_sizes raises ValueError for oversized parameters."""

    def _strategy(self, max_param_chars=200_000):
        return PredictStrategy(config=PredictConfig(max_param_chars=max_param_chars))

    def test_small_param_passes(self):
        strategy = self._strategy()
        call = _make_call(args=("hello",), signature="(text: str)")
        strategy._assert_param_sizes(call)  # must not raise

    def test_small_kwarg_passes(self):
        strategy = self._strategy()
        call = _make_call(kwargs={"data": list(range(10))})
        strategy._assert_param_sizes(call)  # must not raise

    def test_oversized_string_raises(self):
        strategy = self._strategy(max_param_chars=1000)
        call = _make_call(args=("x" * 2000,), signature="(text: str)")
        with pytest.raises(ValueError, match="text"):
            strategy._assert_param_sizes(call)

    def test_oversized_list_raises(self):
        strategy = self._strategy(max_param_chars=500)
        call = _make_call(args=(list(range(10_000)),), signature="(items: list)")
        with pytest.raises(ValueError, match="items"):
            strategy._assert_param_sizes(call)

    def test_error_message_names_the_param(self):
        strategy = self._strategy(max_param_chars=100)
        call = _make_call(
            args=("y" * 500,),
            signature="(document: str)",
        )
        with pytest.raises(ValueError, match="document"):
            strategy._assert_param_sizes(call)

    def test_error_message_mentions_max_param_chars(self):
        strategy = self._strategy(max_param_chars=500)
        call = _make_call(kwargs={"report": "z" * 2000})
        with pytest.raises(ValueError, match="500"):
            strategy._assert_param_sizes(call)

    def test_error_message_suggests_raising_limit(self):
        strategy = self._strategy(max_param_chars=100)
        call = _make_call(kwargs={"x": "a" * 500})
        with pytest.raises(ValueError, match="max_param_chars"):
            strategy._assert_param_sizes(call)

    def test_large_default_allows_document_summarization(self):
        """Default 200K chars covers realistic documents without raising."""
        strategy = self._strategy()  # default 200_000
        # Typical long document: ~150K chars
        call = _make_call(
            args=("word " * 30_000,),  # ~150K chars
            signature="(document: str)",
        )
        strategy._assert_param_sizes(call)  # must not raise

    def test_no_signature_uses_arg_index_names(self):
        """Without a signature, oversized positional arg raises with arg_0 name."""
        strategy = self._strategy(max_param_chars=100)
        call = _make_call(args=("b" * 500,))
        with pytest.raises(ValueError, match="arg_0"):
            strategy._assert_param_sizes(call)

    def test_custom_limit_respected(self):
        """PredictConfig(max_param_chars=N) sets the limit."""
        strategy = self._strategy(max_param_chars=50)
        call = _make_call(args=("c" * 100,), signature="(text: str)")
        with pytest.raises(ValueError):
            strategy._assert_param_sizes(call)

        # With a higher limit, same call passes
        strategy2 = self._strategy(max_param_chars=200)
        strategy2._assert_param_sizes(call)  # must not raise


class TestPredictPromptSizeGuardImages:
    """Images on Task are repr=False — excluded from text rendering.

    Image content blocks bypass the text truncation pipeline entirely —
    they are attached to the LLM message *after* formatting via the
    provider-level _append_images path, not through format_event.
    """

    def test_task_images_excluded_from_plain_formatter(self):
        """Task.images has repr=False — image data never appears in formatted text."""
        from nooa.events import Task
        from nooa.plain_formatter import PlainBlockFormatter

        big_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "A" * 10_000},
        }
        task = Task(prompt="Analyze this image.", images=[big_image])
        text = PlainBlockFormatter().format_event(task)

        # Image bytes must NOT appear in the text block
        assert "AAAAAAA" not in text
        assert "image_url" not in text
        # Prompt is the only content
        assert text == "Analyze this image."

    def test_format_event_output_length_matches_guard_threshold(self):
        """len(format_event(task)) == len(task.prompt) — guard threshold is exact."""
        from nooa.events import Task
        from nooa.plain_formatter import PlainBlockFormatter

        prompt = "Describe the scene in detail."
        big_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "B" * 50_000},
        }
        task = Task(prompt=prompt, images=[big_image])
        formatted = PlainBlockFormatter().format_event(task)

        # The formatted output is exactly the prompt — no XML wrapper, no image overhead
        assert formatted == prompt
        assert len(formatted) == len(prompt)

    def test_long_prompt_with_images_passes_through_verbatim(self):
        """format_event passes long prompts through verbatim regardless of images."""
        from nooa.events import Task
        from nooa.plain_formatter import PlainBlockFormatter

        long_prompt = "x" * 2000
        small_image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
        task = Task(prompt=long_prompt, images=[small_image])

        formatted = PlainBlockFormatter().format_event(task)
        assert long_prompt in formatted  # full prompt present
        assert "image_url" not in formatted  # images still excluded from text

    def test_short_prompt_with_images_not_truncated(self):
        """format_event does not truncate short prompts even when images are large."""
        from nooa.events import Task
        from nooa.plain_formatter import PlainBlockFormatter

        prompt = "Short prompt."
        big_image = {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + "C" * 100_000},
        }
        task = Task(prompt=prompt, images=[big_image])

        formatted = PlainBlockFormatter().format_event(task)
        assert formatted == prompt  # no truncation


class TestPredictNoInputTruncation:
    """Predict renders accepted inputs without truncation and without config."""

    def test_default_runtime_truncates_long_strings(self):
        """Documents the bug source: normal prefill formatting truncates long strings."""
        from nooa.config.truncation_config import TruncationConfig

        long_str = "x" * 5291
        call = _make_call(args=(long_str,), signature="(knowledge_md: str)")

        truncated = call.format_parameters_as_code(tc=TruncationConfig())
        assert "str(len=" in truncated

    def test_predict_uses_no_tc_so_long_strings_are_not_truncated(self):
        """Predict calls format_parameters_as_code() with no tc, so repr() renders in full."""
        long_str = "x" * 5291
        call = _make_call(args=(long_str,), signature="(knowledge_md: str)")

        output = call.format_parameters_as_code()
        assert "str(len=" not in output
        assert long_str in output

    def test_predict_no_tc_preserves_full_content(self):
        """A 5K-char string that passes the param guard renders verbatim in the prompt."""
        content = "Knowledge guide: " + "a" * 5000 + " end."
        call = _make_call(args=(content,), signature="(knowledge_md: str)")

        output = call.format_parameters_as_code()

        assert content in output
