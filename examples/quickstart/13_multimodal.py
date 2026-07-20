# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402,F403,F405
"""Quickstart 13: Multimodal — images with CodeAct and PredictStrategy.

Demonstrates both strategies seeing images:
- CodeAct: prefill auto-calls show() for Image params → LLM sees the image
- PredictStrategy: Image params attached as content blocks to the Task event

Requires a vision-capable model (e.g. gpt-4o with a valid OPENAI_API_KEY).
If the quickstart MODEL doesn't support vision, this example is skipped.

Usage:
  uv run python examples/quickstart/13_multimodal.py
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore", message=".*coroutine.*was never awaited.*", category=RuntimeWarning
)

from nooa.media import Image
from nooa.util.quickstart import *

ASSETS = Path(__file__).parent.parent / "assets"


def _skip_if_vision_unsupported(error: Exception) -> None:
    text = str(error).lower()
    if any(term in text for term in ("vision", "image", "multimodal")) and any(
        term in text for term in ("unsupported", "not support", "does not support", "invalid")
    ):
        print(
            f"SKIP: quickstart MODEL ({MODEL}) does not support vision.\n"
            "Set MODEL to a vision-capable model (e.g. gpt-4o) and configure\n"
            "OPENAI_API_KEY to run this example."
        )
        sys.exit(0)
    raise error


# ---------------------------------------------------------------------------
# CodeAct agent — LLM generates code, show() injects images
# ---------------------------------------------------------------------------
class ImageDescriber(Agent, llm=llm):
    """You are an agent that describes images concisely."""

    async def describe(self, image: Image) -> str:
        """Describe what you see in this image. Be concise (1-2 sentences)."""
        ...


# ---------------------------------------------------------------------------
# PredictStrategy agent — single-shot structured output with image
# ---------------------------------------------------------------------------
class ImageAnalysis(BaseModel):
    """Structured analysis of an image."""

    colors: list[str]
    text_content: list[str]
    description: str


class ImageAnalyzer(Agent, llm=llm):
    """You are an image analysis agent that returns structured data."""

    @strategy(PredictStrategy())
    async def analyze(self, image: Image) -> ImageAnalysis:
        """Analyze this image. Extract all colors, any text content, and a brief description."""
        ...


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@autorun
async def main():
    print(f"Model: {llm.model}\n")

    try:
        # --- CodeAct: describe the shapes image ---
        shapes_image = Image.from_file(ASSETS / "test_image.png")
        print("=== CodeAct: describe(shapes_image) ===")
        describer = ImageDescriber()
        description = await describer.describe(shapes_image)
        print(f"  {description}\n")

        # --- PredictStrategy: structured analysis of the text image ---
        text_image = Image.from_file(ASSETS / "test_text.png")
        print("=== PredictStrategy: analyze(text_image) ===")
        analyzer = ImageAnalyzer()
        analysis = await analyzer.analyze(text_image)
        print(f"  Colors: {analysis.colors}")
        print(f"  Text:   {analysis.text_content}")
        print(f"  Desc:   {analysis.description}\n")
    except Exception as error:
        _skip_if_vision_unsupported(error)
