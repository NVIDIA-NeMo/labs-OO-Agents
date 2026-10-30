# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quickstart 13: Images with CodeAct and structured Predict analysis.

Setup: run ``uv run nooa connect``, save a vision-capable route as ``vision``,
and choose image in its media_inputs (YAML: ``media_inputs: [image]``).
Connect also configures the reply token limit and known or supplied context size.
Enable image inputs for the saved model before running this example.

Usage:
  uv run python examples/quickstart/13_multimodal.py
  uv run python examples/quickstart/13_multimodal.py --model vision --config FILE

Without --config, the saved aliases use normal registry discovery. --config
reloads only FILE. Importing this module neither constructs a client nor runs it.
"""

import argparse
import asyncio
from pathlib import Path

from pydantic import BaseModel

from nooa import Agent, hidden, strategy
from nooa.media import Image
from nooa.strategies import PredictStrategy
from nooa.unifiedllm import UnifiedLLM, get_llm_client, reload_registry

ASSETS = Path(__file__).resolve().parent.parent / "assets"


class ImageDescriber(Agent):
    """Describe images using the default CodeAct strategy."""

    async def describe(self, image: Image) -> str:
        """Describe what you see in this image. Be concise (1-2 sentences)."""
        ...


class ImageAnalysis(BaseModel):
    """Structured analysis of an image."""

    colors: list[str]
    text_content: list[str]
    description: str


class ImageAnalyzer(Agent):
    """Analyze images with structured output and no tool loop."""

    @strategy(PredictStrategy())
    async def analyze(self, image: Image) -> ImageAnalysis:
        """Extract all colors, any text content, and a brief description of the image."""
        ...


@hidden
async def run_example(llm: UnifiedLLM) -> None:
    """Run both image paths; the caller owns the injected client."""
    if "image" not in llm.media_inputs:
        raise ValueError(
            "This example requires media_inputs: [image]. Run `uv run nooa connect`, "
            "save a vision-capable alias (default: vision), and enable image inputs."
        )

    print(f"Model: {llm.model}\n")
    # CodeAct's default prefill calls show(image), attaching the image to the LLM turn.
    describer = ImageDescriber(llm=llm)
    try:
        description = await describer.describe(Image.from_file(ASSETS / "test_image.png"))
        print(f"=== CodeAct: describe ===\n{description}\n")
    finally:
        await describer.aclose()

    # Predict attaches the Image argument directly to its task content blocks.
    analyzer = ImageAnalyzer(llm=llm)
    try:
        analysis = await analyzer.analyze(Image.from_file(ASSETS / "test_text.png"))
        print("=== PredictStrategy: analyze ===")
        print(analysis.model_dump_json(indent=2))
    finally:
        await analyzer.aclose()


@hidden
async def main(model: str = "vision", config: Path | None = None) -> None:
    if config is not None:
        if not config.is_file():
            raise FileNotFoundError(f"Config file does not exist: {config}")
        reload_registry(config)
    llm = get_llm_client(model)
    try:
        await run_example(llm)
    finally:
        await llm.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="vision", help="Saved Connect alias (default: vision)")
    parser.add_argument("--config", type=Path, help="Reload aliases from only this YAML file")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.config))
