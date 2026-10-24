# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native single-attempt attachment access self-reports, not accuracy scores."""

from typing import Annotated

from pydantic import Field

from nooa import Agent, no_trace
from nooa.config.strategy_config import PredictConfig
from nooa.decorators import strategy
from nooa.media import Audio, File, Image, Video
from nooa.strategies.predict import PredictStrategy

from ._baseline import _probe_agent, _typed_call
from ._media import pdf_fixture, png_fixture, video_fixture, wav_fixture


class _ImageAccess(Agent):
    @no_trace
    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def access(self, image: Image) -> Annotated[bool, Field(strict=True)]:
        """Can you access and interpret the attached image? Return false if you cannot access or interpret its contents."""
        ...


class _AudioAccess(Agent):
    @no_trace
    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def access(self, audio: Audio) -> Annotated[bool, Field(strict=True)]:
        """Can you access and interpret the attached audio? Return false if you cannot access or interpret its contents."""
        ...


class _VideoAccess(Agent):
    @no_trace
    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def access(self, video: Video) -> Annotated[bool, Field(strict=True)]:
        """Can you access and interpret the attached video? Return false if you cannot access or interpret its contents."""
        ...


class _DocumentAccess(Agent):
    @no_trace
    @strategy(PredictStrategy(config=PredictConfig(max_retries=1)))
    async def access(self, document: File) -> Annotated[bool, Field(strict=True)]:
        """Can you access and interpret the attached document? Return false if you cannot access or interpret its contents."""
        ...


async def typed_media(client, kind, *, observe_output_model=None):
    """The standard Predict media collector attaches each original argument once."""
    contracts = {
        "image": (_ImageAccess, Image, png_fixture, "image/png"),
        "audio": (_AudioAccess, Audio, wav_fixture, "audio/wav"),
        "video": (_VideoAccess, Video, video_fixture, "video/mp4"),
        "pdf": (_DocumentAccess, File, pdf_fixture, "application/pdf"),
    }
    cls, media_type, fixture, mime = contracts[kind]
    agent = _probe_agent(cls, llm=client)
    return await _typed_call(
        agent,
        agent.access,
        media_type.from_bytes(fixture(), media_type=mime),
        observe_output_model=observe_output_model,
    )
