# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shell-owned numeric calibration, independent of retry buffers and billing.

A frozen call-local proxy captures the association once. Each send explicitly
passes a fresh private owner to the exact built-in implementation and consumes
its single numeric record even when parsing raises. No context-global ownership,
shared adapter/client mutation, callback API, or third-party protocol extension.
Legacy count_tokens still consumes the legacy singleton; route-estimate
consumption and reply clamping belong to the next loader migration step.
"""

from dataclasses import dataclass

from nooa.llm_types import LLMResponse

from ._observations import _AttemptWriter, _ObservationOwner
from .adapters._calibration import _route_calibration, _RouteCalibration
from .adapters.base import Adapter
from .adapters.litellm_chat import LiteLLMChatAdapter
from .adapters.litellm_responses import LiteLLMResponsesAdapter
from .request import LLMRequest


@dataclass(frozen=True, slots=True)
class _CalibrationAdapter:
    adapter: LiteLLMChatAdapter | LiteLLMResponsesAdapter
    record: _RouteCalibration

    @property
    def capabilities(self):
        return self.adapter.capabilities

    def _update(self, owner: _ObservationOwner) -> None:
        observation = owner.writer.record
        estimated = observation.raw_input_estimate
        usage = observation.usage
        actual = usage.input_tokens if usage is not None else None
        if (
            observation.finished
            and observation.response_received
            and estimated is not None
            and actual is not None
            and estimated > 0
            and actual > 0
        ):
            self.record.update(estimated, actual)

    def send_sync(self, request: LLMRequest) -> LLMResponse:
        owner = _ObservationOwner(_AttemptWriter())
        try:
            return self.adapter._send_sync(request, _owner=owner)
        finally:
            self._update(owner)

    async def send(self, request: LLMRequest) -> LLMResponse:
        owner = _ObservationOwner(_AttemptWriter())
        try:
            return await self.adapter._send(request, _owner=owner)
        finally:
            self._update(owner)


def _calibration_adapter(adapter: Adapter) -> Adapter:
    # Exact built-ins only: subclasses/custom adapters retain their send hooks
    # and require no calibration, copying or additional methods.
    if type(adapter) not in (LiteLLMChatAdapter, LiteLLMResponsesAdapter):
        return adapter
    return _CalibrationAdapter(adapter, _route_calibration(adapter))
