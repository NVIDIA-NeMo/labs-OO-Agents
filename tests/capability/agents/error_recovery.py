# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weather lookup agent — tests transient API failure handling."""

from contextvars import ContextVar

from requests.models import Response

from nooa import Agent, hidden

# Task-scoped call counter: each asyncio.Task gets its own copy, so parallel
# eval runs are isolated while instances within the same run share state.
_api_call_count: ContextVar[int] = ContextVar("_api_call_count", default=0)


@hidden
def _call_weather_api() -> Response:
    """
    Simulates an unreliable weather API.

    The first call (per task context) returns a 503 Service Temporarily Unavailable.
    All subsequent calls return a 200 OK with temperature data.

    Returns:
        Response: A Response object with the status code and content.
    """
    call_num = _api_call_count.get()
    _api_call_count.set(call_num + 1)

    if call_num == 0:
        resp = Response()
        resp.status_code = 503
        resp.reason = "Service Temporarily Unavailable"
        resp.headers["Retry-After"] = "Retry after 0.1 seconds."
        return resp
    else:
        resp = Response()
        resp.status_code = 200
        resp.reason = "OK"
        resp._content = b'{"temperature": 17}'
        return resp


class WeatherLookupAgent(Agent):
    """You are an agent that retrieves the current temperature from a weather service."""

    async def get_temperature(self) -> int:
        """Get the current temperature from the weather service."""
        ...

    async def fetch_weather(self) -> int:
        """Fetch the current temperature from the weather API."""
        response = _call_weather_api()
        if response.status_code != 200:
            raise Exception(
                f"Weather API error: status {response.status_code}; "
                f"reason: {response.reason}; headers: {response.headers}"
            )
        return int(response.json()["temperature"])


class WeatherLookupAgentWrapper(Agent):
    async def get_temperature(self) -> int:
        """Get the current temperature from the weather service."""
        _api_call_count.set(0)
        agent = WeatherLookupAgent()
        return await agent.get_temperature()
