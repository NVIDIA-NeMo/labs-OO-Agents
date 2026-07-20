# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import httpx
from pydantic import BaseModel, ConfigDict


class HttpConfig(BaseModel):
    """Per-client HTTP connection-pool and timeout settings.

    Each ``CompletionClient`` / ``ResponsesClient`` builds its **own** httpx
    client from its ``HttpConfig`` and hands it to litellm on a per-call basis
    (via litellm's caller-provided-client support). The settings therefore apply
    **only to that client's requests** — there is no global state, no
    module-level ``_active_http_config``, and no monkey-patch of
    ``httpx.AsyncClient``. Two clients constructed with different ``HttpConfig``s
    are fully independent, and an unrelated ``httpx.AsyncClient`` created by other
    code in the process is never affected.

    The default (``max_keepalive_connections=0``) disables connection pooling to
    avoid CLOSE_WAIT hangs. ``keepalive_expiry`` defaults to httpx's normal
    5-second idle timeout so a client that opts into pooling by overriding
    ``max_keepalive_connections`` gets actual connection reuse without also
    having to remember to override the expiry.
    """

    model_config = ConfigDict(frozen=True)

    max_connections: int = 100
    max_keepalive_connections: int = 0  # 0 = disabled, prevents CLOSE_WAIT
    keepalive_expiry: float = 5.0
    connect_timeout: float = 10.0
    read_timeout: float = 60.0  # catches CLOSE_WAIT hangs
    write_timeout: float = 10.0
    pool_timeout: float = 10.0

    def to_httpx_limits(self) -> httpx.Limits:
        """Build the httpx connection-pool limits for this config."""
        return httpx.Limits(
            max_connections=self.max_connections,
            max_keepalive_connections=self.max_keepalive_connections,
            keepalive_expiry=self.keepalive_expiry,
        )

    def to_httpx_timeout(self) -> httpx.Timeout:
        """Build the httpx timeout for this config.

        ``read_timeout`` is the gap between received bytes (reset on every byte),
        which catches frozen CLOSE_WAIT connections without killing long but
        healthy streaming responses.
        """
        return httpx.Timeout(
            connect=self.connect_timeout,
            read=self.read_timeout,
            write=self.write_timeout,
            pool=self.pool_timeout,
        )
