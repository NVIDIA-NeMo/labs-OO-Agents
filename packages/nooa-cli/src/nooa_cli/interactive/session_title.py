# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request session housekeeping when either host consumes its first prompt."""

from typing import Any


class SessionTitleRequest:
    """Request a title once per attached session, preserving existing names."""

    def __init__(self) -> None:
        self._manager: Any = None

    def request(self, agent: Any, opening_message: str) -> bool:
        manager = getattr(agent, "_session_manager", None)
        if manager is None or manager is self._manager:
            return False
        self._manager = manager
        info = manager.info
        if info.title_is_user_set or (info.title or "").strip():
            return False
        request_title = getattr(agent, "request_session_title", None)
        if not callable(request_title):
            return False
        request_title(opening_message)
        return True
