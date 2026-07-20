# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""No-op context manager for backward compatibility.

``with visible:`` is a no-op because everything is visible by default in
nooa. This exists purely for compatibility with agent files that were
written against an older API that had an explicit ``visible`` context manager.
"""

from typing import Literal, Self


class _Visible:
    """No-op context manager — everything is visible by default."""

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> Literal[False]:
        return False

    def __repr__(self) -> str:
        return "visible"


visible = _Visible()
