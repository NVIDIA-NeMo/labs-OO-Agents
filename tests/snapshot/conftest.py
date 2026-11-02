# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry and render-hook fixtures that undo their registrations, even when a test fails."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from nooa.snapshot import capture


@pytest.fixture
def registered() -> Iterator[Any]:
    """``capture.register(T, shape)`` with cleanup."""
    registered: list[type[Any]] = []

    def register(cls: type[Any], shape: Any) -> None:
        capture.register(cls, shape)
        registered.append(cls)

    yield register
    for cls in registered:
        capture.unregister(cls)


@pytest.fixture
def hooks() -> Iterator[Any]:
    """Register render hooks on a renderer with cleanup: ``hooks(renderer, kind, fn)``."""
    added: list[tuple[Any, str]] = []

    def register(renderer: Any, kind: str, function: Any) -> None:
        renderer.register(kind, function)
        added.append((renderer, kind))

    yield register
    for renderer, kind in added:
        renderer.unregister(kind)
