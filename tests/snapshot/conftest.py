# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry and render-hook fixtures that undo their registrations, even when a test fails."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

from nooa.snapshot import capture


@pytest.fixture
def lazy() -> Iterator[Any]:
    """``capture.register_lazy`` with cleanup."""
    names: list[tuple[str, str]] = []

    def register(module: str, qualname: str, function: Any) -> None:
        capture.register_lazy(module, qualname, function)
        names.append((module, qualname))

    yield register
    for module, qualname in names:
        capture.unregister_lazy(module, qualname)


@pytest.fixture
def eager() -> Iterator[Any]:
    registered: list[type[Any]] = []

    def register(cls: type[Any], function: Any) -> None:
        capture.register(cls, function)
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
