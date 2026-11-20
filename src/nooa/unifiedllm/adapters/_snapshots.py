# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provider-neutral container snapshots; opaque resources keep their identity."""

from collections.abc import Mapping
from typing import Any


def snapshot_options(value: Any) -> Any:
    """Detach settings containers, preserving opaque SDK clients and transports."""
    if isinstance(value, Mapping):
        return {key: snapshot_options(item) for key, item in value.items()}
    if isinstance(value, list):
        return [snapshot_options(item) for item in value]
    if isinstance(value, tuple):
        return tuple(snapshot_options(item) for item in value)
    return value
