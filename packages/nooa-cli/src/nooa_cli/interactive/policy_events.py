# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Semantic policy notices shared by presentation adapters."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TurnStatus:
    """A turn-policy status, distinct from the agent's reply to the user."""

    kind: str
    explanation: str
