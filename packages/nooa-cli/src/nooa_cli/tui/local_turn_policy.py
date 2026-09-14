# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapt shared turn-policy notices to terminal outputs."""

from nooa_cli.interactive.local_turn_policy import LocalTurnPolicy as SharedTurnPolicy

from .output import StopReasonOutput


class LocalTurnPolicy(SharedTurnPolicy):
    def __init__(self, agent, runtime, *, emit_output, invalidate):
        async def emit(status):
            await emit_output(StopReasonOutput(status.kind, status.explanation))

        super().__init__(agent, runtime, emit_output=emit, invalidate=invalidate)
