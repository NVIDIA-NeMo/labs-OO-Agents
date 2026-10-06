# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical coding-agent identities and legacy spellings (defined in ``session.loader``)."""

from nooa_coder.session.loader import (
    CODING_AGENT,
    EXPERIMENTAL_CODING_AGENT,
    LEGACY_AGENT_SPECS,
    canonical_agent_spec,
)

__all__ = [
    "CODING_AGENT",
    "EXPERIMENTAL_CODING_AGENT",
    "LEGACY_AGENT_SPECS",
    "canonical_agent_spec",
]
