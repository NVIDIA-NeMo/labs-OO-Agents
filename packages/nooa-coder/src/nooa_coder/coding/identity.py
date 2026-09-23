# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical coding-agent identities and legacy configuration spellings."""

CODING_AGENT = "nooa_coder.coding.agent:CodingAgent"
EXPERIMENTAL_CODING_AGENT = "nooa_coder.coding.experimental_agent:ExperimentalCodingAgent"
LEGACY_AGENT_SPECS = {
    "nooa_cli.tui.agent:TUIAgent": CODING_AGENT,
    "nooa_coder.coding.legacy_agent:TUIAgent": CODING_AGENT,
    "nooa_cli.tui.experimental_agent:ExperimentalTUIAgent": EXPERIMENTAL_CODING_AGENT,
    "nooa_coder.coding.experimental_agent:ExperimentalTUIAgent": EXPERIMENTAL_CODING_AGENT,
}


def canonical_agent_spec(spec: str) -> str:
    return LEGACY_AGENT_SPECS.get(spec, spec)
