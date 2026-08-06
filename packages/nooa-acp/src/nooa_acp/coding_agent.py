# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility exports for the host-neutral NOOA coding agent."""

from nooa_cli.coding import CodingAgent

from nooa.interactive import RespondReason

# Keep the original public name for callers of the preview ACP package while
# making the implementation identical across protocol and terminal hosts.
CodingInteractiveAgent = CodingAgent

__all__ = ["CodingAgent", "CodingInteractiveAgent", "RespondReason"]
