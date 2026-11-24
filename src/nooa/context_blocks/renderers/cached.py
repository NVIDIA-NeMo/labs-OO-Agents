# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility wrapper for the historical cache-oriented formatter import."""

from nooa.context_blocks.formatter import XMLBlockFormatter


class CachedBlockFormatter(XMLBlockFormatter):
    """XML compatibility wrapper; cache placement remains view-owned."""
