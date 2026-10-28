# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compatibility import for the former cache-oriented XML formatter."""

from nooa.context_blocks.formatter import XMLBlockFormatter


class CachedBlockFormatter(XMLBlockFormatter):
    """Legacy XML formatter name; caching is controlled by context views and clients.

    Serialization and format guidance are identical to :class:`XMLBlockFormatter`.
    XML and Markdown both preserve the cache boundaries selected by the view.
    """


__all__ = ["CachedBlockFormatter"]
