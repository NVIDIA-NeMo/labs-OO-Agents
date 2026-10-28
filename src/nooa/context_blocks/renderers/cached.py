# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XML formatter retained for cache-oriented configurations.

Context views choose prefix, event, boundary, and trailing placement. This
formatter only serializes that order; UnifiedLLM maps cache boundaries to the
selected provider.
"""

from nooa.context_blocks.formatter import XMLBlockFormatter


class CachedBlockFormatter(XMLBlockFormatter):
    """XML formatting with the cache-oriented configuration's prompt description."""

    def format_description(self) -> str:
        return (
            "Your prompt is organized in XML context blocks: `<name>CONTENT</name>`.\n"
            "Expression context blocks carry an "
            '`expr="..."` attribute whose value is the Python expression '
            "re-evaluated each turn.\n"
            'Event history: system entries in `<sys tag="N">`; '
            'reference via `self.events["N"]`.'
        )
