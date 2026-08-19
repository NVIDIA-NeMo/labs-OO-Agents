# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XML formatter retained for cache-oriented configurations.

Context views now choose prefix, event, and trailing placement. This formatter
only serializes that order and leaves provider cache annotation to adapters.
"""

from nooa.context_blocks.formatter import (
    FORMAT_XML,
    BlockFormatter,
    FormatType,
    _build_messages,
    _xml_message_content,
    _xml_system_block,
)
from nooa.context_blocks.models import RenderedMessage, ResolvedBlock


class CachedBlockFormatter(BlockFormatter):
    """Serialize an already assembled XML context without reordering it."""

    @property
    def format_type(self) -> FormatType:
        return FORMAT_XML

    def format_description(self) -> str:
        return (
            "Your prompt is organized in XML context blocks: `<name>CONTENT</name>`.\n"
            "Blocks produced by `self.context.set_dynamic()` carry an "
            '`expr="..."` attribute whose value is the Python expression '
            "re-evaluated each turn.\n"
            'Event history: system entries in `<sys tag="N">`; '
            'reference via `self.events["N"]`.'
        )

    def format(self, blocks: list[ResolvedBlock]) -> list[RenderedMessage]:
        return _build_messages(
            blocks,
            wrap_system=_xml_system_block,
            wrap_message=_xml_message_content,
        )
