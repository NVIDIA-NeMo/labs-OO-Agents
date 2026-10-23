# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PlainBlockFormatter — renders conversation messages as plain text.

Serializes event fields as human-readable text while retaining XML block wrappers.
Event fields marked repr=False in the model are excluded from field serialization;
canonical assistant responses instead retain their original replay content.

Select through RenderConfig, independently of the context view::

    agent = MyAgent(render_config=RenderConfig(block_formatter=PlainBlockFormatter()))

Import RenderConfig from nooa.context_blocks and PlainBlockFormatter from this
module. Materialized blocks retain XML wrappers; event fields use the plain
serialization described below. Cache placement remains the view's decision.
"""

from typing import TYPE_CHECKING

from nooa.agentdoc import pformat
from nooa.context_blocks.events import EventBase
from nooa.context_blocks.formatter import (
    FORMAT_PLAIN,
    FormatType,
    XMLBlockFormatter,
    _is_llm_response,
)
from nooa.context_blocks.roles import Role

if TYPE_CHECKING:
    from nooa.config.truncation_config import FormatConfig


class PlainBlockFormatter(XMLBlockFormatter):  # type: ignore[misc]
    """Renders materialized blocks with XMLBlockFormatter's XML wrappers.

    Event fields are serialized below; event role/tag metadata may still add
    the stock XML message wrapper. This is not a tag-free message format.

    Produces cleaner, more token-efficient messages for the LLM.
    Which fields are shown is controlled by repr=False markers on the event
    model — fields marked repr=False are infrastructure and are skipped.

    Single-field events return the value directly (no wrapper tag).
    Multi-field events use <field>value</field> XML element tags.
    """

    @property
    def format_type(self) -> FormatType:
        return FORMAT_PLAIN

    def format_event(
        self,
        event: EventBase,
        event_format: "FormatConfig | None" = None,
    ) -> str:
        """Serialize event fields not marked repr=False as plain XML elements.

        ``event_format`` provides structural bounds (max_string / max_length /
        max_depth) for nested values within event fields.
        """
        # Canonical assistant text must remain unchanged so the shared replay
        # path can retain native reasoning and exact tool-call data.
        if _is_llm_response(event) or (
            event.event_role is Role.ASSISTANT and isinstance(getattr(event, "content", None), str)
        ):
            return super().format_event(event, event_format=event_format)

        fc_kwargs = event_format.model_dump() if event_format is not None else {}

        def render(value: object) -> str:
            if isinstance(value, str):
                return value
            return pformat(value, **fc_kwargs)

        def is_empty(value: object) -> bool:
            return value is None or value == ""

        items = [
            (name, getattr(event, name))
            for name, fi in type(event).model_fields.items()
            if fi.repr is not False and not is_empty(getattr(event, name))
        ]

        if not items:
            return "(no output)"

        if len(items) == 1:
            return render(items[0][1])

        return "\n".join(f"<{name}>{render(value).rstrip()}</{name}>" for name, value in items)
