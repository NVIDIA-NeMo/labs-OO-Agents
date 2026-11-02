# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composable context assembly contracts and reusable mechanisms."""

import logging
from collections.abc import AsyncIterator, Iterable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from nooa.context_blocks.events import EventBase, ResultStatus, ToolCallEvent, ToolResult
from nooa.context_blocks.models import BlockMetadata
from nooa.context_blocks.roles import Role
from nooa.llm_types import CacheBoundary, LLMResponse

if TYPE_CHECKING:
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)


class Block(BaseModel):
    """One materialized context block."""

    model_config = ConfigDict(frozen=True)

    key: str
    content: object
    role: Role = Role.SYSTEM
    metadata: BlockMetadata | None = None


type ContextItem = Block | EventBase | CacheBoundary


@runtime_checkable
class ContextView[Owner](Protocol):
    """Produce an ordered context stream for an owner."""

    def assemble(
        self,
        owner: Owner,
        call: "CurrentCall",
    ) -> AsyncIterator[ContextItem]: ...


def is_model_visible(event: EventBase) -> bool:
    """Whether an event carries model-facing content, without selecting history."""
    return event._role not in (Role.RUNTIME_EVENT, Role.METADATA) and not event.is_empty


def replayable(events: Iterable[EventBase]) -> tuple[EventBase, ...]:
    """Make a selection of events sendable without changing stored history.

    Every tool call needs its result and every linked result its call. A call
    without a recorded result gets an outcome-unknown result on a copy. A
    tool-call turn missing any of its executions is omitted with its results,
    as are results whose turn is not in the selection. Run it after any
    filtering, since filtering can separate calls from results.
    """
    events = tuple(events)
    executions: dict[str, dict[str, ToolCallEvent]] = {}
    for event in events:
        if isinstance(event, ToolCallEvent) and event.llm_response_id is not None:
            executions.setdefault(event.llm_response_id, {})[event.tool_call_id] = event
    complete: set[str] = set()
    for event in events:
        if isinstance(event, LLMResponse) and event.replay_tool_calls:
            ids = [call.id for call in event.replay_tool_calls]
            # Duplicate ids cannot be paired with executions.
            if len(set(ids)) == len(ids) and set(ids) <= executions.get(event.id, {}).keys():
                complete.add(event.id)
            else:
                logger.warning(
                    "LLMResponse %s has an incomplete execution batch; omitting the "
                    "tool-call turn and its results from replay.",
                    event.id,
                )

    selected: list[EventBase] = []
    for event in events:
        if isinstance(event, LLMResponse) and event.replay_tool_calls:
            if event.id not in complete:
                continue
        elif isinstance(event, ToolCallEvent):
            if event.llm_response_id is not None and event.llm_response_id not in complete:
                continue
            if event.result is None:
                event = event.model_copy(
                    update={
                        "result": ToolResult(
                            tool_call_id=event.tool_call_id,
                            content=(
                                "Tool result unavailable in the active history; "
                                "execution outcome is unknown."
                            ),
                            result_status=ResultStatus.ERROR,
                        )
                    }
                )
        selected.append(event)
    return tuple(selected)


async def collect_context_items[Owner](
    view: ContextView[Owner], owner: Owner, call: "CurrentCall"
) -> tuple[ContextItem, ...]:
    """Collect and validate a view into one immutable snapshot."""
    items: list[ContextItem] = []
    async for item in view.assemble(owner, call):
        if not isinstance(item, (Block, EventBase, CacheBoundary)):
            raise TypeError(
                f"{type(view).__name__}.assemble() yielded {type(item).__name__}; "
                "expected Block, EventBase, or CacheBoundary"
            )
        items.append(item)
    return tuple(items)


def context_text(value: Any, *, call: "CurrentCall") -> str:
    """Render a value with the call's bounded context-block format."""
    if isinstance(value, str):
        return value

    from nooa.agentdoc import pformat

    kwargs = call.context_format.model_dump() if call.context_format is not None else {}
    return pformat(value, unquote_strings=True, **kwargs)


async def evaluate_context_expression(
    expression: str,
    *,
    owner: Any,
    call: "CurrentCall",
) -> Any:
    """Evaluate a declarative expression with ``self`` bound to ``owner``."""
    agent = call.agent
    runtime = getattr(agent, "runtime", None)
    if runtime is None:
        raise RuntimeError("Expression evaluation requires an agent-bound CurrentCall")

    return await runtime.evaluate_expression(
        expression,
        extra_context={
            "self": owner,
            "call": call,
            "method": call.method,
            "call_args": call.args,
            "call_kwargs": call.kwargs,
            "strategy": call.strategy,
            "datetime": datetime,
            "runtime": runtime,
        },
        error_mode="raise",
    )


def apply_context_budget(
    items: Sequence[ContextItem],
    *,
    call: "CurrentCall",
    evictable: Sequence[Block],
) -> tuple[ContextItem, ...]:
    """Apply the call budget by replacing candidates in the supplied priority."""
    limit = call.context_budget
    if limit is None:
        return tuple(items)
    counter = call.context_token_counter
    if counter is None:
        from nooa.token_counter import char_approximate_token_counter

        counter = char_approximate_token_counter

    result = list(items)
    total = sum(
        counter(context_text(item.content, call=call)) for item in result if isinstance(item, Block)
    )
    if total <= limit:
        return tuple(result)

    by_identity = {id(item): index for index, item in enumerate(result)}
    for candidate in evictable:
        if total <= limit:
            break
        index = by_identity.get(id(candidate))
        if index is None:
            continue
        block = result[index]
        if not isinstance(block, Block):
            continue
        size = counter(context_text(block.content, call=call))
        replacement = f"EVICTED: over context budget (block_tokens={size:,})"
        update: dict[str, Any] = {"content": replacement}
        metadata = block.metadata if block.metadata is not None else BlockMetadata()
        update["metadata"] = metadata.model_copy(update={"truncated": True})
        result[index] = block.model_copy(update=update)
        total += counter(replacement) - size
    return tuple(result)


__all__ = [
    "Block",
    "CacheBoundary",
    "ContextItem",
    "ContextView",
    "apply_context_budget",
    "collect_context_items",
    "context_text",
    "evaluate_context_expression",
    "is_model_visible",
    "replayable",
]
