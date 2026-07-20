# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core types for context blocks.

DynamicContext: Marks a context block for dynamic evaluation each turn.
ResolvedBlock: A fully-resolved block ready for rendering.
BlockMetadata: Typed metadata for resolved blocks.
RenderedMessage: Neutral in-memory message emitted by a BlockFormatter,
    consumed by a ProviderFormatter to produce provider-specific output.
ToolCallInfo: Structured tool-call payload carried on a RenderedMessage.
Role: Re-exported from roles.py for backward compatibility.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# Import EventBase here (not forward ref) — possible because events.py imports
# Role from roles.py, breaking the circular dependency.
from nooa.context_blocks.events import EventBase
from nooa.context_blocks.exceptions import BlockSyntaxError
from nooa.context_blocks.roles import Role  # noqa: F401 — re-exported for backward compat


class DynamicContext(BaseModel):
    """Marks a context block for dynamic evaluation each turn.

    Wraps a Python expression string that will be evaluated by the runtime
    at each LLM turn. The expression is validated at creation time.

    Usage:
        self.context.set_dynamic("status", "self.format_status()")
        self.context.set_dynamic("progress", "self.todo.show_active()")

    The expression must be valid Python (compilable as an eval expression).
    """

    model_config = ConfigDict(frozen=True)

    expr: Annotated[str, Field(description="Python expression to evaluate each turn")]

    def __init__(self, expr: str, **kwargs: Any):
        """Create a DynamicContext block marker.

        Args:
            expr: Python expression to evaluate each turn.

        Raises:
            BlockSyntaxError: If expr is not valid Python syntax.
        """
        try:
            compile(expr, "<block_expr>", "eval")
        except SyntaxError as e:
            raise BlockSyntaxError(key="<dynamic>", expr=expr, original_error=e) from e
        super().__init__(expr=expr, **kwargs)

    def __repr__(self) -> str:
        return f"DynamicContext({self.expr!r})"


class Context(BaseModel):
    """Unified context block value — controls content and placement.

    Two orthogonal axes:
    - **Content**: literal text (``value``) or re-evaluated expression (``expr``)
    - **Placement**: cacheable prefix (``prefix=True``) or volatile suffix (default)

    Usage in context dicts (class-level, instance-level, @strategy, ScopedContext, runtime)::

        from nooa import Context

        context={
            "role": "You are an expert",                              # bare str shorthand (suffix, literal)
            "shell": Context(expr="doc(self.shell)", prefix=True),    # prefix, evaluated each turn
            "config": Context("stable config", prefix=True),          # prefix, literal
            "status": Context(expr="f'{self.done}/{self.total}'"),    # suffix, evaluated
            "self": None,                                             # suppress block
        }

    At runtime::

        self.context["shell"] = Context(expr="doc(self.shell)", prefix=True)
        self.context["status"] = Context(expr="f'{self.done}/{self.total}'")
        self.context["self"] = None  # suppress
    """

    model_config = ConfigDict(frozen=True)

    value: str | None = Field(default=None, description="Literal text content")
    expr: str | None = Field(
        default=None, description="Python expression re-evaluated each LLM turn"
    )
    prefix: bool = Field(
        default=False, description="Place in cacheable prefix if True, volatile suffix if False"
    )

    def __init__(
        self,
        value: str | None = None,
        *,
        expr: str | None = None,
        prefix: bool = False,
        **kwargs: Any,
    ):
        """Create a context block value.

        Args:
            value: Literal text content (mutually exclusive with expr).
            expr: Python expression re-evaluated each LLM turn (mutually exclusive with value).
            prefix: If True, place in the cacheable prefix partition.

        Raises:
            TypeError: If both value and expr are given, or neither is given.
            BlockSyntaxError: If expr is not valid Python syntax.
        """
        if value is not None and expr is not None:
            raise TypeError("Context() takes value or expr, not both")
        if value is None and expr is None:
            raise TypeError("Context() requires value or expr")
        if expr is not None:
            try:
                compile(expr, "<block_expr>", "eval")
            except SyntaxError as e:
                raise BlockSyntaxError(key="<context>", expr=expr, original_error=e) from e
        super().__init__(value=value, expr=expr, prefix=prefix, **kwargs)

    @property
    def is_dynamic(self) -> bool:
        """True if this block uses an expression (re-evaluated each turn)."""
        return self.expr is not None

    def to_dynamic_context(self) -> "DynamicContext | None":
        """Convert to a DynamicContext if this is an expression block, else None."""
        if self.expr is not None:
            return DynamicContext(self.expr)
        return None

    def __repr__(self) -> str:
        if self.expr is not None:
            parts = [f"expr={self.expr!r}"]
        else:
            parts = [repr(self.value)]
        if self.prefix:
            parts.append("prefix=True")
        return f"Context({', '.join(parts)})"


class BlockMetadata(BaseModel):
    """Typed metadata for resolved blocks.

    Replaces the untyped dict[str, Any] with well-defined fields.
    Used by formatters and provider formatters to render blocks correctly.
    """

    model_config = ConfigDict(frozen=True)

    expr: str | None = Field(default=None, description="Python expression for accessing this block")
    tag: str | None = Field(default=None, description="Event position tag")
    truncated: bool = Field(default=False, description="Whether content was truncated")
    user_block: bool = Field(
        default=False,
        description="Whether this is a user-set block (from self.context). "
        "User blocks are dropped first during context truncation.",
    )
    static: bool = Field(
        default=False,
        description="Author's declaration that the block's content is stable across "
        "turns. Renderers use this to place the block in a cacheable prefix. "
        "Does not affect truncation or correctness — only caching behavior.",
    )
    source_dynamic: bool = Field(
        default=False,
        description="Whether this block came from self.context.set_dynamic(). "
        "Only these blocks render their ``expr`` attribute — static blocks, "
        "framework blocks, strategy overrides, and events suppress it.",
    )


class ResolvedBlock(BaseModel):
    """A fully-resolved block ready for rendering.

    All content has been evaluated — no expressions, no Dynamic markers.
    The renderer receives these and formats them without any evaluation.

    For event blocks, the original event is carried through on the ``event``
    field. Provider formatters use this for tool call events (which need
    structured fields like function name, arguments, tool_call_id) and to
    read the event's ``_role``.
    """

    model_config = ConfigDict(frozen=True)

    key: Annotated[str, Field(description="Unique identifier for the block")]
    content: Annotated[str, Field(description="Pre-resolved string content")]
    role: Role = Field(
        default=Role.SYSTEM,
        description="Message role (SYSTEM for system prompt, USER/ASSISTANT/TOOL for messages)",
    )
    metadata: BlockMetadata = Field(
        default_factory=BlockMetadata, description="Block metadata for rendering"
    )
    event: EventBase | None = Field(
        default=None, description="Original event, if this block represents one"
    )


class ToolCallInfo(BaseModel):
    """Structured tool-call payload on a :class:`RenderedMessage`.

    Provider formatters reshape this into the appropriate wire format
    (OpenAI ``tool_calls`` array entry, Anthropic ``tool_use`` content block).
    """

    model_config = ConfigDict(frozen=True)

    id: Annotated[str, Field(description="Tool call id (matches the result's tool_call_id)")]
    name: Annotated[str, Field(description="Tool name")]
    arguments: Annotated[dict[str, Any], Field(description="Tool arguments as a plain dict")]


class TextPart(BaseModel):
    """Literal text inside a :class:`RenderedMessage`.

    The ``kind`` tag exists so this type can participate in a
    discriminated union with :class:`BlockPart` without Pydantic having
    to introspect structure.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["text"] = "text"
    text: Annotated[str, Field(description="Literal text")]


class BlockPart(BaseModel):
    """A reference to a :class:`ResolvedBlock` inside a :class:`RenderedMessage`.

    Emitted by a block-aware ``BlockFormatter`` alongside (or instead of)
    the bulk ``content`` string so downstream consumers — the journal
    publisher, trace viewer reconstruction — can identify block
    boundaries within the rendered message and content-address each
    block separately for dedup across turns.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["block"] = "block"
    key: Annotated[str, Field(description="Block key (e.g. 'system_prompt')")]
    content: Annotated[str, Field(description="Rendered representation of the block")]


MessagePart = Annotated[TextPart | BlockPart, Field(discriminator="kind")]


class RenderedMessage(BaseModel):
    """Neutral message emitted by a BlockFormatter.

    The BlockFormatter is responsible for ordering the system prompt, the
    event history, and any additional context messages into a single
    ``list[RenderedMessage]``. The ProviderFormatter is a thin adapter that
    converts this list into provider-specific wire format.

    Fields are optional and combine based on message kind:

    * A plain text message sets ``role`` and ``content``.
    * An assistant tool call sets ``role=ASSISTANT`` and ``tool_call`` (and
      leaves ``content=None``).
    * A tool result sets ``role=TOOL``, ``tool_call_id`` to the matching call
      id, and ``content`` to the result text.
    * A multimodal message sets ``content`` to the text and ``images`` to a
      list of provider-agnostic image part dicts (LiteLLM's ``image_url``
      shape, which all providers support).

    Block-aware formatters additionally populate ``parts``, a list of
    :class:`TextPart` / :class:`BlockPart` entries whose concatenated
    text equals ``content``. ``parts`` is what the journal publisher
    walks to build a skeleton + content-addressed block store. Provider
    formatters continue to read ``content`` and are unaware of parts.
    """

    model_config = ConfigDict(frozen=True)

    role: Role = Field(description="Message role (SYSTEM / USER / ASSISTANT / TOOL)")
    content: str | None = Field(
        default=None, description="Text content, pre-serialized by the BlockFormatter"
    )
    parts: list[MessagePart] | None = Field(
        default=None,
        description=(
            "Optional block-aware breakdown of ``content`` into literal "
            "text and block references. Present on messages emitted by "
            "block-aware formatters; absent for plain event / tool-result "
            "messages where no blocks are involved."
        ),
    )
    tool_call: ToolCallInfo | None = Field(
        default=None, description="Assistant tool-call payload, if any"
    )
    tool_call_id: str | None = Field(
        default=None, description="Tool-call id this message is a result for"
    )
    images: list[dict[str, Any]] | None = Field(
        default=None, description="Optional image parts (LiteLLM shape)"
    )


class ContextWindowStats(BaseModel):
    """Context window utilization snapshot.

    The single source of truth for token usage is ``prompt_tokens`` — the
    exact prompt-token count reported by the provider in the response usage
    of the most recent successful call. There is **no local token estimate**:
    before the first response ``prompt_tokens`` is ``None`` and all derived
    token figures are ``None`` too.

    ``render_context()`` populates only the structural fields (block/event
    *counts* and raw *character* sizes); the runtime writes ``prompt_tokens``
    back after the provider returns usage. The per-category token breakdown
    (context blocks vs. events) is **attributed** from the authoritative
    ``prompt_tokens`` total in proportion to each category's character share —
    the provider reports a single prompt-token number, never a breakdown, so
    the split is approximate but always sums exactly to ``prompt_tokens``.
    """

    # extra="forbid": reject the removed token-estimate kwargs
    # (total_tokens/context_blocks_tokens/events_tokens/max_event_tokens) so a
    # stale caller fails loudly instead of silently producing None.
    model_config = ConfigDict(frozen=True, extra="forbid")

    context_blocks_count: Annotated[
        int, Field(description="Number of system blocks (post-eviction)")
    ]
    events_count: Annotated[int, Field(description="Number of event blocks")]
    prompt_tokens: Annotated[
        int | None,
        Field(
            description=(
                "Provider-reported prompt tokens from the latest response usage. "
                "None until the first successful call returns usage."
            )
        ),
    ] = None
    context_blocks_chars: Annotated[
        int, Field(description="Raw character size of system blocks (for breakdown attribution)")
    ] = 0
    events_chars: Annotated[
        int, Field(description="Raw character size of event blocks (for breakdown attribution)")
    ] = 0
    max_context_tokens: Annotated[
        int | None,
        Field(description="Configured eviction budget for context blocks (informational)"),
    ] = None
    model_context_window: Annotated[
        int | None, Field(description="Model's total context window size (the % denominator)")
    ] = None
    context_blocks_dropped: Annotated[
        int, Field(description="System blocks marked EVICTED during eviction")
    ] = 0
    events_dropped: Annotated[int, Field(description="Events dropped during truncation")] = 0
    reserved_output_tokens: Annotated[
        int | None,
        Field(
            description=(
                "Tokens reserved for the model's response on the next call "
                "(the call's max_tokens when known, else the configured "
                "response reserve). Subtracted from the window when computing "
                "utilization: a prompt only fits if prompt + output ≤ window."
            )
        ),
    ] = None

    @property
    def total_tokens(self) -> int | None:
        """Provider-reported prompt tokens, or None before the first response."""
        return self.prompt_tokens

    @property
    def context_blocks_tokens(self) -> int | None:
        """Provider total attributed to context blocks by character share.

        None until ``prompt_tokens`` is available.
        """
        if self.prompt_tokens is None:
            return None
        total_chars = self.context_blocks_chars + self.events_chars
        if total_chars <= 0:
            return 0
        return round(self.prompt_tokens * self.context_blocks_chars / total_chars)

    @property
    def events_tokens(self) -> int | None:
        """Provider total attributed to events (= prompt_tokens − context blocks).

        Defined as the remainder so the breakdown sums exactly to
        ``prompt_tokens``. None until ``prompt_tokens`` is available.
        """
        if self.prompt_tokens is None:
            return None
        return self.prompt_tokens - (self.context_blocks_tokens or 0)

    @property
    def effective_window(self) -> int | None:
        """Usable input window: model window minus the output-token reserve.

        The provider rejects any call where prompt + completion budget exceeds
        the window, so the honest denominator for "how full are we" is
        ``model_context_window - reserved_output_tokens``. Falls back to the
        raw window when no reserve is known. None if the window is unknown.
        """
        if not self.model_context_window:
            return None
        reserve = self.reserved_output_tokens or 0
        return max(1, self.model_context_window - reserve)

    @property
    def overall_utilization(self) -> float | None:
        """Fraction of the *usable* context window consumed, or None if unknown.

        Usable = model window minus ``reserved_output_tokens`` (see
        :attr:`effective_window`). Can exceed 1.0 when the prompt has grown
        into the output reserve — the next call at full ``max_tokens`` would
        be rejected by the provider.
        """
        if self.prompt_tokens is None:
            return None
        window = self.effective_window
        if not window:
            return None
        return self.prompt_tokens / window

    def format(self) -> str:
        """Human-readable context window summary, suitable for a context block.

        Before the first provider response there is no token count yet::

            Context usage: awaiting first model response (no provider token count yet)

        With provider usage and a known model window::

            Context usage: 12,450 / 200,000 tokens (6.2%) [provider-reported]
              Context blocks: ~8,200 tokens — 6 blocks
              Events:         ~4,250 tokens — 18 events

        The header total is the exact provider count; the per-category lines
        are attributed from it by character share (prefixed ``~``).
        """
        if self.prompt_tokens is None:
            return (
                "Context usage: awaiting first model response (no provider token count yet)\n"
                "Free space by collapsing older event history with "
                "self.events.collapse(start_tag, end_tag, summary_text=...); "
                "use doc(self.events) for the available event-history tools. "
                "Use self.context (ContextApi) to summarize or remove large "
                "context blocks."
            )

        lines: list[str] = []

        # --- Header line: exact provider total over the usable window ---
        window = self.model_context_window
        usable = self.effective_window
        if window and usable:
            pct = self.prompt_tokens / usable * 100
            reserve = self.reserved_output_tokens or 0
            if reserve:
                lines.append(
                    f"Context usage: {self.prompt_tokens:,} / {usable:,} usable tokens "
                    f"({pct:.1f}%) [provider-reported; {reserve:,} of the "
                    f"{window:,}-token window reserved for output]"
                )
            else:
                lines.append(
                    f"Context usage: {self.prompt_tokens:,} / {window:,} tokens "
                    f"({pct:.1f}%) [provider-reported]"
                )
        else:
            lines.append(f"Context usage: {self.prompt_tokens:,} tokens [provider-reported]")

        # --- Context blocks line (attributed by character share) ---
        cb = self.context_blocks_tokens or 0
        cb_parts = [f"~{cb:,} tokens", f"{self.context_blocks_count} blocks"]
        if self.context_blocks_dropped:
            cb_parts.append(f"{self.context_blocks_dropped} EVICTED")
        lines.append(f"  Context blocks: {' — '.join(cb_parts)}")

        # --- Events line (attributed by character share) ---
        ev = self.events_tokens or 0
        ev_parts = [f"~{ev:,} tokens", f"{self.events_count} events"]
        if self.events_dropped:
            ev_parts.append(f"{self.events_dropped} dropped")
        lines.append(f"  Events:         {' — '.join(ev_parts)}")

        # --- Warning (only when hot or something was evicted/dropped) ---
        util = self.overall_utilization
        hot = util is not None and util > 0.8
        if self.context_blocks_dropped or self.events_dropped or hot:
            lines.append("Context is nearly full. Context blocks over budget are labeled EVICTED.")

        # --- Cleanup guidance ---
        lines.append(
            "Free space by collapsing older event history with "
            "self.events.collapse(start_tag, end_tag, summary_text=...); "
            "use doc(self.events) for the available event-history tools. "
            "Use self.context (ContextApi) to summarize or remove large "
            "context blocks."
        )

        return "\n".join(lines)
