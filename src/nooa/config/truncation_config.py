# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for output truncation across the agent framework.

Two layers of bounds:

1. **Whole-context token budget** (optional, applied at assembly):
   ``max_context_tokens``, ``max_event_tokens``.
2. **Sub-configs by mechanism**:
   - ``capture`` (head/tail truncation of raw text streams via
     ``TruncatingStringIO`` — used for execute_python's stdout / stderr /
     error fields).
   - ``media_capture`` (``MediaCaptureConfig``): cap on multimodal content
     blocks (images, audio, files) captured by ``show()`` per
     ``execute_python`` cell.
   - ``event_format`` (``FormatConfig``): defaults for event-field rendering at
     trajectory-build time. Renders every turn for the rest of the run, so
     these need to be generous enough that the LLM can read its own outputs
     (especially ``PythonOutput.value``).
   - ``prefill_format`` (``FormatConfig``): defaults for parameter rendering at
     method invocation. One-shot per call; the agent author controls what's
     passed in, so tighter defaults are fine.
   - ``context_block_format`` (``FormatConfig``): defaults for non-string values
     passed to ``self.context[key] = value``. Few blocks per agent, set
     intentionally by the author, often large (e.g. ``doc(self)``,
     formatted state). Defaults to fully unlimited; whole-block eviction
     handles overflow at the assembly step.
"""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaptureConfig(BaseModel):
    """Char budgets for raw text streams captured via ``TruncatingStringIO``.

    These apply at *capture time* (when execute_python writes to stdout / stderr
    or formats an exception). The captured text is head/tail-truncated to fit
    the budget, with a ``<truncated>...</truncated>`` wrapper indicating where
    content was elided.

    Field naming: ``max_<stream>`` — char count, no ``_chars`` suffix needed
    inside the capture config since every field here is char-based.
    """

    model_config = ConfigDict(frozen=True)

    max_stdout: Annotated[int, Field(description="Max chars of stdout per cell")] = 50_000
    max_stderr: Annotated[int, Field(description="Max chars of stderr per cell")] = 2_000
    max_error: Annotated[int, Field(description="Max chars of error/traceback per cell")] = 10_000
    tail: Annotated[
        int | None,
        Field(
            description=(
                "Chars reserved for the tail window in head/tail truncation. "
                "Applies to stdout / stderr / error capture. "
                "None = half of each stream's total budget (50/50 head/tail split). "
                "Wired through for the file-writing escape hatch (#150)."
            ),
        ),
    ] = None
    file_backed: Annotated[
        bool,
        Field(
            description=(
                "When True, stdout/stderr capture uses FileBackedTruncatingStringIO "
                "which streams full output to a temp file on disk while still "
                "truncating the in-memory representation. The temp file path is "
                "included in the truncation notice so the LLM can reference it."
            ),
        ),
    ] = False

    @model_validator(mode="after")
    def _check(self) -> "CaptureConfig":
        errors = []
        for name in ("max_stdout", "max_stderr", "max_error"):
            if getattr(self, name) <= 0:
                errors.append(f"capture.{name} must be > 0, got {getattr(self, name)}")
        if self.tail is not None:
            if self.tail < 0:
                errors.append(f"capture.tail must be >= 0, got {self.tail}")
            else:
                # tail must be strictly less than each stream's total
                for name in ("max_stdout", "max_stderr", "max_error"):
                    if self.tail >= getattr(self, name):
                        errors.append(
                            f"capture.tail ({self.tail:,}) must be less than "
                            f"capture.{name} ({getattr(self, name):,})"
                        )
        if errors:
            raise ValueError("Invalid CaptureConfig:\n" + "\n".join(f"  - {e}" for e in errors))
        return self


class MediaCaptureConfig(BaseModel):
    """Cap on multimodal content captured by ``show()`` during execution.

    Applies at *capture time*: every ``show(image | audio | file)`` call inside
    an ``execute_python`` cell appends a content block to a per-cell buffer.
    Once the cap is reached, further ``show()`` calls in the same cell are
    dropped and ``[show() limit reached (N), attachment not added]`` is printed
    to stdout so the LLM can observe the spillover.

    **Scope: per ``execute_python`` cell, not per turn or per agent run.** A
    fresh buffer is allocated for each cell, so an LLM turn that emits multiple
    ``execute_python`` tool calls can attach up to
    ``max_attachments_per_execution`` blocks *each*. This is by design — the
    cap lives on the runtime (which only sees one cell at a time), and
    cross-cell aggregation would belong on the strategy. The default (5) is
    intentionally small so a single-cell ``for`` loop dumping attachments
    stays bounded.
    """

    model_config = ConfigDict(frozen=True)

    max_attachments_per_execution: Annotated[
        int,
        Field(description="Max media attachments captured by show() per execute_python cell"),
    ] = 5

    @model_validator(mode="after")
    def _check(self) -> "MediaCaptureConfig":
        if self.max_attachments_per_execution <= 0:
            raise ValueError(
                "Invalid MediaCaptureConfig:\n"
                "  - media_capture.max_attachments_per_execution must be > 0, "
                f"got {self.max_attachments_per_execution}"
            )
        return self


class FormatConfig(BaseModel):
    """Recursive bounds for ``pformat``-based rendering.

    Used at multiple rendering moments (see :attr:`TruncationConfig.event_format`
    and :attr:`TruncationConfig.prefill_format`). Field names match the underlying
    ``pformat()`` kwargs exactly so that ``cfg.event_format.model_dump()`` /
    ``cfg.prefill_format.model_dump()`` can be splatted directly into
    ``pformat(**...)``.

    A per-field ``Annotated[T, spec(max_string=N, ...)]`` annotation overrides
    these defaults on that single field.
    """

    model_config = ConfigDict(frozen=True)

    max_string: Annotated[
        int | None, Field(description="Max chars per string before str(len=N, [:H], [-T:]) marker")
    ] = 500
    max_length: Annotated[
        int | None,
        Field(description="Max items per container before type(len=N, [:H], [-T:]) marker"),
    ] = 50
    max_depth: Annotated[int | None, Field(description="Max nesting depth")] = 4

    @model_validator(mode="after")
    def _check(self) -> "FormatConfig":
        errors = []
        for name in ("max_string", "max_length", "max_depth"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                errors.append(f"{name} must be > 0 or None, got {v}")
        if errors:
            raise ValueError("Invalid FormatConfig:\n" + "\n".join(f"  - {e}" for e in errors))
        return self


class TruncationConfig(BaseModel):
    """Controls output size at render time.

    See module docstring for the layering. The sub-configs (``capture``,
    ``media_capture``, ``event_format``, ``prefill_format``,
    ``context_block_format``) cover the distinct mechanisms / rendering
    moments; the top-level fields are cross-cutting (token budgets).
    """

    model_config = ConfigDict(frozen=True)

    max_context_tokens: Annotated[int | None, Field(description="Total context token budget")] = (
        None
    )
    max_event_tokens: Annotated[int | None, Field(description="Total event token budget")] = None
    # L4 eviction: never evict fewer than this many recent events. Guarantees
    # the model always sees its current Task + recent reasoning even when older
    # events get evicted to fit the budget.
    min_preserved_events: Annotated[
        int,
        Field(description="Minimum number of recent events preserved during eviction"),
    ] = 5
    # Output-token planning reserve. When a call sets ``max_tokens``
    # explicitly, THAT value is reserved out of the model window (the provider
    # rejects prompt + max_tokens > window); when it doesn't, this value is
    # used instead. The reserve shrinks the usable window for the default
    # context-block budget (``(context_window - reserve) // 2``), for the
    # ``ctx N%`` utilization / nearly-full warning, and for post-error archive
    # sizing. Set to 0 to disable both the reserve and the auto-derived
    # default budget.
    response_reserve_tokens: Annotated[
        int,
        Field(
            description=(
                "Tokens reserved for the LLM response when the call does not "
                "set max_tokens explicitly. Shrinks the usable context window "
                "for budgeting and utilization. 0 disables."
            )
        ),
    ] = 4_096
    capture: CaptureConfig = CaptureConfig()
    media_capture: MediaCaptureConfig = MediaCaptureConfig()
    # Generous: rendered every turn, especially PythonOutput.value (the LLM
    # needs to see what its code returned).
    event_format: FormatConfig = FormatConfig(max_string=10_000, max_length=200, max_depth=5)
    # Prefill: one-shot per method call; the agent author controls inputs.
    prefill_format: FormatConfig = FormatConfig(max_string=2_000, max_length=25, max_depth=4)
    # Unlimited: agent-author-curated values placed via self.context[key]=value.
    # If you put a structured object in context, you meant for it to render in
    # full. L4 (eviction) handles whole-block overflow at the assembly step.
    context_block_format: FormatConfig = FormatConfig(
        max_string=None, max_length=None, max_depth=None
    )

    @model_validator(mode="after")
    def _check(self) -> "TruncationConfig":
        errors = []
        for name in ("max_context_tokens", "max_event_tokens"):
            v = getattr(self, name)
            if v is not None and v <= 0:
                errors.append(f"{name} must be > 0 or None, got {v}")
        if errors:
            raise ValueError("Invalid TruncationConfig:\n" + "\n".join(f"  - {e}" for e in errors))
        return self

    def merge_with(self, other: "TruncationConfig | None") -> "TruncationConfig":
        """Merge with another config (other takes precedence for explicitly-set fields).

        Sub-configs (``capture``, ``media_capture``, ``event_format``,
        ``prefill_format``, ``context_block_format``) merge field-by-field —
        supplying ``other.event_format`` with only ``max_string`` set does NOT
        clobber the base's ``max_length``. Top-level fields use the same
        explicit-fields semantics.
        """
        if other is None:
            return self
        if not other.model_fields_set:
            raise ValueError(
                "merge_with() received a config with no model_fields_set. "
                "Was it constructed from model_dump() or model_validate()? "
                "Config objects must be freshly constructed: TruncationConfig(field=value)."
            )
        sub_field_names = (
            "capture",
            "media_capture",
            "event_format",
            "prefill_format",
            "context_block_format",
        )
        updates: dict = {
            k: getattr(other, k) for k in other.model_fields_set if k not in sub_field_names
        }
        # Sub-config merge: only apply if user explicitly passed a sub-config.
        for sub in sub_field_names:
            if sub in other.model_fields_set:
                base_sub = getattr(self, sub)
                other_sub = getattr(other, sub)
                updates[sub] = base_sub.model_copy(
                    update={k: getattr(other_sub, k) for k in other_sub.model_fields_set}
                )
        return self.model_copy(update=updates)


# Default configuration
DEFAULT_TRUNCATION_CONFIG = TruncationConfig()
