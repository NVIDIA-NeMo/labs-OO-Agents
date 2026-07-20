# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strategy configuration for CodeAct, Predict, and Reflexion strategies."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nooa.runtime.restrictions import RestrictionsConfig
from nooa.runtime.sandbox.config import SandboxConfig
from nooa.strategy_validation import (
    MethodPostcondition,
    MethodPrecondition,
    normalize_conditions,
)

if TYPE_CHECKING:
    from nooa.strategies.prefill import Prefill  # noqa: F401


def _default_prefill() -> Any:
    """Lazy default-factory for ``CodeActConfig.prefill``.

    Imports inside the function so we avoid the
    ``config -> strategies.prefill -> strategies.__init__ -> config`` cycle.
    """
    from nooa.strategies.prefill import InspectInputsPrefill

    return InspectInputsPrefill()


class CodeActConfig(BaseModel):
    """Config for CodeActStrategy."""

    # ``arbitrary_types_allowed`` lets us put a ``Prefill`` protocol instance
    # in the config (Pydantic doesn't validate protocols natively).
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    max_iterations: int | None = None
    max_retries: int = 3
    # Maximum consecutive turns where the LLM returns plain text instead of a
    # tool call before the run is aborted. A real tool call resets the counter.
    # Set to 0 to disable the guard (legacy behavior). See also
    # text_only_stop_behavior for how each text-only response is handled.
    max_consecutive_text_only: int = 3
    # How to handle finish_reason="stop" (text-only, no tool call) responses:
    # - "return_result": Route through return_result(content) validation. If the
    #   return type matches, the session terminates cleanly. If not, the LLM gets
    #   an actionable validation error to self-correct. (Recommended — breaks
    #   loops faster and often terminates successfully.)
    # - "synthetic_comment": Convert to an execute_python call whose code is the
    #   text as a `#` comment — a no-op synthetic call that preserves the text
    #   in traces. The LLM sees "status: complete" and must still call
    #   return_result() explicitly.
    text_only_stop_behavior: Literal["return_result", "synthetic_comment"] = "return_result"

    @field_validator("text_only_stop_behavior", mode="before")
    @classmethod
    def _migrate_synthetic_reasoning(cls, v: str) -> str:
        if v == "synthetic_reasoning":
            return "synthetic_comment"
        return v

    cell_timeout: float | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tool_calls: int | None = None
    translate_tool_calls: bool = False
    restrictions: RestrictionsConfig = RestrictionsConfig()

    # Execution backend for execute_python cells:
    #   * "inprocess" (default) — run cells in the agent's own process/event loop.
    #     Zero behavior change; the historical path.
    #   * "sandbox" — run each cell in a locked-down worker process with
    #     OS-enforced guardrails (hard timeout, memory/CPU caps, filesystem
    #     confinement, network off). Turns ``cell_timeout`` into a *hard* bound.
    # The ``sandbox`` sub-config below is ignored unless this is "sandbox".
    execution_backend: Literal["inprocess", "sandbox"] = "inprocess"
    sandbox: SandboxConfig = SandboxConfig()

    # Method-local deterministic validators (see nooa.strategy_validation).
    # ``preconditions`` run before generation (fail-fast); ``postconditions`` run
    # after return-type validation and raise ``InvariantError`` for a
    # model-correctable retry. Attach per-method via the strategy config, e.g.
    # ``@strategy(CodeActStrategy(config=CodeActConfig(postconditions=[fn])))``.
    preconditions: Sequence[MethodPrecondition] = ()
    postconditions: Sequence[MethodPostcondition] = ()

    @field_validator("preconditions", "postconditions", mode="before")
    @classmethod
    def _normalize_conditions(cls, value: Any, info: Any) -> Sequence[Any]:
        return normalize_conditions(info.field_name, value)

    # Prefill plugin to run before the main generation loop.
    #
    #   * Default — ``InspectInputsPrefill()`` auto-renders every parameter via
    #     ``pformat`` (the slice-keys / items markers from truncation 3.0).
    #   * ``None`` — disable prefill entirely. Pre-ellipsis code (statements
    #     between the docstring and the ``...`` marker) still runs regardless.
    #   * Custom ``Prefill`` instance — full control over turn-1 setup. The
    #     ``Prefill`` protocol is a single method::
    #
    #         def get_code(self, call, config=None) -> str | None: ...
    #
    #     Returning a Python source string runs it as a synthetic prefill step
    #     in the REPL session; returning ``None`` skips that step.
    #
    # To override the *strategy prompt* (the "## Strategy" instruction block),
    # use the existing ``@strategy(..., context=ScopedContext(...))`` decorator
    # API — the decorator-context phase runs after strategy overrides and wins:
    #
    #     @strategy(
    #         CodeActStrategy(),
    #         ScopedContext(context={"strategy_prompt": "Custom instructions."}),
    #     )
    #     async def my_method(self, ...): ...
    # Typed as ``Any`` to sidestep Pydantic's forward-reference resolution for
    # the ``Prefill`` protocol (which lives in ``strategies.prefill`` and would
    # create an import cycle if eagerly resolved here). The runtime contract
    # is ``Prefill | None`` — a duck-typed object with a ``get_code`` method
    # or ``None`` to disable.
    prefill: Any = Field(default_factory=_default_prefill)

    def merge_with(self, other: "CodeActConfig") -> "CodeActConfig":
        if not other.model_fields_set:
            raise ValueError(
                "merge_with() received a config with no model_fields_set. "
                "Was it constructed from model_dump() or model_validate()? "
                "Config objects must be freshly constructed: CodeActConfig(field=value)."
            )
        return self.model_copy(update={k: getattr(other, k) for k in other.model_fields_set})


class PredictConfig(BaseModel):
    """Config for PredictStrategy."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # Method-local deterministic validators (see nooa.strategy_validation);
    # currently consumed by the CodeAct strategy (Predict wiring lands with the
    # full validator MR, !444).
    preconditions: Sequence[MethodPrecondition] = ()
    postconditions: Sequence[MethodPostcondition] = ()

    @field_validator("preconditions", "postconditions", mode="before")
    @classmethod
    def _normalize_conditions(cls, value: Any, info: Any) -> Sequence[Any]:
        return normalize_conditions(info.field_name, value)

    max_retries: int = 10
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_error_chars: int = 1000
    # ``None`` = unconstrained (parameter-size guard disabled).
    max_param_chars: int | None = 200_000
    # How the Predict output is serialized back into the conversation history:
    # - "event": The LLMOutput event stays; it replays as a plain assistant
    #   message (raw JSON content, no wrapper).
    # - "tool_call": Replace with a synthetic return_result() ToolCallEvent that
    #   renders natively through the provider formatter — clearer for downstream
    #   tool-using models that read this history.
    output_serialization: Literal["event", "tool_call"] = "event"

    def merge_with(self, other: "PredictConfig") -> "PredictConfig":
        if not other.model_fields_set:
            raise ValueError(
                "merge_with() received a config with no model_fields_set. "
                "Was it constructed from model_dump() or model_validate()? "
                "Config objects must be freshly constructed: PredictConfig(field=value)."
            )
        return self.model_copy(update={k: getattr(other, k) for k in other.model_fields_set})


class ReflexionConfig(BaseModel):
    """Config for ReflexionStrategy.

    Deprecated: ReflexionStrategy is now experimental.
    This config class is kept here for backward compatibility.
    """

    model_config = ConfigDict(frozen=True)

    max_iterations: int = 3
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None

    def merge_with(self, other: "ReflexionConfig") -> "ReflexionConfig":
        if not other.model_fields_set:
            raise ValueError(
                "merge_with() received a config with no model_fields_set. "
                "Was it constructed from model_dump() or model_validate()? "
                "Config objects must be freshly constructed: ReflexionConfig(field=value)."
            )
        return self.model_copy(update={k: getattr(other, k) for k in other.model_fields_set})
