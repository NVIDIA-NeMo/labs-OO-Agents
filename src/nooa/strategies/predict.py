# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PREDICT strategy - Single-shot LLM call that tries to solve the task in one go.

Flow:
1. Inspect method's return type annotation
2. Convert to Pydantic model (wrap basic types if needed)
3. Call LLM with output_model for structured output
4. Validate response against model (if needed)
5. If validation fails → add error event, retry
6. Return validated data

The return type annotation is used to validate the output. Supports Pydantic models,
basic types (str, int, bool, dict, list), Optional[T], and nested models.
"""

import json
import logging
import types
from typing import TYPE_CHECKING, Any, Union, cast, get_args, get_origin
from uuid import uuid4

from pydantic import BaseModel, RootModel, create_model
from pydantic import ValidationError as PydanticValidationError

from nooa.agentdoc import pformat
from nooa.agentdoc.visibility import is_hidden_field
from nooa.context_blocks import DynamicContext, ResultStatus, ToolCallEvent, ToolResult
from nooa.decorators import strategy
from nooa.errors import GenerationError
from nooa.events import AfterTurn, BeforeTurn, Error, Task
from nooa.runtime.harness_metrics import get_harness_metrics
from nooa.strategies.base import (
    GenerationStrategy,
    RuntimeServices,
    build_sampling_kwargs,
)
from nooa.strategies.template import TemplateStrategy

if TYPE_CHECKING:
    from nooa.config.strategy_config import PredictConfig
    from nooa.strategies.current_call import CurrentCall

logger = logging.getLogger(__name__)


class PredictStrategy(GenerationStrategy):
    """Single-shot LLM call that tries to solve the task in one go (no loop/iteration).

    The return type annotation is used to validate the output. Supports:
    - Pydantic models (used directly)
    - Basic types (str, int, bool, dict, list) - wrapped automatically
    - Optional[T] - unwrapped to T
    - Nested models and complex types

    The strategy handles validation retry with clear error feedback if the LLM's
    output doesn't match the schema.

    Configuration:
        max_retries: Maximum validation retry attempts

    Example:
        from nooa.config import PredictConfig

        @strategy(PredictStrategy(config=PredictConfig(max_retries=3)))
        def analyze(self, data: str) -> AnalysisResult:
            '''Analyze data and return structured results.'''
    """

    def __init__(self, config: "PredictConfig | None" = None):
        """Initialize PREDICT strategy.

        Args:
            config: PredictConfig with retry limits and sampling params.
        """
        from nooa.config.strategy_config import PredictConfig as _PC

        self.config = config or _PC()

    def _build_sampling_kwargs(self) -> dict[str, Any]:
        """Build sampling kwargs for llm calls, excluding None values."""
        return build_sampling_kwargs(self.config)

    @property
    def name(self) -> str:
        """Strategy name."""
        return "PREDICT"

    def get_block_overrides(self) -> dict[str, str | DynamicContext | None]:
        return {
            "strategy_prompt": DynamicContext("strategy.strategy_instructions(runtime)"),
        }

    @strategy(TemplateStrategy())
    async def strategy_instructions(self, runtime: RuntimeServices) -> str:
        """You are generating structured data matching the specified return type.
        Output will be validated against the type schema.

        Ensure your response:
        1. Includes all required fields
        2. Uses correct types for each field
        3. Follows the exact schema structure
        4. Contains no extra fields beyond the schema

        The LLM will automatically format your output as JSON."""
        ...

    @strategy(TemplateStrategy())
    async def error_pydantic_validation(self, runtime: RuntimeServices, error_list: str) -> str:
        """Your output failed Pydantic validation. Please fix the following errors:

        {error_list}

        Ensure your JSON output:
        1. Includes all required fields
        2. Uses correct types for each field
        3. Follows the exact schema structure
        4. Contains no extra fields

        Try again with corrected output."""
        ...

    @strategy(TemplateStrategy())
    async def error_type_validation(
        self, runtime: RuntimeServices, return_type: str, error: str
    ) -> str:
        """Your output failed type validation for return type `{return_type}`.

        Error: {error}

        Ensure your JSON output:
        1. Matches the expected type structure
        2. Contains valid data for the specified type
        3. Is properly formatted JSON

        Try again with corrected output."""
        ...

    @strategy(TemplateStrategy())
    async def error_json_parsing(self, runtime: RuntimeServices, error: str) -> str:
        """Your output could not be parsed as JSON.

        Error: {error}

        Requirements:
        1. Output ONLY valid JSON - no markdown, no code fences, no explanatory text
        2. Ensure proper JSON syntax (quotes around strings, commas between items)
        3. Do not include any text before or after the JSON object

        Try again with valid JSON output."""
        ...

    async def execute(self, runtime: RuntimeServices, call: "CurrentCall") -> Any:
        """Execute PREDICT strategy.

        Args:
            runtime: RuntimeServices providing LLM, execution, and event management.
            call: CurrentCall with method details and arguments.

        Returns:
            Validated structured data matching return type.

        Raises:
            GenerationError: If no return type or validation fails after max retries.
        """
        # Emit BeforeTurn/AfterTurn so observability consumers (e.g. the ATIF
        # exporter) can detect this single-shot generation turn.
        # PredictStrategy is always one inference; turn_number=1 and
        # AfterTurn(is_final=True).
        _gen_id = runtime.get_generation_id() or ""
        _parent_gen_id = runtime.get_parent_generation_id()
        runtime.event_manager.add(
            BeforeTurn(
                method_name=call.method_name,
                strategy=self.name,
                generation_id=_gen_id,
                parent_generation_id=_parent_gen_id,
                turn_number=1,
            ),
            record=False,
        )
        _turn_success = False
        _turn_exception_type: str | None = None
        try:
            result = await self._execute_inner(runtime, call)
            _turn_success = True
            return result
        except BaseException as exc:
            _turn_exception_type = type(exc).__name__
            raise
        finally:
            runtime.event_manager.add(
                AfterTurn(
                    method_name=call.method_name,
                    strategy=self.name,
                    generation_id=_gen_id,
                    parent_generation_id=_parent_gen_id,
                    turn_number=1,
                    is_final=True,
                    success=_turn_success,
                    exception_type=_turn_exception_type,
                ),
                record=False,
            )

    async def _execute_inner(self, runtime: RuntimeServices, call: "CurrentCall") -> Any:
        """Actual PREDICT body — wrapped by :meth:`execute` for BeforeTurn/AfterTurn."""
        # Return type is pre-resolved by _execute_with_generation (handles PEP 563).
        return_type = call.return_type
        if return_type is None:
            raise GenerationError(
                f"Method `{call.method_name}` has no return type annotation. "
                f"PREDICT strategy requires a return type (Pydantic model or basic type)."
            )

        # Convert return type to Pydantic model (wraps basic types if needed)
        response_model = self._create_response_model(return_type, call.method_name)

        # Fail fast with a clear, actionable error if the return type can't be turned
        # into a JSON schema. This catches Pydantic *model* return types whose fields
        # are non-serializable (e.g. a model with a pandas.DataFrame field) — those
        # pass _create_response_model (the model is returned as-is) but blow up later
        # in the request layer with a cryptic Pydantic error. See KDD-cup feedback.
        try:
            response_model.model_json_schema()
        except Exception as e:  # noqa: BLE001 — surface as a clear GenerationError
            raise GenerationError(
                f"PredictStrategy cannot build a JSON schema for return type {return_type}: {e}\n"
                f"PredictStrategy requires a JSON-serializable return type. Non-serializable "
                f"types such as pandas.DataFrame or numpy.ndarray — including Pydantic models "
                f"that contain such a field — are not supported. Use CodeActStrategy (which can "
                f"build and return the object in code), or return a serializable proxy — e.g. a "
                f"model with `columns: list[str]` and `rows: list[list]` instead of a DataFrame."
            ) from e

        # Collect Media params as multimodal content blocks for the task
        from nooa.media import Media
        from nooa.runtime.media_capture import media_to_content_block

        # bound_parameters() attaches each effective input once (positional args
        # also appear in kwargs, so a raw args+kwargs union would double-attach).
        media_blocks = [
            media_to_content_block(v)
            for v in call.bound_parameters().values()
            if isinstance(v, Media)
        ]

        # Guard against oversized parameters before building the prompt.
        # PredictStrategy is single-shot — a truncated input produces silently wrong output.
        self._assert_param_sizes(call)

        # Build task prompt first so we can check its size before adding it to events.
        task_prompt = await self._build_task_message(runtime, original_call=call)

        # Add task event (with media attachments if any)
        runtime.event_manager.add(
            Task(
                prompt=task_prompt,
                images=media_blocks,
            )
        )

        # Collect all failed attempts for span attributes (added at the end)
        failed_attempts: list[dict[str, Any]] = []

        # Validation retry loop
        for attempt in range(1, self.config.max_retries + 1):
            raw_response_content = None  # Track raw response for error reporting
            llm_response = None  # Track raw LLM response

            try:
                # Step 1: Call LLM (this doesn't parse, so it won't throw parse errors)
                llm_response, _event_id = await self._call_llm_raw(runtime, response_model)

                # Step 2: Extract raw content BEFORE any parsing (for error reporting)
                raw_response_content = self._extract_raw_from_llm_response(llm_response)
                logger.debug(
                    f"[PREDICT attempt={attempt}] Got raw content: "
                    f"len={len(raw_response_content) if raw_response_content else 0}"
                )

                # Step 3: Parse the response (this may throw JSONDecodeError)
                parsed_response = self._parse_llm_response(llm_response, call.method_name)

                # Step 4: Validate the parsed response (this may throw PydanticValidationError)
                validated_data = self._validate_response(
                    parsed_response, response_model, return_type
                )

                logger.debug(f"[PREDICT attempt={attempt}] Validation successful")

                if self.config.output_serialization == "tool_call":
                    self._replace_with_tool_call(runtime, _event_id, validated_data)

                return validated_data

            except (PydanticValidationError, ValueError, TypeError, json.JSONDecodeError) as e:
                # If we don't have raw_response_content yet but do have llm_response, try to extract it
                if raw_response_content is None and llm_response is not None:
                    logger.debug("[PREDICT] Extracting raw content in exception handler")
                    raw_response_content = self._extract_raw_from_llm_response(llm_response)

                # Use descriptive placeholder if still None
                if raw_response_content is None:
                    # Try to get more info about what the LLM returned
                    if llm_response is None:
                        raw_response_content = "(llm_response was None)"
                    else:
                        # Try to capture all available fields from the response
                        tc = runtime.truncation_config
                        response_info = []
                        for field in ["content", "reasoning", "raw", "text", "message"]:
                            val = getattr(llm_response, field, "<missing>")
                            if val != "<missing>":
                                val_repr = pformat(val, **tc.event_format.model_dump())
                                response_info.append(f"{field}={val_repr}")
                        if response_info:
                            raw_response_content = (
                                f"(extraction failed, fields: {', '.join(response_info)})"
                            )
                        else:
                            raw_response_content = (
                                f"(extraction failed, type={type(llm_response).__name__})"
                            )

                # Validation or JSON parsing failed
                logger.debug(
                    f"[PREDICT attempt={attempt}] Validation/parsing failed: "
                    f"{type(e).__name__}: {e}, raw_len={len(raw_response_content)}"
                )

                get_harness_metrics().predict_retry(f"{type(e).__name__}: {str(e)[:200]}")

                # Collect failed attempt info (will add to span at the end)
                failed_attempts.append(
                    {
                        "attempt": attempt,
                        "raw_output": raw_response_content,
                        "error_type": type(e).__name__,
                        "error_message": str(e)[:500],
                    }
                )

                if attempt >= self.config.max_retries:
                    # Add ALL failed attempts to span before raising
                    self._add_all_failed_attempts_to_span(failed_attempts)

                    error_details = await self._format_validation_error(e, return_type, runtime)
                    raise GenerationError(
                        f"Structured output validation failed after {self.config.max_retries} attempts.\n"
                        f"Last error:\n{error_details}"
                    ) from e

                # Add error feedback for next attempt
                error_msg = await self._format_validation_error(e, return_type, runtime)

                # Include raw output in error event for visibility
                error_content = (
                    f"Validation error (attempt {attempt}/{self.config.max_retries}):\n{error_msg}"
                )
                # Truncate if very long
                max_err = self.config.max_error_chars
                truncated_content = raw_response_content[:max_err]
                if len(raw_response_content) > max_err:
                    truncated_content += (
                        f"\n... (truncated, {len(raw_response_content)} chars total)"
                    )
                error_content += f"\n\nRaw LLM output:\n{truncated_content}"

                runtime.event_manager.add(Error(content=error_content))

        # Should never reach here
        raise GenerationError(
            f"Structured output generation failed after {self.config.max_retries} attempts"
        )

    @strategy(TemplateStrategy())
    async def _build_task_message(
        self, runtime: RuntimeServices, original_call: "CurrentCall"
    ) -> str:
        """
        # Your task
        {original_call.docstring}

        ## Input parameters:
        {original_call.format_parameters_as_code()}

        Perform the task and return the result directly.
        """
        ...

    def _assert_param_sizes(self, call: "CurrentCall") -> None:
        """Raise ValueError if any parameter's repr size exceeds max_param_chars.

        PredictStrategy makes a single-shot call — a silently truncated input produces
        wrong output with no indication of failure.  Fail loudly instead so callers can
        chunk, summarise, or raise max_param_chars in their PredictConfig.

        Size is measured by writing repr(value) into a TruncatingStringIO and checking
        was_truncated — the same mechanism used throughout the formatting pipeline.
        repr() matches the conservative lower bound of what format_parameters_as_code
        puts into the prompt (truncating_pformat can produce larger output for complex objects).

        String fast-path: plain strings skip the repr() allocation.  len(s) ≤ len(repr(s))
        always, so a string that already exceeds the limit certainly has an oversized repr.
        """
        from nooa.agentdoc import TruncatingStringIO

        limit = self.config.max_param_chars
        # ``None`` disables the guard entirely — used by the summarizer.
        if limit is None:
            return

        # Check each effective input once (bound_parameters() de-duplicates the
        # positional/keyword overlap).
        for name, value in call.bound_parameters().items():
            # Fast path for strings: avoids repr() allocation for large inputs.
            # len(s) ≤ len(repr(s)) always, so if the string itself exceeds limit
            # its repr certainly does too.
            if isinstance(value, str) and len(value) > limit:
                raise ValueError(
                    f"PredictStrategy: parameter '{name}' is {len(value):,} chars, "
                    f"exceeding max_param_chars={limit:,}. "
                    f"Chunk or summarise the input before calling this method, "
                    f"or raise PredictConfig(max_param_chars=...) if the size is intentional."
                )
            stream = TruncatingStringIO(limit=limit)
            stream.write(repr(value))
            if stream.was_truncated:
                raise ValueError(
                    f"PredictStrategy: parameter '{name}' is {stream.chars_written:,} chars (repr), "
                    f"exceeding max_param_chars={limit:,}. "
                    f"Chunk or summarise the input before calling this method, "
                    f"or raise PredictConfig(max_param_chars=...) if the size is intentional."
                )

    def _jsonable(self, value: Any) -> Any:
        """Convert a validated Predict result into JSON-compatible tool arguments."""
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {str(k): self._jsonable(v) for k, v in value.items()}
        if isinstance(value, set):
            converted = [self._jsonable(v) for v in value]
            return sorted(converted, key=lambda item: json.dumps(item, sort_keys=True))
        if isinstance(value, (list, tuple)):
            return [self._jsonable(v) for v in value]
        try:
            json.dumps(value)
            return value
        except TypeError:
            return str(value)

    def _replace_with_tool_call(self, runtime: RuntimeServices, event_id: str, result: Any) -> None:
        """Replace Predict's LLMOutput with a synthetic return_result tool call."""
        runtime.event_manager.remove(event_id)
        tool_call_id = f"predict_{uuid4().hex[:8]}"
        runtime.event_manager.add(
            ToolCallEvent(
                tool_call_id=tool_call_id,
                name="return_result",
                arguments={"result": self._jsonable(result)},
                result=ToolResult(
                    tool_call_id=tool_call_id,
                    content="Result accepted.",
                    result_status=ResultStatus.COMPLETE,
                ),
                metadata={"synthetic": True, "synthetic_type": "predict_return_result"},
            )
        )

    def _extract_raw_from_llm_response(self, llm_response: Any) -> str | None:
        """Extract raw content from LLMResponse object for error reporting.

        This is used when parsing fails and we need to extract the raw output
        directly from the LLMResponse object.

        Args:
            llm_response: LLMResponse object

        Returns:
            String representation of the raw response, or None if not extractable
        """
        try:
            # Log what we're working with
            has_reasoning = hasattr(llm_response, "reasoning")
            reasoning_value = getattr(llm_response, "reasoning", None)
            has_content = hasattr(llm_response, "content")
            content_value = getattr(llm_response, "content", None)

            logger.debug(
                f"[PREDICT] _extract_raw_from_llm_response: "
                f"has_reasoning={has_reasoning}, "
                f"reasoning_type={type(reasoning_value).__name__}, "
                f"reasoning_len={len(str(reasoning_value)) if reasoning_value else 0}, "
                f"has_content={has_content}, "
                f"content_type={type(content_value).__name__}, "
                f"content_len={len(str(content_value)) if content_value else 0}"
            )

            # For extraction (error reporting), capture BOTH fields if available
            # to show what the LLM actually returned
            parts = []

            # Content field (this is where structured output SHOULD be)
            if has_content and content_value:
                if isinstance(content_value, str):
                    parts.append(f"[CONTENT]: {content_value}")
                elif isinstance(content_value, BaseModel):
                    parts.append(f"[CONTENT (Pydantic)]: {content_value.model_dump_json(indent=2)}")
                elif isinstance(content_value, dict):
                    parts.append(f"[CONTENT (dict)]: {json.dumps(content_value, indent=2)}")
                else:
                    parts.append(f"[CONTENT ({type(content_value).__name__})]: {content_value}")

            # Reasoning field (model's thinking - may contain clues for debugging)
            if has_reasoning and reasoning_value:
                # Truncate reasoning if very long (it's often thousands of chars)
                reasoning_str = str(reasoning_value)
                if len(reasoning_str) > 1000:
                    original_len = len(reasoning_str)
                    reasoning_str = reasoning_str[:1000] + f"... (truncated, {original_len} total)"
                parts.append(f"[REASONING]: {reasoning_str}")

            if parts:
                result = "\n".join(parts)
                logger.debug(
                    f"[PREDICT] Extracted response with {len(parts)} parts, total len={len(result)}"
                )
                return result

            # No content found
            logger.warning("[PREDICT] No content found in LLM response!")
            return "(empty response - both content and reasoning were empty)"
        except Exception as e:
            logger.warning(f"[PREDICT] Failed to extract raw from LLMResponse: {e}")
            return f"(error extracting: {e})"

    def _add_all_failed_attempts_to_span(self, failed_attempts: list[dict[str, Any]]) -> None:
        """Add all failed attempt info to current OpenTelemetry span.

        This is called once at the end of the retry loop with all collected failures.
        This ensures we capture all attempts before the span ends.

        Args:
            failed_attempts: List of dicts with attempt, raw_output, error_type, error_message
        """
        try:
            from opentelemetry import trace as otel_trace

            span = otel_trace.get_current_span()
            span_ctx = span.get_span_context() if span else None

            logger.debug(
                f"[PREDICT] Adding {len(failed_attempts)} failed attempts to span: "
                f"span_id={format(span_ctx.span_id, '016x') if span_ctx else 'None'}, "
                f"is_recording={span.is_recording() if span else False}"
            )

            if span and span.is_recording():
                for attempt_info in failed_attempts:
                    attempt = attempt_info["attempt"]
                    raw_output = attempt_info["raw_output"]
                    error_type = attempt_info["error_type"]
                    error_message = attempt_info["error_message"]

                    # Truncate raw output if very long
                    truncated_output = raw_output[:2000]
                    if len(raw_output) > 2000:
                        truncated_output += f"\n... (truncated from {len(raw_output)} chars)"

                    span.set_attribute(
                        f"predict.failed_attempt_{attempt}.raw_output", truncated_output
                    )
                    span.set_attribute(f"predict.failed_attempt_{attempt}.error_type", error_type)
                    span.set_attribute(
                        f"predict.failed_attempt_{attempt}.error_message", error_message
                    )

                    logger.debug(f"[PREDICT] Added attempt {attempt} to span")

                # Also add a summary
                span.set_attribute("predict.total_failed_attempts", len(failed_attempts))

                logger.debug(
                    f"[PREDICT] Successfully added {len(failed_attempts)} failed attempts to span"
                )
            else:
                logger.debug("[PREDICT] Cannot add to span: span is None or not recording")
        except ImportError:
            logger.debug("[PREDICT] OpenTelemetry not available")
        except Exception as e:
            logger.debug(f"[PREDICT] Failed to add span attributes: {e}")

    def _strip_xml_wrapper(self, content: str | None) -> str:
        """Strip XML wrapper tags from content if present (lenient mode).

        Delegates core matching to the shared ``strip_xml_wrapper`` function.
        Unlike the PurePython version, this does NOT raise on malformed XML —
        it silently returns the original content.
        """
        from nooa.runtime.response_cleanup import strip_xml_wrapper

        content = (content or "").strip()
        inner_content, tag_name = strip_xml_wrapper(content)
        if tag_name:
            get_harness_metrics().xml_wrapper_stripped(tag_name)
            logger.debug(
                f"[PREDICT] Stripped XML wrapper <{tag_name}>, extracted {len(inner_content)} chars"
            )
        return inner_content

    async def _call_llm_raw(
        self,
        runtime: RuntimeServices,
        response_model: type[BaseModel],
    ) -> tuple[Any, str]:
        """Call LLM with structured output format, returning raw response.

        This method only calls the LLM and returns the raw response object.
        It does NOT parse the response - that's done by _parse_llm_response().
        This separation ensures we always have access to the raw response
        for error reporting even when parsing fails.

        Args:
            runtime: RuntimeServices for LLM access.
            response_model: Pydantic model for response schema.

        Returns:
            Tuple of (llm_response_object, event_id).
        """
        response, event_id = await runtime.generate(
            tools=None,
            output_model=response_model,
            **self._build_sampling_kwargs(),
        )

        logger.debug(
            f"[PREDICT] LLM Response: content_type={type(response.content).__name__}, "
            f"content_len={len(str(response.content)) if response.content else 0}, "
            f"reasoning_len={len(str(response.reasoning)) if response.reasoning else 0}"
        )

        return response, event_id

    # ── Post-response cleanup (Predict) ────────────────────────
    # Intercept point: strategy-specific response transforms.
    # Handles XML wrapper stripping and content-to-reasoning fallback.
    # Consider making extensible in the future.

    def _parse_llm_response(self, llm_response: Any, method_name: str) -> dict[str, Any]:
        """Parse LLM response into a dict for validation.

        This method may throw JSONDecodeError if the response cannot be parsed.
        The caller should capture raw_response_content BEFORE calling this method
        to ensure we have it for error reporting.

        Args:
            llm_response: Raw LLMResponse object from _call_llm_raw()
            method_name: Method name (for error messages)

        Returns:
            Parsed response as dict

        Raises:
            json.JSONDecodeError: If string content cannot be parsed as JSON
            GenerationError: If response type is unexpected
        """
        # For structured output, the JSON result should be in the content field.
        # The reasoning field contains the model's thinking process (not JSON).
        # Try content first, then fall back to reasoning only if content is empty.
        if llm_response.content:
            content_to_parse = llm_response.content
            source = "content"
        elif llm_response.reasoning:
            # Only use reasoning as fallback - some models might put JSON there
            get_harness_metrics().record_content_to_reasoning_fallback()
            content_to_parse = llm_response.reasoning
            source = "reasoning (fallback)"
        else:
            content_to_parse = ""
            source = "empty"

        logger.debug(f"[PREDICT] Parsing: using={source}, type={type(content_to_parse).__name__}")

        # Validated Pydantic model directly in response.content
        # Convert to dict for our validation layer
        if isinstance(content_to_parse, BaseModel):
            return content_to_parse.model_dump()

        # If content is a dict (defensive)
        if isinstance(content_to_parse, dict):
            return content_to_parse.copy()

        # For test mocking and string responses (FakeLLMClient or reasoning models)
        if isinstance(content_to_parse, str):
            logger.debug(
                f"[PREDICT] Parsing string content for {method_name}: "
                f"len={len(content_to_parse)}, preview=|{content_to_parse[:200]}|"
            )
            # Strip potential XML wrapper tags (e.g., <assistant_message expr="...">JSON</assistant_message>)
            cleaned_content = self._strip_xml_wrapper(content_to_parse)

            # This may raise json.JSONDecodeError
            parsed_data = json.loads(cleaned_content)

            # Ensure we return a dict
            if isinstance(parsed_data, dict):
                return parsed_data
            else:
                # For non-dict responses, wrap in dict
                return {"value": parsed_data}

        # Shouldn't reach here with output_model
        raise GenerationError(
            f"Unexpected response type: content={type(llm_response.content).__name__}, "
            f"reasoning={type(llm_response.reasoning).__name__ if llm_response.reasoning else None}"
        )

    def _create_response_model(self, return_type: Any, method_name: str) -> type[BaseModel]:
        """Create Pydantic model from return type annotation.

        Handles:
        - Existing Pydantic models (use directly)
        - dict types - use RootModel (LLM returns the object directly, not wrapped)
        - list/tuple/set and basic types (str, int, bool) - wrap in model with a
          `value` field so the root schema stays `type: object` (the Responses API
          rejects array-rooted schemas; see issue 232)
        - Optional[X] - unwrap to X
        - Nested types - preserve structure

        Args:
            return_type: Return type annotation from method
            method_name: Method name (for generated model name)

        Returns:
            Pydantic model class (always a BaseModel subclass)

        Raises:
            GenerationError: If return type is None or unsupported Union type.
        """
        # Check for None type (type(None) is NoneType singleton)
        if return_type is type(None):
            raise GenerationError(
                f"Method `{method_name}` has return type None. "
                f"PREDICT requires a concrete return type."
            )

        # Get origin and args for generic types (list, dict, Union, etc.)
        origin = get_origin(return_type)
        args = get_args(return_type)

        # Handle Optional[X] → Union[X, None] - unwrap to inner type
        # For Union types with multiple non-None types (e.g., int | float), pass through
        # to create_model which handles them correctly via Pydantic's anyOf schema.
        # Note: Python 3.10+ uses types.UnionType for X | Y syntax, while typing.Union is for Union[X, Y]
        if origin is Union or origin is types.UnionType:
            # Filter out None from Union (type(None) is NoneType singleton)
            non_none_types = [t for t in args if t is not type(None)]
            if len(non_none_types) == 1:
                # Optional[X] case - recurse with the inner type
                return self._create_response_model(non_none_types[0], method_name)
            # Multiple non-None types: fall through to create_model which handles Union types

        # If already a Pydantic model, use it or a public-only subset
        if isinstance(return_type, type) and issubclass(return_type, BaseModel):
            hidden_names = {
                name for name in return_type.model_fields if is_hidden_field(return_type, name)
            }
            if not hidden_names:
                return return_type
            # Build a model with only public fields so the LLM schema omits hidden fields
            from pydantic_core import PydanticUndefined

            public_fields = {}
            for name, field_info in return_type.model_fields.items():
                if name in hidden_names:
                    continue
                ann = field_info.annotation
                default = field_info.default
                if default is PydanticUndefined:
                    public_fields[name] = (ann, ...)
                else:
                    public_fields[name] = (ann, default)
            public_name = f"{return_type.__name__}Public"
            public_model = create_model(public_name, **cast(dict[str, Any], public_fields))
            public_model.model_rebuild(_types_namespace=vars(__import__("typing")))
            public_model._agentdoc_original_model = return_type  # pyright: ignore[reportAttributeAccessIssue]
            return public_model  # type: ignore[no-any-return]

        # For dict types, use RootModel so the LLM returns the object directly
        # (not wrapped in {"value": ...}). A dict RootModel's root schema is
        # `type: object`, which the OpenAI/Azure Responses API accepts.
        #
        # Preserve the declared key/value types (e.g. `dict[str, Person]`) so Pydantic
        # validates and constructs the values — otherwise `dict[str, Person]` returns
        # plain dicts instead of Person instances. Bare `dict` falls back to dict[str, Any].
        if return_type is dict or origin is dict:
            model_name = f"{method_name.title().replace('_', '')}Response"
            key_type, value_type = (args[0], args[1]) if len(args) == 2 else (str, Any)
            dict_type = dict[key_type, value_type]  # type: ignore[valid-type]

            class DictRootModel(RootModel[dict_type]):
                pass

            DictRootModel.__name__ = model_name
            DictRootModel.__qualname__ = model_name
            DictRootModel.model_rebuild(_types_namespace=vars(__import__("typing")))
            return DictRootModel

        # NOTE: `list` / `list[T]` are intentionally NOT given a RootModel here.
        # A `RootModel[list[...]]` produces a top-level `{"type": "array"}` schema,
        # which the OpenAI/Azure Responses API rejects (`response_format` schemas must
        # be object-rooted). Instead we let lists fall through to the generic
        # `value`-wrapper below (the same path used for tuple/set/scalars): the list
        # lives under a `value` property, keeping the root an object, and
        # `_validate_response` unwraps `value` before returning the bare list to the
        # caller. See GitLab issue 232.

        # For basic types (str, int, bool, etc.) and container types without a
        # dedicated branch (list, tuple, set), wrap in a Pydantic model with a `value`
        # field so the root schema is an object.
        model_name = f"{method_name.title().replace('_', '')}Response"

        try:
            response_model = create_model(
                model_name,
                value=(return_type, ...),  # Required field with the return type
            )
            # Rebuild with typing namespace so Pydantic can resolve types like
            # Literal, Annotated, etc. that may come from dynamically-defined methods.
            # Without this, create_model's __module__ points to this file, and types
            # not imported here cause "model is not fully defined" errors.
            import typing as _typing

            response_model.model_rebuild(_types_namespace=vars(_typing))
            return response_model
        except Exception as e:
            raise GenerationError(
                f"PredictStrategy cannot build a JSON schema for return type {return_type}: {e}\n"
                f"PredictStrategy requires a JSON-serializable return type (Pydantic model, "
                f"dataclass, or basic/container type). Non-serializable types such as "
                f"pandas.DataFrame or numpy.ndarray (including models with such fields) are not "
                f"supported. Use CodeActStrategy (which can build and return the object in code), "
                f"or return a serializable proxy — e.g. a model with `columns: list[str]` and "
                f"`rows: list[list]` instead of a DataFrame."
            ) from e

    def _validate_response(
        self, response_data: Any, response_model: type[BaseModel], original_return_type: Any
    ) -> Any:
        """Validate response data against Pydantic model and convert to original type.

        Args:
            response_data: Parsed JSON data (dict or list)
            response_model: Pydantic model to validate against
            original_return_type: Original return type annotation (for unwrapping)

        Returns:
            Validated data in the original return type format

        Raises:
            PydanticValidationError: If validation fails
        """
        # Check if response_model is a RootModel (for dict/list return types)
        is_root_model = isinstance(response_model, type) and issubclass(response_model, RootModel)

        if is_root_model:
            # RootModel takes the value directly, not as kwargs
            root_model_cls: type[RootModel[Any]] = response_model  # type: ignore[assignment]
            validated_root = root_model_cls(response_data)
            # Return the unwrapped root value
            return validated_root.root

        # Regular model validation
        validated = response_model(**response_data)

        # If we used a public-only subset (hidden fields stripped for LLM), rebuild original type
        original_cls = getattr(response_model, "_agentdoc_original_model", None)
        if original_cls is not None:
            return original_cls(**validated.model_dump())

        # If original return type is already a Pydantic model, return as-is
        if isinstance(original_return_type, type) and issubclass(original_return_type, BaseModel):
            return validated

        # Otherwise, unwrap the "value" field from wrapper model
        if hasattr(validated, "value"):
            return getattr(validated, "value")  # noqa: B009

        # Fallback: return the model as-is (shouldn't happen)
        return validated

    async def _format_validation_error(
        self, error: Exception, return_type: Any, runtime: RuntimeServices
    ) -> str:
        """Format validation error details for LLM retry.

        Uses error_pydantic_validation, error_type_validation, or error_json_parsing
        @strategy methods with TemplateStrategy.

        Args:
            error: Validation error
            return_type: Target return type
            runtime: RuntimeServices for @strategy method calls

        Returns:
            Formatted error details
        """
        if isinstance(error, json.JSONDecodeError):
            # JSON parsing errors - provide clear feedback about JSON format
            return await self.error_json_parsing(runtime, error=str(error))

        if isinstance(error, PydanticValidationError):
            # Pydantic validation errors
            error_details = []
            for err in error.errors():
                location = " -> ".join(str(loc) for loc in err["loc"])
                msg = err["msg"]
                error_type = err["type"]
                error_details.append(f"  - Field `{location}`: {msg} (type: {error_type})")

            error_list = "\n".join(error_details)

            # Use @strategy method for Pydantic error message
            return await self.error_pydantic_validation(runtime, error_list=error_list)

        else:
            # Basic type validation errors (dict, list, etc.)
            # Use @strategy method for type error message
            return await self.error_type_validation(
                runtime, return_type=str(return_type), error=str(error)
            )
