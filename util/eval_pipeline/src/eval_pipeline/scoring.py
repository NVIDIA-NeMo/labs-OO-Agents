# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoring stage of the pipeline.

Includes:
- ScorerConfig: Configuration for scorers
- ExactMatchScorer: Simple string matching (handles Pydantic models)
- TypeMatchScorer: Structure/type validation (checks schema, not values)
- LLMJudgeScorer: LLM-as-judge evaluation
- LLMMethodologyScorer: LLM-based evaluation of code execution process
- score_task(): Run all scorers on a context
- build_scoring_context(): Build context from execution result

Debug Features:
- Set DEBUG_JUDGE_INPUT=./tmp/judge_debug to dump judge inputs to files
- Set DEBUG_JUDGE_VERBOSE=1 for additional debug prints
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nooa import Agent, strategy
from nooa.agentdoc import pformat as _pformat
from nooa.strategies import PredictStrategy

from .model_factory import client as model_client
from .models import ExecutionResult, ScoreResult, ScoringContext

if TYPE_CHECKING:
    from nooa.trace_explorer import TraceExplorer


def _client_from_spec(spec):
    """Create LLM client from ModelSpec."""
    import os

    from nooa.unifiedllm import CompletionClient

    api_key = os.environ.get(spec.api_key_env, "")

    # Build extra_body for reasoning model parameters
    extra_body = {}
    if spec.reasoning_effort:
        extra_body["reasoning_effort"] = spec.reasoning_effort
    if spec.max_thinking_tokens:
        extra_body["nvext"] = {"max_thinking_tokens": spec.max_thinking_tokens}

    kwargs = {
        "model": spec.model_name,
        "api_base": spec.endpoint,
        "api_key": api_key,
        "max_tokens": spec.max_tokens or 4096,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    return CompletionClient(**kwargs)


# Scorers take the full ScoringContext and extract what they need.
#
# Example:
#
#   class ExactMatchScorer:
#       def score(self, ctx: ScoringContext) -> ScoreResult:
#           return ScoreResult(
#               score=1.0 if ctx.expected == ctx.actual else 0.0,
#               reasoning="exact match",
#           )
#
#   class CodeQualityScorer:
#       async def score(self, ctx: ScoringContext) -> ScoreResult:
#           # Uses ctx.code, ctx.actual
#           ...


@dataclass
class ScorerConfig:
    """Configuration for a scorer."""

    name: str
    weight: float
    scorer: Any  # Any object with a score() method
    # Serializable spec for subprocess reconstruction (populated by config.py)
    scorer_class: str = ""
    scorer_kwargs: dict[str, Any] = field(default_factory=dict)


def scorer_from_spec(spec) -> ScorerConfig:
    """Reconstruct a scorer from a ScorerSpec in a subprocess.

    Imports the scorer class by dotted path and instantiates with kwargs.
    """
    import importlib

    mod_path, cls_name = spec.scorer_class.rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)
    scorer = cls(**spec.scorer_kwargs)
    return ScorerConfig(
        name=spec.name,
        weight=spec.weight,
        scorer=scorer,
        scorer_class=spec.scorer_class,
        scorer_kwargs=dict(spec.scorer_kwargs),
    )


def _iter_all_sessions(trace: TraceExplorer):
    """Yield all AgentSessions depth-first, including nested children."""
    from nooa.trace_explorer.explorer import AgentSession as _AgentSession

    def _recurse(session: _AgentSession):
        yield session
        for child in session.children:
            yield from _recurse(child)

    for root in trace.sessions:
        yield from _recurse(root)


def _count_tokens(trace: TraceExplorer) -> tuple[int, int, int] | None:
    """Sum token counts across all LLM turns in the trace."""
    from nooa.trace_explorer.explorer import LLMTurn as _LLMTurn

    input_t = output_t = total_t = 0
    for session in _iter_all_sessions(trace):
        for turn in session.turns:
            if isinstance(turn, _LLMTurn) and turn.token_counts:
                input_t += turn.token_counts.get("prompt", 0)
                output_t += turn.token_counts.get("completion", 0)
                total_t += turn.token_counts.get("total", 0)
    if input_t or output_t or total_t:
        return (input_t, output_t, total_t)
    return None


def _get_code_executions(
    trace: TraceExplorer, *, skip_prefill: bool = False
) -> list[dict[str, Any]]:
    """Extract all code executions from all sessions in the trace."""
    from nooa.trace_explorer.explorer import ExecutionTurn as _ExecutionTurn

    executions = []
    for session in _iter_all_sessions(trace):
        for turn in session.turns:
            if isinstance(turn, _ExecutionTurn):
                if skip_prefill and turn.tool_call_id.startswith("prefill_"):
                    continue
                executions.append(
                    {
                        "code": turn.code,
                        "result": turn.returned_value,
                        "error": turn.error,
                        "span_name": "code_execution",
                        "timestamp": session.start_time,
                    }
                )
    executions.sort(key=lambda x: x.get("timestamp", 0))
    return executions


def _get_code(trace: TraceExplorer, *, skip_prefill: bool = False) -> str | None:
    """Extract generated code from the trace.

    For CodeAct/PurePython: returns concatenated code from ExecutionTurn blocks.
    Direct assistant response text is not code and returns None.
    """
    from nooa.trace_explorer.explorer import ExecutionTurn as _ExecutionTurn

    code_blocks: list[str] = []

    for session in _iter_all_sessions(trace):
        for turn in session.turns:
            if isinstance(turn, _ExecutionTurn):
                if skip_prefill and turn.tool_call_id.startswith("prefill_"):
                    continue
                if turn.code:
                    code_blocks.append(turn.code)

    if code_blocks:
        return "\n\n".join(code_blocks)
    return None


def build_scoring_context(
    result: ExecutionResult,
    *,
    metadata: dict[str, Any] | None = None,
    trace: TraceExplorer | None = None,
    use_otlp: bool = False,
) -> ScoringContext:
    """Build scoring context from execution result.

    Token counts are extracted from the TraceExplorer instance (when provided).

    Args:
        metadata: Task metadata dict (from Task.metadata) forwarded to
            ScoringContext.metadata so custom scorers can access per-task
            rubrics, difficulty tags, etc.
        trace: TraceExplorer instance for token extraction and code analysis.
    """
    token_data = _count_tokens(trace) if trace is not None else None

    input_tokens = None
    output_tokens = None
    total_tokens = None
    if token_data:
        input_tokens, output_tokens, total_tokens = token_data

    return ScoringContext(
        task_id=result.task_id,
        input=result.input,
        expected=result.expected,
        actual=result.actual,
        trace=trace,
        latency_ms=result.latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        metadata=metadata or {},
        error=result.error,
        session_id=result.session_id,
        use_otlp=use_otlp,
    )


async def score_task(
    ctx: ScoringContext,
    scorers: list[ScorerConfig],
) -> dict[str, dict[str, Any]]:
    """Run all scorers on a context.

    Each scorer receives the full ScoringContext and extracts what it needs.

    Returns dict mapping scorer name to score details.
    """
    import asyncio

    results: dict[str, dict[str, Any]] = {}

    for config in scorers:
        scorer = config.scorer

        # Handle both sync and async scorers
        if asyncio.iscoroutinefunction(getattr(scorer, "score", None)):
            result = await scorer.score(ctx)
        else:
            result = scorer.score(ctx)

        results[config.name] = {
            "score": result.score,
            "weight": config.weight,
            "reasoning": result.reasoning,
            "metadata": result.metadata,
        }

    return results


def compute_weighted_score(scores: dict[str, dict[str, Any]]) -> float:
    """Compute weighted average of scores."""
    if not scores:
        return 0.0

    total_weight = sum(s["weight"] for s in scores.values())
    if total_weight == 0:
        return 0.0

    weighted_sum = sum(s["score"] * s["weight"] for s in scores.values())
    return weighted_sum / total_weight


def _prepare_rubric_context(
    rubric: str,
    output_val: Any,
    expected_val: Any,
    **extra_context: Any,
) -> tuple[str, dict[str, Any]]:
    """Prepare rubric and template context for LLM judge scoring.

    Flattens nested dict access and transforms {foo[bar]} syntax to {foo_bar}.

    Args:
        rubric: The rubric template with placeholders
        output_val: The output value (will be flattened with 'output_' prefix)
        expected_val: The expected value (will be flattened with 'expected_' prefix)
        **extra_context: Additional context keys to include in template

    Returns:
        Tuple of (transformed_rubric, template_context)
    """
    import re

    # Build context dict for template substitution
    # Use empty dict for None values to avoid TypeError on subscript access
    if output_val is None:
        output_val = {}
    if expected_val is None:
        expected_val = {}

    template_context: dict[str, Any] = {
        "output": output_val,
        "expected": expected_val,
        **extra_context,
    }

    # Flatten nested dict access: {expected[key]} -> expected_key in context
    # This allows rubrics to use {expected[outcome]} syntax
    def flatten_nested(obj: Any, prefix: str, context: dict[str, Any]) -> None:
        """Add flattened keys for nested dict access."""
        if isinstance(obj, dict):
            for key, value in obj.items():
                context[f"{prefix}_{key}"] = value

    flatten_nested(output_val, "output", template_context)
    flatten_nested(expected_val, "expected", template_context)

    # Transform rubric: replace {foo[bar]} with {foo_bar}
    transformed_rubric = re.sub(r"\{(\w+)\[(\w+)\]\}", r"{\1_\2}", rubric)

    return transformed_rubric, template_context


def _format_rubric_with_context(rubric: str, template_context: dict[str, Any]) -> str:
    """Format rubric with template context, with fallback for missing keys.

    Args:
        rubric: The rubric template (already transformed by _prepare_rubric_context)
        template_context: Dictionary of values to substitute

    Returns:
        Formatted rubric string
    """
    try:
        return rubric.format(**template_context)
    except (KeyError, TypeError):
        # Fallback: use partial formatting for missing keys or type errors
        prompt = rubric
        for key, value in template_context.items():
            prompt = prompt.replace(f"{{{key}}}", str(value) if value is not None else "")
        return prompt


# =============================================================================
# Scorer Implementations
# =============================================================================


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two values for equality, handling numeric type differences.

    576 == 576.0, [1, 2.0] == [1.0, 2], etc.
    String comparison is case-insensitive and whitespace-trimmed.
    """
    # Both numbers - compare numerically
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b

    # Both strings - case-insensitive comparison (matching original behavior)
    if isinstance(a, str) and isinstance(b, str):
        return a.lower().strip() == b.lower().strip()

    # Both lists - compare element-wise
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_values_equal(x, y) for x, y in zip(a, b, strict=True))

    # Both dicts - subset matching (expected keys must exist in actual with matching values)
    # This allows actual to have extra keys beyond what's specified in expected
    if isinstance(a, dict) and isinstance(b, dict):
        # Check all expected keys exist in actual
        if not set(a.keys()).issubset(set(b.keys())):
            return False
        # Check values match for expected keys
        return all(_values_equal(a[k], b[k]) for k in a)

    # Same type - direct comparison
    if type(a) is type(b):
        return a == b

    # Different types - try string comparison (case-insensitive)
    return str(a).lower().strip() == str(b).lower().strip()


def _parse_value(val: Any) -> Any:
    """Try to parse a value as JSON/Python literal.

    Returns parsed value or original if parsing fails.
    Handles Pydantic models by converting them to dicts (recursively).
    """
    # Handle Pydantic models - convert to dict (recursively handles nested models)
    if hasattr(val, "model_dump"):
        # Pydantic v2 - model_dump() recursively converts nested models
        return val.model_dump()
    elif hasattr(val, "dict"):
        # Pydantic v1 - dict() recursively converts nested models
        return val.dict()

    # Handle dicts - recursively parse values (in case of nested Pydantic models)
    if isinstance(val, dict):
        return {k: _parse_value(v) for k, v in val.items()}

    # Handle lists - recursively parse elements
    if isinstance(val, list):
        return [_parse_value(item) for item in val]

    if not isinstance(val, str):
        return val

    s = val.strip()

    # Try JSON first
    try:
        parsed = json.loads(s)
        # Recursively parse the result in case it contains nested structures
        return _parse_value(parsed)
    except (json.JSONDecodeError, ValueError):
        pass

    # Try Python literal (handles things like Python's repr of lists)
    try:
        parsed = ast.literal_eval(s)
        return _parse_value(parsed)
    except (ValueError, SyntaxError):
        pass

    return val


def _describe_mismatch(expected: Any, actual: Any) -> str:
    """Produce a concise description of WHY two values differ.

    For lists, pinpoints the first differing index so the message is never
    "Expected [a, b, c, ...+37], got [a, b, c, ...+37]" (i.e. looks identical).
    """
    # List-vs-list: find the first element that differs
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return f"List length mismatch: expected {len(expected)}, got {len(actual)}"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            if not _values_equal(e, a):
                e_repr = repr(e)
                a_repr = repr(a)
                type_hint = ""
                if type(e) is not type(a):
                    type_hint = f" (types: {type(e).__name__} vs {type(a).__name__})"
                return (
                    f"Lists differ at index {i}/{len(expected)}: "
                    f"expected {e_repr}, got {a_repr}{type_hint}"
                )
        return "Output mismatch (no element diff found)"

    # Dict-vs-dict: report first key difference
    if isinstance(expected, dict) and isinstance(actual, dict):
        missing = set(expected.keys()) - set(actual.keys())
        if missing:
            return f"Missing keys: {missing}"
        for k in expected:
            if not _values_equal(expected[k], actual[k]):
                return f"Dict differs at key {k!r}: expected {expected[k]!r}, got {actual[k]!r}"
        return "Output mismatch (no key diff found)"

    # Scalar / short values: show both
    exp_brief = _pformat(expected, max_string=40, max_length=3, max_depth=2)
    act_brief = _pformat(actual, max_string=40, max_length=3, max_depth=2)
    if len(exp_brief) + len(act_brief) < 60:
        return f"Expected {exp_brief}, got {act_brief}"
    return "Output mismatch"


class ExactMatchScorer:
    """Exact match scorer.

    Compares actual output to expected output.
    Handles numeric type differences (576 == 576.0).
    Handles Pydantic models by converting them to dicts (recursively).
    Falls back to case-insensitive string comparison.
    Returns score of 1.0 for match, 0.0 for mismatch.
    """

    def score(self, ctx: ScoringContext) -> ScoreResult:
        expected_raw = ctx.expected
        actual_raw = ctx.actual

        # Parse values if they're strings representing structured data
        # This also handles Pydantic models by calling model_dump()
        expected = _parse_value(expected_raw)
        actual = _parse_value(actual_raw)

        # Auto-extract answer from dict outputs with common response patterns
        # This handles agents that return {"response": X, "answer": X, ...}
        if isinstance(actual, dict) and not isinstance(expected, dict):
            # Try common answer keys in order of preference
            for key in ("answer", "response", "result", "output"):
                if key in actual:
                    actual = _parse_value(actual[key])
                    break

        # Compare with numeric-aware equality
        match = _values_equal(expected, actual)

        # For display, show parsed values (truncated if large)
        def _truncate(val: Any, max_len: int = 500) -> str:
            s = json.dumps(val, default=str) if isinstance(val, (dict, list)) else str(val)
            if len(s) > max_len:
                return s[:max_len] + "..."
            return s

        expected_str = _truncate(expected)
        actual_str = _truncate(actual)

        # Track type info for debugging
        raw_type = type(actual_raw).__name__
        parsed_type = type(actual).__name__

        # Generate concise, human-readable reasoning
        if match:
            reasoning = "Output matches"
        else:
            reasoning = _describe_mismatch(expected, actual)

        return ScoreResult(
            score=1.0 if match else 0.0,
            reasoning=reasoning,
            metadata={
                "result": actual_str,
                "expected": expected_str,
                "output_correct": match,
                "raw_type": raw_type,
                "parsed_type": parsed_type,
            },
        )


class TypeMatchScorer:
    """Type/structure matching scorer.

    Validates that the actual output has the same structure and types as expected,
    without requiring exact value matches. Perfect for structured output tests
    where LLM-generated content may vary but structure must be correct.

    Checks:
    - All expected keys exist in actual (for dicts)
    - Types match (str, int, float, bool, list, dict)
    - Nested structures are recursively validated
    - Lists have correct element types (checks first element)

    Returns score of 1.0 if structure matches, 0.0 otherwise.
    """

    def _get_type_name(self, val: Any) -> str:
        """Get a simplified type name for a value."""
        if val is None:
            return "null"
        if isinstance(val, bool):  # Must check before int since bool is subclass of int
            return "bool"
        if isinstance(val, int):
            return "int"
        if isinstance(val, float):
            return "number"
        if isinstance(val, str):
            return "str"
        if isinstance(val, list):
            return "list"
        if isinstance(val, dict):
            return "dict"
        return type(val).__name__

    def _check_type_match(
        self, expected: Any, actual: Any, path: str = ""
    ) -> tuple[bool, list[str]]:
        """Recursively check if actual matches expected structure/types.

        Args:
            expected: Expected value (defines the schema)
            actual: Actual value to validate
            path: Current path for error messages (e.g., "user.name")

        Returns:
            Tuple of (matches: bool, errors: list[str])
        """
        errors = []
        current_path = path or "root"

        # Handle None
        if expected is None:
            if actual is not None:
                errors.append(f"{current_path}: expected null, got {self._get_type_name(actual)}")
            return len(errors) == 0, errors

        # Get types
        expected_type = self._get_type_name(expected)
        actual_type = self._get_type_name(actual)

        # Allow int/float interchangeability for numbers
        if expected_type in ("int", "number") and actual_type in ("int", "number"):
            return True, []

        # Type mismatch
        if expected_type != actual_type:
            errors.append(f"{current_path}: expected {expected_type}, got {actual_type}")
            return False, errors

        # Dict - check all expected keys exist and have correct types
        if isinstance(expected, dict) and isinstance(actual, dict):
            for key in expected:
                key_path = f"{path}.{key}" if path else key
                if key not in actual:
                    errors.append(f"{key_path}: missing key")
                else:
                    _, sub_errors = self._check_type_match(expected[key], actual[key], key_path)
                    errors.extend(sub_errors)
            return len(errors) == 0, errors

        # List - check element types match (using first element as reference)
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) > 0 and len(actual) > 0:
                # Check first element type as representative
                elem_path = f"{path}[0]" if path else "[0]"
                _, sub_errors = self._check_type_match(expected[0], actual[0], elem_path)
                errors.extend(sub_errors)
            elif len(expected) > 0 and len(actual) == 0:
                errors.append(f"{current_path}: expected non-empty list, got empty list")
            return len(errors) == 0, errors

        # Primitive types match
        return True, []

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """Score based on type/structure matching.

        Args:
            ctx: Scoring context with expected and actual

        Returns:
            ScoreResult with score 1.0 if structure matches, 0.0 otherwise
        """
        expected_raw = ctx.expected
        actual_raw = ctx.actual

        # Parse values (handles Pydantic models, JSON strings, etc.)
        expected = _parse_value(expected_raw)
        actual = _parse_value(actual_raw)

        # Check type match
        matches, errors = self._check_type_match(expected, actual)

        if matches:
            reasoning = "Structure and types match expected schema"
        else:
            reasoning = f"Type/structure mismatch: {'; '.join(errors[:5])}"
            if len(errors) > 5:
                reasoning += f" (and {len(errors) - 5} more errors)"

        return ScoreResult(
            score=1.0 if matches else 0.0,
            reasoning=reasoning,
            metadata={
                "type_match": matches,
                "errors": errors,
                "error_count": len(errors),
            },
        )


class ModeSelectionScorer:
    """Check if agent used expected execution mode (internal vs code).

    Validates whether the agent correctly chose to use internal reasoning
    (no code execution) vs code execution based on the task requirements.

    Trivial code (no operations - just assignments, prints, returns) is treated
    as internal mode since it's functionally a direct answer, not real code execution.

    Usage in config:
        scorers:
          - name: mode_check
            class: ModeSelectionScorer
            weight: 0.4
            expected: internal  # or "code"
    """

    def __init__(self, expected: str):
        """Initialize scorer.

        Args:
            expected: Either "internal" (no code) or "code" (executed code)
        """
        if expected not in ("internal", "code"):
            raise ValueError(f"expected must be 'internal' or 'code', got {expected}")
        self.expected = expected

    def _has_operations(self, node: ast.AST) -> bool:
        """Check if AST node contains any operations (loops, conditionals, comparisons, etc.).

        Returns True if node contains non-trivial operations that indicate computation.
        """
        if isinstance(
            node,
            (
                ast.For,
                ast.While,
                ast.AsyncFor,  # Loops
                ast.If,
                ast.IfExp,  # Conditionals
                ast.BinOp,  # +, -, *, /, etc.
                ast.Compare,  # ==, !=, <, >, in, not in, etc.
                ast.BoolOp,  # and, or
            ),
        ):
            return True

        if isinstance(node, ast.UnaryOp) and not isinstance(node.operand, ast.Constant):
            return True

        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in ("print", "pprint", "return_result"):
                    return True
            else:
                return True

        if isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            return True

        for child in ast.iter_child_nodes(node):
            if self._has_operations(child):
                return True

        return False

    def _is_trivial_code(self, code: str | None) -> bool:
        """Check if code is trivial (just packaging an answer, no real computation).

        Trivial code is functionally equivalent to internal reasoning - no real
        computation, loops, or data processing.

        The implementation uses a simple heuristic: code is trivial if it contains
        no operations (loops, conditionals, comparisons, arithmetic, function calls).

        Examples of trivial code:
        - return 'negative'
        - return [1, 2, 3]
        - return {'a': 1}
        - 'negative'
        - return_result('negative')
        - sentiment = 'negative'; return_result(sentiment)
        - text_content = text; sentiment = 'positive'; return sentiment
        - print('debug'); return_result('neutral')
        - # comment; sentiment = 'neutral'; return_result(sentiment)

        Examples of non-trivial code (contains operations):
        - for item in items: ...  (loop)
        - if 'love' in text: ...  (conditional + comparison)
        - result = process(data)  (function call)
        - pos_count = sum(1 for word in words if word in text)  (comprehension + comparison)
        - x + y  (arithmetic)
        - 'love' in text  (comparison operation)
        """
        if code is None:
            return True

        # Strip common prefixes
        for prefix in ("# Generated code:", "# Code executed:"):
            if code.startswith(prefix):
                code = code[len(prefix) :].strip()

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False

        if len(tree.body) == 0:
            return True

        if self._has_operations(tree):
            return False

        return True

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """Score based on whether code execution matches expectation.

        Args:
            ctx: Scoring context with trace_file

        Returns:
            ScoreResult with score 1.0 if mode matches, 0.0 otherwise
        """

        if ctx.trace is not None:
            code = _get_code(ctx.trace, skip_prefill=True)
        else:
            return ScoreResult(
                score=0.0,
                reasoning="No trace data available",
                metadata={},
            )
        # Trivial code (just return literal) counts as internal, not code
        is_trivial = self._is_trivial_code(code)
        executed_real_code = code is not None and not is_trivial

        if self.expected == "internal":
            passed = not executed_real_code
            reasoning = "Expected no code execution"
        else:
            passed = executed_real_code
            reasoning = "Expected code execution"

        reasoning += f". Code was {'executed' if executed_real_code else 'not executed'}."

        return ScoreResult(
            score=1.0 if passed else 0.0,
            reasoning=reasoning,
            metadata={
                "expected_mode": self.expected,
                "executed_code": code is not None,
                "is_trivial_code": is_trivial,
                "executed_real_code": executed_real_code,
                "mode_correct": passed,
                "code": code,
            },
        )


class FastAgentHelpScorer:
    """Deterministically score fast_agent help/no-help behavior from traces.

    This scorer deliberately avoids LLM judging. It ignores CodeAct prefill
    executions and checks only whether agent-initiated code called
    ``self.call_for_help(...)``. Direct text answers therefore count as no-help
    behavior instead of being misclassified as executed code.
    """

    def __init__(self, expect_help: bool):
        self.expect_help = expect_help

    @staticmethod
    def _code_calls_help(code: str) -> bool:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fallback for malformed/incomplete snippets in traces.
            return "self.call_for_help" in code

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "call_for_help"
                and isinstance(func.value, ast.Name)
                and func.value.id == "self"
            ):
                return True
        return False

    def score(self, ctx: ScoringContext) -> ScoreResult:
        if ctx.trace is None:
            return ScoreResult(
                score=0.0,
                reasoning="No trace data available",
                metadata={"expected_help": self.expect_help},
            )

        executions = _get_code_executions(ctx.trace, skip_prefill=True)
        help_executions = [e for e in executions if self._code_calls_help(str(e.get("code") or ""))]
        called_help = bool(help_executions)

        if self.expect_help:
            passed = called_help
            reasoning = (
                "Expected self.call_for_help(...) and found an agent-initiated call"
                if passed
                else "Expected self.call_for_help(...) but no agent-initiated call was found"
            )
        else:
            passed = not called_help
            if passed:
                reasoning = "Expected direct answer/no help; no self.call_for_help(...) call found"
            else:
                reasoning = "Expected direct answer/no help, but self.call_for_help(...) was called"

        return ScoreResult(
            score=1.0 if passed else 0.0,
            reasoning=reasoning,
            metadata={
                "expected_help": self.expect_help,
                "called_help": called_help,
                "execution_count": len(executions),
                "help_execution_count": len(help_executions),
                "help_codes": [e.get("code") for e in help_executions],
            },
        )


@dataclass
class Judgment:
    """Structured output from LLM judge.

    Attributes:
        passed: Whether the evaluation passed
        reasoning: Explanation for the judgment
    """

    passed: bool
    reasoning: str


class LLMJudgeAgent(Agent):
    """Agent that evaluates content against a rubric.

    The prompt is fully specified by the caller - no hardcoded template.
    """

    def __init__(self, llm_client):
        super().__init__(llm=llm_client)

    @strategy(PredictStrategy())
    async def judge(self, prompt: str) -> Judgment:
        """{prompt}"""
        ...


class DumbCodeScorer:
    """Penalize recursion and naive classifiers — accept direct answers and code-as-interface.

    Inspects executed code for bad patterns:
    - Recursion: calling self.<method_name>() (the method being tested)
    - Naive classifiers: if/elif chains with string-in-string checks

    Passes when:
    - No code was executed (direct return_result tool call)
    - Code is trivial (just return_result with a literal)
    - Code uses legitimate reasoning without bad patterns

    Usage in config:
        scorers:
          - name: methodology
            class: DumbCodeScorer
            weight: 0.5
    """

    def __init__(self, **kwargs):
        pass

    def score(self, ctx: ScoringContext) -> ScoreResult:
        """Score based on absence of dumb code patterns."""
        if ctx.trace is not None:
            code = _get_code(ctx.trace, skip_prefill=True)
        else:
            return ScoreResult(score=1.0, reasoning="No trace — pass by default", metadata={})

        # No code executed → direct answer → always OK
        if code is None:
            return ScoreResult(
                score=1.0,
                reasoning="No code execution (direct answer) — accepted",
                metadata={"mode": "direct", "dumb_patterns": []},
            )

        # Check for dumb patterns in the code
        dumb_patterns = []

        # Pattern 1: Recursion — self.classify(), self.answer(), etc.
        import re as _re

        recursion_matches = _re.findall(r"self\.[a-z_]+\(", code)
        # Filter out allowed self.* calls (self.call_for_help is OK in help tests)
        bad_recursion = [
            m
            for m in recursion_matches
            if not any(ok in m for ok in ["self.call_for_help", "self.notes", "self.context"])
        ]
        if bad_recursion:
            dumb_patterns.append(f"recursion: {bad_recursion}")

        # Pattern 2: Naive classifier — if "word" in text / if text contains
        naive_matches = _re.findall(r'if\s+["\'].+?["\']\s+in\s+', code)
        naive_matches += _re.findall(r"if\s+.*\.(contains|find|index)\(", code)
        # Also catch elif chains with string literals
        elif_with_strings = _re.findall(r'elif\s+["\'].+?["\']\s+in\s+', code)
        naive_matches += elif_with_strings
        if naive_matches:
            dumb_patterns.append(f"naive_classifier: {naive_matches}")

        # Pattern 3: Importing NLP libraries
        nlp_imports = _re.findall(r"import\s+(nltk|spacy|transformers|sklearn|textblob)", code)
        if nlp_imports:
            dumb_patterns.append(f"nlp_import: {nlp_imports}")

        if dumb_patterns:
            return ScoreResult(
                score=0.0,
                reasoning=f"Dumb code detected: {'; '.join(dumb_patterns)}",
                metadata={"mode": "code", "dumb_patterns": dumb_patterns, "code": code},
            )

        return ScoreResult(
            score=1.0,
            reasoning="Code-as-interface without dumb patterns — accepted",
            metadata={"mode": "code", "dumb_patterns": [], "code": code},
        )


class LLMJudgeScorer:
    """LLM-as-judge scorer.

    Uses an LLM to evaluate content against a rubric with template placeholders.
    Judge traces are written to a separate file from agent traces.

    The rubric can contain placeholders that get filled from ScoringContext:
        {input}    - The task input
        {output}   - The actual output from the agent
        {expected} - The expected output
        {code}     - Generated code (if available)

    Example rubric:
        '''
        Evaluate if the output matches the expected result.

        Input: {input}
        Expected: {expected}
        Actual: {output}

        Return passed=true if the output is semantically equivalent.
        '''

    NOTE: A fresh judge agent is created for each score() call to avoid
    conversation history accumulation across samples.
    """

    def __init__(
        self,
        rubric: str,
        model: str | None = None,
        model_spec=None,
        skip_prefill: bool = False,
    ):
        """Initialize LLM judge scorer.

        Args:
            rubric: Evaluation rubric with optional {field} placeholders
            model: Model ID to look up in models.yaml (deprecated)
            model_spec: ModelSpec with full configuration (preferred)
            skip_prefill: If True, exclude prefill code from {code} in rubric
        """
        if model_spec:
            # Accept dict (from subprocess reconstruction) or ModelSpec
            if isinstance(model_spec, dict):
                from .eval_types import ModelSpec

                model_spec = ModelSpec(**model_spec)
            self._llm = _client_from_spec(model_spec)
        elif model:
            self._llm = model_client(model)
        else:
            raise ValueError("LLMJudgeScorer requires either 'model' or 'model_spec'")
        self._rubric = rubric
        self._skip_prefill = skip_prefill

        # Debug: Print if verbose debug mode is enabled
        import os

        debug_judge = os.environ.get("DEBUG_JUDGE_INPUT")
        debug_verbose = os.environ.get("DEBUG_JUDGE_VERBOSE", "0") == "1"
        if debug_judge and debug_verbose:
            print(
                f"[DEBUG] LLMJudgeScorer initialized with DEBUG_JUDGE_INPUT={debug_judge}, skip_prefill={skip_prefill}",
                flush=True,
            )

    def _create_agent(self):
        """Create a fresh judge agent instance to avoid history accumulation."""
        return LLMJudgeAgent(llm_client=self._llm)

    async def score(self, ctx: ScoringContext) -> ScoreResult:
        """Score the output using LLM judge.

        Formats the rubric template with context fields and runs the judge.
        """
        # DEBUG: Entry point (only if verbose mode)
        import os

        debug_judge = os.environ.get("DEBUG_JUDGE_INPUT")
        debug_verbose = os.environ.get("DEBUG_JUDGE_VERBOSE", "0") == "1"
        if debug_judge and debug_verbose:
            print(
                f"[DEBUG] LLMJudgeScorer.score() called for task_id={ctx.task_id}, skip_prefill={self._skip_prefill}",
                flush=True,
            )

        # Get code, optionally filtering out prefill
        if self._skip_prefill:
            code = _get_code(ctx.trace, skip_prefill=True) if ctx.trace else None
        else:
            code = _get_code(ctx.trace) if ctx.trace else None

        # Prepare rubric and template context using shared helper
        rubric, template_context = _prepare_rubric_context(
            self._rubric,
            ctx.actual,
            ctx.expected,
            input=ctx.input or "",
            code=code or "",
        )

        # Format rubric with template context
        prompt = _format_rubric_with_context(rubric, template_context)

        # DEBUG: Dump judge input to file for inspection
        # This helps diagnose issues where judge sees different data than expected
        import uuid

        if debug_judge:
            try:
                debug_dir = Path(debug_judge)
                debug_dir.mkdir(parents=True, exist_ok=True)
                debug_file = debug_dir / f"judge_input_{uuid.uuid4().hex[:8]}.txt"
                debug_content = (
                    f"=== JUDGE INPUT DEBUG ===\n\n"
                    f"Task ID: {ctx.task_id}\n"
                    f"Skip prefill: {self._skip_prefill}\n\n"
                    f"=== CODE EXTRACTED ===\n{code or '(no code)'}\n\n"
                    f"=== FULL PROMPT TO JUDGE ===\n{prompt}\n"
                )
                debug_file.write_text(debug_content)
                # Always print file write (this is the key info users need)
                print(f"Judge debug: {debug_file}", flush=True)
            except Exception as e:
                print(f"[ERROR] Failed to write judge debug file: {e}", flush=True)

        # Switch to judge session (separate from agent session)
        agent_session = None
        judge_session = None
        try:
            from nooa.tracing import get_session, set_session

            agent_session = get_session()
            judge_session = f"{agent_session}_judge" if agent_session else f"{ctx.task_id}_judge"
            set_session(judge_session)
        except Exception:
            pass

        # Create fresh agent for this call to avoid history accumulation
        agent = self._create_agent()

        # Run the judge with retry logic
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                result = await agent.judge(prompt)
                passed = result.passed
                reasoning = result.reasoning
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    continue
        else:
            error_msg = str(last_error)
            passed = False
            reasoning = f"Judge failed after {max_retries} attempts: {error_msg[:200]}"

        # Switch back to agent session
        if agent_session:
            try:
                from nooa.tracing import set_session

                set_session(agent_session)
            except Exception:
                pass

        # Flush judge traces
        try:
            from nooa.tracing import flush_traces

            flush_traces()
        except Exception:
            pass

        return ScoreResult(
            score=1.0 if passed else 0.0,
            reasoning=reasoning,
            metadata={
                "judge_session": judge_session,
            },
        )


class LLMMethodologyScorer:
    """Methodology scorer.

    Uses an LLM to evaluate all code executions from the trace against a rubric.
    Unlike LLMJudgeScorer which evaluates final output, this scorer validates
    the execution process - all intermediate code executions and their results.

    The rubric can contain placeholders that get filled from ScoringContext:
        {input}        - The task input
        {output}       - The final output
        {expected}     - The expected output
        {executions}   - All code executions with results and errors

    Example rubric:
        '''
        Evaluate if the code executions demonstrate proper error handling.

        Task: {input}

        Code Executions:
        {executions}

        Return passed=true if:
        1. Errors are caught and handled gracefully
        2. Recovery attempts are made when failures occur
        3. The agent doesn't give up after first failure
        '''

    NOTE: A fresh judge agent is created for each score() call to avoid
    conversation history accumulation across samples.
    """

    def __init__(
        self,
        rubric: str,
        model: str | None = None,
        model_spec=None,
        skip_prefill: bool = True,
    ):
        """Initialize methodology scorer.

        Args:
            rubric: Evaluation rubric with optional {field} placeholders
            model: Model ID to look up in models.yaml (deprecated)
            model_spec: ModelSpec with full configuration (preferred)
            skip_prefill: If True (default), exclude prefill/inspection code
                from the executions shown to the judge. Prefill is auto-generated
                by CodeActStrategy and not agent-initiated.
        """
        if model_spec:
            if isinstance(model_spec, dict):
                from .eval_types import ModelSpec

                model_spec = ModelSpec(**model_spec)
            self._llm = _client_from_spec(model_spec)
        elif model:
            self._llm = model_client(model)
        else:
            raise ValueError("LLMMethodologyScorer requires either 'model' or 'model_spec'")
        self._rubric = rubric
        self._skip_prefill = skip_prefill

    def _create_agent(self):
        """Create a fresh judge agent instance to avoid history accumulation."""
        return LLMJudgeAgent(llm_client=self._llm)

    def _format_executions(self, executions: list[dict[str, Any]]) -> str:
        """Format execution records into readable text."""
        if not executions:
            return "(No code executions found)"

        lines = []
        for i, exec_record in enumerate(executions, 1):
            lines.append(f"Execution {i}:")
            lines.append("```python")
            lines.append(exec_record["code"])
            lines.append("```")

            if exec_record.get("error"):
                lines.append(f"Error: {exec_record['error']}")
            elif exec_record.get("result") is not None:
                result_str = str(exec_record["result"])
                # Truncate long results
                if len(result_str) > 200:
                    result_str = result_str[:200] + "..."
                lines.append(f"Result: {result_str}")
            else:
                lines.append("(No result captured)")

            if exec_record.get("assistant"):
                lines.append(f"Assistant response: {exec_record['assistant']}")

            lines.append("")  # Blank line between executions

        return "\n".join(lines)

    async def score(self, ctx: ScoringContext) -> ScoreResult:
        """Score the execution process using LLM judge.

        Extracts all code executions from trace and validates against rubric.
        """
        # Extract all code executions from trace
        if ctx.trace:
            executions = _get_code_executions(ctx.trace, skip_prefill=self._skip_prefill)
        else:
            return ScoreResult(
                score=0.0,
                reasoning="No trace data available for method evaluation",
                metadata={"execution_count": 0},
            )
        formatted_executions = self._format_executions(executions)

        # Prepare rubric and template context using shared helper
        rubric, template_context = _prepare_rubric_context(
            self._rubric,
            ctx.actual,
            ctx.expected,
            input=ctx.input,
            executions=formatted_executions,
            execution_count=len(executions),
        )

        # Format rubric with template context
        prompt = _format_rubric_with_context(rubric, template_context)

        # Switch to judge trace file (separate from agent trace)
        agent_session = None
        judge_session = None
        try:
            from nooa.tracing import get_session, set_session

            agent_session = get_session()
            judge_session = (
                f"{agent_session}_method_judge" if agent_session else f"{ctx.task_id}_method_judge"
            )
            set_session(judge_session)
        except Exception:
            pass

        # Create fresh agent for this call to avoid history accumulation
        agent = self._create_agent()

        # Run the judge with retry logic
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                result = await agent.judge(prompt)
                passed = result.passed
                reasoning = result.reasoning
                break
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    continue
        else:
            error_msg = str(last_error)
            passed = False
            reasoning = f"Judge failed after {max_retries} attempts: {error_msg[:200]}"

        # Switch back to agent session
        if agent_session:
            try:
                from nooa.tracing import set_session

                set_session(agent_session)
            except Exception:
                pass

        # Flush judge traces
        try:
            from nooa.tracing import flush_traces

            flush_traces()
        except Exception:
            pass

        return ScoreResult(
            score=1.0 if passed else 0.0,
            reasoning=reasoning,
            metadata={
                "judge_session": judge_session,
                "execution_count": len(executions),
            },
        )
