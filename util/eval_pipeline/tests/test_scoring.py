# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for eval_pipeline scoring."""

import pytest

from eval_pipeline.models import ExecutionResult, ScoreResult, ScoringContext
from eval_pipeline.scoring import (
    ExactMatchScorer as RealExactMatchScorer,
)
from eval_pipeline.scoring import (
    ModeSelectionScorer,
    ScorerConfig,
    TypeMatchScorer,
    _get_code,
    _parse_value,
    _values_equal,
    build_scoring_context,
    compute_weighted_score,
    score_task,
)


def _make_trace(code: str):
    """Create a minimal TraceExplorer with a single ExecutionTurn."""
    from nooa.trace_explorer import TraceExplorer
    from nooa.trace_explorer.explorer import AgentSession, ExecutionTurn

    session = AgentSession(
        session_id="test",
        agent_name="TestAgent",
        method_name="run",
        parent_session_id=None,
    )
    session.turns.append(
        ExecutionTurn(
            code=code,
            stdout="",
            error=None,
            returned_value=None,
            tool_call_id="call_test",
        )
    )
    return TraceExplorer(sessions=[session], trace_file="test://trace")


# Example scorers - simple classes that take ctx and extract what they need


class ExactMatchScorer:
    """Simple exact match scorer."""

    def __init__(self, case_insensitive: bool = False):
        self.case_insensitive = case_insensitive

    def score(self, ctx: ScoringContext) -> ScoreResult:
        expected = str(ctx.expected)
        actual = str(ctx.actual)

        if self.case_insensitive:
            expected = expected.lower()
            actual = actual.lower()

        match = expected == actual
        return ScoreResult(
            score=1.0 if match else 0.0,
            reasoning=f"{'Match' if match else 'No match'}: {actual!r} vs {expected!r}",
        )


class CodeQualityScorer:
    """Scorer that checks code doesn't use keyword lists."""

    def score(self, ctx: ScoringContext) -> ScoreResult:
        code = _get_code(ctx.trace) or "" if ctx.trace else ""

        # Check for keyword list patterns
        bad_patterns = ["positive_words", "negative_words", "keywords"]
        has_bad_pattern = any(p in code.lower() for p in bad_patterns)

        return ScoreResult(
            score=0.0 if has_bad_pattern else 1.0,
            reasoning="Uses keyword list" if has_bad_pattern else "Direct approach",
        )


class TestExactMatchScorer:
    def test_exact_match(self):
        ctx = ScoringContext(
            task_id="t1",
            input="classify this",
            expected="positive",
            actual="positive",
        )
        scorer = ExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_no_match(self):
        ctx = ScoringContext(
            task_id="t1",
            input="classify this",
            expected="positive",
            actual="negative",
        )
        scorer = ExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 0.0

    def test_case_insensitive(self):
        ctx = ScoringContext(
            task_id="t1",
            input="classify this",
            expected="Positive",
            actual="positive",
        )
        scorer = ExactMatchScorer(case_insensitive=True)
        result = scorer.score(ctx)
        assert result.score == 1.0


class TestCodeQualityScorer:
    def test_direct_approach(self):
        trace = _make_trace('return "positive"')
        ctx = ScoringContext(
            task_id="t1",
            input="classify this",
            expected="positive",
            actual="positive",
            trace=trace,
        )
        scorer = CodeQualityScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_keyword_list_detected(self):
        trace = _make_trace(
            'positive_words = ["good", "great"]\nif any(w in text for w in positive_words): return "positive"',
        )
        ctx = ScoringContext(
            task_id="t1",
            input="classify this",
            expected="positive",
            actual="positive",
            trace=trace,
        )
        scorer = CodeQualityScorer()
        result = scorer.score(ctx)
        assert result.score == 0.0


class TestScoreTask:
    @pytest.mark.asyncio
    async def test_single_scorer(self):
        ctx = ScoringContext(
            task_id="t1",
            input="x",
            expected="positive",
            actual="positive",
        )
        scorers = [
            ScorerConfig(name="exact_match", weight=1.0, scorer=ExactMatchScorer()),
        ]
        results = await score_task(ctx, scorers)

        assert "exact_match" in results
        assert results["exact_match"]["score"] == 1.0
        assert results["exact_match"]["weight"] == 1.0

    @pytest.mark.asyncio
    async def test_multiple_scorers(self):
        trace = _make_trace('return "positive"')
        ctx = ScoringContext(
            task_id="t1",
            input="x",
            expected="positive",
            actual="positive",
            trace=trace,
        )
        scorers = [
            ScorerConfig(name="exact_match", weight=0.5, scorer=ExactMatchScorer()),
            ScorerConfig(name="code_quality", weight=0.5, scorer=CodeQualityScorer()),
        ]
        results = await score_task(ctx, scorers)

        assert len(results) == 2
        assert results["exact_match"]["score"] == 1.0
        assert results["code_quality"]["score"] == 1.0


class TestComputeWeightedScore:
    def test_equal_weights(self):
        scores = {
            "a": {"score": 1.0, "weight": 1.0},
            "b": {"score": 0.0, "weight": 1.0},
        }
        assert compute_weighted_score(scores) == 0.5

    def test_unequal_weights(self):
        scores = {
            "a": {"score": 1.0, "weight": 0.8},
            "b": {"score": 0.0, "weight": 0.2},
        }
        assert compute_weighted_score(scores) == 0.8

    def test_empty_scores(self):
        assert compute_weighted_score({}) == 0.0


class TestBuildScoringContext:
    def test_from_execution_result(self, tmp_path):
        trace_file = tmp_path / "trace.jsonl"
        trace_file.write_text("")  # Empty trace

        result = ExecutionResult(
            task_id="t1",
            input="hello",
            expected="world",
            actual="world",
            trace_file=trace_file,
            latency_ms=100.0,
        )

        ctx = build_scoring_context(result)

        assert ctx.task_id == "t1"
        assert ctx.input == "hello"
        assert ctx.expected == "world"
        assert ctx.actual == "world"
        assert ctx.latency_ms == 100.0
        assert ctx.trace is None


class TestValuesEqual:
    """Tests for numeric-aware value comparison."""

    def test_int_equals_float(self):
        assert _values_equal(576, 576.0) is True

    def test_float_equals_int(self):
        assert _values_equal(576.0, 576) is True

    def test_int_not_equals_different_float(self):
        assert _values_equal(576, 576.5) is False

    def test_list_int_float_mixed(self):
        """The original bug: [576, 2, 39] should equal [576.0, 2.0, 39.0]"""
        assert _values_equal([576, 2, 39], [576.0, 2.0, 39.0]) is True

    def test_list_with_mixed_types(self):
        """Full example from the bug report."""
        actual = [19635204.0, 576, 2, 39, 17287.0]
        expected = [19635204.0, 576.0, 2.0, 39.0, 17287.0]
        assert _values_equal(actual, expected) is True

    def test_list_different_length(self):
        assert _values_equal([1, 2, 3], [1, 2]) is False

    def test_nested_list(self):
        assert _values_equal([[1, 2], [3, 4]], [[1.0, 2.0], [3.0, 4.0]]) is True

    def test_dict_int_float(self):
        assert _values_equal({"a": 1, "b": 2}, {"a": 1.0, "b": 2.0}) is True

    def test_dict_different_keys(self):
        assert _values_equal({"a": 1}, {"b": 1}) is False

    def test_string_comparison(self):
        assert _values_equal("hello", "hello") is True
        assert _values_equal("Hello", "hello") is True  # case-insensitive

    def test_string_with_whitespace(self):
        """String comparison is whitespace-trimmed."""
        assert _values_equal("  hello  ", "hello") is True
        assert _values_equal("hello", "  hello  ") is True


class TestParseValue:
    """Tests for value parsing."""

    def test_parse_json_list(self):
        result = _parse_value("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_parse_json_dict(self):
        result = _parse_value('{"a": 1}')
        assert result == {"a": 1}

    def test_parse_number(self):
        assert _parse_value("42") == 42
        assert _parse_value("3.14") == 3.14

    def test_non_string_passthrough(self):
        """Non-string values pass through unchanged."""
        assert _parse_value([1, 2, 3]) == [1, 2, 3]
        assert _parse_value(42) == 42

    def test_unparseable_string(self):
        """Unparseable strings return as-is."""
        assert _parse_value("just a string") == "just a string"


class TestRealExactMatchScorer:
    """Tests for the actual ExactMatchScorer from scoring module."""

    def test_numeric_list_comparison(self):
        """The original bug: int/float mismatch in lists."""
        ctx = ScoringContext(
            task_id="t1",
            input="calculate",
            expected=[19635204.0, 576.0, 2.0, 39.0, 17287.0],
            actual=[19635204.0, 576, 2, 39, 17287.0],
        )
        scorer = RealExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_exact_string_match(self):
        ctx = ScoringContext(
            task_id="t1",
            input="classify",
            expected="positive",
            actual="positive",
        )
        scorer = RealExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_case_insensitive_string_match(self):
        ctx = ScoringContext(
            task_id="t1",
            input="classify",
            expected="Positive",
            actual="positive",
        )
        scorer = RealExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_no_match(self):
        ctx = ScoringContext(
            task_id="t1",
            input="classify",
            expected="positive",
            actual="negative",
        )
        scorer = RealExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 0.0

    def test_metadata_contains_output_correct(self):
        ctx = ScoringContext(
            task_id="t1",
            input="classify",
            expected="positive",
            actual="positive",
        )
        scorer = RealExactMatchScorer()
        result = scorer.score(ctx)
        assert result.metadata["output_correct"] is True

    def test_pydantic_model_comparison(self):
        """Test that Pydantic models are converted to dicts for comparison."""
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str
            age: int
            email: str

        # Create a Pydantic model instance
        actual = UserInfo(name="Alice Johnson", age=28, email="alice.j@example.com")

        # Expected is a dict
        expected = {"name": "Alice Johnson", "age": 28, "email": "alice.j@example.com"}

        ctx = ScoringContext(
            task_id="t1",
            input="extract user info",
            expected=expected,
            actual=actual,
        )
        scorer = RealExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_pydantic_model_no_match(self):
        """Test that Pydantic model comparison correctly identifies mismatches."""
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str
            age: int
            email: str

        # Create a Pydantic model instance
        actual = UserInfo(name="Bob Smith", age=35, email="bob.smith@company.org")

        # Expected is different
        expected = {"name": "Alice Johnson", "age": 28, "email": "alice.j@example.com"}

        ctx = ScoringContext(
            task_id="t1",
            input="extract user info",
            expected=expected,
            actual=actual,
        )
        scorer = RealExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 0.0

    def test_nested_pydantic_model_comparison(self):
        """Test that nested Pydantic models are recursively converted to dicts."""
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str
            age: int
            email: str

        class ReviewInfo(BaseModel):
            product_name: str
            rating: int
            key_points: list[str]

        class CombinedResult(BaseModel):
            user: UserInfo
            review: ReviewInfo
            summary: str

        # Create nested Pydantic model instance
        actual = CombinedResult(
            user=UserInfo(name="Bob Martinez", age=35, email="bob@test.com"),
            review=ReviewInfo(
                product_name="Widget Pro", rating=4, key_points=["Great quality", "Fast delivery"]
            ),
            summary="Bob loves the Widget Pro",
        )

        # Expected is a nested dict
        expected = {
            "user": {"name": "Bob Martinez", "age": 35, "email": "bob@test.com"},
            "review": {
                "product_name": "Widget Pro",
                "rating": 4,
                "key_points": ["Great quality", "Fast delivery"],
            },
            "summary": "Bob loves the Widget Pro",
        }

        ctx = ScoringContext(
            task_id="t1",
            input="process review",
            expected=expected,
            actual=actual,
        )
        scorer = RealExactMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0
        assert result.metadata["raw_type"] == "CombinedResult"
        assert result.metadata["parsed_type"] == "dict"


class TestTypeMatchScorer:
    """Tests for TypeMatchScorer - validates structure/types, not values."""

    def test_simple_dict_match(self):
        """Test matching simple dict structure."""
        expected = {"name": "Alice", "age": 30}
        actual = {"name": "Bob", "age": 25}  # Different values, same structure

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_nested_dict_match(self):
        """Test matching nested dict structure."""
        expected = {
            "user": {"name": "Alice", "age": 30},
            "review": {"rating": 5, "text": "Great!"},
        }
        actual = {
            "user": {"name": "Bob", "age": 25},
            "review": {"rating": 3, "text": "OK"},
        }

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_list_type_match(self):
        """Test matching list element types."""
        expected = {"items": ["apple", "banana"], "count": 2}
        actual = {"items": ["orange", "grape", "mango"], "count": 3}

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_missing_key_fails(self):
        """Test that missing keys fail."""
        expected = {"name": "Alice", "age": 30, "email": "alice@test.com"}
        actual = {"name": "Bob", "age": 25}  # Missing email

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 0.0
        assert "email: missing key" in result.reasoning

    def test_wrong_type_fails(self):
        """Test that wrong types fail."""
        expected = {"name": "Alice", "age": 30}
        actual = {"name": "Bob", "age": "twenty-five"}  # String instead of int

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 0.0
        assert "age: expected int, got str" in result.reasoning

    def test_int_float_interchangeable(self):
        """Test that int and float are treated as compatible."""
        expected = {"value": 42}
        actual = {"value": 42.0}

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_bool_not_int(self):
        """Test that bool is not confused with int."""
        expected = {"flag": True}
        actual = {"flag": 1}  # Int, not bool

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 0.0
        assert "expected bool, got int" in result.reasoning

    def test_pydantic_model_type_match(self):
        """Test TypeMatchScorer with Pydantic models."""
        from pydantic import BaseModel

        class UserInfo(BaseModel):
            name: str
            age: int

        class ReviewInfo(BaseModel):
            rating: int
            text: str

        class CombinedResult(BaseModel):
            user: UserInfo
            review: ReviewInfo

        # Actual is a Pydantic model with different values
        actual = CombinedResult(
            user=UserInfo(name="Bob Martinez", age=35),
            review=ReviewInfo(rating=2, text="Not great"),
        )

        # Expected is a dict with same structure
        expected = {
            "user": {"name": "Alice Johnson", "age": 28},
            "review": {"rating": 5, "text": "Excellent!"},
        }

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0

    def test_extra_keys_allowed(self):
        """Test that actual can have extra keys not in expected."""
        expected = {"name": "Alice"}
        actual = {"name": "Bob", "age": 25, "email": "bob@test.com"}  # Extra keys

        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected=expected,
            actual=actual,
        )
        scorer = TypeMatchScorer()
        result = scorer.score(ctx)
        assert result.score == 1.0  # Extra keys are OK


class TestModeSelectionScorer:
    """Tests for ModeSelectionScorer._is_trivial_code."""

    def test_single_return_literal_is_trivial(self):
        """Single return with literal value should be trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code("return 'negative'")
        assert scorer._is_trivial_code("return 42")
        assert scorer._is_trivial_code("return [1, 2, 3]")
        assert scorer._is_trivial_code("return {'a': 1}")

    def test_bare_expression_is_trivial(self):
        """Bare literal expression should be trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code("'negative'")
        assert scorer._is_trivial_code("[1, 2, 3]")

    def test_assignment_plus_return_literal_is_trivial(self):
        """Assignment + return should be trivial (just packaging the answer, even if variable is unused)."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code("x = 'negative'\nreturn x")
        # Even if variable is unused, it's still trivial (no computation)
        assert scorer._is_trivial_code("x = 'temp'\nreturn 'negative'")

    def test_loop_is_non_trivial(self):
        """Code with loops should be non-trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert not scorer._is_trivial_code("for i in range(10):\n    print(i)\nreturn 'done'")
        assert not scorer._is_trivial_code("while True:\n    break\nreturn 1")

    def test_function_call_is_non_trivial(self):
        """Code with function calls should be non-trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert not scorer._is_trivial_code("result = process(data)\nreturn result")
        assert not scorer._is_trivial_code("return len([1, 2, 3])")

    def test_conditional_is_non_trivial(self):
        """Code with conditionals should be non-trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert not scorer._is_trivial_code("if True:\n    return 'yes'\nelse:\n    return 'no'")
        assert not scorer._is_trivial_code("x = 'a' if True else 'b'\nreturn x")

    def test_complex_expression_is_non_trivial(self):
        """Code with complex expressions should be non-trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert not scorer._is_trivial_code("return x + y")
        assert not scorer._is_trivial_code("result = a * b\nreturn result")

    def test_in_operator_is_non_trivial(self):
        """Code with 'in' operator should be non-trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert not scorer._is_trivial_code("if 'love' in text:\n    return 'positive'")
        assert not scorer._is_trivial_code(
            "result = 'positive' if 'love' in text else 'negative'\nreturn result"
        )
        assert not scorer._is_trivial_code("word in words\nreturn word")

    def test_return_unassigned_variable_is_trivial(self):
        """Returning a variable that wasn't assigned is still trivial (no computation, just a reference)."""
        scorer = ModeSelectionScorer(expected="internal")
        # Returning 'y' which was never assigned - still trivial (no computation, will just fail at runtime)
        assert scorer._is_trivial_code("x = 5\nreturn y")

    def test_empty_code_is_trivial(self):
        """Empty or None code should be trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code(None)
        assert scorer._is_trivial_code("")

    def test_generated_code_prefix_is_stripped(self):
        """Code with '# Generated code:' prefix should be handled."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code("# Generated code:\nreturn 'negative'")

    def test_syntax_error_is_non_trivial(self):
        """Code with syntax errors should be non-trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert not scorer._is_trivial_code("return )")

    def test_score_no_trace_file(self):
        """Should handle missing trace gracefully."""
        scorer = ModeSelectionScorer(expected="internal")
        ctx = ScoringContext(
            task_id="t1",
            input="test",
            expected="expected",
            actual="actual",
            trace=None,
        )
        result = scorer.score(ctx)
        # No trace means score of 0.0 (no code available)
        assert result.score == 0.0
        assert "No trace" in result.reasoning

    def test_print_statements_are_trivial(self):
        """Print statements should be filtered out as trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code("print('debug')\nreturn_result('neutral')")
        assert scorer._is_trivial_code(
            "print(f'Analyzing: {text}')\nsentiment = 'neutral'\nreturn_result(sentiment)"
        )

    def test_comments_are_trivial(self):
        """Comments should be filtered out as trivial (handled by AST parsing)."""
        scorer = ModeSelectionScorer(expected="internal")
        # Comments are removed by AST parsing, so code with only comments is empty
        assert scorer._is_trivial_code("# This is a comment\nreturn_result('neutral')")
        assert scorer._is_trivial_code(
            "# Reasoning here\nsentiment = 'neutral'\nreturn_result(sentiment)"
        )

    def test_mixed_prints_comments_assignment_is_trivial(self):
        """Mixed prints, comments, and literal assignment should be trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        code = """print(f"Analyzing sentiment of: {text}")

# Simple rule-based sentiment analysis
sentiment = 'neutral'

print(f"Determined sentiment: {sentiment}")
return_result(sentiment)
"""
        assert scorer._is_trivial_code(code.strip())

    def test_print_with_computation_is_non_trivial(self):
        """Print statements don't make computation trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert not scorer._is_trivial_code(
            "result = process(data)\nprint(result)\nreturn_result(result)"
        )

    def test_pprint_is_trivial(self):
        """pprint() statements should also be filtered out as trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code("pprint('debug')\nreturn_result('neutral')")
        assert scorer._is_trivial_code(
            "pprint({'key': 'value'})\nsentiment = 'neutral'\nreturn_result(sentiment)"
        )

    def test_multiple_prints_are_trivial(self):
        """Multiple print statements should all be filtered."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code(
            "print('start')\nprint('middle')\nprint('end')\nreturn_result('done')"
        )
        assert scorer._is_trivial_code(
            "print('debug1')\nprint('debug2')\nsentiment = 'neutral'\nprint('debug3')\nreturn_result(sentiment)"
        )

    def test_print_with_f_string_is_trivial(self):
        """F-strings in print statements should be allowed (just formatting)."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code('print(f"Value: {value}")\nreturn_result("done")')
        assert scorer._is_trivial_code(
            'print(f"Analyzing: {text}")\nsentiment = "neutral"\nprint(f"Result: {sentiment}")\nreturn_result(sentiment)'
        )

    def test_variable_to_variable_assignment_is_trivial(self):
        """Variable-to-variable assignments should be trivial (no computation)."""
        scorer = ModeSelectionScorer(expected="internal")
        assert scorer._is_trivial_code(
            "text_content = text\nsentiment = 'positive'\nreturn sentiment"
        )
        assert scorer._is_trivial_code("var1 = input_var\nvar2 = 'value'\nreturn_result(var2)")

    def test_multiple_trivial_assignments_with_return_is_trivial(self):
        """Multiple trivial assignments followed by return should be trivial."""
        scorer = ModeSelectionScorer(expected="internal")
        code = """text_content = text

print(f"Text to classify: {text_content}")

sentiment = "positive"

print(f"Sentiment classification: {sentiment}")

return sentiment
"""
        assert scorer._is_trivial_code(code.strip())
