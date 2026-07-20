# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for CurrentCall dataclass.

TDD: Write these tests first, then implement current_call.py to make them pass.
"""

import pytest


class TestCurrentCallBasic:
    """Basic CurrentCall tests."""

    def test_create_with_required_fields(self):
        """CurrentCall should require id, method_name, decorator."""
        from nooa.strategies.current_call import CurrentCall

        call = CurrentCall(
            id="call_123",
            method_name="process",
            decorator="plan",
        )

        assert call.id == "call_123"
        assert call.method_name == "process"
        assert call.decorator == "plan"

    def test_optional_fields_default_to_none_or_empty(self):
        """Optional fields should have sensible defaults."""
        from nooa.strategies.current_call import CurrentCall

        call = CurrentCall(
            id="call_123",
            method_name="process",
            decorator="plan",
        )

        assert call.signature is None
        assert call.docstring is None
        assert call.args == ()
        assert call.kwargs == {}
        assert call.parent_id is None

    def test_create_with_all_fields(self):
        """CurrentCall should accept all fields."""
        from nooa.strategies.current_call import CurrentCall

        call = CurrentCall(
            id="call_123",
            method_name="analyze",
            decorator="plan",
            signature="(self, data: str, limit: int = 10) -> dict",
            docstring="Analyze the data and return results.",
            args=("test data",),
            kwargs={"limit": 5},
            parent_id="parent_call_456",
        )

        assert call.id == "call_123"
        assert call.method_name == "analyze"
        assert call.decorator == "plan"
        assert call.signature == "(self, data: str, limit: int = 10) -> dict"
        assert call.docstring == "Analyze the data and return results."
        assert call.args == ("test data",)
        assert call.kwargs == {"limit": 5}
        assert call.parent_id == "parent_call_456"


class TestCurrentCallFromMethod:
    """Tests for CurrentCall.from_method() factory."""

    def test_from_method_extracts_signature(self):
        """from_method should extract method signature."""
        from nooa.strategies.current_call import CurrentCall

        def example_method(self, data: str, count: int = 10) -> list:
            """Process data and return list."""
            pass

        call = CurrentCall.from_method(
            method=example_method,
            args=("test",),
            kwargs={"count": 5},
        )

        assert call.method_name == "example_method"
        assert call.args == ("test",)
        # After rebase: positional args are merged into kwargs for template expansion
        assert call.kwargs == {"data": "test", "count": 5}
        assert "data: str" in call.signature
        assert "count: int = 10" in call.signature
        assert "-> list" in call.signature

    def test_from_method_extracts_docstring(self):
        """from_method should extract method docstring."""
        from nooa.strategies.current_call import CurrentCall

        def documented_method(self):
            """This is the docstring for the method."""
            pass

        call = CurrentCall.from_method(method=documented_method)

        assert call.docstring == "This is the docstring for the method."

    def test_from_method_handles_no_docstring(self):
        """from_method should handle methods without docstrings."""
        from nooa.strategies.current_call import CurrentCall

        def no_docs(self):
            pass

        call = CurrentCall.from_method(method=no_docs)

        assert call.docstring is None

    def test_from_method_generates_unique_id(self):
        """from_method should generate unique call ID."""
        from nooa.strategies.current_call import CurrentCall

        def test_method(self):
            pass

        call1 = CurrentCall.from_method(method=test_method)
        call2 = CurrentCall.from_method(method=test_method)

        assert call1.id != call2.id
        assert call1.id.startswith("call_")
        assert call2.id.startswith("call_")

    def test_from_method_accepts_decorator_type(self):
        """from_method should accept decorator type."""
        from nooa.strategies.current_call import CurrentCall

        def test_method(self):
            pass

        call = CurrentCall.from_method(method=test_method, decorator="agent")

        assert call.decorator == "agent"

    def test_from_method_accepts_parent_id(self):
        """from_method should accept parent_id for nested calls."""
        from nooa.strategies.current_call import CurrentCall

        def child_method(self):
            pass

        call = CurrentCall.from_method(
            method=child_method,
            parent_id="parent_call_789",
        )

        assert call.parent_id == "parent_call_789"


class TestCurrentCallBoundParameters:
    """Tests for CurrentCall.bound_parameters() — effective inputs, each once.

    from_method() mirrors positional args into kwargs, so concatenating
    args + kwargs.values() double-counts a positional argument. bound_parameters()
    must yield each input exactly once.
    """

    def test_positional_arg_mirrored_into_kwargs_appears_once(self):
        """The issue's core case: a positional arg is in both args and kwargs."""
        from nooa.strategies.current_call import CurrentCall

        def analyze(self, image: str) -> dict:
            """Analyze {image}."""
            pass

        sentinel = object()
        call = CurrentCall.from_method(analyze, args=(sentinel,), kwargs={})

        # from_method mirrors the positional into kwargs (and keeps it in args)
        assert call.args == (sentinel,)
        assert call.kwargs == {"image": sentinel}

        bound = call.bound_parameters()
        assert bound == {"image": sentinel}
        # The de-dup invariant: exactly one value, not two.
        assert list(bound.values()) == [sentinel]

    def test_positional_and_keyword_mix(self):
        """Positional + keyword args map to the right names with no duplicates."""
        from nooa.strategies.current_call import CurrentCall

        def analyze(self, data: str, count: int = 10) -> list:
            """..."""
            pass

        call = CurrentCall.from_method(analyze, args=("hello",), kwargs={"count": 5})
        bound = call.bound_parameters()
        assert bound == {"data": "hello", "count": 5}

    def test_var_positional_args_each_appear_once(self):
        """*args: extra positionals land under arg_<i>; none are duplicated."""
        from nooa.strategies.current_call import CurrentCall

        def analyze(self, *imgs) -> dict:
            """..."""
            pass

        a, b = object(), object()
        call = CurrentCall.from_method(analyze, args=(a, b), kwargs={})

        bound = call.bound_parameters()
        # First positional maps to the var-positional name; the rest get arg_<i>.
        assert list(bound.values()) == [a, b]
        # No object appears twice.
        assert len(bound) == 2

    def test_keyword_only_param(self):
        """Keyword-only param: no stray '*' key leaks into the mapping."""
        from nooa.strategies.current_call import CurrentCall

        def analyze(self, *, image: str) -> dict:
            """..."""
            pass

        sentinel = object()
        call = CurrentCall.from_method(analyze, args=(), kwargs={"image": sentinel})
        bound = call.bound_parameters()
        assert bound == {"image": sentinel}

    def test_positional_only_param(self):
        """Positional-only params ('/') map to their names via param_names."""
        from nooa.strategies.current_call import CurrentCall

        def analyze(self, a, b, /) -> dict:
            """..."""
            pass

        call = CurrentCall.from_method(analyze, args=("x", "y"), kwargs={})
        bound = call.bound_parameters()
        assert bound == {"a": "x", "b": "y"}
        assert "/" not in bound

    def test_no_signature_uses_arg_index_names(self):
        """Without a signature, positionals become arg_<i> and kwargs keep names."""
        from nooa.strategies.current_call import CurrentCall

        call = CurrentCall(
            id="c",
            method_name="m",
            decorator="agent",
            signature=None,
            args=("p0", "p1"),
            kwargs={"k": "v"},
        )
        bound = call.bound_parameters()
        assert bound == {"arg_0": "p0", "arg_1": "p1", "k": "v"}

    def test_signature_with_commas_in_annotations(self):
        """Commas inside type annotations don't corrupt the name→value mapping.

        bound_parameters() relies on param_names captured from inspect (not on
        splitting the signature string on commas), so a parameter annotated with a
        comma-containing generic still maps to the right name with no duplicates.
        """
        from nooa.strategies.current_call import CurrentCall

        def analyze(self, a: dict[str, int], b: int) -> dict:
            """..."""
            pass

        payload = {"x": 1}
        call = CurrentCall.from_method(analyze, args=(payload, 2), kwargs={})
        assert call.bound_parameters() == {"a": payload, "b": 2}

    def test_uses_param_names_not_signature_string(self):
        """bound_parameters relies on the authoritative param_names field, not on
        parsing the (possibly awkward) signature string — so even a signature with
        a '->' inside a default value maps positionals to the right names."""
        from nooa.strategies.current_call import CurrentCall

        call = CurrentCall(
            id="c",
            method_name="m",
            decorator="agent",
            signature="(self, x: str = 'a->b', y: int = 3) -> dict",
            param_names=["x", "y"],
            args=("first", 7),
            kwargs={},
        )
        bound = call.bound_parameters()
        assert bound == {"x": "first", "y": 7}

    def test_kwarg_collision_with_synthetic_key_kwarg_wins(self):
        """A kwarg named arg_<i> overrides the synthetic positional key (documented)."""
        from nooa.strategies.current_call import CurrentCall

        call = CurrentCall(
            id="c",
            method_name="m",
            decorator="agent",
            signature=None,
            args=("positional",),
            kwargs={"arg_0": "keyword"},
        )
        bound = call.bound_parameters()
        assert bound == {"arg_0": "keyword"}


class TestCurrentCallEquality:
    """Tests for CurrentCall equality and hashing."""

    def test_equality_by_id(self):
        """Two CurrentCall with same id should be equal."""
        from nooa.strategies.current_call import CurrentCall

        call1 = CurrentCall(id="same_id", method_name="test", decorator="plan")
        call2 = CurrentCall(id="same_id", method_name="test", decorator="plan")

        assert call1 == call2

    def test_inequality_by_id(self):
        """Two CurrentCall with different id should not be equal."""
        from nooa.strategies.current_call import CurrentCall

        call1 = CurrentCall(id="id_1", method_name="test", decorator="plan")
        call2 = CurrentCall(id="id_2", method_name="test", decorator="plan")

        assert call1 != call2

    def test_hashable(self):
        """CurrentCall should be hashable for use in sets/dicts."""
        from nooa.strategies.current_call import CurrentCall

        call = CurrentCall(id="call_123", method_name="test", decorator="plan")

        # Should not raise
        hash(call)

        # Should be usable in set
        call_set = {call}
        assert call in call_set


class TestCurrentCallImmutability:
    """Tests for CurrentCall immutability."""

    def test_fields_are_frozen(self):
        """CurrentCall should be frozen (immutable)."""
        from nooa.strategies.current_call import CurrentCall

        call = CurrentCall(id="call_123", method_name="test", decorator="plan")

        with pytest.raises(AttributeError):
            call.id = "new_id"

        with pytest.raises(AttributeError):
            call.method_name = "new_method"


def test_from_method_captures_param_names_from_live_signature():
    """from_method captures ordered param names from the live signature.

    This is the authoritative path used by every real agent call: param names
    come from inspect.signature(method), so format_parameters_as_code never parses
    a signature string. Covers Annotated[str, spec(...)] and a comma-in-default —
    the cases that previously produced phantom 'spec(...' params.
    """
    from typing import Annotated

    from nooa.agentdoc import spec
    from nooa.strategies.current_call import CurrentCall

    def method(
        self,
        chunk_text: Annotated[str, spec(max_string=None)],
        chunk_index: int,
        opts: dict = {"a": 1, "b": 2},  # noqa: B006 - exercises comma-in-default
    ) -> dict: ...

    call = CurrentCall.from_method(method, args=("hello", 3))
    assert call.param_names == ["chunk_text", "chunk_index", "opts"]

    rendered = call.format_parameters_as_code()
    assert "chunk_text = 'hello'" in rendered
    assert "chunk_index = 3" in rendered
    assert "spec(max_string" not in rendered
    assert "Annotated" not in rendered


def test_format_parameters_uses_param_names_not_signature_string():
    """With param_names present, the signature string is never re-parsed.

    Build via a real method (the production path), then confirm a deliberately
    wrong signature string on the same call is ignored in favour of param_names.
    """
    import dataclasses

    from nooa.strategies.current_call import CurrentCall

    def method(self, a: int, b: str) -> None: ...

    call = CurrentCall.from_method(method, args=(1, "x"))
    # Corrupt the signature string; param_names must still drive the rendering.
    call = dataclasses.replace(call, signature="(self, GARBAGE not a real sig")
    rendered = call.format_parameters_as_code()
    assert "a = 1" in rendered
    assert "b = 'x'" in rendered
    assert "GARBAGE" not in rendered
