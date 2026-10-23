# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope-sensitive coverage for CodeAct's reserved completion binding."""

import textwrap

import pytest

from nooa.strategies._completion_guard import completion_binding_errors


@pytest.mark.parametrize(
    "code",
    [
        "return_result = 42",
        "await helper()\nreturn_result = 42",
        "return_result = 42\nreturn 42",
        "return_result = answer = 42",
        "return_result: int",
        "return_result: int = 42",
        "return_result += 1",
        "(a, (b, *return_result)) = (1, (2, 3))",
        "[return_result] = [42]",
        "del return_result",
        "del (answer, return_result)",
        "if False:\n    return_result = 42",
        "while False:\n    return_result = 42",
        "for return_result in []:\n    pass",
        "async for return_result in stream:\n    pass",
        "with manager as return_result:\n    pass",
        "async with manager as return_result:\n    pass",
        "try:\n    pass\nexcept Exception as return_result:\n    pass",
        "try:\n    pass\nexcept* Exception as return_result:\n    pass",
        "match value:\n    case return_result:\n        pass",
        "match value:\n    case [*return_result]:\n        pass",
        "match value:\n    case {'x': x, **return_result}:\n        pass",
        "match value:\n    case [x] as return_result:\n        pass",
        "import math as return_result",
        "from math import sqrt as return_result",
        "import return_result.submodule",
        "from package import return_result",
        "def return_result():\n    pass",
        "async def return_result():\n    pass",
        "class return_result:\n    pass",
        "type return_result = int",
        "(return_result := 42)",
        "[(return_result := value) for value in values]",
        "{(return_result := value) for value in values}",
        "{key: (return_result := value) for key, value in values}",
        "{(return_result := key): value for key, value in values}",
        "((return_result := value) for value in values)",
        "[value for value in values if (return_result := value)]",
        "[[(return_result := value) for value in values] for _ in values]",
        "def helper(value=(return_result := 42)):\n    pass",
        "async def helper(*, value=(return_result := 42)):\n    pass",
        "def helper(value: (return_result := int)):\n    pass",
        "def helper() -> (return_result := int):\n    pass",
        "@((return_result := decorator))\ndef helper():\n    pass",
        "class Helper((return_result := Base)):\n    pass",
        "class Helper(metaclass=(return_result := Meta)):\n    pass",
        "value = lambda x=(return_result := 42): x",
        "def helper():\n    global return_result\n    return_result = 42",
        "def helper():\n    if False:\n        global return_result\n    return_result = 42",
        "def helper():\n    global return_result\n    import math as return_result",
        "def helper():\n    global return_result\n    del return_result",
        "class Helper:\n    global return_result\n    return_result = 42",
        "def helper():\n    def nested():\n        global return_result\n        return_result = 42",
        "def helper():\n    global return_result\n    [(return_result := value) for value in values]",
    ],
)
def test_rejects_writes_to_completion_binding(code):
    """Every Python binding form is guarded in the cell or an explicit global scope."""
    errors = completion_binding_errors(code)
    assert errors
    assert all("`return_result` is reserved" in error for error in errors)
    assert all("return_result(result)" in error for error in errors)


@pytest.mark.parametrize(
    "code",
    [
        "",
        "answer = 42",
        "return_result(answer)",
        "submit = return_result",
        "self.return_result = 42",
        "values['return_result'] = 42",
        "import math as result",
        "class Helper:\n    return_result = 42",
        "class Helper:\n    def return_result(self):\n        pass",
        "def helper(return_result):\n    return return_result",
        "async def helper(return_result):\n    return return_result",
        "def helper():\n    return_result = 42\n    return return_result",
        "def helper():\n    global return_result\n    return_result(42)",
        "def helper():\n    global return_result\n    def nested():\n        return_result = 42",
        "def helper():\n    return_result = 42\n    def nested():\n        nonlocal return_result\n        return_result += 1",
        "def helper():\n    return_result = 42\n    def nested():\n        global return_result\n        return_result(42)",
        "answer = lambda return_result: return_result",
        "answer = lambda: (return_result := 42)",
        "[return_result for return_result in values]",
        "{return_result for return_result in values}",
        "{return_result: return_result for return_result in values}",
        "(return_result for return_result in values)",
        "[return_result for return_result in return_result(values)]",
        "[return_result for return_result in values if return_result]",
        "[return_result for x in values for return_result in x]",
        "[(lambda return_result: return_result)(x) for x in values]",
        "[(lambda: (return_result := x))() for x in values]",
        "def helper():\n    [(return_result := x) for x in values]",
        "type Helper[return_result] = list[return_result]",
        "def helper[return_result](value: return_result):\n    return value",
    ],
)
def test_preserves_independent_names_and_scopes(code):
    """Attributes, reads, nested locals and comprehension targets remain ordinary Python."""
    assert completion_binding_errors(code) == []


def test_reports_the_binding_source_line():
    """The model sees the bad target's line instead of a later not-callable failure."""
    [error] = completion_binding_errors("answer = 42\nif ready:\n    return_result = answer")
    assert error.startswith("Line 3:")


def test_syntax_errors_keep_the_existing_execution_diagnostic():
    """Malformed Python is left to the normal cell traceback path."""
    assert completion_binding_errors("return_result = (") == []


@pytest.mark.parametrize(
    "code",
    [
        "return_result = 1\nglobal return_result",
        "return_result(1)\nglobal return_result\nreturn_result = 1",
        "return_result: int = 1\nglobal return_result",
        "def helper(return_result):\n    global return_result\n    return_result = 1",
        "def return_result(value):\n    return value\nglobal return_result",
        "return_result = 1\nnonlocal return_result",
        "await helper()\nreturn_result = 1\nglobal return_result",
        "return_result = 1\nreturn 42\nglobal return_result",
    ],
)
def test_semantic_syntax_errors_preserve_python_diagnostics(code):
    """Compiler-only errors retain their native message and original cell source line."""
    with pytest.raises(SyntaxError) as expected:
        compile("async def cell():\n" + textwrap.indent(code, "    "), "<cell>", "exec")

    with pytest.raises(SyntaxError) as actual:
        completion_binding_errors(code)

    assert actual.value.msg == expected.value.msg
    assert actual.value.lineno == expected.value.lineno - 1


@pytest.mark.parametrize("binding_first", [False, True])
@pytest.mark.parametrize(
    "deep_code",
    [
        pytest.param("answer = " + "+".join(["1"] * 1500), id="expression-traversal"),
        pytest.param("answer = " + "+".join(["1"] * 10000), id="parser"),
        pytest.param(
            "if False:\n    pass\n" + "elif False:\n    pass\n" * 1500,
            id="elif-traversal",
        ),
        pytest.param(
            "def helper():\n    answer = " + "+".join(["1"] * 1500),
            id="scope-discovery",
        ),
    ],
)
def test_recursion_errors_reject_the_whole_cell(deep_code, binding_first):
    """Incomplete parsing or traversal must not allow unchecked or partially checked code."""
    binding = "return_result = 0"
    code = "\n".join([binding, deep_code] if binding_first else [deep_code, binding])

    [error] = completion_binding_errors(code)

    assert "too deeply" in error
    assert "return_result" in error
    assert "split" in error
    assert "retry" in error
