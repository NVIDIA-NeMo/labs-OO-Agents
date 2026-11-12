# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scope-sensitive coverage for CodeAct's advertised framework helper bindings."""

import textwrap
from unittest.mock import patch

import pytest

from nooa.strategies._completion_guard import completion_binding_errors
from nooa.strategies.codeact_v2 import CodeActV2
from nooa.strategies.generated_code import GeneratedCodeValidator

RESERVED = frozenset(name.removesuffix("()") for name in CodeActV2()._always_available_builtins())


@pytest.mark.parametrize("name", sorted(RESERVED))
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
def test_rejects_writes_to_completion_binding(code, name):
    """Every Python binding form is guarded in the cell or an explicit global scope."""
    errors = completion_binding_errors(code.replace("return_result", name), reserved=RESERVED)
    assert errors
    assert all(f"`{name}` is reserved" in error for error in errors)
    assert all("use a different variable name" in error for error in errors)


@pytest.mark.parametrize("name", sorted(RESERVED))
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
def test_preserves_independent_names_and_scopes(code, name):
    """Attributes, reads, nested locals and comprehension targets remain ordinary Python."""
    assert completion_binding_errors(code.replace("return_result", name), reserved=RESERVED) == []


def test_reports_the_binding_source_line():
    """The model sees the bad target's line instead of a later not-callable failure."""
    [error] = completion_binding_errors(
        "answer = 42\nif ready:\n    return_result = answer", reserved=RESERVED
    )
    assert error.startswith("Line 3:")


def test_syntax_errors_keep_the_existing_execution_diagnostic():
    """Malformed Python is left to the normal cell traceback path."""
    assert completion_binding_errors("return_result = (", reserved=RESERVED) == []


@pytest.mark.parametrize("name", sorted(RESERVED))
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
def test_semantic_syntax_errors_preserve_python_diagnostics(code, name):
    """Compiler-only errors retain their native message and original cell source line."""
    code = code.replace("return_result", name)
    with pytest.raises(SyntaxError) as expected:
        compile("async def cell():\n" + textwrap.indent(code, "    "), "<cell>", "exec")

    with pytest.raises(SyntaxError) as actual:
        completion_binding_errors(code, reserved=RESERVED)

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
def test_deep_cells_with_bindings_are_rejected(deep_code, binding_first):
    """Deep bindings are rejected whether analysis completes or CPython hits its limit."""
    binding = "return_result = 0"
    code = "\n".join([binding, deep_code] if binding_first else [deep_code, binding])

    [error] = completion_binding_errors(code, reserved=RESERVED)
    if "too deeply" in error:
        assert "split" in error
        assert "retry" in error
    else:
        assert "`return_result` is reserved" in error


@pytest.mark.parametrize("mention", ["", "print('ready')\n", "# return_result stays callable\n"])
@pytest.mark.parametrize(
    "deep_code",
    [
        pytest.param("answer = " + "+".join(["1"] * 1500), id="expression"),
        pytest.param("if False:\n    pass\n" + "elif False:\n    pass\n" * 1500, id="elif"),
    ],
)
def test_deep_valid_cells_without_rebinding_remain_valid(deep_code, mention):
    """The binding guard must not reject deep cells that CPython and policy checks accept."""

    class PolicyAgent:
        """Keep the missing-await walk active even for deep code without method calls."""

        async def work(self):
            """Provide a visible async method for the policy validator."""
            pass

    code = mention + deep_code
    compile("async def cell():\n" + textwrap.indent(code, "    "), "<cell>", "exec")
    assert GeneratedCodeValidator().validate(code, PolicyAgent()) == []
    assert completion_binding_errors(code, reserved=RESERVED) == []


@pytest.mark.parametrize("reserved", [RESERVED, frozenset()])
def test_unrelated_source_skips_ast_parsing(reserved):
    """The source pre-check avoids parsing when no advertised helper can be rebound."""
    with patch("nooa.strategies._completion_guard.ast.parse", side_effect=AssertionError):
        assert completion_binding_errors("answer = 42", reserved=reserved) == []


@pytest.mark.parametrize(
    "code,name",
    [
        ("ｐｒｉｎｔ = 1", "print"),
        ("ｄｏｃ = 1", "doc"),
        ("ｒｅｔｕｒｎ＿ｒｅｓｕｌｔ = 1", "return_result"),
    ],
)
def test_source_precheck_preserves_normalized_identifiers(code, name):
    """Unicode identifier normalization cannot bypass the literal-name fast path."""
    [error] = completion_binding_errors(code, reserved=RESERVED)
    assert f"`{name}` is reserved" in error


@pytest.mark.parametrize(
    "global_name,local_name",
    [
        (global_name, local_name)
        for global_name in sorted(RESERVED)
        for local_name in sorted(RESERVED)
        if global_name != local_name
    ],
)
def test_global_declarations_only_protect_the_declared_name(global_name, local_name):
    """A global declaration for one helper must not reserve independent function locals."""
    code = f"def helper():\n    global {global_name}\n    {local_name} = 1"
    assert completion_binding_errors(code, reserved=RESERVED) == []
    [error] = completion_binding_errors(code + f"\n    {global_name} = 1", reserved=RESERVED)
    assert f"`{global_name}` is reserved" in error


def test_multiple_global_helpers_keep_source_order_and_independent_locals():
    """Collect every declared helper while preserving source ordering and ordinary locals."""
    errors = completion_binding_errors(
        "def helper():\n    global doc, print\n    doc = 1\n    print = 2\n    return_result = 3",
        reserved=RESERVED,
    )
    assert len(errors) == 2
    assert errors[0].startswith("Line 3: `doc`")
    assert errors[1].startswith("Line 4: `print`")


def test_reserved_names_are_supplied_by_the_caller():
    """A custom advertised helper is protected without hard-coding it into the visitor."""
    reserved = frozenset({"custom_helper"})
    assert completion_binding_errors("print = 1", reserved=reserved) == []
    [error] = completion_binding_errors("custom_helper = 1", reserved=reserved)
    assert "`custom_helper` is reserved" in error
