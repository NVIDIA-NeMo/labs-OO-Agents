# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native Python differentials for token-preserving owned-cell compilation."""

import ast
import asyncio
import inspect
import linecache
import traceback

import pytest

from nooa.runtime.execution_namespace import ExecutionNamespace

LITERALS = [
    pytest.param('"""first\nsecond\n    third\n"""', id="text"),
    pytest.param("'" * 3 + "first\nsecond\n    third\n" + "'" * 3, id="single-quoted-text"),
    pytest.param('b"""first\nsecond\n    third\n"""', id="bytes"),
    pytest.param('r"""first\\n\nsecond\n    third\n"""', id="raw-text"),
    pytest.param('rb"""first\\n\nsecond\n    third\n"""', id="raw-bytes"),
    pytest.param('f"""first {label}\n{label.upper()}\n    {len(label):03d}\n"""', id="f-string"),
    pytest.param('rf"""first\\n\n{label}\n    \\path\n"""', id="raw-f-string"),
    pytest.param('f"""first\n{(\n    label\n)=}\nlast"""', id="multiline-debug-f-string"),
    pytest.param('"""first\n \t\n\nlast\n"""', id="whitespace-only-lines"),
    pytest.param('"""first\r\n\tsecond\r\n    third\r\n"""', id="crlf-and-tab"),
    pytest.param('"first\\\nsecond\\\n    third"', id="continued-text"),
    pytest.param('b"first\\\nsecond"', id="continued-bytes"),
    pytest.param('r"first\\\nsecond"', id="continued-raw-text"),
    pytest.param('f"first\\\n{label}"', id="continued-f-string"),
    pytest.param('"""\\\nfirst\nsecond\n"""', id="escaped-opening-newline"),
    pytest.param('("first"\n """second\nthird\n"""\n r"\\tail")', id="adjacent-literals"),
    pytest.param('"""first\n\u03bb\n    \u96ea\n"""', id="unicode"),
]


@pytest.mark.parametrize("literal", LITERALS)
@pytest.mark.parametrize("scope", ["cell", "helper", "async-helper", "class", "method"])
async def test_literal_values_match_native_python(literal, scope):
    expected = eval(literal, {"label": "value"})
    if scope == "cell":
        source = "value = " + literal + "\nreturn value"
    elif scope == "helper":
        source = "def helper():\n    return " + literal + "\nreturn helper()"
    elif scope == "async-helper":
        source = "async def helper():\n    return " + literal + "\nreturn await helper()"
    elif scope == "class":
        source = "class Holder:\n    value = " + literal + "\nreturn Holder.value"
    else:
        source = "class Holder:\n    def helper(self):\n        return " + literal
        source += "\nreturn Holder().helper()"
    ns = ExecutionNamespace(reserved=frozenset(), initial={"label": "value"})
    with ns.enter({}):
        actual = await ns.cell(source)
    assert type(actual) is type(expected)
    assert actual == expected
    if scope in {"helper", "async-helper", "method"}:
        helper = ns.globals["Holder"].helper if scope == "method" else ns.globals["helper"]
        # Strategy helpers consume this metadata: its literals must retain the
        # same values too, without the injected runtime decorators/transforms.
        restored = {"label": "value"}
        exec(helper._generated_source, restored)
        restored_value = restored["helper"](None) if scope == "method" else restored["helper"]()
        if scope == "async-helper":
            restored_value = await restored_value
        assert restored_value == expected


async def test_heredoc_command_retains_delimiter_and_payload():
    literal = '"""cat <<\'EOF\'\nfirst\n    second\nEOF\n"""'
    expected = eval(literal)
    ns = ExecutionNamespace(reserved=frozenset())
    with ns.enter({}):
        command = await ns.cell("command = " + literal + "\nreturn command")
    assert command == expected
    process = await asyncio.create_subprocess_exec(
        "bash", "-c", command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5)
    assert process.returncode == 0
    assert stdout == b"first\n    second\n" and stderr == b""


async def test_original_helper_source_locations_and_literal_metadata():
    helper_source = (
        'def helper():\n    """first\nsecond\n    third\n"""\n'
        '    value = """one\ntwo\n"""\n'
        "    return 1 / 0  # original failure\n"
    )
    source = "# before helper\n\n" + helper_source + "return helper"
    filename = "<owned-cell-literal-source>"
    ns = ExecutionNamespace(reserved=frozenset())
    with ns.enter({}):
        helper = await ns.cell(source, filename)
    assert helper.__doc__ == "first\nsecond\n    third\n"
    assert helper.__code__.co_firstlineno == 3
    assert inspect.getsource(helper) == helper_source
    assert "".join(linecache.getlines(filename)) == source
    assert ast.dump(ast.parse(helper._generated_source)) == ast.dump(ast.parse(helper_source))
    with ns.enter({}):
        await ns.cell("unrelated = 'later cell'", "<owned-cell-literal-later>")
        with pytest.raises(ZeroDivisionError) as caught:
            await ns.cell("helper()", "<owned-cell-literal-caller>")
    frame = traceback.extract_tb(caught.value.__traceback__)[-1]
    assert frame.filename == filename and frame.lineno == 11
    assert frame.line == "return 1 / 0  # original failure"
    assert frame.colno == 11 and frame.end_colno == 16


async def test_await_return_and_finally_keep_literal_values_and_global_writes():
    literal = '"""first\nsecond\n    third\n"""'
    expected = eval(literal)
    ns = ExecutionNamespace(reserved=frozenset(), initial={"asyncio": asyncio})
    with ns.enter({}):
        value = await ns.cell(
            "try:\n    return await asyncio.sleep(0, result=" + literal + ")\n"
            "finally:\n    saved = " + literal
        )
        assert await ns.cell("return saved") == expected
    assert value == expected


@pytest.mark.parametrize("source", ["", "\n", "# comment only\n", "  # indented comment\n"])
async def test_empty_cells_still_have_an_async_body(source):
    ns = ExecutionNamespace(reserved=frozenset())
    with ns.enter({}):
        assert await ns.cell(source) is None
