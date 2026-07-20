# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for sandbox validation and execution."""

import json  # noqa: F401 — used by test_sandbox_has_agent_module_imports (visible by default)

import pytest

from nooa.agent import Agent
from nooa.runtime.actor import ActorRuntime
from nooa.runtime.code_validator import ValidationError, validate_code
from nooa.unifiedllm import FakeLLMClient

# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()


def test_validator_allows_safe_code():
    """Test that validator allows safe code."""
    code = """
result = [1, 2, 3]
for i in result:
    print(i)
total = sum(result)
"""
    # Should not raise
    validate_code(code)


def test_validator_forbids_import_when_not_available():
    """Test that validator forbids import of restricted modules."""
    code = "import os"

    with pytest.raises(ValidationError, match="import of 'os' is restricted"):
        validate_code(code, restricted_imports=frozenset({"os"}))


def test_validator_forbids_from_import_when_not_available():
    """Test that validator forbids from...import for restricted modules."""
    code = "from os import path"

    with pytest.raises(ValidationError, match="import of 'os' is restricted"):
        validate_code(code, restricted_imports=frozenset({"os"}))


def test_validator_allows_import_when_available():
    """Test that validator allows imports for modules in available_names."""
    code = "import asyncio"

    # Should not raise when asyncio is in available_names
    validate_code(code, available_names=["asyncio"])


def test_validator_allows_from_import_when_available():
    """Test that validator allows from...import for modules in available_names."""
    code = "from json import dumps"

    # Should not raise when json is in available_names
    validate_code(code, available_names=["json"])


def test_validator_forbids_import_star():
    """Test that validator always forbids 'from X import *' even if X is available."""
    code = "from json import *"

    with pytest.raises(ValidationError, match="from ... import \\*.*forbidden"):
        validate_code(code, available_names=["json"])


def test_validator_allows_nested_module_import():
    """Test that import X.Y.Z is allowed when top-level X is available."""
    code = "import asyncio.tasks"

    # Should not raise when asyncio is available
    validate_code(code, available_names=["asyncio"])


def test_validator_forbids_exec():
    """Test that validator forbids exec()."""
    code = "exec(\"print('hello')\")"

    with pytest.raises(ValidationError, match="exec\\(\\) is forbidden"):
        validate_code(code)


def test_validator_forbids_eval():
    """Test that validator forbids eval()."""
    code = "result = eval('1 + 1')"

    with pytest.raises(ValidationError, match="eval\\(\\) is forbidden"):
        validate_code(code)


def test_validator_forbids_compile():
    """Test that validator forbids compile()."""
    code = "compile('code', 'file', 'exec')"

    with pytest.raises(ValidationError, match="compile\\(\\) is forbidden"):
        validate_code(code)


def test_validator_forbids_aliased_forbidden_builtins():
    """Test that validator catches aliased forbidden builtins.

    When someone does `from builtins import input as get_input`, calling
    get_input() should still be forbidden since it's an alias for input().
    """
    # Test aliased import
    code = """
from builtins import input as get_input
name = get_input("Enter name: ")
"""
    with pytest.raises(ValidationError) as exc_info:
        validate_code(code, importable_modules={"builtins"})

    error_msg = str(exc_info.value)
    assert "get_input()" in error_msg
    assert "alias for input" in error_msg


def test_validator_execution_count_in_cell_name():
    """Test that execution_count parameter controls Cell In[N] name."""
    code = "eval('1+1')"

    # Default execution_count is 1
    with pytest.raises(ValidationError) as exc_info:
        validate_code(code)
    assert "Cell In[1]" in str(exc_info.value)

    # Custom execution_count
    with pytest.raises(ValidationError) as exc_info:
        validate_code(code, execution_count=42)
    assert "Cell In[42]" in str(exc_info.value)


def test_validator_allows_calling_plan_method():
    """Test that validator allows calling self.@strategy methods (runtime handles nesting)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        async def my_plan(self):
            return 42

    code = "await self.my_plan()"

    # Should NOT raise - @strategy methods can now be called from generated code
    validate_code(code, agent_class=TestAgent)


def test_validator_multiple_errors():
    """Test that validator shows IPython-style error for first violation.

    Note: We show the first error encountered in IPython style with source
    context and caret indicator. This matches IPython/Jupyter behavior.
    """
    code = """
import os
eval('1 + 1')
exec('pass')
"""

    with pytest.raises(ValidationError) as exc_info:
        validate_code(code, restricted_imports=frozenset({"os"}))

    error_msg = str(exc_info.value)
    # First error is the restricted import (line 2)
    assert "import" in error_msg and "os" in error_msg
    assert "Cell In[1], line 2" in error_msg
    # IPython-style shows source line and caret
    assert "import os" in error_msg
    assert "^" in error_msg


def test_validator_shows_available_modules_in_error():
    """Test that validator shows restricted module in error message."""
    code = "import numpy"

    with pytest.raises(ValidationError) as exc_info:
        validate_code(code, restricted_imports=frozenset({"numpy"}))

    error_msg = str(exc_info.value)
    assert "numpy" in error_msg
    assert "restricted" in error_msg


@pytest.mark.asyncio
async def test_sandbox_executes_safe_code():
    """Test that ActorRuntime can execute safe code."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.result = 0

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
self.result = 42
"""

    result = await runtime.execute_code(code)
    assert result.error is None
    assert agent_instance.result == 42


@pytest.mark.asyncio
async def test_sandbox_has_safe_builtins():
    """Test that ActorRuntime has access to safe builtins."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.data = []

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
# Test various builtins
numbers = list(range(5))
total = sum(numbers)
self.data = [str(n) for n in numbers]
"""

    result = await runtime.execute_code(code)
    assert result.error is None
    assert agent_instance.data == ["0", "1", "2", "3", "4"]


@pytest.mark.asyncio
async def test_sandbox_has_agent_module_imports():
    """Test that ActorRuntime has access to imports from the agent's module."""
    # The json import at the top of this test file will be available to the agent
    # defined in this file, demonstrating the "whatever was imported" approach

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.result = ""

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
# json is available because it's imported in the test_sandbox module
data = {"key": "value"}
self.result = json.dumps(data)
"""

    result = await runtime.execute_code(code)
    assert result.error is None
    assert agent_instance.result == '{"key": "value"}'


@pytest.mark.asyncio
async def test_sandbox_has_asyncio_gather():
    """Test that ActorRuntime has asyncio.gather for parallel child agents."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.results = []

        async def task(self, n):
            return n * 2

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
# Test asyncio.gather availability
# Note: actual parallel execution would happen via asyncio.gather in real usage
self.results = [2, 4, 6]
"""

    result = await runtime.execute_code(code)
    assert result.error is None
    assert agent_instance.results == [2, 4, 6]


@pytest.mark.asyncio
async def test_sandbox_can_access_agent_context():
    """Test that ActorRuntime can access agent's context dict."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    # Set up agent context (block-based prompt context)
    agent_instance.test_value = "test_data"

    runtime = ActorRuntime(agent_instance)

    code = """
# Access agent attributes
result = self.test_value
"""

    exec_result = await runtime.execute_code(code)
    assert exec_result.error is None
    # Verify agent attributes are accessible
    assert agent_instance.test_value == "test_data"


@pytest.mark.asyncio
async def test_sandbox_logger_available():
    """Test that logger is available in ActorRuntime execution context."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
logger.info("Test message", extra_data="value")
logger.debug("Debug message")
logger.warning("Warning message")
logger.error("Error message")
"""

    # Should not raise - but ActorRuntime doesn't provide logger by default
    # This test documents that logger is NOT available in execute_code context
    result = await runtime.execute_code(code)
    # ActorRuntime doesn't inject logger into exec_globals, so this will error
    assert result.error is not None
    assert "logger" in str(result.error)


# =============================================================================
# Tests for import whitelist validation
# =============================================================================


@pytest.mark.asyncio
async def test_sandbox_allows_import_asyncio():
    """Test that 'import asyncio' is allowed when asyncio is in scope."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.result = None

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    # asyncio is always in exec_globals, so this should work
    code = """
import asyncio
self.result = "success"
"""

    result = await runtime.execute_code(code)
    assert result.error is None
    assert agent_instance.result == "success"


@pytest.mark.asyncio
async def test_sandbox_allows_from_json_import():
    """Test that 'from json import X' works when json module is available."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.result = None

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    # json is imported at the top of this test file, so it's in scope
    code = """
from json import dumps
self.result = dumps({"key": "value"})
"""

    result = await runtime.execute_code(code)
    assert result.error is None
    assert agent_instance.result == '{"key": "value"}'


@pytest.mark.asyncio
async def test_sandbox_forbids_unavailable_imports():
    """Test that imports for unavailable modules still fail."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    # Use a definitely-nonexistent module to test runtime import failure
    code = """
import _nonexistent_module_xyzzy_12345
result = _nonexistent_module_xyzzy_12345.foo()
"""

    result = await runtime.execute_code(code)
    # With deny-list import policy (empty restricted_imports default), the module passes
    # AST validation but fails at runtime since it doesn't exist.
    assert result.error is not None
    assert isinstance(result.error, ModuleNotFoundError)


@pytest.mark.asyncio
async def test_sandbox_forbids_import_star():
    """Test that from X import * is still forbidden even if X is available."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    # json is in scope but import * is still forbidden
    code = """
from json import *
result = dumps({})
"""

    result = await runtime.execute_code(code)
    assert result.error is not None
    assert "from ... import *" in str(result.error)


@pytest.mark.asyncio
async def test_sandbox_allows_import_with_alias():
    """Test that 'import X as Y' works for available modules."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.result = None

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
import asyncio as aio
self.result = "success"
"""

    result = await runtime.execute_code(code)
    assert result.error is None
    assert agent_instance.result == "success"


@pytest.mark.asyncio
async def test_sandbox_allows_submodule_import():
    """Test that 'import X.Y' works when top-level X is available."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.result = None

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
import asyncio.tasks
self.result = "success"
"""

    result = await runtime.execute_code(code)
    assert result.error is None
    assert agent_instance.result == "success"


# =============================================================================
# Tests for REPL-style implicit last expression return
# =============================================================================


@pytest.mark.asyncio
async def test_implicit_return_last_expression():
    """Test that last expression is implicitly returned (REPL style)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.value = 42

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
result = self.value * 2
result
"""

    result = await runtime.execute_code(code, wrap_in_function=True)
    assert result.error is None
    assert result.has_return
    assert result.returned_value == 84


@pytest.mark.asyncio
async def test_implicit_return_function_call():
    """Test that function call as last expression is implicitly returned."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
len([1, 2, 3, 4, 5])
"""

    result = await runtime.execute_code(code, wrap_in_function=True)
    assert result.error is None
    assert result.has_return
    assert result.returned_value == 5


@pytest.mark.asyncio
async def test_no_implicit_return_for_assignment():
    """Test that assignment as last statement does NOT implicitly return."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
x = 42
"""

    result = await runtime.execute_code(code, wrap_in_function=True)
    assert result.error is None
    assert not result.has_return


@pytest.mark.asyncio
async def test_explicit_return_still_works():
    """Test that explicit return statements still work correctly."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
x = 10
y = 20
return x + y
"""

    result = await runtime.execute_code(code, wrap_in_function=True)
    assert result.error is None
    assert result.has_return
    assert result.returned_value == 30


@pytest.mark.asyncio
async def test_implicit_return_print_suppresses_none():
    """Test that print() as last expression does NOT return None (matches IPython)."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
print("hello")
"""

    result = await runtime.execute_code(code, wrap_in_function=True)
    assert result.error is None
    # print() returns None, which is suppressed for implicit returns (like IPython)
    assert not result.has_return
    assert "hello" in result.stdout


@pytest.mark.asyncio
async def test_implicit_return_mixed_code():
    """Test implicit return with mixed statements and expression at end."""

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.items = [1, 2, 3]

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
total = sum(self.items)
doubled = total * 2
self.result = doubled
doubled
"""

    result = await runtime.execute_code(code, wrap_in_function=True)
    assert result.error is None
    assert result.has_return
    assert result.returned_value == 12
    assert agent_instance.result == 12


@pytest.mark.asyncio
async def test_implicit_return_method_attribute():
    """Test implicit return when last statement is an attribute access expression.

    This tests REPL-style behavior where the last expression is implicitly
    returned, enabling patterns like `doc(self.method)` or `self.name` to
    return their values without an explicit return statement.
    """

    class TestAgent(Agent, llm=_TEST_LLM):
        def __init__(self):
            super().__init__()
            self.name = "TestAgent"

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
self.name
"""

    result = await runtime.execute_code(code, wrap_in_function=True)
    assert result.error is None
    assert result.has_return
    assert result.returned_value == "TestAgent"


@pytest.mark.asyncio
async def test_no_implicit_return_without_wrap_in_function():
    """Test that implicit return only works when wrap_in_function=True."""

    class TestAgent(Agent, llm=_TEST_LLM):
        pass

    agent_instance = TestAgent()
    runtime = ActorRuntime(agent_instance)

    code = """
42
"""

    # Without wrap_in_function, no return capture happens
    result = await runtime.execute_code(code, wrap_in_function=False)
    assert result.error is None
    assert not result.has_return


# =============================================================================
# Tests for importable_modules (aliased module imports)
# =============================================================================


def test_validator_importable_modules_allows_actual_name():
    """Test that importable_modules allows imports by actual module name."""
    code = "import pandas"

    # pd is the alias, pandas is the actual module name
    # The validator should allow 'import pandas' when importable_modules contains 'pandas'
    validate_code(
        code,
        available_names=["pd"],  # alias
        importable_modules={"pandas"},  # actual module name
    )


def test_validator_importable_modules_allows_with_new_alias():
    """Test that importable_modules allows import X as Y for actual module name."""
    code = "import pandas as foo"

    # Even with a different alias, if pandas is in importable_modules, allow it
    validate_code(
        code,
        available_names=["pd"],
        importable_modules={"pandas"},
    )


def test_validator_importable_modules_blocks_alias_as_module():
    """Test that import using alias name fails when it's in restricted_imports."""
    code = "import pd"

    # pd is restricted — should be blocked regardless of available_names
    with pytest.raises(ValidationError, match="import of 'pd' is restricted"):
        validate_code(
            code,
            available_names=["pd"],
            restricted_imports=frozenset({"pd"}),
        )


def test_validator_importable_modules_prevents_wrong_module():
    """Test that restricted modules are blocked even if aliased in available_names.

    'import os' should FAIL when os is in restricted_imports,
    regardless of available_names or importable_modules.
    """
    code_os = "import os"
    code_numpy = "import numpy"

    # os is restricted
    with pytest.raises(ValidationError, match="import of 'os' is restricted"):
        validate_code(
            code_os,
            available_names=["os"],
            restricted_imports=frozenset({"os"}),
        )

    # numpy is not restricted — should work
    validate_code(
        code_numpy,
        available_names=["os"],
        restricted_imports=frozenset({"os"}),
    )


def test_validator_importable_modules_from_import():
    """Test that from-imports also use importable_modules."""
    code = "from pandas import DataFrame"

    validate_code(
        code,
        available_names=["pd"],
        importable_modules={"pandas"},
    )


def test_validator_importable_modules_submodule():
    """Test that submodule imports work with importable_modules."""
    code = "import pandas.core"

    # pandas is in importable_modules, so pandas.core should be allowed
    validate_code(
        code,
        available_names=["pd"],
        importable_modules={"pandas"},
    )


def test_validator_backwards_compatible_without_importable_modules():
    """Test that validator works without importable_modules (backwards compat)."""
    code = "import asyncio"

    # When importable_modules is not provided, falls back to available_names
    validate_code(
        code,
        available_names=["asyncio"],
    )


def test_validator_importable_modules_from_import_with_alias():
    """Test that from-imports work with aliased modules.

    If agent has 'import numpy as np', LLM should be able to do 'from numpy import array'.
    """
    code = "from numpy import array"

    validate_code(
        code,
        available_names=["np"],  # alias
        importable_modules={"numpy"},  # actual module name
    )


def test_validator_importable_modules_from_import_blocks_alias():
    """Test that from-imports using a restricted module name fails."""
    code = "from np import array"

    with pytest.raises(ValidationError, match="import of 'np' is restricted"):
        validate_code(
            code,
            available_names=["np"],
            restricted_imports=frozenset({"np"}),
        )


def test_validator_importable_modules_exact_submodule_match():
    """Test that exact submodule names in importable_modules work.

    If importable_modules contains 'pandas.core' specifically,
    'import pandas.core' should work.
    """
    code = "import pandas.core"

    # Only the exact submodule is whitelisted, not the parent
    validate_code(
        code,
        available_names=[],
        importable_modules={"pandas.core"},
    )


def test_validator_importable_modules_exact_submodule_blocks_parent():
    """Test that restricting a parent blocks the parent import.

    If 'pandas' is in restricted_imports, 'import pandas' should fail.
    """
    code = "import pandas"

    with pytest.raises(ValidationError, match="import of 'pandas' is restricted"):
        validate_code(
            code,
            available_names=[],
            restricted_imports=frozenset({"pandas"}),
        )


def test_validator_importable_modules_multiple_modules():
    """Test that multiple modules can be whitelisted."""
    code1 = "import pandas"
    code2 = "import numpy"
    code3 = "from json import dumps"

    importable = {"pandas", "numpy", "json"}

    validate_code(code1, available_names=["pd", "np"], importable_modules=importable)
    validate_code(code2, available_names=["pd", "np"], importable_modules=importable)
    validate_code(code3, available_names=["pd", "np"], importable_modules=importable)
