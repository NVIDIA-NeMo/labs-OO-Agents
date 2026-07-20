# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the unified code validator.

This module tests the UnifiedCodeValidator and its component validators:
- SecurityValidator: Prevents security risks
- BlockingCallValidator: Prevents blocking calls that freeze the event loop
- REPLPolicyValidator: Enforces REPL-style coding conventions

Each test class has two sections:
1. Patterns to REJECT (should raise ValidationError)
2. Patterns to ALLOW (should NOT raise)
"""

import pytest

from nooa.runtime.code_validator import (
    BlockingCallValidator,
    REPLPolicyValidator,
    SecurityValidator,
    UnifiedCodeValidator,
    ValidationContext,
    ValidationError,
    strip_redundant_imports,
)
from nooa.unifiedllm import FakeLLMClient

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def validator() -> UnifiedCodeValidator:
    """Create a unified validator with security + async validators (default for execute_code)."""
    return UnifiedCodeValidator()


@pytest.fixture
def full_validator() -> UnifiedCodeValidator:
    """Create a unified validator with ALL validators including REPL policy."""
    return UnifiedCodeValidator(include_repl_policy=True)


@pytest.fixture
def security_validator() -> SecurityValidator:
    """Create a security validator for isolated testing."""
    return SecurityValidator()


@pytest.fixture
def blocking_call_validator() -> BlockingCallValidator:
    """Create a blocking call validator for isolated testing."""
    return BlockingCallValidator()


@pytest.fixture
def repl_validator() -> REPLPolicyValidator:
    """Create a REPL policy validator for isolated testing."""
    return REPLPolicyValidator()


@pytest.fixture
def default_context() -> ValidationContext:
    """Create a default validation context with deny-list import restrictions."""
    from nooa.runtime.restrictions import DEFAULT_BLOCKED_MODULES

    # Explicit deny list for testing — DEFAULT_RESTRICTED_IMPORTS is empty by design
    # (no restrictions by default), but tests need specific modules restricted.
    return ValidationContext(
        code="",
        restricted_imports=frozenset({"os", "shutil", "pathlib", "sys", "ctypes", "importlib"}),
        blocked_modules=DEFAULT_BLOCKED_MODULES,
    )


@pytest.fixture
def agent_context():
    """Create an agent and validation context for ClassAssignmentValidator tests.

    Returns a tuple of (context, TestAgent_class) for tests that need to
    reference the class name in generated code.
    """
    from nooa.agent import Agent

    class TestAgent(Agent, llm=FakeLLMClient()):
        async def process(self) -> dict:
            """Process something."""
            ...

    agent = TestAgent()
    context = ValidationContext(code="", agent=agent)
    return context, TestAgent


# =============================================================================
# SecurityValidator Tests
# =============================================================================


class TestSecurityValidator:
    """Tests for security-related validations."""

    # -------------------------------------------------------------------------
    # Patterns to REJECT
    # -------------------------------------------------------------------------

    class TestPatternsToReject:
        """Patterns that MUST be rejected for security reasons."""

        def test_reject_exec(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """exec() allows arbitrary code execution."""
            code = "exec('print(1)')"
            with pytest.raises(ValidationError, match="exec.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_eval(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """eval() allows arbitrary expression evaluation."""
            code = "result = eval('1 + 1')"
            with pytest.raises(ValidationError, match="eval.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_compile(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """compile() can be used to create executable code objects."""
            code = "code = compile('x = 1', '<string>', 'exec')"
            with pytest.raises(ValidationError, match="compile.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_dunder_import(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """__import__ allows dynamic imports bypassing restrictions."""
            code = "os = __import__('os')"
            with pytest.raises(ValidationError, match="__import__.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_input(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """input() blocks waiting for stdin, hanging the event loop."""
            code = "name = input('Enter name: ')"
            with pytest.raises(ValidationError, match="input.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_breakpoint(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """breakpoint() opens debugger, blocking on stdin."""
            code = "breakpoint()"
            with pytest.raises(ValidationError, match="breakpoint.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_import_unavailable_module(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Import of modules not in importable_modules must be rejected."""
            code = "import os"
            with pytest.raises(ValidationError, match="import.*'os'.*restricted"):
                validator.validate(code, default_context)

        def test_reject_from_import_unavailable_module(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """From-import of restricted modules must be rejected."""
            code = "from os import path"
            with pytest.raises(ValidationError, match="import.*'os'.*restricted"):
                validator.validate(code, default_context)

        def test_reject_import_star(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """'from X import *' is always forbidden, even for available modules."""
            code = "from json import *"
            with pytest.raises(ValidationError, match="import \\*.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_aliased_forbidden_builtin(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Aliased forbidden builtins must still be caught."""
            # When builtins is in importable_modules, track the alias
            context = ValidationContext(
                code="",
            )
            code = """
from builtins import input as get_input
name = get_input("Enter: ")
"""
            with pytest.raises(ValidationError, match="get_input.*alias.*input"):
                validator.validate(code, context)

        def test_reject_class_bases_jailbreak(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """__class__.__bases__ is a common sandbox escape technique."""
            code = "().__class__.__bases__[0].__subclasses__()"
            with pytest.raises(ValidationError, match="__class__|__bases__|__subclasses__"):
                validator.validate(code, default_context)

        def test_reject_globals_call(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """globals() gives access to the global namespace."""
            code = "x = globals()['__builtins__']"
            with pytest.raises(ValidationError, match="globals.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_locals_call(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """locals() gives access to the local namespace."""
            code = "x = locals()"
            with pytest.raises(ValidationError, match="locals.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_dunder_globals_attr(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """func.__globals__ exposes the function's global namespace."""
            code = "x = some_func.__globals__"
            with pytest.raises(ValidationError, match="__globals__.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_dunder_code_attr(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """func.__code__ allows code object manipulation."""
            code = "x = some_func.__code__"
            with pytest.raises(ValidationError, match="__code__.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_dunder_builtins_attr(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """__builtins__ gives access to built-in functions."""
            code = "x = __builtins__"
            with pytest.raises(ValidationError, match="__builtins__.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_dunder_dict_attr(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """obj.__dict__ exposes the raw attribute dictionary, bypassing visibility controls."""
            code = "x = self.__dict__"
            with pytest.raises(ValidationError, match="__dict__.*forbidden"):
                validator.validate(code, default_context)

        def test_reject_recursive_self_call(self, validator: UnifiedCodeValidator):
            """Calling the method being generated causes infinite recursion."""
            context = ValidationContext(
                code="",
                forbidden_self_calls={"my_method"},
            )
            code = "result = await self.my_method()"
            with pytest.raises(ValidationError, match="self\\.my_method.*forbidden.*recursion"):
                validator.validate(code, context)

    # -------------------------------------------------------------------------
    # Patterns to ALLOW
    # -------------------------------------------------------------------------

    class TestPatternsToAllow:
        """Safe patterns that MUST NOT be rejected."""

        def test_allow_basic_arithmetic(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Basic arithmetic operations are safe."""
            code = """
x = 1 + 2
y = x * 3
z = y / 2
"""
            validator.validate(code, default_context)  # Should not raise

        def test_allow_list_operations(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """List operations are safe."""
            code = """
items = [1, 2, 3]
items.append(4)
total = sum(items)
"""
            validator.validate(code, default_context)

        def test_allow_dict_operations(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Dict operations are safe."""
            code = """
data = {"key": "value"}
data["new_key"] = 123
keys = list(data.keys())
"""
            validator.validate(code, default_context)

        def test_allow_string_operations(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """String operations are safe."""
            code = """
text = "hello world"
upper = text.upper()
parts = text.split()
"""
            validator.validate(code, default_context)

        def test_allow_import_available_module(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Import of modules in importable_modules is allowed."""
            code = "import asyncio"
            validator.validate(code, default_context)

        def test_allow_from_import_available_module(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """From-import of modules in importable_modules is allowed."""
            code = "from json import dumps"
            validator.validate(code, default_context)

        def test_allow_import_with_alias(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Import with alias for available module is allowed."""
            code = "import asyncio as aio"
            validator.validate(code, default_context)

        def test_allow_submodule_import(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Submodule import when parent is available is allowed."""
            code = "import asyncio.tasks"
            validator.validate(code, default_context)

        def test_allow_for_loops(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """For loops are safe."""
            code = """
for i in range(10):
    print(i)
"""
            validator.validate(code, default_context)

        def test_allow_while_loops_with_break(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """While loops with break conditions are safe."""
            code = """
count = 0
while count < 10:
    count += 1
"""
            validator.validate(code, default_context)

        def test_allow_function_definitions(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Helper function definitions are allowed."""
            code = """
def helper(x):
    return x * 2

result = helper(5)
"""
            validator.validate(code, default_context)

        def test_allow_async_function_definitions(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Async helper function definitions are allowed."""
            code = """
async def async_helper(x):
    return x * 2
"""
            validator.validate(code, default_context)

        def test_allow_lambda(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Lambda expressions are allowed."""
            code = "double = lambda x: x * 2"
            validator.validate(code, default_context)

        def test_allow_list_comprehension(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """List comprehensions are allowed."""
            code = "squares = [x**2 for x in range(10)]"
            validator.validate(code, default_context)

        def test_allow_dict_comprehension(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Dict comprehensions are allowed."""
            code = "squares = {x: x**2 for x in range(10)}"
            validator.validate(code, default_context)

        def test_allow_try_except(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Try-except blocks are allowed."""
            code = """
try:
    result = 1 / 0
except ZeroDivisionError:
    result = 0
"""
            validator.validate(code, default_context)

        def test_allow_with_statement(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """With statements are allowed (for context managers)."""
            # Note: We don't define a class here since class definitions are forbidden
            # in REPL-style code. Just test that with statement syntax passes validation.
            code = """
# With statement using existing context manager
with open_context() as ctx:
    process(ctx)
"""
            validator.validate(code, default_context)

        def test_allow_self_attribute_access(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Accessing self attributes is allowed."""
            code = """
x = self.data
self.result = x * 2
"""
            validator.validate(code, default_context)

        def test_allow_calling_other_self_methods(self, validator: UnifiedCodeValidator):
            """Calling other self methods (not the current one) is allowed."""
            context = ValidationContext(
                code="",
                forbidden_self_calls={"my_method"},  # Only this is forbidden
            )
            code = "result = await self.other_method()"  # This is fine
            validator.validate(code, context)

        def test_allow_regular_dunder_methods(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Regular dunder methods like __str__ are allowed."""
            code = """
text = str(obj)
length = len(items)
"""
            validator.validate(code, default_context)

        def test_allow_isinstance_issubclass(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """isinstance and issubclass are safe type checks."""
            code = """
is_list = isinstance(x, list)
is_sub = issubclass(MyClass, BaseClass)
"""
            validator.validate(code, default_context)

        def test_allow_getattr_without_dunder(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """getattr() for non-dunder attributes is allowed."""
            code = "value = getattr(obj, 'name', None)"
            validator.validate(code, default_context)

        def test_allow_hasattr(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """hasattr() is safe for attribute checking."""
            code = "has_name = hasattr(obj, 'name')"
            validator.validate(code, default_context)

        def test_allow_type_call(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """type() for type checking is allowed."""
            code = "t = type(obj)"
            validator.validate(code, default_context)

        def test_allow_dir_call(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """dir() is allowed (useful for debugging)."""
            code = "attrs = dir(obj)"
            validator.validate(code, default_context)


class TestSecurityValidatorWarnings:
    """Tests for security patterns that generate warnings (not errors)."""

    def test_warn_open_call(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """open() should warn - agents should use file tools instead."""
        code = "f = open('file.txt', 'r')"
        # Should pass validation but log a warning
        # For now, we don't have warning capture in tests, so just verify it doesn't error
        validator.validate(code, default_context)


class TestSecurityValidatorAdditionalRejects:
    """Additional security patterns that must be rejected."""

    def test_reject_vars_call(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """vars() gives access to object's __dict__, similar to locals()."""
        code = "x = vars()"
        with pytest.raises(ValidationError, match="vars.*forbidden"):
            validator.validate(code, default_context)

    def test_reject_exit_call(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """exit() would terminate the process."""
        code = "exit(0)"
        with pytest.raises(ValidationError, match="exit.*forbidden"):
            validator.validate(code, default_context)

    def test_reject_quit_call(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """quit() would terminate the process."""
        code = "quit()"
        with pytest.raises(ValidationError, match="quit.*forbidden"):
            validator.validate(code, default_context)

    def test_reject_setattr_with_dunder(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """setattr() with dunder names is dangerous."""
        code = "setattr(obj, '__class__', Evil)"
        with pytest.raises(ValidationError, match="setattr.*dunder|__class__.*forbidden"):
            validator.validate(code, default_context)

    def test_reject_delattr_with_dunder(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """delattr() with dunder names is dangerous."""
        code = "delattr(obj, '__dict__')"
        with pytest.raises(ValidationError, match="delattr.*dunder|__dict__.*forbidden"):
            validator.validate(code, default_context)

    def test_reject_getattr_with_dunder(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """getattr() with dunder names bypasses direct attribute access checks."""
        code = "d = getattr(obj, '__dict__')"
        with pytest.raises(ValidationError, match="getattr.*dunder|__dict__.*forbidden"):
            validator.validate(code, default_context)

    def test_allow_setattr_with_normal_name(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """setattr() with normal attribute names is allowed."""
        code = "setattr(obj, 'name', 'value')"
        validator.validate(code, default_context)

    def test_allow_delattr_with_normal_name(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """delattr() with normal attribute names is allowed."""
        code = "delattr(obj, 'temp_attr')"
        validator.validate(code, default_context)


class TestSecurityValidatorProcessTermination:
    """E005 — process termination escapes (raise SystemExit, sys.exit(), etc.).

    AST detection is a fast-fail for the literal forms; indirect forms
    (aliasing the callable, raising a name bound elsewhere) are deliberately not
    caught here — the runtime converts any SystemExit/KeyboardInterrupt that
    reaches it into an ordinary execution error.
    """

    def test_reject_raise_systemexit(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        with pytest.raises(ValidationError, match="SystemExit.*forbidden"):
            validator.validate("raise SystemExit", default_context)

    def test_reject_raise_systemexit_with_code(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        with pytest.raises(ValidationError, match="SystemExit.*forbidden"):
            validator.validate("raise SystemExit(3)", default_context)

    def test_reject_raise_keyboardinterrupt(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        with pytest.raises(ValidationError, match="KeyboardInterrupt.*forbidden"):
            validator.validate("raise KeyboardInterrupt", default_context)

    def test_reject_sys_exit(self, security_validator: SecurityValidator):
        import ast

        issues = security_validator.validate(ast.parse("sys.exit()"), ValidationContext())
        assert any(i.code == "E005" and "sys.exit" in i.message for i in issues)

    def test_reject_os_underscore_exit(self, security_validator: SecurityValidator):
        import ast

        issues = security_validator.validate(ast.parse("os._exit(1)"), ValidationContext())
        assert any(i.code == "E005" and "os._exit" in i.message for i in issues)

    def test_reject_os_abort(self, security_validator: SecurityValidator):
        import ast

        issues = security_validator.validate(ast.parse("os.abort()"), ValidationContext())
        assert any(i.code == "E005" and "os.abort" in i.message for i in issues)

    def test_reject_sys_exit_module_alias(self, security_validator: SecurityValidator):
        import ast

        issues = security_validator.validate(
            ast.parse("import sys as s\ns.exit()"), ValidationContext()
        )
        assert any(i.code == "E005" and "alias for sys.exit" in i.message for i in issues)

    def test_reject_os_underscore_exit_module_alias(self, security_validator: SecurityValidator):
        import ast

        issues = security_validator.validate(
            ast.parse("import os as o\no._exit(1)"), ValidationContext()
        )
        assert any(i.code == "E005" and "alias for os._exit" in i.message for i in issues)

    def test_reject_os_abort_module_alias(self, security_validator: SecurityValidator):
        import ast

        issues = security_validator.validate(
            ast.parse("import os as o\no.abort()"), ValidationContext()
        )
        assert any(i.code == "E005" and "alias for os.abort" in i.message for i in issues)

    def test_reject_from_sys_import_exit_alias(self, security_validator: SecurityValidator):
        import ast

        issues = security_validator.validate(
            ast.parse("from sys import exit as bye\nbye()"), ValidationContext()
        )
        assert any(i.code == "E005" and "alias for sys.exit" in i.message for i in issues)

    def test_reject_from_os_import_underscore_exit_alias(
        self, security_validator: SecurityValidator
    ):
        import ast

        issues = security_validator.validate(
            ast.parse("from os import _exit as die\ndie(1)"), ValidationContext()
        )
        assert any(i.code == "E005" and "alias for os._exit" in i.message for i in issues)

    def test_reject_from_os_import_abort_alias(self, security_validator: SecurityValidator):
        import ast

        issues = security_validator.validate(
            ast.parse("from os import abort as die\ndie()"), ValidationContext()
        )
        assert any(i.code == "E005" and "alias for os.abort" in i.message for i in issues)

    def test_allow_raise_ordinary_exception(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Ordinary exceptions are fine — only process-termination is blocked."""
        validator.validate("raise ValueError('nope')", default_context)

    def test_allow_raise_cancellederror(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """asyncio.CancelledError must NOT be blocked — cancellation is legitimate."""
        validator.validate("import asyncio\nraise asyncio.CancelledError", default_context)


# =============================================================================
# BlockingCallValidator Tests
# =============================================================================


class TestBlockingCallValidator:
    """Tests for blocking call validations (replaces TestAsyncSafetyValidator).

    The BlockingCallValidator resolves names against exec_globals to determine
    module of origin. Tests provide the relevant modules in exec_globals.
    """

    # -------------------------------------------------------------------------
    # Patterns to REJECT
    # -------------------------------------------------------------------------

    class TestPatternsToReject:
        """Blocking patterns that freeze the event loop and must be rejected."""

        def test_reject_asyncio_run(self, validator: UnifiedCodeValidator):
            """asyncio.run() can't be used inside an already-running event loop."""
            import asyncio

            ctx = ValidationContext(exec_globals={"asyncio": asyncio})
            code = "result = asyncio.run(fetch_data())"
            with pytest.raises(ValidationError, match="asyncio.*run.*blocks"):
                validator.validate(code, ctx)

        def test_reject_asyncio_run_with_alias(self, validator: UnifiedCodeValidator):
            """asyncio.run() must be detected even with exec_globals alias."""
            import asyncio

            ctx = ValidationContext(exec_globals={"aio": asyncio})
            code = "result = aio.run(fetch_data())"
            with pytest.raises(ValidationError, match="asyncio.*run.*blocks"):
                validator.validate(code, ctx)

        def test_reject_from_import_run(self, validator: UnifiedCodeValidator):
            """asyncio.run() must be detected when the function is directly in exec_globals."""
            import asyncio

            ctx = ValidationContext(exec_globals={"run": asyncio.run})
            code = "result = run(fetch_data())"
            with pytest.raises(ValidationError, match="asyncio.*run.*blocks"):
                validator.validate(code, ctx)

        def test_reject_run_until_complete(self, validator: UnifiedCodeValidator):
            """run_until_complete() blocks the event loop."""
            import asyncio

            ctx = ValidationContext(exec_globals={"asyncio": asyncio})
            code = """
loop = asyncio.get_event_loop()
result = loop.run_until_complete(coro())
"""
            with pytest.raises(ValidationError, match="run_until_complete.*blocks"):
                validator.validate(code, ctx)

        def test_reject_run_until_complete_chained(self, validator: UnifiedCodeValidator):
            """Chained run_until_complete() must be detected."""
            import asyncio

            ctx = ValidationContext(exec_globals={"asyncio": asyncio})
            code = "asyncio.get_event_loop().run_until_complete(coro())"
            with pytest.raises(ValidationError, match="run_until_complete.*blocks"):
                validator.validate(code, ctx)

        def test_reject_run_forever(self, validator: UnifiedCodeValidator):
            """run_forever() blocks the event loop forever."""
            import asyncio

            ctx = ValidationContext(exec_globals={"asyncio": asyncio})
            code = """
loop = asyncio.get_event_loop()
loop.run_forever()
"""
            with pytest.raises(ValidationError, match="run_forever.*blocks"):
                validator.validate(code, ctx)

        def test_reject_run_coroutine_threadsafe(self, validator: UnifiedCodeValidator):
            """run_coroutine_threadsafe() blocks in async context."""
            import asyncio

            ctx = ValidationContext(exec_globals={"asyncio": asyncio})
            code = """
loop = asyncio.get_running_loop()
future = asyncio.run_coroutine_threadsafe(coro(), loop)
"""
            with pytest.raises(ValidationError, match="run_coroutine_threadsafe.*blocks"):
                validator.validate(code, ctx)

        def test_reject_time_sleep(self, validator: UnifiedCodeValidator):
            """time.sleep() blocks the event loop."""
            import time

            ctx = ValidationContext(exec_globals={"time": time})
            code = "time.sleep(5)"
            with pytest.raises(ValidationError, match="sleep.*blocks"):
                validator.validate(code, ctx)

        def test_reject_thread_join(self, validator: UnifiedCodeValidator):
            """Thread.join() blocks in async context."""
            import threading

            ctx = ValidationContext(exec_globals={"threading": threading})
            code = """
t = threading.Thread(target=work)
t.start()
t.join()
"""
            with pytest.raises(ValidationError, match="join.*blocks"):
                validator.validate(code, ctx)

        def test_reject_process_join(self, validator: UnifiedCodeValidator):
            """Process.join() blocks in async context."""
            import multiprocessing

            ctx = ValidationContext(exec_globals={"multiprocessing": multiprocessing})
            code = """
p = multiprocessing.Process(target=work)
p.start()
p.join()
"""
            with pytest.raises(ValidationError, match="join.*blocks"):
                validator.validate(code, ctx)

    # -------------------------------------------------------------------------
    # Patterns to ALLOW
    # -------------------------------------------------------------------------

    class TestPatternsToAllow:
        """Patterns that are safe and must be allowed."""

        def test_allow_await_coroutine(self, validator: UnifiedCodeValidator):
            """await on a coroutine is the correct async pattern."""
            ctx = ValidationContext(exec_globals={"asyncio": __import__("asyncio")})
            code = "result = await fetch_data()"
            validator.validate(code, ctx)

        def test_allow_asyncio_gather(self, validator: UnifiedCodeValidator):
            """asyncio.gather() is the correct way to run parallel tasks."""
            ctx = ValidationContext(exec_globals={"asyncio": __import__("asyncio")})
            code = """
results = await asyncio.gather(
    task1(),
    task2(),
    task3(),
)
"""
            validator.validate(code, ctx)

        def test_allow_asyncio_create_task(self, validator: UnifiedCodeValidator):
            """asyncio.create_task() is correct for background tasks."""
            ctx = ValidationContext(exec_globals={"asyncio": __import__("asyncio")})
            code = """
task = asyncio.create_task(background_work())
await task
"""
            validator.validate(code, ctx)

        def test_allow_asyncio_sleep(self, validator: UnifiedCodeValidator):
            """asyncio.sleep() is the correct async sleep."""
            ctx = ValidationContext(exec_globals={"asyncio": __import__("asyncio")})
            code = "await asyncio.sleep(1)"
            validator.validate(code, ctx)

        def test_allow_asyncio_wait(self, validator: UnifiedCodeValidator):
            """asyncio.wait() is correct for waiting on multiple tasks."""
            ctx = ValidationContext(exec_globals={"asyncio": __import__("asyncio")})
            code = """
done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
"""
            validator.validate(code, ctx)

        def test_allow_asyncio_wait_for(self, validator: UnifiedCodeValidator):
            """asyncio.wait_for() is correct for timeouts."""
            ctx = ValidationContext(exec_globals={"asyncio": __import__("asyncio")})
            code = """
try:
    result = await asyncio.wait_for(slow_operation(), timeout=5.0)
except asyncio.TimeoutError:
    result = None
"""
            validator.validate(code, ctx)

        def test_allow_async_for(self, validator: UnifiedCodeValidator):
            """async for loops are allowed."""
            ctx = ValidationContext(exec_globals={})
            code = """
async for item in async_iterator:
    process(item)
"""
            validator.validate(code, ctx)

        def test_allow_async_with(self, validator: UnifiedCodeValidator):
            """async with statements are allowed."""
            ctx = ValidationContext(exec_globals={})
            code = """
async with async_lock:
    await do_work()
"""
            validator.validate(code, ctx)

        def test_allow_asyncio_to_thread(self, validator: UnifiedCodeValidator):
            """asyncio.to_thread() is correct for running sync code."""
            ctx = ValidationContext(exec_globals={"asyncio": __import__("asyncio")})
            code = "result = await asyncio.to_thread(blocking_func, arg1, arg2)"
            validator.validate(code, ctx)

        def test_allow_asyncio_wrap_future(self, validator: UnifiedCodeValidator):
            """asyncio.wrap_future() is correct for awaiting futures."""
            ctx = ValidationContext(exec_globals={"asyncio": __import__("asyncio")})
            code = "result = await asyncio.wrap_future(future)"
            validator.validate(code, ctx)

        def test_allow_str_join(self, validator: UnifiedCodeValidator):
            """str.join() should NOT be flagged (not a blocking operation)."""
            ctx = ValidationContext(exec_globals={})
            code = """
items = ["a", "b", "c"]
result = ", ".join(items)
"""
            validator.validate(code, ctx)

        def test_allow_list_join_method(self, validator: UnifiedCodeValidator):
            """Arbitrary .join() calls should NOT be flagged unless on Thread/Process."""
            ctx = ValidationContext(exec_globals={})
            code = """
separator = "-"
result = separator.join(["x", "y", "z"])
"""
            validator.validate(code, ctx)

        def test_allow_arbitrary_submit_result(self, validator: UnifiedCodeValidator):
            """form.submit() should NOT be tracked as a future (not an executor)."""
            ctx = ValidationContext(exec_globals={})
            code = """
form = FormBuilder()
response = form.submit()
data = response.result()  # This is fine - not an executor future
"""
            validator.validate(code, ctx)


# =============================================================================
# REPLPolicyValidator Tests
# =============================================================================


class TestREPLPolicyValidator:
    """Tests for REPL-style policy validations.

    Note: REPLPolicyValidator is NOT included by default in UnifiedCodeValidator.
    It's used by strategies (PurePythonStrategy, CodeActStrategy) separately.
    These tests use full_validator which includes REPL policy.
    """

    # -------------------------------------------------------------------------
    # Patterns to REJECT
    # -------------------------------------------------------------------------

    class TestPatternsToReject:
        """REPL patterns that violate policy and must be rejected."""

        def test_reject_missing_await_on_async_method_only(
            self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Only missing-await errors are raised; class defs are now allowed."""
            # Class defs no longer rejected — this section only has await errors
            pass

        def test_reject_missing_await_on_async_method(self, full_validator: UnifiedCodeValidator):
            """Calling async method without await is an error."""
            # Create a mock agent class to test against
            from nooa.agent import Agent

            class TestAgent(Agent, llm=FakeLLMClient()):
                async def fetch_data(self):
                    return []

            agent = TestAgent()
            context = ValidationContext(
                code="",
                agent=agent,
            )
            code = "result = self.fetch_data()"  # Missing await
            with pytest.raises(ValidationError, match="fetch_data.*async.*await"):
                full_validator.validate(code, context)

    # -------------------------------------------------------------------------
    # Patterns to ALLOW
    # -------------------------------------------------------------------------

    class TestPatternsToAllow:
        """REPL patterns that are valid and must be allowed."""

        def test_allow_class_definition(
            self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Class definitions are now allowed in REPL-style code."""
            code = """
class MyHelper:
    def __init__(self, value):
        self.value = value

    def double(self):
        return self.value * 2

result = MyHelper(21).double()
"""
            full_validator.validate(code, default_context)  # must not raise

        def test_allow_nested_class_definition(
            self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Nested class definitions inside functions are also allowed."""
            code = """
def factory(n):
    class Counter:
        def __init__(self):
            self.count = n
        def increment(self):
            self.count += 1
    return Counter()

c = factory(0)
"""
            full_validator.validate(code, default_context)  # must not raise

        def test_allow_helper_function(
            self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Helper function definitions are allowed."""
            code = """
def process_item(item):
    return item * 2

results = [process_item(x) for x in items]
"""
            full_validator.validate(code, default_context)

        def test_allow_helper_async_function(
            self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Async helper function definitions are allowed."""
            code = """
async def fetch_one(url):
    return await http_get(url)

results = await asyncio.gather(*[fetch_one(u) for u in urls])
"""
            full_validator.validate(code, default_context)

        def test_allow_self_method_helper(
            self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Helper methods with self parameter are allowed."""
            code = """
def helper(self, x):
    return self.multiplier * x

result = helper(self, 5)
"""
            full_validator.validate(code, default_context)

        def test_allow_await_on_async_method(self, full_validator: UnifiedCodeValidator):
            """Correctly awaited async method calls are allowed."""
            from nooa.agent import Agent

            class TestAgent(Agent, llm=FakeLLMClient()):
                async def fetch_data(self):
                    return []

            agent = TestAgent()
            context = ValidationContext(
                code="",
                agent=agent,
            )
            code = "result = await self.fetch_data()"
            full_validator.validate(code, context)

        def test_allow_async_method_in_gather(self, full_validator: UnifiedCodeValidator):
            """Async methods in gather patterns don't need individual await."""
            from nooa.agent import Agent

            class TestAgent(Agent, llm=FakeLLMClient()):
                async def process(self, item):
                    return item * 2

            agent = TestAgent()
            context = ValidationContext(
                code="",
                agent=agent,
            )
            # In list comprehension for gather, await is on gather, not individual calls
            code = """
tasks = [self.process(x) for x in items]
results = await asyncio.gather(*tasks)
"""
            full_validator.validate(code, context)

        def test_allow_async_method_in_generator(self, full_validator: UnifiedCodeValidator):
            """Async methods in generator expressions for gather are allowed."""
            from nooa.agent import Agent

            class TestAgent(Agent, llm=FakeLLMClient()):
                async def process(self, item):
                    return item * 2

            agent = TestAgent()
            context = ValidationContext(
                code="",
                agent=agent,
            )
            code = "results = await asyncio.gather(*(self.process(x) for x in items))"
            full_validator.validate(code, context)

        def test_allow_return_statement(
            self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Return statements are allowed (explicit completion)."""
            code = """
result = compute_value()
return result
"""
            full_validator.validate(code, default_context)

        def test_allow_implicit_return_expression(
            self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """Last expression as implicit return is allowed."""
            code = """
result = 1 + 2
result
"""
            full_validator.validate(code, default_context)


class TestREPLPolicyAdditional:
    """Additional REPL policy tests for enhanced features."""

    def test_reject_infinite_while_true_loop(
        self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """while True without break is a potential infinite loop."""
        code = """
while True:
    process()
"""
        with pytest.raises(ValidationError, match="infinite loop|break"):
            full_validator.validate(code, default_context)

    def test_allow_while_true_with_break(
        self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """while True with break is allowed."""
        code = """
while True:
    if done:
        break
    process()
"""
        full_validator.validate(code, default_context)

    def test_allow_while_true_with_return(
        self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """while True with return is allowed."""
        code = """
while True:
    result = process()
    if result:
        return result
"""
        full_validator.validate(code, default_context)

    def test_allow_while_condition_loop(
        self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """while with a condition is allowed (not infinite)."""
        code = """
count = 0
while count < 10:
    count += 1
"""
        full_validator.validate(code, default_context)

    def test_reject_while_true_with_return_in_nested_function(
        self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """while True with return only in nested function is still infinite."""
        code = """
while True:
    def helper():
        return "done"  # This return is in helper(), not the while loop!
    process()
"""
        with pytest.raises(ValidationError, match="infinite loop|break"):
            full_validator.validate(code, default_context)

    def test_reject_while_true_with_break_in_nested_function(
        self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """while True with break only in nested loop is still infinite at outer level."""
        code = """
while True:
    for i in range(10):
        if i == 5:
            break  # This breaks the for loop, not the while loop!
    process()
"""
        with pytest.raises(ValidationError, match="infinite loop|break"):
            full_validator.validate(code, default_context)

    def test_allow_while_true_with_raise(
        self, full_validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """while True with raise is allowed (exception exits the loop)."""
        code = """
while True:
    result = process()
    if not result:
        raise ValueError("Failed")
"""
        full_validator.validate(code, default_context)

    def test_warn_unnecessary_await_on_sync_method(self, full_validator: UnifiedCodeValidator):
        """Awaiting a sync method is unnecessary and wastes tokens."""
        from nooa.agent import Agent

        class TestAgent(Agent, llm=FakeLLMClient()):
            def sync_method(self):
                """A synchronous method."""
                return 42

        agent = TestAgent()
        context = ValidationContext(
            code="",
            agent=agent,
        )
        # This should generate a warning (not error) about unnecessary await
        # For now, just verify it doesn't crash - warning capture is separate
        code = "result = await self.sync_method()"
        # The validator should handle this gracefully
        # Note: await on non-coroutine raises TypeError at runtime, not validation time
        full_validator.validate(code, context)


# =============================================================================
# Error Formatting Tests
# =============================================================================


class TestErrorFormatting:
    """Tests for error message formatting."""

    def test_error_includes_line_number(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Error message should include the line number."""
        code = """
x = 1
eval('2')
y = 3
"""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(code, default_context)
        assert "line 3" in str(exc_info.value)

    def test_error_includes_source_line(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Error message should include the source line."""
        code = "forbidden_eval = eval('1+1')"
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(code, default_context)
        assert "eval('1+1')" in str(exc_info.value)

    def test_error_includes_caret(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Error message should include caret pointing to issue."""
        code = "x = eval('1')"
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(code, default_context)
        assert "^" in str(exc_info.value)

    def test_error_includes_cell_in_format(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Error message should use Cell In[N] format."""
        context = ValidationContext(
            code="",
            execution_count=42,
        )
        code = "exec('pass')"
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(code, context)
        assert "Cell In[42]" in str(exc_info.value)

    def test_error_shows_available_modules(self, validator: UnifiedCodeValidator):
        """Import error should show restricted module in message."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset({"numpy"}),
        )
        code = "import numpy"
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(code, context)
        error_msg = str(exc_info.value)
        assert "numpy" in error_msg
        assert "restricted" in error_msg


# =============================================================================
# Integration Tests
# =============================================================================


class TestValidatorIntegration:
    """Integration tests for the unified validator."""

    def test_multiple_errors_reports_first(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """With multiple errors, the first one is reported."""
        code = """
import os
eval('1')
exec('2')
"""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(code, default_context)
        # First error is the import on line 2
        assert "import" in str(exc_info.value) or "os" in str(exc_info.value)

    def test_syntax_error_takes_precedence(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Syntax errors should be caught before other validations."""
        code = "def incomplete("
        with pytest.raises(ValidationError, match="[Ss]yntax"):
            validator.validate(code, default_context)

    def test_validators_run_in_order(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Validators should run in defined order."""
        # Security check should run before async check
        code = """
exec('bad')
asyncio.run(coro())
"""
        with pytest.raises(ValidationError) as exc_info:
            validator.validate(code, default_context)
        # exec (security) should be caught before asyncio.run (async)
        assert "exec" in str(exc_info.value)

    def test_empty_code_is_valid(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Empty code should pass validation."""
        validator.validate("", default_context)

    def test_whitespace_only_is_valid(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Whitespace-only code should pass validation."""
        validator.validate("   \n\n   ", default_context)

    def test_comment_only_is_valid(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Comment-only code should pass validation."""
        code = """
# This is a comment
# Another comment
"""
        validator.validate(code, default_context)

    def test_complex_safe_code_passes(
        self, validator: UnifiedCodeValidator, default_context: ValidationContext
    ):
        """Complex but safe code should pass all validations."""
        code = """
# Complex but safe code example
import asyncio
from json import dumps

async def process_items(items):
    \"\"\"Process items in parallel.\"\"\"
    async def process_one(item):
        await asyncio.sleep(0.1)
        return item * 2

    tasks = [process_one(i) for i in items]
    return await asyncio.gather(*tasks)

# Main logic
data = [1, 2, 3, 4, 5]
results = await process_items(data)
output = dumps({"results": results})
return output
"""
        validator.validate(code, default_context)


# =============================================================================
# ClassAssignmentValidator Tests
# =============================================================================


class TestClassAssignmentValidator:
    """Tests for blocking class attribute assignments.

    The LLM can generate code like `ClassName.method = value` which corrupts
    the class definition, affecting all instances. This must be blocked.
    """

    # -------------------------------------------------------------------------
    # Patterns to REJECT
    # -------------------------------------------------------------------------

    class TestPatternsToReject:
        """Class assignment patterns that MUST be rejected."""

        def test_reject_direct_class_assignment(self, validator: UnifiedCodeValidator):
            """Validator blocks: ClassName.method = value"""
            from nooa.agent import Agent

            class TestAgent(Agent, llm=FakeLLMClient()):
                async def process(self) -> dict:
                    """Process something."""
                    ...

            agent = TestAgent()
            context = ValidationContext(code="", agent=agent)

            code = "TestAgent.process = lambda self: None"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class"):
                validator.validate(code, context)

        def test_reject_factory_pattern_class_assignment(self, validator: UnifiedCodeValidator):
            """Validator blocks factory function that assigns to class."""
            from nooa.agent import Agent

            class TestAgent(Agent, llm=FakeLLMClient()):
                async def process(self) -> dict:
                    """Process something."""
                    ...

            agent = TestAgent()
            context = ValidationContext(code="", agent=agent)

            code = """
def _make_method():
    async def process(self):
        return {}
    return process

TestAgent.process = _make_method()
"""

            with pytest.raises(ValidationError, match="[Cc]annot assign to class"):
                validator.validate(code, context)

        def test_reject_subagent_class_assignment(self, validator: UnifiedCodeValidator):
            """Validator blocks assignment to sub-agent classes."""
            from nooa.agent import Agent

            class ChildAgent(Agent, llm=FakeLLMClient()):
                async def work(self) -> str:
                    """Do work."""
                    ...

            class ParentAgent(Agent, llm=FakeLLMClient()):
                async def process(self) -> dict:
                    """Process something."""
                    ...

            # Assign after class definition to avoid NameError
            ParentAgent.ChildAgent = ChildAgent  # type: ignore[attr-defined]

            agent = ParentAgent()
            context = ValidationContext(code="", agent=agent)

            code = "ChildAgent.work = lambda self: 'hacked'"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class"):
                validator.validate(code, context)

        def test_reject_augmented_class_assignment(self, validator: UnifiedCodeValidator):
            """Validator blocks augmented assignment to class attributes."""
            from nooa.agent import Agent

            class TestAgent(Agent, llm=FakeLLMClient()):
                counter = 0

                async def process(self) -> dict:
                    """Process something."""
                    ...

            agent = TestAgent()
            context = ValidationContext(code="", agent=agent)

            # Augmented assignment like += also modifies the class
            code = "TestAgent.counter += 1"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class"):
                validator.validate(code, context)

        def test_reject_annotated_class_assignment(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """Validator blocks annotated assignment: ClassName.attr: Type = value"""
            context, TestAgent = agent_context

            # Annotated assignment pattern
            code = "TestAgent.counter: int = 42"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class"):
                validator.validate(code, context)

        def test_reject_parent_class_assignment_via_mro(self, validator: UnifiedCodeValidator):
            """Validator blocks assignment to parent classes via MRO."""
            from nooa.agent import Agent

            class BaseAgent(Agent, llm=FakeLLMClient()):
                async def base_method(self) -> str:
                    """Base method."""
                    ...

            class ChildAgent(BaseAgent, llm=FakeLLMClient()):
                async def process(self) -> dict:
                    """Process something."""
                    ...

            agent = ChildAgent()
            context = ValidationContext(code="", agent=agent)

            # LLM tries to assign to parent class - should be blocked
            code = "BaseAgent.base_method = lambda self: 'hacked'"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class"):
                validator.validate(code, context)

        def test_reject_agent_base_class_assignment(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """Validator blocks assignment to Agent base class itself."""
            context, _ = agent_context

            # LLM tries to assign to the Agent base class
            code = "Agent.some_method = lambda self: None"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class"):
                validator.validate(code, context)

    # -------------------------------------------------------------------------
    # Patterns to ALLOW (False Positive Checks)
    # -------------------------------------------------------------------------

    class TestPatternsToAllow:
        """Patterns that MUST NOT be rejected (false positive prevention)."""

        def test_allow_self_assignment(self, validator: UnifiedCodeValidator, agent_context):
            """self.attr = value is allowed (instance assignment)."""
            context, _ = agent_context
            code = "self._helper = lambda: 42"
            validator.validate(code, context)  # Should NOT raise

        def test_allow_non_class_attribute_assignment(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """obj.attr = value is allowed when obj is not a known class."""
            context, _ = agent_context
            # 'config' is not a known class name
            code = "config.value = 42"
            validator.validate(code, context)  # Should NOT raise

        def test_allow_dict_key_assignment(self, validator: UnifiedCodeValidator, agent_context):
            """data["key"] = value is allowed (subscript, not attribute)."""
            context, _ = agent_context
            code = 'data["TestAgent"] = "value"'
            validator.validate(code, context)  # Should NOT raise

        def test_allow_local_variable_with_class_name(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """Local variable shadowing class name is allowed if assigned first."""
            context, _ = agent_context
            # This creates a local variable, then assigns to it
            # The validator should ideally track this, but for now
            # we test that simple local var assignment works
            code = """
result = {}
result.value = 42
"""
            validator.validate(code, context)  # Should NOT raise

        def test_allow_chained_attribute_assignment(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """self.obj.attr = value is allowed."""
            context, _ = agent_context
            code = "self.config.value = 42"
            validator.validate(code, context)  # Should NOT raise

        def test_allow_annotation_only(self, validator: UnifiedCodeValidator, agent_context):
            """ClassName.attr: Type (no assignment) is allowed."""
            context, TestAgent = agent_context
            # Just an annotation, no assignment - should be allowed
            code = "TestAgent.counter: int"
            validator.validate(code, context)  # Should NOT raise

        def test_allow_module_attribute_assignment(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """module.attr = value is allowed for non-class modules."""
            # asyncio.current_task = ... would be weird but not our class
            code = "some_module.setting = True"
            validator.validate(code, default_context)  # Should NOT raise

        def test_allow_without_agent_context(
            self, validator: UnifiedCodeValidator, default_context: ValidationContext
        ):
            """When no agent is provided, class names can't be detected."""
            # Without agent context, we can't know what's a class
            code = "SomeClass.method = lambda: None"
            validator.validate(code, default_context)  # Should NOT raise (no agent)

    # -------------------------------------------------------------------------
    # Additional Patterns to Reject (setattr, dynamic class references)
    # -------------------------------------------------------------------------

    class TestSetattrAndDynamicClassPatterns:
        """Tests for setattr and dynamic class reference patterns."""

        def test_reject_setattr_to_class(self, validator: UnifiedCodeValidator, agent_context):
            """setattr(ClassName, 'attr', value) must be rejected.

            This pattern modifies the class just like direct assignment.
            """
            context, TestAgent = agent_context

            code = "setattr(TestAgent, 'process', lambda self: None)"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class|setattr.*class"):
                validator.validate(code, context)

        def test_reject_setattr_to_subagent_class(self, validator: UnifiedCodeValidator):
            """setattr on sub-agent classes must also be rejected."""
            from nooa.agent import Agent

            class ChildAgent(Agent, llm=FakeLLMClient()):
                async def work(self) -> str: ...

            class ParentAgent(Agent, llm=FakeLLMClient()):
                async def process(self) -> dict: ...

            # Assign after class definition to avoid NameError
            ParentAgent.ChildAgent = ChildAgent  # type: ignore[attr-defined]

            agent = ParentAgent()
            context = ValidationContext(code="", agent=agent)

            code = "setattr(ChildAgent, 'work', lambda self: 'hacked')"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class|setattr.*class"):
                validator.validate(code, context)

        def test_reject_dynamic_class_via_type_self(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """cls = type(self); cls.attr = value must be rejected.

            Variables assigned from type(self) hold a class reference.
            """
            context, _ = agent_context

            code = """
cls = type(self)
cls.process = lambda self: None
"""

            with pytest.raises(ValidationError, match="[Cc]annot assign|class reference"):
                validator.validate(code, context)

        def test_reject_dynamic_class_via_dunder_class(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """cls = self.__class__; cls.attr = value - __class__ blocked by SecurityValidator."""
            context, _ = agent_context

            # __class__ access is blocked by SecurityValidator, so this should fail
            code = """
cls = self.__class__
cls.process = lambda self: None
"""

            with pytest.raises(ValidationError, match="__class__.*forbidden"):
                validator.validate(code, context)

        def test_reject_inline_type_self_assignment(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """type(self).attr = value must be rejected (inline pattern)."""
            context, _ = agent_context

            code = "type(self).process = lambda self: None"

            with pytest.raises(ValidationError, match="[Cc]annot assign to class|type\\(self\\)"):
                validator.validate(code, context)

        def test_reject_nested_self_reference_type_call(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """agent = self; type(agent).attr = value must be rejected.

            When a variable is assigned from `self`, calling type() on it
            is equivalent to type(self) and should be blocked.
            """
            context, _ = agent_context

            code = """
agent = self
type(agent).process = lambda self: None
"""

            with pytest.raises(ValidationError, match="[Cc]annot assign|type\\("):
                validator.validate(code, context)

        def test_reject_setattr_with_nested_self_reference(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """setattr(type(agent), ...) where agent = self must be rejected."""
            context, _ = agent_context

            code = """
agent = self
setattr(type(agent), 'process', lambda self: None)
"""

            with pytest.raises(ValidationError, match="[Cc]annot|setattr"):
                validator.validate(code, context)

        def test_self_class_blocked_by_security_validator(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """self.__class__.attr = value IS caught by SecurityValidator.

            The __class__ dunder attribute is in DANGEROUS_DUNDER_ATTRS,
            so this pattern is blocked before ClassAssignmentValidator runs.
            This is defense in depth - SecurityValidator catches it first.
            """
            context, _ = agent_context

            # This pattern is blocked because __class__ access is forbidden
            code = "self.__class__.process = lambda self: None"

            # Blocked by SecurityValidator's dunder check
            with pytest.raises(ValidationError, match="__class__.*forbidden"):
                validator.validate(code, context)

    # -------------------------------------------------------------------------
    # Known False Positives (legitimate patterns incorrectly blocked)
    # -------------------------------------------------------------------------

    class TestKnownFalsePositives:
        """Document patterns that SHOULD be allowed but might be incorrectly blocked.

        These tests verify the current behavior is correct (no false positives).
        If any of these start failing, it indicates a regression.
        """

        def test_no_false_positive_local_var_same_name_as_class(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """Local variable with same name as class should be allowed AFTER reassignment.

            Edge case: If LLM creates a local variable that shadows a class name,
            subsequent assignments to that variable should be allowed.

            Note: This is tricky - we can't do full data flow analysis in AST validation.
            Current behavior: We reject this as a false positive because we can't
            distinguish between the class and a local variable with the same name.
            """
            context, TestAgent = agent_context

            # This creates a local variable shadowing the class name
            # Ideally this should be allowed, but we can't distinguish it from class
            code = """
TestAgent = {"mock": True}  # Local var shadows class
TestAgent.method = lambda: None  # Assignment to local... but looks like class
"""
            # Current behavior: This IS rejected (false positive)
            # This is acceptable because:
            # 1. Shadowing class names is bad practice
            # 2. LLM shouldn't be doing this anyway
            # 3. Better to be safe than sorry
            with pytest.raises(ValidationError, match="[Cc]annot assign to class"):
                validator.validate(code, context)

        def test_allow_method_call_that_looks_like_assignment(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """Method calls on class are allowed (only assignments are blocked)."""
            context, TestAgent = agent_context

            # Calling a class method is fine
            code = "result = TestAgent.some_classmethod()"
            validator.validate(code, context)  # Should NOT raise

        def test_allow_class_attribute_read(self, validator: UnifiedCodeValidator, agent_context):
            """Reading class attributes is allowed (only assignments are blocked)."""
            context, TestAgent = agent_context

            # Reading from class is fine
            code = "value = TestAgent.some_attribute"
            validator.validate(code, context)  # Should NOT raise

        def test_allow_instantiation(self, validator: UnifiedCodeValidator, agent_context):
            """Instantiating the class is allowed."""
            context, TestAgent = agent_context

            # Creating instances is fine
            code = "instance = TestAgent()"
            validator.validate(code, context)  # Should NOT raise

        def test_allow_isinstance_check(self, validator: UnifiedCodeValidator, agent_context):
            """isinstance/issubclass checks are allowed."""
            context, TestAgent = agent_context

            code = """
is_agent = isinstance(obj, TestAgent)
is_sub = issubclass(SomeClass, TestAgent)
"""
            validator.validate(code, context)  # Should NOT raise

        def test_allow_class_in_type_annotation(
            self, validator: UnifiedCodeValidator, agent_context
        ):
            """Using class in type annotations is allowed."""
            context, TestAgent = agent_context

            # Type annotations referencing the class are fine
            code = """
def helper(agent: TestAgent) -> TestAgent:
    return agent
"""
            validator.validate(code, context)  # Should NOT raise

        def test_allow_class_as_dict_value(self, validator: UnifiedCodeValidator, agent_context):
            """Storing class reference in a dict is allowed."""
            context, TestAgent = agent_context

            code = 'registry = {"agent": TestAgent}'
            validator.validate(code, context)  # Should NOT raise

        def test_allow_class_in_list(self, validator: UnifiedCodeValidator, agent_context):
            """Storing class reference in a list is allowed."""
            context, TestAgent = agent_context

            code = "classes = [TestAgent, OtherClass]"
            validator.validate(code, context)  # Should NOT raise


# =============================================================================
# strip_redundant_imports tests
# =============================================================================


class TestStripRedundantImports:
    """Tests for the import pre-processing that removes redundant imports."""

    def test_strips_from_import_when_all_names_in_scope(self):
        code = "from typing import Literal\nx = Literal['a', 'b']"
        result, _ = strip_redundant_imports(code, {"Literal", "typing"})
        assert "from typing" not in result
        assert "Literal['a', 'b']" in result

    def test_strips_plain_import_when_name_in_scope(self):
        code = "import asyncio\nawait asyncio.sleep(1)"
        result, _ = strip_redundant_imports(code, {"asyncio"})
        assert "import asyncio" not in result
        assert "asyncio.sleep" in result

    def test_strips_from_non_module_name(self):
        """'from strategy import X' where strategy is a function, not a module."""
        code = "from strategy import PredictStrategy, strategy\nx = 1"
        available = {"strategy", "PredictStrategy", "x"}
        result, _ = strip_redundant_imports(code, available)
        assert "from strategy" not in result
        assert "x = 1" in result

    def test_keeps_import_when_some_names_not_in_scope(self):
        code = "from typing import Literal, TypeVar\nx = 1"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing import Literal, TypeVar" in result

    def test_keeps_import_when_no_names_in_scope(self):
        code = "import pandas\ndf = pandas.DataFrame()"
        result, _ = strip_redundant_imports(code, set())
        assert "import pandas" in result

    def test_does_not_strip_star_imports(self):
        code = "from typing import *\nx = 1"
        result, _ = strip_redundant_imports(code, {"Literal", "typing"})
        assert "from typing import *" in result

    def test_handles_aliased_import(self):
        code = "import numpy as np\nx = np.array([1])"
        result, _ = strip_redundant_imports(code, {"np"})
        assert "import numpy" not in result

    def test_passes_through_syntax_errors(self):
        code = "this is not valid python {{{"
        result, _ = strip_redundant_imports(code, {"x"})
        assert result == code

    def test_strips_multiple_redundant_imports(self):
        code = "from typing import Literal\nfrom strategy import strategy\nx = 1"
        result, _ = strip_redundant_imports(code, {"Literal", "strategy", "typing"})
        assert "from typing" not in result
        assert "from strategy" not in result
        assert "x = 1" in result

    def test_preserves_non_import_code(self):
        code = "from typing import Literal\n\n@strategy(PredictStrategy())\nasync def classify(self, t: str) -> Literal['a']:\n    ...\n\nresults = []"
        result, _ = strip_redundant_imports(
            code, {"Literal", "typing", "strategy", "PredictStrategy"}
        )
        assert "from typing" not in result
        assert "@strategy" in result
        assert "results = []" in result

    def test_preserves_inline_comments(self):
        # ast.unparse strips all comments; line-based removal must not.
        # Comments in LLM-generated code are rare but present, and stripped
        # comments corrupt the error-message line numbers shown back to the LLM.
        code = "from typing import Literal\n# explain the logic\nx = 1  # inline note"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "# explain the logic" in result
        assert "# inline note" in result

    def test_preserves_blank_lines(self):
        # ast.unparse collapses all blank lines; line-based removal must not.
        # A stripped blank line shifts every subsequent line number, making
        # validation error messages reference wrong lines.
        code = "from typing import Literal\n\nx = 1"
        result, _ = strip_redundant_imports(code, {"Literal"})
        lines = result.splitlines()
        # Blank line between import and code must survive
        assert lines[0] == ""
        assert lines[1] == "x = 1"

    def test_removes_all_lines_of_multiline_parenthesized_import(self):
        # A parenthesized import spans multiple lines; all of them must be removed.
        code = "from typing import (\n    Literal,\n    Optional,\n)\nx = 1"
        result, _ = strip_redundant_imports(code, {"Literal", "Optional"})
        assert "from typing" not in result
        assert "Literal" not in result
        assert "x = 1" in result

    def test_removes_all_lines_of_backslash_continuation_import(self):
        # Backslash-continued imports also span multiple AST end_lineno lines.
        code = "from typing import \\\n    Literal\nx = 1"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing" not in result
        assert "Literal" not in result
        assert "x = 1" in result

    def test_import_only_code_returns_empty(self):
        # Stripping the only statement should return empty string, not crash.
        code = "from typing import Literal\n"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert result.strip() == ""

    def test_semicolon_same_line_preserves_non_import_code(self):
        # `from X import Y; real_code` on one line: the import is stripped but
        # the non-import statement on the same line must be preserved.
        code = "from typing import Literal; x = 1"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing import Literal" not in result
        assert "x = 1" in result

    def test_semicolon_multiple_statements_after_import(self):
        # Multiple non-import statements after a removed import on the same line.
        code = "from typing import Literal; x = 1; y = 2"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing import Literal" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_semicolon_only_some_imports_removed(self):
        # `import os; import pandas` where only os is in scope — pandas import
        # is kept, os import is removed, both on same line.
        code = "import os; import pandas"
        result, _ = strip_redundant_imports(code, {"os"})
        assert "import os" not in result
        assert "import pandas" in result

    def test_semicolon_multiline_kept_node_starting_on_removed_line(self):
        # The non-import statement spans multiple lines but its opening is on
        # the same line as the removed import.
        code = "from typing import Literal; result = (\n    1 + 2\n)"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing import Literal" not in result
        assert "result" in result
        assert "1 + 2" in result

    def test_nested_import_inside_function_not_stripped(self):
        # Imports inside function bodies are in a nested scope and must never
        # be stripped, even if the name is in exec_globals.
        code = "def foo():\n    from typing import Literal\n    return Literal['a']"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing import Literal" in result

    def test_mid_file_import_is_stripped(self):
        # An import that appears mid-file (not at the top) is still redundant
        # if the name is already in scope and must be stripped.
        code = "x = 1\nfrom typing import Literal\ny = 2"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing import Literal" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_code_before_import_on_same_line(self):
        # Non-import statement appears BEFORE the import on the same line.
        code = "x = 1; from typing import Literal"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing import Literal" not in result
        assert "x = 1" in result

    def test_comment_on_semicolon_line_is_preserved(self):
        # When kept code shares a line with a removed import, the inline
        # comment that follows the kept statement must be preserved.
        code = "from typing import Literal; x = 1  # important note"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert "from typing import Literal" not in result
        assert "x = 1" in result
        assert "# important note" in result

    def test_multiline_kept_node_preserves_original_formatting(self):
        # When a multi-line kept node starts on the same line as a removed
        # import, its original multi-line formatting must be preserved —
        # not flattened to a single line by ast.unparse.
        code = "from typing import Literal; result = (\n    1 + 2\n)"
        result, _ = strip_redundant_imports(code, {"Literal"})
        assert result == "result = (\n    1 + 2\n)"

    def test_line_numbers_shift_by_exactly_one_removed_line(self):
        # The whole point of the fix: after stripping a 1-line import, code
        # that was on line 2 moves to line 1 — shifted by exactly 1, not
        # collapsed to line 1 as ast.unparse would do with the whole file.
        code = "from typing import Literal\nx = 1\ny = 2"
        result, _ = strip_redundant_imports(code, {"Literal"})
        lines = result.splitlines()
        assert lines[0] == "x = 1"  # was line 2, now line 1
        assert lines[1] == "y = 2"  # was line 3, now line 2

    def test_line_numbers_shift_by_multiline_import_size(self):
        # A 4-line parenthesized import is removed; subsequent code shifts
        # by exactly 4 lines, not more.
        code = "from typing import (\n    Literal,\n    Optional,\n)\nx = 1\ny = 2"
        result, _ = strip_redundant_imports(code, {"Literal", "Optional"})
        lines = result.splitlines()
        assert lines[0] == "x = 1"  # was line 5, now line 1
        assert lines[1] == "y = 2"  # was line 6, now line 2

    def test_lines_before_mid_file_import_keep_original_numbers(self):
        # Lines before a stripped import are unaffected; lines after shift by 1.
        code = "a = 1\nb = 2\nfrom typing import Literal\nc = 3\nd = 4"
        result, _ = strip_redundant_imports(code, {"Literal"})
        lines = result.splitlines()
        assert lines[0] == "a = 1"  # was line 1, still line 1
        assert lines[1] == "b = 2"  # was line 2, still line 2
        assert lines[2] == "c = 3"  # was line 4, now line 3
        assert lines[3] == "d = 4"  # was line 5, now line 4

    def test_blank_lines_keep_relative_spacing(self):
        # Blank lines within the code body are preserved, maintaining
        # the relative line distance between statements.
        code = "from typing import Literal\n\na = 1\n\nb = 2"
        result, _ = strip_redundant_imports(code, {"Literal"})
        lines = result.splitlines()
        assert lines[0] == ""  # blank line (was line 2, now line 1)
        assert lines[1] == "a = 1"  # was line 3, now line 2
        assert lines[2] == ""  # blank line preserved
        assert lines[3] == "b = 2"  # was line 5, now line 4


# =============================================================================
# ReturnTypeShadowValidator Tests
# =============================================================================


class TestReturnTypeShadowValidator:
    """Generated code must not redefine the method's declared return type.

    Inside ``__repl_wrapper__``, a local ``class Answer(BaseModel)`` becomes
    ``__repl_wrapper__.<locals>.Answer`` — structurally identical to the original
    ``Answer`` but a distinct class object. ``return_result()`` then fails
    Pydantic's ``isinstance`` check with an unhelpful "Expected: Answer / Got:
    Answer" error and the LLM cannot recover (issue gl-143).
    """

    @staticmethod
    def _ctx_with_return_type(return_type):
        """Build a ValidationContext seeded with a return type annotation."""
        return ValidationContext(
            code="",
            return_type=return_type,
        )

    class TestPatternsToReject:
        """Definitions that shadow the return type must be rejected."""

        def test_reject_class_shadow_of_pydantic_return_type(self, validator: UnifiedCodeValidator):
            from pydantic import BaseModel

            class Answer(BaseModel):
                answer: int
                reason: str

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = (
                "class Answer(BaseModel):\n"
                "    answer: int\n"
                "    reason: str\n"
                "result = Answer(answer=42, reason='because')"
            )

            with pytest.raises(ValidationError) as exc:
                validator.validate(code, context)
            assert "Cannot redefine 'Answer'" in str(exc.value)
            assert "return_result" in str(exc.value)
            assert "E501" not in str(exc.value)  # error code surfaces via field, not message
            # iPython-style cell location is included.
            assert "Cell In[" in str(exc.value)

        def test_reject_class_shadow_inside_optional(self, validator: UnifiedCodeValidator):
            """``Answer | None`` annotations still protect ``Answer``."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer | None)
            code = "class Answer(BaseModel):\n    value: int"
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate(code, context)

        def test_reject_class_shadow_inside_list(self, validator: UnifiedCodeValidator):
            """``list[Answer]`` annotations still protect ``Answer``."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(list[Answer])
            code = "class Answer(BaseModel):\n    value: int"
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate(code, context)

        def test_reject_function_shadow_of_return_type(self, validator: UnifiedCodeValidator):
            """A def with the same name as the return type also shadows it."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "def Answer(value):\n    return value"
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate(code, context)

        def test_reject_assignment_shadow_of_return_type(self, validator: UnifiedCodeValidator):
            """Answer = ... overwrites the return type and must be rejected."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "Answer = None"
            with pytest.raises(ValidationError, match="Cannot reassign 'Answer'"):
                validator.validate(code, context)

        def test_reject_assignment_shadow_expression(self, validator: UnifiedCodeValidator):
            """Answer = <expr> also caught even with complex RHS."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = (
                "Answer = BaseModel.__getattr__('Answer') if hasattr(BaseModel, 'Answer') else None"
            )
            with pytest.raises(ValidationError, match="Cannot reassign 'Answer'"):
                validator.validate(code, context)

        def test_reject_annotated_assignment_shadow(self, validator: UnifiedCodeValidator):
            """Answer: type = ... (annotated assignment) also caught."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "Answer: type = type('Answer', (), {})"
            with pytest.raises(ValidationError, match="Cannot reassign 'Answer'"):
                validator.validate(code, context)

        def test_reject_tuple_unpack_shadow(self, validator: UnifiedCodeValidator):
            """Answer, other = ... (tuple unpacking) also caught."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "Answer, other = None, 42"
            with pytest.raises(ValidationError, match="Cannot reassign 'Answer'"):
                validator.validate(code, context)

    class TestPatternsToAllow:
        """Definitions that do not shadow the return type must be accepted."""

        def test_allow_helper_class_with_unrelated_name(self, validator: UnifiedCodeValidator):
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "class Helper:\n    pass"
            validator.validate(code, context)  # must NOT raise

        def test_allow_helper_function_with_unrelated_name(self, validator: UnifiedCodeValidator):
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "def gcd(a, b):\n    while b: a, b = b, a % b\n    return a"
            validator.validate(code, context)  # must NOT raise

        def test_allow_construction_of_existing_return_type(self, validator: UnifiedCodeValidator):
            """Constructing the existing class is the correct fix and must pass."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "result = Answer(value=42)"
            validator.validate(code, context)  # must NOT raise

        def test_allow_when_return_type_is_unset(self, validator: UnifiedCodeValidator):
            """Without a known return type, validator is a no-op."""
            context = ValidationContext(
                code="",
                return_type=None,
            )
            code = "class Answer:\n    pass"
            validator.validate(code, context)  # must NOT raise

        def test_allow_when_return_type_is_builtin(self, validator: UnifiedCodeValidator):
            """Builtin return types like ``str``/``int`` aren't user-redefinable in practice."""
            context = TestReturnTypeShadowValidator._ctx_with_return_type(str)
            code = "def helper(): return 1"
            validator.validate(code, context)  # must NOT raise

    class TestTelemetry:
        """Shadow rejections increment the harness metric."""

        def test_redefinition_increments_harness_metric(self, validator: UnifiedCodeValidator):
            from pydantic import BaseModel

            from nooa.runtime.harness_metrics import (
                HarnessMetrics,
                _harness_metrics_var,
            )

            class Answer(BaseModel):
                value: int

            metrics = HarnessMetrics()
            token = _harness_metrics_var.set(metrics)
            try:
                context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
                code = "class Answer:\n    value: int"
                with pytest.raises(ValidationError):
                    validator.validate(code, context)
                assert metrics.return_types_redefined == ["Answer"]
            finally:
                _harness_metrics_var.reset(token)

    class TestNonPydanticTypeKinds:
        """The shadow rule applies to any class/function used as a return type,
        not just Pydantic. The runtime symptom differs (TypedDict survives via
        structural validation; Pydantic fails identity), but redefining the name
        in the cell is wrong in every case because the user-redefined symbol is
        what the LLM ends up using in its own code, not the framework's."""

        def test_reject_typeddict_shadow(self, validator: UnifiedCodeValidator):
            from typing import TypedDict

            class AnswerTD(TypedDict):
                answer: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(AnswerTD)
            code = "class AnswerTD(TypedDict):\n    answer: int"
            with pytest.raises(ValidationError, match="Cannot redefine 'AnswerTD'"):
                validator.validate(code, context)

        def test_reject_dataclass_shadow(self, validator: UnifiedCodeValidator):
            from dataclasses import dataclass

            @dataclass
            class AnswerDC:
                answer: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(AnswerDC)
            # Decorated class definition: AST lineno points at the `class`
            # keyword, not the decorator; the iPython renderer still produces a
            # readable error.
            code = "@dataclass\nclass AnswerDC:\n    answer: int"
            with pytest.raises(ValidationError, match="Cannot redefine 'AnswerDC'"):
                validator.validate(code, context)

        def test_reject_plain_class_shadow(self, validator: UnifiedCodeValidator):
            class AnswerPlain:
                def __init__(self, answer: int) -> None:
                    self.answer = answer

            context = TestReturnTypeShadowValidator._ctx_with_return_type(AnswerPlain)
            code = (
                "class AnswerPlain:\n    def __init__(self, answer):\n        self.answer = answer"
            )
            with pytest.raises(ValidationError, match="Cannot redefine 'AnswerPlain'"):
                validator.validate(code, context)

        def test_function_shadow_message_says_function(self, validator: UnifiedCodeValidator):
            """The wording must say 'scoped function', not 'scoped class', when
            the shadow is a def. Regression: an earlier version always said
            'scoped class', which was misleading for def shadows."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                answer: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "def Answer(value):\n    return value"
            with pytest.raises(ValidationError) as exc:
                validator.validate(code, context)
            msg = str(exc.value)
            assert "A local function definition" in msg
            assert "scoped function" in msg
            assert "scoped class" not in msg

    class TestComplexReturnTypeAnnotations:
        """`_collect_type_names` must walk the full annotation surface."""

        def test_protects_class_inside_annotated(self, validator: UnifiedCodeValidator):
            from typing import Annotated

            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(
                Annotated[Answer, "the final answer"]
            )
            code = "class Answer(BaseModel):\n    value: int"
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate(code, context)

        def test_protects_class_inside_typing_optional(self, validator: UnifiedCodeValidator):
            """The legacy ``Optional[T]`` spelling protects T just like ``T | None``."""
            from typing import Optional

            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Optional[Answer])  # noqa: UP045
            code = "class Answer(BaseModel):\n    value: int"
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate(code, context)

        def test_protects_class_inside_dict_value(self, validator: UnifiedCodeValidator):
            from pydantic import BaseModel

            class Entry(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(dict[str, Entry])
            code = "class Entry(BaseModel):\n    value: int"
            with pytest.raises(ValidationError, match="Cannot redefine 'Entry'"):
                validator.validate(code, context)

        def test_protects_class_inside_nested_generics(self, validator: UnifiedCodeValidator):
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            # ``list[dict[str, Answer]]`` — the validator must recurse all the way.
            context = TestReturnTypeShadowValidator._ctx_with_return_type(list[dict[str, Answer]])
            code = "class Answer(BaseModel):\n    value: int"
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate(code, context)

        def test_protects_all_classes_in_a_union(self, validator: UnifiedCodeValidator):
            """Both arms of ``A | B`` should be protected independently."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                a: int

            class Question(BaseModel):
                q: str

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer | Question)

            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate("class Answer(BaseModel):\n    a: int", context)
            with pytest.raises(ValidationError, match="Cannot redefine 'Question'"):
                validator.validate("class Question(BaseModel):\n    q: str", context)

        def test_dict_of_builtins_protects_nothing(self, validator: UnifiedCodeValidator):
            """`dict[str, int]` collects only builtins, which are skipped."""
            context = TestReturnTypeShadowValidator._ctx_with_return_type(dict[str, int])
            # Defining a helper class is fine — nothing in the annotation is a
            # user class to protect.
            code = "class Helper:\n    pass"
            validator.validate(code, context)  # must NOT raise

        def test_protects_all_arms_of_three_way_union(self, validator: UnifiedCodeValidator):
            from pydantic import BaseModel

            class Alpha(BaseModel):
                a: int

            class Beta(BaseModel):
                b: int

            class Gamma(BaseModel):
                c: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Alpha | Beta | Gamma)
            for name in ("Alpha", "Beta", "Gamma"):
                with pytest.raises(ValidationError, match=f"Cannot redefine '{name}'"):
                    validator.validate(f"class {name}(BaseModel):\n    x: int", context)

        def test_protects_mixed_kind_union(self, validator: UnifiedCodeValidator):
            """A union of Pydantic + TypedDict + dataclass should protect every arm,
            since the ``__repl_wrapper__`` shadow problem is type-system-agnostic."""
            from dataclasses import dataclass
            from typing import TypedDict

            from pydantic import BaseModel

            class Pyd(BaseModel):
                p: int

            class TD(TypedDict):
                t: int

            @dataclass
            class DC:
                d: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Pyd | TD | DC)
            with pytest.raises(ValidationError, match="Cannot redefine 'Pyd'"):
                validator.validate("class Pyd(BaseModel):\n    p: int", context)
            with pytest.raises(ValidationError, match="Cannot redefine 'TD'"):
                validator.validate("class TD(TypedDict):\n    t: int", context)
            with pytest.raises(ValidationError, match="Cannot redefine 'DC'"):
                validator.validate("class DC:\n    d: int", context)

        def test_protects_union_with_none(self, validator: UnifiedCodeValidator):
            """`A | B | None` protects A and B; None is silently dropped."""
            from pydantic import BaseModel

            class Alpha(BaseModel):
                a: int

            class Beta(BaseModel):
                b: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Alpha | Beta | None)
            with pytest.raises(ValidationError, match="Cannot redefine 'Alpha'"):
                validator.validate("class Alpha(BaseModel):\n    a: int", context)
            with pytest.raises(ValidationError, match="Cannot redefine 'Beta'"):
                validator.validate("class Beta(BaseModel):\n    b: int", context)

        def test_protects_union_nested_in_generic(self, validator: UnifiedCodeValidator):
            """A union sitting inside ``dict[str, A | B]`` still protects A and B."""
            from pydantic import BaseModel

            class Alpha(BaseModel):
                a: int

            class Beta(BaseModel):
                b: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(dict[str, Alpha | Beta])
            with pytest.raises(ValidationError, match="Cannot redefine 'Alpha'"):
                validator.validate("class Alpha(BaseModel):\n    a: int", context)
            with pytest.raises(ValidationError, match="Cannot redefine 'Beta'"):
                validator.validate("class Beta(BaseModel):\n    b: int", context)

        def test_protects_type_brackets(self, validator: UnifiedCodeValidator):
            """`type[Answer]` (the *class* of Answer) still protects Answer.

            Origin is the builtin ``type`` (filtered); the argument is Answer
            (protected). The agent-method declares it returns the class, but
            the LLM redefining ``Answer`` is still wrong."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(type[Answer])
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate("class Answer(BaseModel):\n    value: int", context)

        def test_literal_protects_nothing(self, validator: UnifiedCodeValidator):
            """`Literal["yes", "no"]` has string args — no classes to protect."""
            from typing import Literal

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Literal["yes", "no"])
            code = "class Helper:\n    pass"
            validator.validate(code, context)  # must NOT raise

        def test_typevar_protects_nothing(self, validator: UnifiedCodeValidator):
            """A TypeVar return type isn't a class — nothing to protect."""
            from typing import TypeVar

            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            T = TypeVar("T", bound=Answer)
            context = TestReturnTypeShadowValidator._ctx_with_return_type(T)
            # Defining a helper class is fine; nothing was extracted from T.
            code = "class Helper:\n    pass"
            validator.validate(code, context)  # must NOT raise

        def test_iterable_origin_does_not_leak_as_protected(self, validator: UnifiedCodeValidator):
            """Regression: `Iterable[Answer]` previously injected ``Iterable``
            (from ``collections.abc``) into the protected set. The arg
            ``Answer`` should be protected, but a helper named ``Iterable``
            should still be allowed."""
            from collections.abc import Iterable

            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Iterable[Answer])
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate("class Answer(BaseModel):\n    value: int", context)
            # A class named after the abc origin must NOT trigger the validator.
            validator.validate("class Iterable:\n    pass", context)

        def test_callable_origin_does_not_leak_as_protected(self, validator: UnifiedCodeValidator):
            """Same regression as Iterable, for ``Callable[..., T]``."""
            from collections.abc import Callable

            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Callable[..., Answer])
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate("class Answer(BaseModel):\n    value: int", context)
            # A class named ``Callable`` is not flagged by the validator
            # (the abc origin is filtered).
            validator.validate("class Callable:\n    pass", context)

        def test_future_annotations_resolves_through_get_type_hints(self):
            """End-to-end sanity: when an agent file uses ``from __future__
            import annotations``, ``inspect.signature(method).return_annotation``
            is the raw string ``'Answer'`` — but ``typing.get_type_hints()``
            resolves it back to the class. The framework's CurrentCall
            construction relies on this. If this test fails, the dabstep-style
            agents (which all use future annotations) will silently bypass
            return-type validation in the existing Pydantic path AND in
            ReturnTypeShadowValidator."""
            import inspect
            from typing import get_type_hints

            ns: dict = {}
            exec(
                compile(
                    "from __future__ import annotations\n"
                    "from pydantic import BaseModel\n"
                    "class Answer(BaseModel):\n"
                    "    value: int\n"
                    "class A:\n"
                    "    async def m(self) -> Answer: ...\n",
                    "<future-ann-test>",
                    "exec",
                ),
                ns,
            )
            method = ns["A"].m
            assert isinstance(inspect.signature(method).return_annotation, str)
            hints = get_type_hints(method, include_extras=True)
            assert hints["return"] is ns["Answer"]

        def test_string_forward_reference_resolves_via_namespace(
            self, validator: UnifiedCodeValidator
        ):
            """Defensive: if a string forward ref slips through (e.g.
            ``get_type_hints`` raised and the framework fell back to the raw
            ``sig.return_annotation``), the walker should still try to resolve
            it against the agent's exec_globals before giving up."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            # ``return_type`` is the bare string the framework couldn't resolve;
            # but the agent's exec_globals contain the actual class.
            context = ValidationContext(
                code="",
                return_type="Answer",
                exec_globals={"Answer": Answer},
                execution_count=2,
            )
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate("class Answer:\n    value: int\nresult = Answer()", context)

        def test_string_forward_reference_no_op_when_namespace_lacks_name(
            self, validator: UnifiedCodeValidator
        ):
            """If the string can't be resolved (name not in exec_globals),
            the walker degrades to a no-op rather than raising. This is the
            'genuinely unresolvable' case — we'd rather under-protect than
            crash on a perfectly-fine helper-class definition."""
            context = ValidationContext(
                code="",
                return_type="Mystery",
                exec_globals={},
                execution_count=2,
            )
            # No raise — the validator can't determine if 'Mystery' is the
            # type and conservatively allows it. Better to under-protect than
            # to false-positive on agent code that happens to use that name.
            validator.validate("class Mystery:\n    pass", context)

        def test_string_forward_reference_in_a_generic(self, validator: UnifiedCodeValidator):
            """Forward refs nested inside generics (``list['Answer']``,
            ``dict[str, 'Answer']``) also resolve via the namespace."""
            from typing import ForwardRef

            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            # ``list['Answer']`` materializes a list with a ForwardRef arg.
            list_of_ref = list[ForwardRef("Answer")]
            context = ValidationContext(
                code="",
                return_type=list_of_ref,
                exec_globals={"Answer": Answer},
                execution_count=2,
            )
            with pytest.raises(ValidationError, match="Cannot redefine 'Answer'"):
                validator.validate("class Answer(BaseModel):\n    value: int", context)

    class TestScopeBoundaries:
        """Only top-level definitions in the cell shadow the return type."""

        def test_allow_class_def_nested_inside_function(self, validator: UnifiedCodeValidator):
            """A class with the protected name defined inside ``def helper()``
            doesn't leak to module scope, so it doesn't shadow."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = (
                "def helper():\n"
                "    class Answer:\n"  # local to helper, not a real shadow
                "        value: int\n"
                "    return Answer\n"
            )
            validator.validate(code, context)  # must NOT raise

        def test_allow_class_def_nested_inside_class_body(self, validator: UnifiedCodeValidator):
            """A nested class inside another class body is also not a top-level
            shadow."""
            from pydantic import BaseModel

            class Answer(BaseModel):
                value: int

            context = TestReturnTypeShadowValidator._ctx_with_return_type(Answer)
            code = "class Outer:\n    class Answer:\n        value: int\n"
            validator.validate(code, context)  # must NOT raise

    class TestCollectTypeNamesUnit:
        """Direct unit tests for the type-name walker."""

        def test_collects_bare_user_class(self):
            from nooa.runtime.code_validator import _collect_type_names

            class Foo:
                pass

            assert _collect_type_names(Foo) == {"Foo"}

        def test_skips_builtins(self):
            from nooa.runtime.code_validator import _collect_type_names

            assert _collect_type_names(str) == set()
            assert _collect_type_names(int) == set()
            # ``list[int]`` — origin is ``list`` (builtin), arg is ``int`` (builtin):
            assert _collect_type_names(list[int]) == set()

        def test_skips_none(self):
            from nooa.runtime.code_validator import _collect_type_names

            assert _collect_type_names(None) == set()
            assert _collect_type_names(type(None)) == set()

        def test_unwraps_annotated(self):
            from typing import Annotated

            from nooa.runtime.code_validator import _collect_type_names

            class Foo:
                pass

            assert _collect_type_names(Annotated[Foo, "tag"]) == {"Foo"}

        def test_collects_inside_optional_and_union(self):
            from typing import Optional

            from nooa.runtime.code_validator import _collect_type_names

            class Foo:
                pass

            class Bar:
                pass

            assert _collect_type_names(Optional[Foo]) == {"Foo"}  # noqa: UP045
            assert _collect_type_names(Foo | Bar) == {"Foo", "Bar"}

        def test_recurses_into_generics(self):
            from nooa.runtime.code_validator import _collect_type_names

            class Item:
                pass

            assert _collect_type_names(list[Item]) == {"Item"}
            assert _collect_type_names(dict[str, Item]) == {"Item"}
            assert _collect_type_names(list[dict[str, Item]]) == {"Item"}

    class TestActorWiring:
        """The actor.py validation block must thread ``return_type`` through.

        We don't run the full strategy here — that's covered by the
        existing CodeAct integration suites. The narrow risk we want to catch
        is the wiring drifting (e.g. someone renames the field on
        ``ValidationContext`` and forgets the call site)."""

        def test_actor_passes_return_type_into_validation_context(self):
            """If actor.py stops forwarding return_type, the shadow validator
            silently becomes a no-op. Anchor the wiring with a textual check
            so a future refactor that drops the kwarg is caught here, not in
            production traces."""
            import inspect

            from nooa.runtime import actor

            src = inspect.getsource(actor)
            # actor.py builds a ValidationContext for the unified validator;
            # the contract is that it forwards `return_type=...` from the
            # current call. If this assertion fires, the wiring is broken
            # and the shadow validator no longer sees the return type.
            assert "return_type=return_type" in src, (
                "actor.py no longer forwards return_type into ValidationContext; "
                "ReturnTypeShadowValidator will silently no-op"
            )
            assert 'getattr(current_call, "return_type"' in src, (
                "actor.py no longer reads return_type from the current call"
            )
