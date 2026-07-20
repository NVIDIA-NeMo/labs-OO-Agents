# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast
import subprocess
import time

import pytest

from nooa.runtime.code_validator import (
    BlockingCallValidator,
    ValidationContext,
)
from nooa.runtime.restrictions import DEFAULT_BLOCKED_MODULES, RestrictionsConfig


@pytest.fixture
def validator():
    return BlockingCallValidator()


class TestFullyBlockedModules:
    """Tests for modules in DEFAULT_BLOCKED_MODULES."""

    def test_reject_subprocess_run(self, validator):
        code = "subprocess.run(['ls'])"
        ctx = ValidationContext(exec_globals={"subprocess": subprocess})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0
        assert "subprocess" in issues[0].message

    def test_reject_subprocess_aliased(self, validator):
        code = "sp.run(['ls'])"
        ctx = ValidationContext(exec_globals={"sp": subprocess})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0

    def test_reject_subprocess_function_directly_imported(self, validator):
        code = "run(['ls'])"
        ctx = ValidationContext(exec_globals={"run": subprocess.run})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0

    def test_reject_socket_connect(self, validator):
        import socket

        code = "sock.connect(('localhost', 80))"
        ctx = ValidationContext(exec_globals={"sock": socket})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0


class TestPartiallyBlockedCalls:
    """Tests for specific calls on allowed modules."""

    def test_reject_time_sleep(self, validator):
        code = "time.sleep(5)"
        ctx = ValidationContext(exec_globals={"time": time})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0
        assert "sleep" in issues[0].message

    def test_allow_time_time(self, validator):
        code = "time.time()"
        ctx = ValidationContext(exec_globals={"time": time})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0

    def test_allow_time_monotonic(self, validator):
        code = "time.monotonic()"
        ctx = ValidationContext(exec_globals={"time": time})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0

    def test_reject_os_system(self, validator):
        import os

        code = "os.system('ls')"
        ctx = ValidationContext(exec_globals={"os": os})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0

    def test_allow_os_path_join(self, validator):
        import os

        code = "os.path.join('a', 'b')"
        ctx = ValidationContext(exec_globals={"os": os})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0

    def test_reject_asyncio_run(self, validator):
        import asyncio

        code = "asyncio.run(coro())"
        ctx = ValidationContext(exec_globals={"asyncio": asyncio})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0


class TestLocalVariableTracking:
    """Tests for instance methods on locally-created objects."""

    def test_reject_thread_join(self, validator):
        import threading

        code = "t = threading.Thread(target=fn)\nt.join()"
        ctx = ValidationContext(exec_globals={"threading": threading})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0
        assert "join" in issues[0].message

    def test_reject_lock_acquire(self, validator):
        import threading

        code = "lock = threading.Lock()\nlock.acquire()"
        ctx = ValidationContext(exec_globals={"threading": threading})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0

    def test_allow_str_join(self, validator):
        code = "','.join(['a', 'b'])"
        ctx = ValidationContext(exec_globals={})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0


class TestAllowedPatterns:
    """Patterns that should NOT be blocked."""

    def test_allow_asyncio_sleep(self, validator):
        import asyncio

        wrapped = "async def _():\n    await asyncio.sleep(1)"
        ctx = ValidationContext(exec_globals={"asyncio": asyncio})
        issues = validator.validate(ast.parse(wrapped), ctx)
        assert len(issues) == 0

    def test_allow_asyncio_gather(self, validator):
        import asyncio

        code = "async def _():\n    await asyncio.gather(a(), b())"
        ctx = ValidationContext(exec_globals={"asyncio": asyncio})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0

    def test_allow_json_loads(self, validator):
        import json

        code = "json.loads(data)"
        ctx = ValidationContext(exec_globals={"json": json})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0

    def test_allow_task_result_after_await(self, validator):
        """task.result() after awaiting the task is safe — task is already done."""
        import asyncio

        code = "task = asyncio.create_task(work())\nawait task\nresult = task.result()"
        ctx = ValidationContext(exec_globals={"asyncio": asyncio})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0

    def test_allow_unresolvable_call(self, validator):
        code = "unknown_func()"
        ctx = ValidationContext(exec_globals={})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0


class TestRuntimePatchResponsibility:
    """Patterns handled by runtime patches in async_safety.py, NOT by BlockingCallValidator.

    These tests document the contract: the AST validator intentionally does not
    flag these patterns because they are caught at runtime instead.
    """

    def test_executor_submit_result_not_flagged(self, validator):
        """executor.submit().result() is a deadlock risk but handled by runtime patch."""
        code = (
            "executor = concurrent.futures.ThreadPoolExecutor()\n"
            "future = executor.submit(lambda: 42)\n"
            "future.result()"
        )
        ctx = ValidationContext(exec_globals={"concurrent": __import__("concurrent.futures")})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0, (
            "executor.submit().result() should NOT be flagged by BlockingCallValidator; "
            "this pattern is handled by runtime patches in async_safety.py"
        )


class TestChainedCallDetection:
    """Tests for chained calls like asyncio.get_event_loop().run_until_complete()."""

    def test_reject_run_until_complete_chained(self, validator):
        import asyncio

        code = "asyncio.get_event_loop().run_until_complete(coro())"
        ctx = ValidationContext(exec_globals={"asyncio": asyncio})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0
        assert "run_until_complete" in issues[0].message

    def test_reject_run_forever_chained(self, validator):
        import asyncio

        code = "asyncio.get_event_loop().run_forever()"
        ctx = ValidationContext(exec_globals={"asyncio": asyncio})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0
        assert "run_forever" in issues[0].message

    def test_allow_chained_non_blocked_call(self, validator):
        import asyncio

        code = "asyncio.get_event_loop().create_task(coro())"
        ctx = ValidationContext(exec_globals={"asyncio": asyncio})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0


class TestModuleResolutionBoundary:
    """Edge cases for __module__ resolution."""

    def test_builtin_open_not_blocked_despite_io_module(self, validator):
        """open() has __module__='io', but 'io' is not in blocked lists."""
        code = "open('file.txt')"
        ctx = ValidationContext(exec_globals={"open": open})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0

    def test_directly_imported_sleep_blocked(self, validator):
        """sleep imported from time should be blocked via __module__ resolution."""
        code = "sleep(5)"
        ctx = ValidationContext(exec_globals={"sleep": time.sleep})
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) > 0


class TestConfigOverride:
    """Tests that config-provided blocked sets are respected."""

    def test_custom_blocked_modules(self):
        code = "subprocess.run(['ls'])"
        ctx = ValidationContext(exec_globals={"subprocess": subprocess})
        validator = BlockingCallValidator(
            restrictions=RestrictionsConfig(blocked_modules=frozenset(), blocked_calls={}),
        )
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0

    def test_custom_blocked_calls(self):
        code = "time.sleep(5)"
        ctx = ValidationContext(exec_globals={"time": time})
        validator = BlockingCallValidator(
            restrictions=RestrictionsConfig(
                blocked_modules=DEFAULT_BLOCKED_MODULES, blocked_calls={}
            ),
        )
        issues = validator.validate(ast.parse(code), ctx)
        assert len(issues) == 0
