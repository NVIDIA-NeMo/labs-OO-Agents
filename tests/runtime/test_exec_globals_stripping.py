# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import time

from nooa.runtime.actor import _strip_blocked_modules


class TestExecGlobalsStripping:
    def test_strips_blocked_module(self):
        globs = {"subprocess": subprocess, "json": __import__("json")}
        result = _strip_blocked_modules(globs, frozenset({"subprocess"}))
        assert "subprocess" not in result
        assert "json" in result

    def test_strips_aliased_module(self):
        globs = {"sp": subprocess}
        result = _strip_blocked_modules(globs, frozenset({"subprocess"}))
        assert "sp" not in result

    def test_strips_function_from_blocked_module(self):
        globs = {"run": subprocess.run}
        result = _strip_blocked_modules(globs, frozenset({"subprocess"}))
        assert "run" not in result

    def test_keeps_non_blocked_module(self):
        globs = {"time": time}
        result = _strip_blocked_modules(globs, frozenset({"subprocess"}))
        assert "time" in result

    def test_keeps_non_module_objects(self):
        globs = {"x": 42, "name": "hello"}
        result = _strip_blocked_modules(globs, frozenset({"subprocess"}))
        assert globs == result

    def test_keeps_parent_module_when_child_is_blocked(self):
        """Parent module 'http' should NOT be stripped if 'http.client' is blocked.

        The parent may have legitimate non-blocking members (e.g. http.HTTPStatus).
        The validator catches 'http.client.X()' calls via child-of-blocked AST matching.
        """
        import http

        globs = {"http": http}
        result = _strip_blocked_modules(globs, frozenset({"http.client"}))
        assert "http" in result

    def test_strips_child_module_when_parent_is_blocked(self):
        """Child module 'http.client' should be stripped if 'http' is blocked."""
        import http.client

        globs = {"hc": http.client}
        result = _strip_blocked_modules(globs, frozenset({"http"}))
        assert "hc" not in result

    def test_empty_blocked_set_strips_nothing(self):
        globs = {"subprocess": subprocess}
        result = _strip_blocked_modules(globs, frozenset())
        assert "subprocess" in result

    def test_default_stripping_matches_default_validation(self):
        """Stripping defaults and validation defaults should agree.

        Both layers should use DEFAULT_BLOCKED_MODULES when no override is
        provided, so a module stripped by one layer is also rejected by the other.
        """
        from nooa.runtime.code_validator import BlockingCallValidator
        from nooa.runtime.restrictions import DEFAULT_BLOCKED_MODULES

        # Stripping removes subprocess by default
        globs = {"subprocess": subprocess}
        stripped = _strip_blocked_modules(globs, DEFAULT_BLOCKED_MODULES)
        assert "subprocess" not in stripped

        # Validator also rejects subprocess by default
        import ast

        from nooa.runtime.code_validator import ValidationContext

        validator = BlockingCallValidator()
        issues = validator.validate(
            ast.parse("subprocess.run(['ls'])"),
            ValidationContext(exec_globals={"subprocess": subprocess}),
        )
        assert len(issues) > 0
