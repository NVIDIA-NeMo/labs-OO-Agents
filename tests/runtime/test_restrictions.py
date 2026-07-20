# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess

import pytest

from nooa.runtime.restrictions import (
    DEFAULT_BLOCKED_CALLS,
    DEFAULT_BLOCKED_MODULES,
    RESTRICTED_MODULES,
    RestrictionsConfig,
    is_from_blocked_module,
    match_blocked_module,
)


class TestConstants:
    def test_blocked_constants_are_well_formed(self):
        """Smoke test: constants have expected types and key entries."""
        assert isinstance(DEFAULT_BLOCKED_MODULES, frozenset)
        assert {"subprocess", "socket"} <= DEFAULT_BLOCKED_MODULES
        assert "time" in DEFAULT_BLOCKED_CALLS
        assert "sleep" in DEFAULT_BLOCKED_CALLS["time"]

    def test_blocked_modules_subset_of_restricted(self):
        assert DEFAULT_BLOCKED_MODULES.issubset(RESTRICTED_MODULES)


class TestMatchBlockedModule:
    """Tests for match_blocked_module parent-matching logic."""

    def test_exact_match(self):
        assert match_blocked_module("subprocess", DEFAULT_BLOCKED_MODULES) == "subprocess"

    def test_child_matches_parent(self):
        """asyncio.runners should match 'asyncio' if it were in the lookup."""
        lookup = frozenset({"asyncio"})
        assert match_blocked_module("asyncio.runners", lookup) == "asyncio"

    def test_parent_does_not_match_child(self):
        """'http' alone should NOT match 'http.client' in the lookup."""
        lookup = frozenset({"http.client"})
        assert match_blocked_module("http", lookup) is None

    def test_no_match(self):
        assert match_blocked_module("json", DEFAULT_BLOCKED_MODULES) is None

    def test_deeply_nested_child_matches(self):
        lookup = frozenset({"urllib"})
        assert match_blocked_module("urllib.request.urlretrieve", lookup) == "urllib"

    def test_works_with_dict_lookup(self):
        """match_blocked_module accepts both frozenset and dict."""
        assert match_blocked_module("time", DEFAULT_BLOCKED_CALLS) == "time"
        assert match_blocked_module("time.struct_time", DEFAULT_BLOCKED_CALLS) == "time"

    def test_empty_lookup(self):
        assert match_blocked_module("subprocess", frozenset()) is None


class TestIsFromBlockedModule:
    """Tests for is_from_blocked_module with module objects and functions."""

    def test_module_object_blocked(self):
        assert is_from_blocked_module(subprocess, DEFAULT_BLOCKED_MODULES) is True

    def test_module_object_not_blocked(self):
        import json

        assert is_from_blocked_module(json, DEFAULT_BLOCKED_MODULES) is False

    def test_function_from_blocked_module(self):
        assert is_from_blocked_module(subprocess.run, DEFAULT_BLOCKED_MODULES) is True

    def test_function_from_non_blocked_module(self):
        import json

        assert is_from_blocked_module(json.loads, DEFAULT_BLOCKED_MODULES) is False

    def test_child_module_blocked_by_parent(self):
        """http.client should be blocked if 'http' is in the set (parent match)."""
        import http.client

        blocked = frozenset({"http"})
        assert is_from_blocked_module(http.client, blocked) is True

    def test_parent_module_not_blocked_by_child(self):
        """http should NOT be blocked if only 'http.client' is in the set."""
        import http

        assert is_from_blocked_module(http, frozenset({"http.client"})) is False

    def test_object_without_module_attr(self):
        assert is_from_blocked_module(42, DEFAULT_BLOCKED_MODULES) is False

    def test_builtin_open_not_blocked(self):
        """open() has __module__='io' which is not in DEFAULT_BLOCKED_MODULES."""
        assert is_from_blocked_module(open, DEFAULT_BLOCKED_MODULES) is False


class TestRestrictionsConfig:
    def test_defaults(self):
        rc = RestrictionsConfig()
        assert rc.blocked_modules == DEFAULT_BLOCKED_MODULES
        assert rc.blocked_calls == DEFAULT_BLOCKED_CALLS
        assert rc.restricted_imports == frozenset()

    def test_frozen(self):
        from pydantic import ValidationError

        rc = RestrictionsConfig()
        with pytest.raises(ValidationError):
            rc.blocked_modules = frozenset()
