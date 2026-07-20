# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for configurable import restrictions.

Tests the new restricted_imports deny-list feature on RestrictionsConfig,
SecurityValidator, and the ValidationContext integration.
"""

import pytest

from nooa.runtime.code_validator import (
    UnifiedCodeValidator,
    ValidationContext,
    ValidationError,
)
from nooa.runtime.restrictions import (
    DEFAULT_BLOCKED_MODULES,
    RestrictionsConfig,
)

# =============================================================================
# RestrictionsConfig — restricted_imports field
# =============================================================================


class TestRestrictedImportsConfig:
    """Tests for the new restricted_imports field on RestrictionsConfig."""

    def test_default_restricted_imports_is_empty(self):
        """Default restricted_imports should be empty (all imports allowed)."""
        rc = RestrictionsConfig()
        assert rc.restricted_imports == frozenset()
        assert isinstance(rc.restricted_imports, frozenset)

    def test_default_restricted_imports_constant_is_empty(self):
        """DEFAULT_RESTRICTED_IMPORTS constant should be empty — no restrictions by default."""
        from nooa.runtime.restrictions import DEFAULT_RESTRICTED_IMPORTS

        assert DEFAULT_RESTRICTED_IMPORTS == frozenset()

    def test_custom_restricted_imports(self):
        """Developers can set a custom restricted_imports set."""
        rc = RestrictionsConfig(restricted_imports=frozenset({"numpy", "pandas"}))
        assert rc.restricted_imports == frozenset({"numpy", "pandas"})

    def test_empty_restricted_imports_allows_all(self):
        """Empty frozenset means no import restrictions."""
        rc = RestrictionsConfig(restricted_imports=frozenset())
        assert rc.restricted_imports == frozenset()

    def test_restricted_imports_frozen(self):
        """restricted_imports field should be immutable (frozen config)."""
        from pydantic import ValidationError as PydanticValidationError

        rc = RestrictionsConfig()
        with pytest.raises(PydanticValidationError):
            rc.restricted_imports = frozenset()

    def test_importlib_in_blocked_calls(self):
        """importlib.import_module should be in DEFAULT_BLOCKED_CALLS."""
        from nooa.runtime.restrictions import DEFAULT_BLOCKED_CALLS

        assert "importlib" in DEFAULT_BLOCKED_CALLS
        assert "import_module" in DEFAULT_BLOCKED_CALLS["importlib"]


# =============================================================================
# SecurityValidator — deny-list import validation
# =============================================================================


class TestSecurityValidatorRestrictedImports:
    """Tests for SecurityValidator using restricted_imports deny list."""

    @pytest.fixture
    def validator(self) -> UnifiedCodeValidator:
        return UnifiedCodeValidator()

    def test_unrestricted_import_allowed(self, validator):
        """Modules not in restricted_imports should be importable."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset({"os", "sys"}),
        )
        code = "import json"
        validator.validate(code, context)  # should not raise

    def test_restricted_import_rejected(self, validator):
        """Modules in restricted_imports should be rejected."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset({"os", "sys"}),
        )
        code = "import os"
        with pytest.raises(ValidationError, match="os.*restricted"):
            validator.validate(code, context)

    def test_restricted_from_import_rejected(self, validator):
        """from-import of restricted modules should be rejected."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset({"os", "sys"}),
        )
        code = "from os import path"
        with pytest.raises(ValidationError, match="os.*restricted"):
            validator.validate(code, context)

    def test_restricted_child_module_rejected(self, validator):
        """Child modules of restricted parents should be rejected."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset({"os"}),
        )
        code = "import os.path"
        with pytest.raises(ValidationError, match="os.*restricted"):
            validator.validate(code, context)

    def test_empty_restricted_allows_everything(self, validator):
        """Empty restricted_imports means all imports are allowed."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset(),
        )
        # os would normally be restricted by default, but empty set = allow all
        code = "import os"
        validator.validate(code, context)  # should not raise

    def test_empty_restricted_allows_any_stdlib(self, validator):
        """With empty restricted_imports, any stdlib module is importable."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset(),
        )
        for mod in ["json", "csv", "re", "collections", "itertools", "math", "os", "sys"]:
            code = f"import {mod}"
            validator.validate(code, context)  # should not raise

    def test_blocked_modules_still_blocked_with_empty_restricted(self, validator):
        """blocked_modules are always blocked regardless of restricted_imports."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset(),
            blocked_modules=DEFAULT_BLOCKED_MODULES,
        )
        # subprocess is in DEFAULT_BLOCKED_MODULES — should still be blocked
        code = "import subprocess"
        with pytest.raises(ValidationError):
            validator.validate(code, context)

    def test_default_config_allows_os(self, validator):
        """With default RestrictionsConfig (empty deny list), 'os' is allowed."""
        rc = RestrictionsConfig()
        context = ValidationContext(
            code="",
            restricted_imports=rc.restricted_imports,
        )
        code = "import os"
        validator.validate(code, context)  # should not raise

    def test_default_config_allows_json(self, validator):
        """With default RestrictionsConfig, 'json' should be allowed."""
        rc = RestrictionsConfig()
        context = ValidationContext(
            code="",
            restricted_imports=rc.restricted_imports,
        )
        code = "import json"
        validator.validate(code, context)  # should not raise

    def test_default_config_allows_csv(self, validator):
        """With default RestrictionsConfig, 'csv' should be allowed."""
        rc = RestrictionsConfig()
        context = ValidationContext(
            code="",
            restricted_imports=rc.restricted_imports,
        )
        code = "import csv"
        validator.validate(code, context)  # should not raise

    def test_parent_not_restricted_by_child(self, validator):
        """If only 'os.path' is restricted, 'os' itself should be allowed."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset({"os.path"}),
        )
        code = "import os"
        validator.validate(code, context)  # should not raise


# =============================================================================
# ValidationContext — restricted_imports field
# =============================================================================


class TestValidationContextRestrictedImports:
    """Tests for the restricted_imports field on ValidationContext."""

    def test_default_restricted_imports_is_empty(self):
        """ValidationContext defaults to empty restricted_imports."""
        ctx = ValidationContext()
        assert ctx.restricted_imports == frozenset()

    def test_restricted_imports_set_explicitly(self):
        """restricted_imports can be set explicitly."""
        ctx = ValidationContext(restricted_imports=frozenset({"os", "sys"}))
        assert ctx.restricted_imports == frozenset({"os", "sys"})


class TestProcessGlobalRestrictedImports:
    """Tests for set_restricted_imports / get_restricted_imports process-global API."""

    def setup_method(self):
        """Clear global override before each test."""
        from nooa.runtime.restrictions import set_restricted_imports

        set_restricted_imports(None)

    def teardown_method(self):
        """Clear global override after each test."""
        from nooa.runtime.restrictions import set_restricted_imports

        set_restricted_imports(None)

    def test_default_is_none(self):
        from nooa.runtime.restrictions import get_restricted_imports

        assert get_restricted_imports() is None

    def test_set_and_get(self):
        from nooa.runtime.restrictions import (
            get_restricted_imports,
            set_restricted_imports,
        )

        deny = frozenset({"os", "sys"})
        set_restricted_imports(deny)
        assert get_restricted_imports() == deny

    def test_clear_with_none(self):
        from nooa.runtime.restrictions import (
            get_restricted_imports,
            set_restricted_imports,
        )

        set_restricted_imports(frozenset({"os"}))
        set_restricted_imports(None)
        assert get_restricted_imports() is None

    def test_global_overrides_config_default(self):
        from nooa.runtime.restrictions import set_restricted_imports

        set_restricted_imports(frozenset({"numpy", "pandas"}))
        rc = RestrictionsConfig()
        assert rc.restricted_imports == frozenset({"numpy", "pandas"})

    def test_explicit_config_overrides_global(self):
        """If restricted_imports is explicitly passed, global is ignored."""
        from nooa.runtime.restrictions import set_restricted_imports

        set_restricted_imports(frozenset({"numpy"}))
        rc = RestrictionsConfig(restricted_imports=frozenset({"os"}))
        assert rc.restricted_imports == frozenset({"os"})

    def test_global_empty_frozenset_overrides_default(self):
        """Setting frozenset() globally should override field default."""
        from nooa.runtime.restrictions import set_restricted_imports

        set_restricted_imports(frozenset())
        rc = RestrictionsConfig()
        assert rc.restricted_imports == frozenset()

    def test_no_global_uses_field_default(self):
        """Without global override, field default is used."""
        rc = RestrictionsConfig()
        assert rc.restricted_imports == frozenset()

    def test_global_affects_validation(self):
        """Process-global restricted_imports flows through to SecurityValidator."""
        from nooa.runtime.restrictions import set_restricted_imports

        set_restricted_imports(frozenset({"pandas"}))
        rc = RestrictionsConfig()

        context = ValidationContext(
            code="",
            restricted_imports=rc.restricted_imports,
        )
        validator = UnifiedCodeValidator()

        # pandas should be restricted
        with pytest.raises(ValidationError):
            validator.validate("import pandas", context)

        # json should be allowed
        validator.validate("import json", context)


class TestSetRestrictedImportsBlocked:
    """Tests that set_restricted_imports/get_restricted_imports are blocked in agent code."""

    @pytest.fixture
    def validator(self) -> UnifiedCodeValidator:
        return UnifiedCodeValidator()

    def test_direct_call_blocked(self, validator):
        """Direct call to set_restricted_imports() is blocked via FORBIDDEN_BUILTINS."""
        context = ValidationContext(code="")
        code = "set_restricted_imports(frozenset())"
        with pytest.raises(ValidationError, match="set_restricted_imports.*forbidden"):
            validator.validate(code, context)

    def test_get_direct_call_blocked(self, validator):
        """Direct call to get_restricted_imports() is blocked via FORBIDDEN_BUILTINS."""
        context = ValidationContext(code="")
        code = "get_restricted_imports()"
        with pytest.raises(ValidationError, match="get_restricted_imports.*forbidden"):
            validator.validate(code, context)

    def test_aliased_import_call_blocked(self, validator):
        """Aliased import of set_restricted_imports is blocked."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset(),  # allow the import
        )
        code = """from nooa.runtime.restrictions import set_restricted_imports as foo
foo(frozenset())"""
        with pytest.raises(ValidationError, match="foo.*forbidden.*set_restricted_imports"):
            validator.validate(code, context)

    def test_attribute_call_blocked(self, validator):
        """Attribute-style call r.set_restricted_imports() is blocked."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset(),  # allow the import
        )
        code = """import nooa.runtime.restrictions as r
r.set_restricted_imports(frozenset())"""
        with pytest.raises(ValidationError, match="set_restricted_imports.*forbidden"):
            validator.validate(code, context)

    def test_attribute_get_call_blocked(self, validator):
        """Attribute-style call r.get_restricted_imports() is blocked."""
        context = ValidationContext(
            code="",
            restricted_imports=frozenset(),
        )
        code = """import nooa.runtime.restrictions as r
r.get_restricted_imports()"""
        with pytest.raises(ValidationError, match="get_restricted_imports.*forbidden"):
            validator.validate(code, context)
