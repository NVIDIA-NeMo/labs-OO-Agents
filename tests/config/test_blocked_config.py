# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess

import pytest

from nooa.config.strategy_config import CodeActConfig
from nooa.runtime.code_validator import (
    UnifiedCodeValidator,
    ValidationContext,
    ValidationError,
)
from nooa.runtime.restrictions import (
    DEFAULT_BLOCKED_CALLS,
    DEFAULT_BLOCKED_MODULES,
    RestrictionsConfig,
)


def test_codeact_config_has_blocked_modules():
    config = CodeActConfig()
    assert config.restrictions.blocked_modules == DEFAULT_BLOCKED_MODULES


def test_codeact_config_has_blocked_calls():
    config = CodeActConfig()
    assert config.restrictions.blocked_calls == DEFAULT_BLOCKED_CALLS


def test_codeact_config_blocked_modules_override():
    custom = DEFAULT_BLOCKED_MODULES - {"subprocess"}
    config = CodeActConfig(restrictions=RestrictionsConfig(blocked_modules=custom))
    assert "subprocess" not in config.restrictions.blocked_modules
    assert "socket" in config.restrictions.blocked_modules


def test_codeact_config_merge_with_blocked():
    base = CodeActConfig()
    override = CodeActConfig(
        restrictions=RestrictionsConfig(blocked_modules=frozenset({"subprocess"}))
    )
    merged = base.merge_with(override)
    assert merged.restrictions.blocked_modules == frozenset({"subprocess"})
    assert merged.max_iterations is None


def test_restrictions_config_reusable():
    """RestrictionsConfig can be constructed independently for reuse."""
    rc = RestrictionsConfig()
    assert rc.blocked_modules == DEFAULT_BLOCKED_MODULES
    assert rc.blocked_calls == DEFAULT_BLOCKED_CALLS

    custom = RestrictionsConfig(blocked_modules=frozenset({"subprocess"}), blocked_calls={})
    assert custom.blocked_modules == frozenset({"subprocess"})
    assert custom.blocked_calls == {}


def test_validator_uses_config_blocked_modules():
    """Validator with empty blocked_modules allows subprocess."""
    validator = UnifiedCodeValidator(
        restrictions=RestrictionsConfig(blocked_modules=frozenset(), blocked_calls={})
    )
    ctx = ValidationContext(exec_globals={"subprocess": subprocess})
    # Should NOT raise — subprocess is not blocked
    validator.validate("subprocess.run(['ls'])", ctx)


def test_validator_uses_default_blocked_modules():
    """Validator with defaults blocks subprocess."""
    validator = UnifiedCodeValidator()
    ctx = ValidationContext(exec_globals={"subprocess": subprocess})
    with pytest.raises(ValidationError):
        validator.validate("subprocess.run(['ls'])", ctx)
