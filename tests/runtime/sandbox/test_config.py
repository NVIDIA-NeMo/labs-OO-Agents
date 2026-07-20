# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for sandbox configuration and spec resolution (no forking)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nooa.config import CodeActConfig
from nooa.runtime.sandbox.config import (
    FileRule,
    SandboxConfig,
    resolve_spec,
)
from nooa.runtime.sandbox.context_block import render_sandbox_block


def test_codeact_defaults_keep_inprocess_backend():
    cfg = CodeActConfig()
    assert cfg.execution_backend == "inprocess"
    assert isinstance(cfg.sandbox, SandboxConfig)


def test_sandbox_config_defaults_are_safe():
    sb = SandboxConfig()
    assert sb.filesystem is True
    assert sb.network is False  # internet off by default
    assert sb.require is True  # fail closed by default
    assert sb.max_memory_mb == 0 and sb.max_cpu_seconds == 0


def test_sandbox_config_is_frozen():
    sb = SandboxConfig()
    with pytest.raises(ValidationError):
        sb.network = True  # type: ignore[misc]


def test_negative_caps_rejected():
    with pytest.raises(ValidationError):
        SandboxConfig(max_memory_mb=-1)
    with pytest.raises(ValidationError):
        SandboxConfig(max_cpu_seconds=-5)


def test_resolve_spec_workspace_is_read_write():
    spec = resolve_spec(SandboxConfig(workspace="/work", system_paths=False))
    rw = [r for r in spec.landlock_rules if r.path == "/work"]
    assert len(rw) == 1 and rw[0].write is True
    assert spec.block_network is True  # network False -> block


def test_resolve_spec_read_only_allow_rule():
    spec = resolve_spec(
        SandboxConfig(
            system_paths=False,
            allow=(FileRule(path="/data", access="read"),),
        )
    )
    r = [x for x in spec.landlock_rules if x.path == "/data"]
    assert len(r) == 1 and r[0].write is False


def test_resolve_spec_read_write_upgrades_read():
    spec = resolve_spec(
        SandboxConfig(
            system_paths=False,
            workspace="/shared",
            allow=(FileRule(path="/shared", access="read"),),
        )
    )
    r = [x for x in spec.landlock_rules if x.path == "/shared"]
    assert len(r) == 1 and r[0].write is True  # read_write wins over read


def test_resolve_spec_network_allowed_does_not_block():
    spec = resolve_spec(SandboxConfig(network=True))
    assert spec.block_network is False


def test_resolve_spec_filesystem_disabled_has_no_rules():
    spec = resolve_spec(SandboxConfig(filesystem=False))
    assert spec.filesystem is False
    assert spec.landlock_rules == ()


def test_context_block_lists_only_active_guards():
    block = render_sandbox_block(SandboxConfig(network=True, filesystem=False), cell_timeout=None)
    assert "Network: disabled" not in block  # network allowed -> omitted
    assert "Filesystem" not in block  # filesystem off -> omitted
    assert "picklable" in block  # always present


def test_context_block_includes_active_guards():
    block = render_sandbox_block(
        SandboxConfig(workspace="/w", network=False, max_memory_mb=512, max_cpu_seconds=30),
        cell_timeout=60.0,
    )
    assert "60s" in block
    assert "512 MB" in block
    assert "30s of CPU" in block
    assert "/w" in block
    assert "Network: disabled" in block


def test_merge_with_preserves_sandbox_fields():
    base = CodeActConfig()
    override = CodeActConfig(execution_backend="sandbox")
    merged = base.merge_with(override)
    assert merged.execution_backend == "sandbox"
