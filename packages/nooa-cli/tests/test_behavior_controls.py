# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Failed live configuration must not persist broken workspace preferences."""

import asyncio
from types import SimpleNamespace

import pytest
from nooa_cli.interactive.controls import MemoryControl, ReflectionControl
from nooa_cli.interactive.options import SessionOptions


@pytest.mark.parametrize(
    "control_type,field",
    [(MemoryControl, "memory_agents"), (ReflectionControl, "reflection_agents")],
)
@pytest.mark.parametrize("cancelled", [False, True])
async def test_failed_control_preserves_saved_and_live_preferences(
    tmp_path, control_type, field, cancelled
):
    config = SessionOptions(working_dir=str(tmp_path))
    agent = SimpleNamespace(_memory_key="current", memory=object())
    settings = tmp_path / ".nooa/settings.yaml"
    settings.parent.mkdir()
    before = "coding:\n  default_model: original\n"
    settings.write_text(before)

    def configure():
        assert "current" in getattr(config, field)
        raise asyncio.CancelledError() if cancelled else RuntimeError("apply failed")

    control = control_type(agent, config, configure_memory=configure, workspace=tmp_path)
    if cancelled:
        with pytest.raises(asyncio.CancelledError):
            await control.run(["on"])
    else:
        result = await control.run(["on"])
        assert not result.success and "apply failed" in str(result)
    assert settings.read_text() == before
    assert getattr(config, field) == {}
    assert config.memory == "off" and config.reflection is False


async def test_memory_control_changes_only_current_agents_default(tmp_path):
    config = SessionOptions(working_dir=str(tmp_path), memory="off")
    agent = SimpleNamespace(_memory_key="current")
    control = MemoryControl(agent, config, configure_memory=lambda: None, workspace=tmp_path)
    result = await control.run(["on"])
    assert result.success, str(result)
    assert config.memory == "off"
    assert config.memory_agents == {"current": "project"}


async def test_persistence_failure_reports_applied_memory_change(tmp_path, monkeypatch):
    config = SessionOptions(working_dir=str(tmp_path))
    control = MemoryControl(
        SimpleNamespace(_memory_key="current"),
        config,
        configure_memory=lambda: None,
        workspace=tmp_path,
    )

    def fail(*args):
        raise OSError("read-only workspace")

    monkeypatch.setattr(control, "_persist_agent_preference", fail)
    result = await control.run(["on"])
    assert result.success
    assert "could not save" in str(result)
    assert config.memory_agents == {"current": "project"}
