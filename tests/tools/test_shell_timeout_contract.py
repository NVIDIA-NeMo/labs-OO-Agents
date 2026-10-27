# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public shell timeout configuration, discovery, and recovery contracts."""

import pytest

from nooa.agentdoc import doc
from nooa.tools.shell_tools import ShellResult, ShellTools


def test_timeout_default_and_instance_documentation():
    assert ShellTools().default_timeout == 30.0
    configured = ShellTools(default_timeout=300)
    other = ShellTools(default_timeout=15)
    class_documentation = doc(ShellTools)
    assert "default_timeout" in class_documentation
    assert "30.0" in class_documentation
    assert "None uses this instance's default_timeout" in class_documentation
    assert "await shell.run" in class_documentation
    assert "default_timeout: float = 300.0" in doc(configured)
    assert "default_timeout: float = 15.0" in doc(other)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True])
def test_default_timeout_requires_positive_finite_seconds(value):
    with pytest.raises(ValueError, match="positive finite"):
        ShellTools(default_timeout=value)


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
@pytest.mark.parametrize("options", [{}, {"timeout": None}], ids=["omitted", "none"])
async def test_inherited_timeout_preserves_partial_output_and_shell_recovers(
    tmp_path, streaming, options
):
    shell = ShellTools(cwd=str(tmp_path), default_timeout=0.1)
    try:
        command = "printf 'partial\\n'; sleep 2"
        if streaming:
            events = [event async for event in shell.run_stream(command, **options)]
            result = events[-1]
            output = "".join(event.text for event in events if event.kind == "stdout")
        else:
            result = await shell.run(command, **options)
            output = result.stdout
            assert result.timeout_seconds == 0.1
            assert "timed_out=True" in str(result)
            assert "timeout_seconds=0.1" in str(result)
        assert result.returncode == 124
        assert result.timed_out
        assert "partial" in output
        assert (await shell.run("printf ready", timeout=5)).stdout == "ready"
    finally:
        await shell.close()


@pytest.mark.parametrize("streaming", [False, True], ids=["buffered", "streaming"])
async def test_explicit_timeout_overrides_instance_default(tmp_path, streaming):
    shell = ShellTools(cwd=str(tmp_path), default_timeout=0.05)
    try:
        command = "sleep 0.2; printf complete"
        if streaming:
            events = [event async for event in shell.run_stream(command, timeout=5)]
            result = events[-1]
            output = "".join(event.text for event in events if event.kind == "stdout")
        else:
            result = await shell.run(command, timeout=5)
            output = result.stdout
            assert result.timeout_seconds == 5
        assert result.returncode == 0
        assert not result.timed_out
        assert output == "complete"
    finally:
        await shell.close()


def test_result_display_distinguishes_timeout_from_exit_124():
    failed = ShellResult("partial", "failure", 124)
    timed = ShellResult("partial", "failure", 124, timed_out=True, timeout_seconds=300)
    assert "returncode=124" in repr(failed)
    assert "return_code=" not in repr(failed)
    assert "timed_out" not in repr(failed)
    assert "timed_out=True" in str(timed)
    assert "timeout_seconds=300" in timed.text
    assert timed.stdout == failed.stdout == "partial"
    assert timed.stderr == failed.stderr == "failure"
    assert not timed.success
