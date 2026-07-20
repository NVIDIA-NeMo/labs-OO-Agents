# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BashSession timeout behavior.

Verifies that on timeout the session resets cleanly: cwd preserved,
commands work again immediately, and the whole cycle completes quickly.
"""

import time

import pytest

from nooa.tools._bash_session import BashSession


@pytest.fixture
async def session(tmp_path):
    """Create a bash session in a temp directory."""
    s = BashSession(cwd=tmp_path)
    await s.start()
    yield s
    await s.close()


class TestTimeoutRecovery:
    """Test that timeout triggers a fast reset and the session stays usable."""

    async def test_timeout_preserves_cwd(self, session, tmp_path):
        """After a timeout, the working directory should be preserved."""
        subdir = tmp_path / "mydir"
        subdir.mkdir()

        await session.run(f"cd {subdir}")
        assert session.cwd == subdir

        # This will timeout and reset
        await session.run("sleep 30", timeout=2)

        # cwd should still be the subdir (preserved across reset)
        out, _, _ = await session.run("pwd")
        assert str(subdir) in out, f"cwd lost after timeout. Got: {out!r}"

    async def test_timeout_returns_partial_output(self, session):
        """Output produced before timeout should be captured."""
        out, err, code = await session.run(
            "echo before_timeout; sleep 30",
            timeout=2,
        )
        assert "before_timeout" in out

    async def test_timeout_returns_quickly(self, session):
        """Timeout + reset should complete well within the timeout window."""
        start = time.monotonic()
        await session.run("sleep 60", timeout=2)
        elapsed = time.monotonic() - start
        # Should complete within timeout + reset overhead (not 60s!)
        assert elapsed < 12, f"Timeout took too long: {elapsed:.1f}s"

    async def test_timeout_exit_code(self, session):
        """Timed-out commands should return exit code 124."""
        _, _, code = await session.run("sleep 30", timeout=2)
        assert code == 124

    async def test_session_works_after_timeout(self, session):
        """After timeout+reset, session should still execute commands."""
        await session.run("sleep 30", timeout=2)

        out, _, code = await session.run("echo still_alive")
        assert code == 0
        assert "still_alive" in out

    async def test_multiple_timeouts_session_survives(self, session):
        """Session should survive multiple consecutive timeouts."""
        for _ in range(3):
            await session.run("sleep 30", timeout=2)

        out, _, code = await session.run("echo works")
        assert code == 0
        assert "works" in out

    async def test_timeout_kills_pipeline(self, session):
        """Timeout should kill all processes in a pipeline."""
        start = time.monotonic()
        await session.run("sleep 30 | cat", timeout=2)
        elapsed = time.monotonic() - start
        assert elapsed < 12, f"Pipeline timeout took too long: {elapsed:.1f}s"

        out, _, code = await session.run("echo alive")
        assert code == 0
        assert "alive" in out

    async def test_timeout_kills_subshell(self, session):
        """Timeout should kill commands in a subshell."""
        await session.run("(sleep 30)", timeout=2)

        out, _, code = await session.run("echo works")
        assert code == 0
        assert "works" in out

    async def test_env_preserved_after_timeout(self, session):
        """Env vars preserved when pgrep-based child kill succeeds."""
        await session.run("export TIMEOUT_TEST=hello")
        out, _, _ = await session.run("echo $TIMEOUT_TEST")
        assert "hello" in out

        # Timeout kills child — env preserved if pgrep works, lost if fallback to reset
        await session.run("sleep 30", timeout=2)
        out, _, code = await session.run("echo $TIMEOUT_TEST")
        assert code == 0  # Session works regardless
        # Env preservation is best-effort (requires pgrep)
        if "hello" not in out:
            import warnings

            warnings.warn(
                "pgrep-based recovery not available; env vars lost on timeout", stacklevel=2
            )

    async def test_timeout_with_nested_processes(self, session):
        """Deeply nested processes (bash -c 'sleep') are also killed."""
        await session.run("export NESTED_VAR=deep")
        await session.run("bash -c 'sleep 30'", timeout=2)

        out, _, code = await session.run("echo $NESTED_VAR")
        assert code == 0  # Session works regardless
        # Env preservation is best-effort (requires pgrep)
        if "deep" not in out:
            import warnings

            warnings.warn(
                "pgrep-based recovery not available; env vars lost on timeout", stacklevel=2
            )

    async def test_fast_command_after_timeout(self, session):
        """Fast commands work immediately after timeout recovery."""
        await session.run("sleep 30", timeout=2)
        # Should return instantly, not wait any residual time
        import time

        t0 = time.monotonic()
        out, _, code = await session.run("echo fast")
        elapsed = time.monotonic() - t0
        assert code == 0
        assert "fast" in out
        assert elapsed < 2, f"Post-timeout command took {elapsed:.1f}s"
