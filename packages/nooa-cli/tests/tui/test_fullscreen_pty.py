# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PTY-level terminal ownership checks for fullscreen mode."""

from __future__ import annotations

import errno
import os
import select
import struct
import subprocess
import sys
import termios
import textwrap
import time

import pytest


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="requires a POSIX pseudo-terminal")
def test_fullscreen_enters_and_leaves_alternate_screen_on_normal_exit() -> None:
    """A real prompt_toolkit run restores the primary screen before exit."""
    script = textwrap.dedent(
        """
        import asyncio

        from nooa_cli.tui.config import DisplayMode
        from nooa_cli.tui.tui_application import TUIApplication

        async def main():
            app = TUIApplication(display_mode=DisplayMode.FULLSCREEN)

            async def exit_after_first_frame():
                await asyncio.sleep(0.05)
                app._app.exit()

            asyncio.create_task(exit_after_first_frame())
            await app.run_async()

        asyncio.run(main())
        """
    )
    master_fd, slave_fd = os.openpty()
    env = {**os.environ, "TERM": "xterm-256color"}
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        env=env,
    )
    os.close(slave_fd)
    output = bytearray()
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master_fd, 65_536)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                output.extend(chunk)
            if process.poll() is not None and not readable:
                break
        try:
            return_code = process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=5)
            pytest.fail(f"fullscreen child did not exit; output={bytes(output)!r}")
    finally:
        os.close(master_fd)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    raw = bytes(output)
    assert return_code == 0, raw.decode(errors="replace")
    entered = raw.find(b"\x1b[?1049h")
    restored = raw.rfind(b"\x1b[?1049l")
    assert entered >= 0, raw
    assert restored > entered, raw


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="requires a POSIX pseudo-terminal")
def test_one_column_fullscreen_never_emits_a_physically_wide_grapheme() -> None:
    """The terminal receives the narrow projection, not the retained CJK source."""
    script = textwrap.dedent(
        """
        import asyncio

        from nooa_cli.tui.config import DisplayMode
        from nooa_cli.tui.tui_application import TUIApplication

        async def main():
            app = TUIApplication(display_mode=DisplayMode.FULLSCREEN)
            app.emit_block("界")

            async def exit_after_frames():
                await asyncio.sleep(0.1)
                app._app.exit()

            asyncio.create_task(exit_after_frames())
            await app.run_async()

        asyncio.run(main())
        """
    )
    master_fd, slave_fd = os.openpty()
    # Force one physical column so the renderer must use its viewport-only
    # fallback rather than trusting prompt_toolkit width bookkeeping.
    import fcntl

    fcntl.ioctl(slave_fd, termios.TIOCSWINSZ, struct.pack("HHHH", 8, 1, 0, 0))
    env = {**os.environ, "TERM": "xterm-256color"}
    process = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        close_fds=True,
        env=env,
    )
    os.close(slave_fd)
    output = bytearray()
    deadline = time.monotonic() + 10
    try:
        while time.monotonic() < deadline:
            readable, _, _ = select.select([master_fd], [], [], 0.1)
            if readable:
                try:
                    chunk = os.read(master_fd, 65_536)
                except OSError as exc:
                    if exc.errno == errno.EIO:
                        break
                    raise
                if not chunk:
                    break
                output.extend(chunk)
            if process.poll() is not None and not readable:
                break
        return_code = process.wait(timeout=1)
    finally:
        os.close(master_fd)
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)

    raw = bytes(output)
    assert return_code == 0, raw.decode(errors="replace")
    assert "…".encode() in raw
    assert "界".encode() not in raw
