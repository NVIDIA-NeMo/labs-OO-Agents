# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""pyp's sources cut a large stderr with nooa.snapshot's render_output line."""

from __future__ import annotations

import sys

import pytest
from nooa_cli.tools.pyp.errors import PipeError
from nooa_cli.tools.pyp.sources import _STDERR_CAP, _stream_bash_lines, run

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="BashSession needs POSIX")

_SIZE = 300_000
_CMD = (
    'python3 -c "import sys; '
    f"sys.stderr.write(''.join(chr(97 + i % 26) for i in range({_SIZE}))); sys.exit(3)\""
)
_WHOLE = "".join(chr(97 + i % 26) for i in range(_SIZE))


async def test_run_keeps_half_the_cap_from_each_end_of_stderr():
    assert _STDERR_CAP == 128 * 1024  # the cap is unchanged
    with pytest.raises(PipeError) as caught:
        await run(_CMD).collect()
    half = _STDERR_CAP // 2
    expected = (
        _WHOLE[: _STDERR_CAP - half]
        + f"\n[... {_SIZE - _STDERR_CAP:,} chars omitted; full output not kept]\n"
        + _WHOLE[-half:]
    )
    assert caught.value.stderr == expected
    assert caught.value.returncode == 3


async def test_shared_sources_keep_the_head_of_stderr_and_say_what_they_cut():
    with pytest.raises(PipeError) as caught:
        async for _ in _stream_bash_lines(_CMD):
            pass
    assert caught.value.stderr == (
        _WHOLE[:_STDERR_CAP]
        + f"\n[... {_SIZE - _STDERR_CAP:,} chars omitted; full output not kept]"
    )


async def test_small_stderr_is_whole():
    with pytest.raises(PipeError) as caught:
        await run("echo oops >&2; false").collect()
    assert caught.value.stderr == "oops\n"
