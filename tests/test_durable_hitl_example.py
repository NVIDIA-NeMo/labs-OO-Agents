# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The durable-HITL example is executable documentation — run it and assert its invariant.

CI lints examples/ but never executes it, so without this test the example rots
silently on the first refactor.
"""

import re
import subprocess
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "advanced" / "durable_hitl.py"


def test_example_resumes_in_a_new_process_without_rerunning_steps():
    result = subprocess.run(
        [sys.executable, str(EXAMPLE)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"example failed:\n{result.stderr}"
    out = result.stdout

    # The parent suspended rather than running to completion.
    assert "SUSPENDED" in out

    # The child restored work it never performed itself.
    assert re.search(r"\[child\].*restored.*step_one.*step_two", out), out

    # The invariant: the child made exactly one LLM call, not three.
    assert re.search(r"\[child\].*LLM calls here\s*:\s*1\b", out), out

    # The parent's narrative must land before the child's. stdout is block-buffered
    # when it is not a terminal, so an unflushed parent reads backwards here while
    # still looking correct in an interactive run.
    assert out.index("SUSPENDED") < out.index("[child]"), out
