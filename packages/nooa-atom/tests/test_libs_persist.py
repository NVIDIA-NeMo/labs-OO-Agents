# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A library written with self.libs is saved in the libs dir and active in a later session."""

import subprocess
import sys
import textwrap

SOURCE = """\
\"\"\"Tiny statistics.\"\"\"
from nooa.skill import Skill


class Stats(Skill):
    \"\"\"Mean of numbers.\"\"\"

    def mean(self, values: list[float]) -> float:
        return sum(values) / len(values)
"""

LATER_SESSION = textwrap.dedent(
    """
    import asyncio, sys
    from pathlib import Path
    from nooa_atom.agent.agent import AtomAgent
    from nooa.unifiedllm import FakeLLMClient

    async def main():
        root = Path(sys.argv[1])
        agent = AtomAgent(llm=FakeLLMClient([]), cwd=root, libs_dir=root / "libs", skills_dirs=[])
        try:
            print("LIST", await agent.libs.list())
            print("MEAN", agent.stats.mean([1.0, 2.0, 3.0]))
        finally:
            await agent.aclose()

    asyncio.run(main())
    """
)


async def test_a_library_is_saved_in_libs_dir_and_active_in_a_new_session(tmp_path):
    from nooa_atom.agent.agent import AtomAgent

    from nooa.unifiedllm import FakeLLMClient

    agent = AtomAgent(
        llm=FakeLLMClient([]), cwd=tmp_path, libs_dir=tmp_path / "libs", skills_dirs=[]
    )
    try:
        await agent.libs.create("stats", "Tiny statistics.")
        path = agent.libs.path("stats", "__init__.py")
        assert path == str(tmp_path / "libs" / "stats" / "__init__.py")
        (tmp_path / "libs" / "stats" / "__init__.py").write_text(SOURCE)
        result = await agent.libs.reload("stats")
        assert "error" not in result.lower(), result
        assert agent.stats.mean([2.0, 4.0]) == 3.0
    finally:
        await agent.aclose()
    assert (tmp_path / "libs" / "stats" / "pyproject.toml").is_file()

    # A new process: nothing to import or activate; the library is self.stats.
    later = subprocess.run(
        [sys.executable, "-c", LATER_SESSION, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert later.returncode == 0, later.stderr[-3000:]
    assert "LIST ['stats']" in later.stdout
    assert "MEAN 2.0" in later.stdout
