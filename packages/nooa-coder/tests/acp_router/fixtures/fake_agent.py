# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa-coder`` in any role, with sessions on a scripted fake model.

Run it as the server command (``python fake_agent.py [nooa-coder options]``).
The router starts its workers from ``sys.orig_argv``, so workers run this
file too and get the same fake model. Options starting with ``--fixture-``
are this file's own and are not passed to ``nooa-coder``:

- ``--fixture-noisy``: print to standard output while the router module is
  imported (in the router, after start-up) and while each session's model is
  built (in the worker), as ``nooa.tracing`` does when it finds an endpoint.
- ``--fixture-hot``: every turn runs a cell that writes the file named by
  ``NOOA_FIXTURE_MARKER`` and then blocks the event loop for two minutes.
"""

import importlib.abc
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # coder_test_agents

from coder_test_agents import CellLLM, cell, reply  # noqa: E402
from nooa_coder.acp import cli  # noqa: E402

from nooa.unifiedllm import FakeLLMClient  # noqa: E402

HOT_CELL = """\
import os, pathlib, time
pathlib.Path(os.environ["NOOA_FIXTURE_MARKER"]).write_text(str(os.getpid()))
time.sleep(120)
"""


class _NoisyImport(importlib.abc.MetaPathFinder):
    def find_spec(self, name: str, path: object, target: object = None) -> None:
        if name == "nooa_coder.acp.router":
            print("stray output while importing the router", flush=True)
        return None


if "--fixture-noisy" in sys.argv:
    sys.meta_path.insert(0, _NoisyImport())


def llm_factory(alias: str | None, workspace: Path) -> FakeLLMClient:
    # CellLLM runs scripted cells in whichever Python tool the agent offers
    # (python_cell on CodeActV2), so the fixture drives the real CodingAgent.
    del alias, workspace
    if "--fixture-noisy" in sys.argv:
        print("stray output while building a model", flush=True)
    if "--fixture-hot" in sys.argv:
        return CellLLM([cell(HOT_CELL)], strict_exhaustion=False)
    return CellLLM([reply("Hi there.") for _ in range(20)], strict_exhaustion=False)


if __name__ == "__main__":
    # The command with this file's model factory (the ctx.obj hook), so
    # nooa-coder's own options parse as usual.
    cli.command.main(
        args=[arg for arg in sys.argv[1:] if not arg.startswith("--fixture-")],
        prog_name="nooa-coder",
        obj={"llm_factory": llm_factory},
    )
