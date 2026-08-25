# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-time regression tests."""

import subprocess
import sys


def _assert_import_does_not_load_litellm(statement: str) -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"{statement}\n"
                "import sys\n"
                "assert 'litellm' not in sys.modules, 'litellm loaded during lightweight import'\n"
            ),
        ],
        check=True,
    )


def test_import_nooa_does_not_load_litellm() -> None:
    _assert_import_does_not_load_litellm("import nooa")


def test_default_strategy_import_does_not_load_litellm() -> None:
    _assert_import_does_not_load_litellm(
        "from nooa import get_default_strategy\n"
        "strategy = get_default_strategy()\n"
        "assert strategy.__class__.__name__ == 'CodeActStrategy'"
    )


def test_unifiedllm_types_do_not_load_litellm() -> None:
    _assert_import_does_not_load_litellm(
        "from nooa.unifiedllm import LLMResponse, Tool, ToolCall\n"
        "assert LLMResponse.__module__ == 'nooa.unifiedllm.types'\n"
        "assert Tool.__module__ == 'nooa.unifiedllm.types'\n"
        "assert ToolCall.__module__ == 'nooa.unifiedllm.types'"
    )


def test_interactive_config_does_not_load_litellm() -> None:
    _assert_import_does_not_load_litellm(
        "from nooa.interactive_config import SummarizationConfig\n"
        "config = SummarizationConfig()\n"
        "assert config.policy == 'token_budget'"
    )
