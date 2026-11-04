# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``import nooa`` does not load the strategies or LiteLLM; the names still resolve."""

import subprocess
import sys

import pytest

_HEAVY = ("litellm", "nooa.strategies", "nooa.unifiedllm")


@pytest.mark.parametrize("statement", ["import nooa", "import nooa.runtime"])
def test_package_import_does_not_load_the_llm_stack(statement):
    code = f"{statement}\nimport sys\nprint([m for m in {_HEAVY!r} if m in sys.modules])\n"
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert out == "[]"


def test_lazy_names_resolve_to_the_defining_objects():
    import nooa
    import nooa.llm_config
    import nooa.strategies
    import nooa.unifiedllm
    from nooa import CodeActStrategy, LLMResponse, llm_config_chain, set_default_strategy

    assert CodeActStrategy is nooa.strategies.CodeActStrategy
    assert set_default_strategy is nooa.strategies.set_default_strategy
    assert LLMResponse is nooa.unifiedllm.LLMResponse
    assert llm_config_chain is nooa.llm_config.llm_config_chain
    for name in nooa._LAZY:
        assert name in nooa.__all__
        assert getattr(nooa, name) is not None


def test_unknown_attribute_still_raises():
    import nooa

    with pytest.raises(AttributeError):
        nooa.does_not_exist  # noqa: B018
