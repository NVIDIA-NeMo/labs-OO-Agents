# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation strategy choices follow the supported public API."""

import pytest

from eval_pipeline.cli import VALID_STRATEGIES, get_strategy_instance
from nooa import CodeActStrategy, CodeActV2
from nooa.strategies import get_default_strategy


def test_v2_replaces_lite_without_changing_the_default():
    assert "codeact_v2" in VALID_STRATEGIES
    assert "codeact_lite" not in VALID_STRATEGIES
    assert isinstance(get_strategy_instance("codeact_v2"), CodeActV2)
    assert type(get_strategy_instance("codeact")) is CodeActStrategy
    assert type(get_default_strategy()) is CodeActStrategy
    with pytest.raises(ValueError, match="Unknown strategy"):
        get_strategy_instance("codeact_lite")
