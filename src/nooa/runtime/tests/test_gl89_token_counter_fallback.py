# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The framework has no local token-counting fallback."""

import nooa


def test_root_has_no_token_counter():
    assert not hasattr(nooa, "char_approximate_token_counter")


def test_runtime_has_no_token_estimator():
    from nooa.runtime.actor import ActorRuntime

    source_names = set(ActorRuntime.__init__.__code__.co_names)
    assert "_tokens_per_char" not in source_names
