# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration merging obeys the same validation rules as construction."""

import pytest
from nooa_memory.config import EmbeddingConfig, MemoryConfig
from pydantic import ValidationError


def test_merge_coerces_boolean_without_mutating_original():
    original = MemoryConfig(enabled=True, path=":memory:")
    merged = original.merge_with(enabled="false")
    assert merged.enabled is False
    assert merged.path == original.path
    assert original.enabled is True


def test_merge_validates_nested_config_as_top_level_replacement():
    original = MemoryConfig(embedding=EmbeddingConfig(dim=64, batch_size=3))
    merged = original.merge_with(embedding={"backend": "hashing", "dim": 16})
    assert isinstance(merged.embedding, EmbeddingConfig)
    assert merged.embedding.dim == 16
    assert merged.embedding.batch_size == EmbeddingConfig().batch_size
    assert original.embedding.dim == 64
    assert original.embedding.batch_size == 3


@pytest.mark.parametrize(
    "overrides", [{"owner": "invalid_owner"}, {"embedding": {"backend": "unknown"}}]
)
def test_merge_rejects_invalid_configuration(overrides):
    with pytest.raises(ValidationError):
        MemoryConfig().merge_with(**overrides)
