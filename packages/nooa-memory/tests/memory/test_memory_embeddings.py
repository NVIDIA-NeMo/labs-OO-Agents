# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the embedding backends."""

import numpy as np
import pytest
from nooa_memory.config import EmbeddingConfig
from nooa_memory.embeddings import HashingEmbedder, get_embedder


def _cos(a, b):
    return float(np.dot(a, b))


def test_hashing_embedder_is_deterministic():
    e = HashingEmbedder(dim=64)
    assert np.array_equal(e.embed("deploy the app"), e.embed("deploy the app"))


def test_hashing_embedder_dim_and_normalization():
    e = HashingEmbedder(dim=128)
    v = e.embed("some non empty text")
    assert v.shape == (128,)
    assert e.dim == 128
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(v))


def test_empty_text_is_zero_vector():
    e = HashingEmbedder(dim=32)
    v = e.embed("")
    assert float(np.linalg.norm(v)) == 0.0


def test_similar_text_has_higher_cosine_than_dissimilar():
    e = HashingEmbedder(dim=512)
    base = e.embed("deploy ship the application to production")
    similar = e.embed("ship the application deploy to production")
    dissimilar = e.embed("banana fruit salad recipe with mango")
    assert _cos(base, similar) > _cos(base, dissimilar)


def test_embed_batch_matches_single():
    e = HashingEmbedder(dim=64)
    texts = ["alpha", "beta gamma", "delta"]
    batch = e.embed_batch(texts)
    assert len(batch) == 3
    assert np.array_equal(batch[1], e.embed("beta gamma"))


def test_get_embedder_selects_backend():
    e = get_embedder(EmbeddingConfig(backend="hashing", dim=77))
    assert isinstance(e, HashingEmbedder)
    assert e.dim == 77


def test_get_embedder_unknown_backend_raises():
    with pytest.raises(ValueError):
        get_embedder(EmbeddingConfig(backend="nope"))  # type: ignore[arg-type]


def test_litellm_embedder_dim_uses_dimensions_not_hashing_dim():
    """Regression: LiteLLMEmbedder.dim must come from `dimensions`, not the hashing `dim`.

    (A sqlite-vec/Chroma table is created from this dim; using the 256 hashing
    default against 1024-d text-embedding-3-large caused a live dim-mismatch.)
    """
    from nooa_memory.embeddings import LiteLLMEmbedder

    e = LiteLLMEmbedder(EmbeddingConfig(backend="litellm", dim=256, dimensions=1024))
    assert e._dim == 1024  # set from dimensions, no network probe
    assert e.dim == 1024

    # dimensions unset -> probe lazily (stays None until first call)
    e2 = LiteLLMEmbedder(EmbeddingConfig(backend="litellm", dim=256, dimensions=None))
    assert e2._dim is None
