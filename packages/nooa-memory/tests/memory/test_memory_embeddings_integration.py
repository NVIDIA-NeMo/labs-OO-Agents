# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Live embedding integration test (NVIDIA gateway, text-embedding-3-large).

Deselected by default (the suite runs with ``-m 'not integration'``). Run with::

    MEM_EMBED_API_KEY=... uv run pytest tests/memory/test_memory_embeddings_integration.py -m integration
"""

import os

import numpy as np
import pytest
from nooa_memory.config import EmbeddingConfig
from nooa_memory.embeddings import LiteLLMEmbedder

pytestmark = pytest.mark.integration


@pytest.fixture
def cfg():
    if not os.environ.get("MEM_EMBED_API_KEY"):
        pytest.skip("MEM_EMBED_API_KEY not set")
    dims = os.environ.get("MEM_EMBED_DIMS")
    return EmbeddingConfig(
        backend="litellm",
        model=os.environ.get("MEM_EMBED_MODEL", "openai/azure/openai/text-embedding-3-large"),
        endpoint=os.environ.get(
            "MEM_EMBED_BASE_URL", "https://inference-api.nvidia.com/v1/embeddings"
        ),
        api_key=os.environ["MEM_EMBED_API_KEY"],
        dimensions=int(dims) if dims else 1024,
    )


def test_live_embedding_is_normalised_and_consistent(cfg):
    emb = LiteLLMEmbedder(cfg)
    v = emb.embed("deploy the service to production")
    assert v.ndim == 1 and v.shape[0] == emb.dim
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-3)  # L2-normalised
    # semantic sanity: related text closer than unrelated
    a = emb.embed("ship a release to prod")
    b = emb.embed("a recipe for banana bread")
    assert float(np.dot(v, a)) > float(np.dot(v, b))
