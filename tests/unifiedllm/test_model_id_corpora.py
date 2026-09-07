# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-id corpus conformance tests.

The corpora are real catalog ids (OpenRouter public catalog, NVIDIA inference
gateway). Ground truth is the vendor segment each catalog itself attaches to
the id — the same signal an adapter would trust at request time. The invariant
under test: the parser never *mis-attributes* an id; ids it cannot resolve
fail closed (provider=None) rather than guessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nooa.unifiedllm.contracts import parse_model_string

FIXTURES = Path(__file__).parent / "fixtures"
CORPORA = json.loads((FIXTURES / "model_id_corpora.json").read_text())

#: Catalog vendor label -> canonical logical provider. These are spelling
#: variants of the SAME logical provider, not misattributions: the catalogs
#: spell the vendor differently than the parser's canonical name
#: ("meta-llama" vs "meta", "mistralai" vs "mistral", "~anthropic" is
#: OpenRouter's variant marker on the vendor segment).
_CANON = {
    "openai": "openai",
    "anthropic": "anthropic",
    "~anthropic": "anthropic",
    "google": "google",
    "meta": "meta",
    "meta-llama": "meta",
    "mistral": "mistral",
    "mistralai": "mistral",
    "x-ai": "xai",
    "~x-ai": "xai",
    "xai": "xai",
    "nvidia": "nvidia",
    "deepseek": "deepseek",
    "deepseek-ai": "deepseek",
    "~deepseek": "deepseek",
    "qwen": "qwen",
    "z-ai": "glm",
    "~z-ai": "glm",
    "zai": "glm",
    "zai-org": "glm",
    "moonshot": "kimi",
    "moonshotai": "kimi",
    "minimaxai": "minimax",
    "microsoft": "microsoft",
}


def _canon(label: str | None) -> str | None:
    if not label:
        return None
    return _CANON.get(label.lower().strip(), label.lower().strip())


def _corpus_ids(name: str) -> list[tuple[str, str | None]]:
    entries = CORPORA[name]
    return [(e["id"], _canon(e.get("vendor"))) for e in entries if e.get("id")]


@pytest.mark.parametrize("name", ["openrouter", "nvidia_gateway"])
def test_no_misattribution_in_corpora(name: str) -> None:
    """Every id the parser resolves must match the catalog's own vendor.

    Fail-closed (provider=None) is acceptable for genuinely-unknown model
    families — it is the safe default, not a bug. Resolving to the WRONG
    provider is a bug; there are zero such ids in either corpus.
    """
    misattributed = []
    for model_id, vendor in _corpus_ids(name):
        if vendor is None:
            continue  # no ground truth for this spelling
        parsed = parse_model_string(model_id)
        if parsed.provider is not None and parsed.provider != vendor:
            misattributed.append((model_id, vendor, parsed.provider))
    assert not misattributed, misattributed[:20]


@pytest.mark.parametrize("name", ["openrouter", "nvidia_gateway"])
def test_corpora_resolution_rate(name: str) -> None:
    """Guard the measured resolution rate against regressions.

    The rates below were measured after the corpus-driven fixes (2026-09-07).
    Fail-closed remainder is dominated by boutique/one-off vendors not worth
    a declaration; the rate guards that a table edit does not silently drop
    a previously-resolving family.
    """
    ids_with_truth = [x for x in _corpus_ids(name) if x[1] is not None]
    resolved = [x for x in ids_with_truth if parse_model_string(x[0]).provider is not None]
    rate = len(resolved) / len(ids_with_truth)
    # OpenRouter ~= 0.74, NVIDIA gateway ~= 0.91 at the time of writing.
    assert rate >= 0.70, f"{name}: resolution rate {rate:.3f} dropped"


def test_deployment_spellings_resolve() -> None:
    """Routing-prefixed deployment spellings collapse to the same identity.

    These are the spellings that actually arrive through gateways and
    deployment-specific clients, drawn from each provider's deployment docs.
    """
    cases = [
        # Azure OpenAI: azure/<deployment-or-model>, incl. region segment.
        ("azure/gpt-4o", "openai", "gpt-4o"),
        ("azure/eu/gpt-4o-2024-08-06", "openai", "gpt-4o-2024-08-06"),
        # Vertex AI: vertex_ai/gemini-*.
        ("vertex_ai/gemini-2.5-pro", "google", "gemini-2.5-pro"),
        # Bedrock: bedrock/<region>/.../vendor.model, and vendor.model alone.
        (
            "bedrock/us-east-1/1-month-commitment/anthropic.claude-3-5-sonnet",
            "anthropic",
            "claude-3-5-sonnet",
        ),
        ("anthropic.claude-3-5-sonnet", "anthropic", "claude-3-5-sonnet"),
        # OpenRouter tier/variant markers.
        ("kimi-k3:free", "kimi", "kimi-k3"),
        ("o3:batch", "openai", "o3"),
        ("~anthropic/claude-fable-latest", "anthropic", "claude-fable-latest"),
        # NVIDIA gateway: nvidia/<vendor>/<model>.
        ("nvidia/zai-org/glm-5.3", "glm", "glm-5.3"),
        (
            "nvidia/nvidia/llama-3.1-nemotron-ultra-253b-v1",
            "nvidia",
            "llama-3.1-nemotron-ultra-253b-v1",
        ),
        ("nvidia/google/gemma-4-31b-it", "google", "gemma-4-31b-it"),
        # o-series with effort suffix.
        ("o4-mini-high", "openai", "o4-mini-high"),
    ]
    for model_string, provider, model in cases:
        parsed = parse_model_string(model_string)
        assert parsed.provider == provider, (model_string, parsed.provider, provider)
        assert parsed.model == model, (model_string, parsed.model, model)


def test_unknown_boutique_vendors_fail_closed() -> None:
    """Undeclared model families resolve to None — never a guess."""
    for model_id in (
        "aion-labs/aion-3.0",
        "amazon/nova-pro-v1",
        "baidu/ernie-4.5-vl-424b-a47b",
        "thinkingmachines/inkling",
        "some-unknown-vendor/model-x",
    ):
        parsed = parse_model_string(model_id)
        assert parsed.provider is None, (model_id, parsed.provider)
