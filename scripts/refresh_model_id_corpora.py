#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Refresh the model-id corpora fixture from the live model catalogs.

Fetches the public OpenRouter catalog (``/v1/models``) and, when
``NVIDIA_INFERENCE_API_KEY`` is set, the NVIDIA inference gateway catalog,
rebuilds the ``openrouter`` / ``nvidia_gateway`` sections of
``tests/unifiedllm/fixtures/model_id_corpora.json``, and prints the
resolution/misattribution summary for the fresh catalog.

Offline-safe: with no network it exits non-zero with a clear message and
never truncates the existing fixture (a failed fetch is never written).

Usage::

    uv run python scripts/refresh_model_id_corpora.py           # refresh
    uv run python scripts/refresh_model_id_corpora.py --check   # report only

``--check`` exits 1 on drift without writing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests/unifiedllm/fixtures/model_id_corpora.json"

OPENROUTER_URL = "https://openrouter.ai/api/v1/models"
NVIDIA_GATEWAY_URL = "https://inference-api.nvidia.com/v1/models"

#: Sections not fetched live (azure/vertex_ai/bedrock come from the transport
#: library's bundled catalog, not a public endpoint): preserved verbatim on
#: rewrite, in fixture key order.
PRESERVED_SECTIONS = ("azure", "vertex_ai", "bedrock")


def fetch_catalog(url: str, headers: dict[str, str] | None = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def openrouter_entries(payload: dict) -> list[dict]:
    """Catalog section entries: ground truth = leading id segment."""
    entries = []
    for model in payload["data"]:
        model_id = model["id"]
        segments = model_id.split("/")
        vendor = segments[0].lstrip("~") if len(segments) > 1 else None
        entries.append({"id": model_id, "vendor": vendor})
    return entries


def nvidia_gateway_entries(payload: dict) -> list[dict]:
    """Catalog section entries.

    Ground truth = the middle vendor segment of ``nvidia/<vendor>/<model>``
    spellings; every other spelling the gateway serves (us/..., gcp/...,
    nvcf/..., bare ids) has no segment the catalog itself vouches for, so
    vendor stays null and conformance treats it as unverifiable.
    """
    entries = []
    for model in payload["data"]:
        model_id = model["id"]
        segments = model_id.split("/")
        vendor = segments[1] if len(segments) == 3 and segments[0] == "nvidia" else None
        entries.append({"id": model_id, "vendor": vendor})
    return entries


def comment(fetch_date: str, openrouter_count: int, nvidia_count: int | None) -> str:
    """Provenance note for the fixture's ``_comment`` member."""
    nvidia = (
        f"nvidia_gateway: {nvidia_count} ids from the NVIDIA inference gateway /v1/models; "
        "ground truth = middle vendor segment for nvidia/<vendor>/<model> spellings. "
        if nvidia_count is not None
        else "nvidia_gateway: not refreshed (NVIDIA_INFERENCE_API_KEY unset); "
        "previous section preserved. "
    )
    return (
        "Model-id corpora for parse_model_string conformance. "
        f"openrouter: {openrouter_count} ids from the public OpenRouter catalog "
        f"(fetched {fetch_date}); ground truth = leading id segment. "
        + nvidia
        + "azure/vertex_ai/bedrock: deployment-prefixed ids from the catalog bundled with the "
        "transport library; ground truth = the logical provider of the served model (bare-id "
        "provider, or the Bedrock vendor.model head). Transport labels (azure/bedrock/...) are "
        "never treated as logical providers. vendor=null means the catalog cannot verify the "
        "logical provider."
    )


#: Catalog vendor label -> canonical logical provider. Mirrors the table in
#: tests/unifiedllm/test_model_id_corpora.py: these are spelling variants of
#: the SAME logical provider ("z-ai" vs "glm", "moonshotai" vs "kimi"), not
#: misattributions, so the summary must not count them as such. Unknown
#: labels pass through verbatim and usually fail closed (unverifiable).
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


def _canon(label: str) -> str:
    return _CANON.get(label.lower().strip(), label.lower().strip())


def summarize(name: str, entries: list[dict]) -> None:
    """Print the resolution/misattribution summary for one catalog section.

    Uses the same canonicalization as the conformance test, so the numbers
    here are the numbers the test would measure.
    """
    from nooa.unifiedllm.contracts import parse_model_string

    misattributed: list[tuple[str, str, str]] = []
    resolved = 0
    with_truth = 0
    for entry in entries:
        vendor = entry["vendor"]
        if not vendor:
            continue  # no ground truth for this spelling
        truth = _canon(vendor)
        with_truth += 1
        parsed = parse_model_string(entry["id"])
        if parsed.provider is not None and parsed.provider != truth:
            misattributed.append((entry["id"], vendor, parsed.provider))
        elif parsed.provider is not None:
            resolved += 1
    rate = resolved / with_truth if with_truth else 0.0
    print(f"{name}: {len(entries)} ids, {with_truth} with catalog ground truth")
    print(f"  resolution rate: {rate:.3f} ({resolved}/{with_truth})")
    print(f"  misattributed: {len(misattributed)}")
    for model_id, vendor, got in misattributed[:20]:
        print(f"    {model_id}: catalog says {vendor}, parser says {got}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="only report drift; exit 1 if the fixture differs from the live catalogs, "
        "without writing",
    )
    parser.add_argument(
        "--openrouter-url",
        default=OPENROUTER_URL,
        help="override the OpenRouter catalog URL",
    )
    args = parser.parse_args(argv)

    try:
        print(f"Fetching OpenRouter catalog: {args.openrouter_url}")
        openrouter_section = openrouter_entries(fetch_catalog(args.openrouter_url))
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        print(
            f"error: could not fetch the OpenRouter catalog ({args.openrouter_url}): {exc}\n"
            "The refresh needs network access; the fixture was left untouched.",
            file=sys.stderr,
        )
        return 1
    if not openrouter_section:
        print(
            "error: the OpenRouter catalog returned no models; refusing to write.", file=sys.stderr
        )
        return 1

    nvidia_section: list[dict] | None = None
    api_key = os.getenv("NVIDIA_INFERENCE_API_KEY")
    if api_key:
        try:
            print(f"Fetching NVIDIA gateway catalog: {NVIDIA_GATEWAY_URL}")
            nvidia_section = nvidia_gateway_entries(
                fetch_catalog(NVIDIA_GATEWAY_URL, headers={"Authorization": f"Bearer {api_key}"})
            )
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            print(
                f"error: could not fetch the NVIDIA gateway catalog ({NVIDIA_GATEWAY_URL}): "
                f"{exc}\nThe fixture was left untouched.",
                file=sys.stderr,
            )
            return 1
    else:
        print(
            "NVIDIA_INFERENCE_API_KEY is unset: keeping the fixture's nvidia_gateway section "
            "and its ground truth as-is."
        )

    for name, section in (("openrouter", openrouter_section), ("nvidia_gateway", nvidia_section)):
        if section is not None:
            summarize(name, section)

    current = json.loads(FIXTURE.read_text())
    # Keys keep the fixture's original order so a refresh diffs only what
    # actually changed. Sections not fetched live (their source is the
    # transport library's bundled catalog, not a public endpoint) are
    # preserved verbatim.
    fresh: dict = {
        "_comment": comment(
            date.today().isoformat(),
            len(openrouter_section),
            len(nvidia_section) if nvidia_section is not None else None,
        ),
        "openrouter": openrouter_section,
        "nvidia_gateway": nvidia_section
        if nvidia_section is not None
        else current["nvidia_gateway"],
        **{name: current[name] for name in PRESERVED_SECTIONS},
    }

    # Drift is about corpus content, not the fixture's provenance note: the
    # fetch date in _comment changes daily, so counting it would make --check
    # fail against an otherwise-identical corpus.
    drifted = [
        key
        for key in ("openrouter", "nvidia_gateway", *PRESERVED_SECTIONS)
        if fresh[key] != current.get(key)
    ]
    if args.check:
        if drifted:
            print(
                "Drift detected in corpus section(s): "
                + ", ".join(drifted)
                + " (rerun without --check to refresh the fixture).",
                file=sys.stderr,
            )
            return 1
        print("No drift: the fixture's corpus already matches the live catalogs.")
        return 0

    # Same layout the fixture has always used (indent=1, one compact entry per
    # model) so the diff stays reviewable.
    FIXTURE.write_text(json.dumps(fresh, indent=1) + "\n")
    print(f"Wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
