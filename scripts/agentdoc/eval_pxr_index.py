#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Offline eval of the pxr docs index: search cases, did-you-mean, doc() coverage.

    PYTHONPATH=<usd-core site dir>:src python scripts/agentdoc/eval_pxr_index.py \
        --coverage-csv reviews/pxr-doc-coverage/coverage.csv

The did-you-mean cases are the 23 names Atom agents invented in OV Gym episodes
(names that do not exist in usd-core), each with the real names an expert would
point to. ``--coverage-csv`` (from the pxr doc coverage study) lists the names
agents used, with use counts and whether doc() had real text before.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys

SEARCH_CASES = [
    (
        "all files a USD asset depends on including MDL",
        [
            "UsdUtils.ComputeAllDependencies",
            "UsdUtils.ExtractExternalReferences",
            "Sdf.Layer.GetExternalAssetDependencies",
            "Sdf.Layer.GetCompositionAssetDependencies",
            "Sdf.Layer.GetExternalReferences",
        ],
    ),
    (
        "create a face subset for material binding",
        ["UsdGeom.Subset.CreateGeomSubset", "UsdShade.MaterialBindingAPI.CreateMaterialBindSubset"],
    ),
    (
        "family type nonOverlapping validate subsets",
        ["UsdGeom.Subset.SetFamilyType", "UsdGeom.Subset.ValidateFamily"],
    ),
]

# invented name -> acceptable real names (any one counts as the right suggestion)
INVENTED = {
    "UsdGeom.Tokens.materialBind": ["UsdShade.Tokens.materialBind"],
    "UsdShade.GetTypeFromValueTypeName": ["Sdf.GetTypeForValueTypeName"],
    "Sdf.Attribute": ["Usd.Attribute", "Sdf.AttributeSpec"],
    "UsdGeom.Subset.CreateSubset": ["UsdGeom.Subset.CreateGeomSubset"],
    "UsdShade.BindingAPI.Bind": ["UsdShade.MaterialBindingAPI"],
    "Usd.Scope.Define": ["UsdGeom.Scope"],
    "UsdGeom.Subset.GetFamilyTypeAttr": ["UsdGeom.Subset.GetFamilyType"],
    "UsdGeom.BBoxCompute": ["UsdGeom.BBoxCache"],
    "Sdf.Layer.Open": ["Sdf.Layer.FindOrOpen", "Sdf.Layer.OpenAsAnonymous"],
    "Vt.Vec2f": ["Gf.Vec2f", "Vt.Vec2fArray"],
    "UsdGeom.Subset.GetGeomSubset": ["UsdGeom.Subset.GetGeomSubsets"],
    "UsdShade.MaterialBindingAPI.GetMaterial": [
        "UsdShade.MaterialBindingAPI.ComputeBoundMaterial",
        "UsdShade.MaterialBindingAPI.GetDirectBinding",
    ],
    "UsdGeom.BindingAPI": ["UsdShade.MaterialBindingAPI"],
    "Usd.Prim.CreateDisplayNameAttr": ["Usd.Prim.SetDisplayName", "Usd.Object.SetDisplayName"],
    "Sdf.Layer.GetField": ["Sdf.Spec.GetInfo", "Sdf.PrimSpec.GetInfo", "Sdf.Layer.GetObjectAtPath"],
    "Sdf.Layer.HasField": ["Sdf.Spec.HasInfo", "Sdf.PrimSpec.HasInfo", "Sdf.Layer.GetObjectAtPath"],
    "Usd.Stage.GetExternalDependencies": [
        "UsdUtils.ComputeAllDependencies",
        "Sdf.Layer.GetExternalAssetDependencies",
        "Sdf.Layer.GetCompositionAssetDependencies",
    ],
    "Sdf.Layer.GetExternalDependencies": [
        "Sdf.Layer.GetExternalAssetDependencies",
        "Sdf.Layer.GetCompositionAssetDependencies",
        "Sdf.Layer.GetExternalReferences",
    ],
    "UsdShade.CollectionAPI": ["Usd.CollectionAPI"],
    "Gf.BBox": ["Gf.BBox3d"],
    "Usd.ShadeOutput.ConnectToSource": ["UsdShade.Output"],
    "UsdShade.AttributeType.Color3f": ["Sdf.ValueTypeNames.Color3f"],
    "UsdShade.MaterialBindingAPI.GetBoundMaterial": [
        "UsdShade.MaterialBindingAPI.ComputeBoundMaterial"
    ],
}

FALLBACK = "No docstring or signature is available"


def _failing_access(name: str):
    """Resolve ``name`` step by step; return the AttributeError of the missing step."""
    import importlib

    parts = name.split(".")
    obj = importlib.import_module("pxr." + parts[0])
    for part in parts[1:]:
        try:
            obj = getattr(obj, part)
        except AttributeError as error:
            return error
    return None


def _mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", text) is not None


def run_search() -> list[dict]:
    from nooa.agentdoc import doc

    rows = []
    for query, expected in SEARCH_CASES:
        out = doc.search(query)
        names = [line.split("(")[0].split(" ")[0] for line in out.splitlines() if " — " in line]
        ranks = {e: names.index(e) + 1 for e in expected if e in names}
        rows.append({"query": query, "output": out, "expected_ranks": ranks})
    return rows


def run_did_you_mean() -> list[dict]:
    from nooa.agentdoc.adapters import pxr_index
    from nooa.agentdoc.core import attribute_error_hint
    from nooa.agentdoc.registry import _owner_key  # noqa: PLC2701

    rows = []
    for name, expected in INVENTED.items():
        error = _failing_access(name)
        if error is None:
            rows.append({"name": name, "error": "resolves", "generic_ok": False, "shown_ok": False})
            continue
        shown = attribute_error_hint(error) or ""
        # Generic provider alone (what the index gives, without the curated static hints).
        msg = re.match(r"^module '([^']+)' has no attribute ", str(error))
        obj = error.obj
        if msg:
            key = msg.group(1)
        else:
            key = _owner_key(obj if isinstance(obj, type) else type(obj))
        generic = pxr_index._attribute_hint(obj, error.name, key) or ""  # noqa: SLF001
        rows.append(
            {
                "name": name,
                "expected": expected,
                "generic_hint": generic,
                "generic_ok": any(_mentions(generic, e) for e in expected),
                "shown_hint": shown,
                "shown_ok": any(_mentions(shown, e) for e in expected),
            }
        )
    return rows


def _resolve(name: str):
    import importlib

    parts = name.split(".")
    obj = importlib.import_module("pxr." + parts[0])
    parent = None
    for part in parts[1:]:
        parent, obj = obj, getattr(obj, part)
    return parent, obj


def _has_text(name: str, page_cache: dict) -> bool:
    """doc() shows real descriptive text for ``name`` (its own page, or for a value its parent's)."""
    from nooa.agentdoc import doc

    parent, obj = _resolve(name)
    if isinstance(obj, type) or callable(obj):
        key = id(obj)
        if key not in page_cache:
            page_cache[key] = doc(obj)
        page = page_cache[key]
        if FALLBACK in page[:600]:
            return False
        # Real text = more than the def line and an empty docstring.
        body = page.split("\n", 1)[1] if "\n" in page else ""
        return len(re.sub(r'[\s"]', "", body)) > 20
    key = id(parent)
    if key not in page_cache:
        page_cache[key] = doc(parent)
    page = page_cache[key]
    return FALLBACK not in page[:600] and _mentions(page, name.rsplit(".", 1)[-1])


def run_coverage(csv_path: str) -> dict:
    rows = list(csv.DictReader(open(csv_path)))
    resolved = [r for r in rows if r["resolved"] == "True"]
    cache: dict = {}
    before_n = now_n = before_u = now_u = total_u = 0
    gained, still = [], []
    for row in resolved:
        uses = int(row["uses"])
        total_u += uses
        before = row["curated"] == "True" or row["has_docstring"] == "True"
        try:
            now = _has_text(row["name"], cache)
        except Exception:  # noqa: BLE001
            now = False
        before_n += before
        before_u += uses * before
        now_n += now
        now_u += uses * now
        if now and not before:
            gained.append(row["name"])
        if not now:
            still.append(row["name"])
    return {
        "resolved_names": len(resolved),
        "resolved_uses": total_u,
        "before_names": before_n,
        "now_names": now_n,
        "before_uses": before_u,
        "now_uses": now_u,
        "gained": gained,
        "still_without_text": still,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--coverage-csv", help="coverage.csv from the pxr doc coverage study")
    parser.add_argument("--json", help="write all results here")
    args = parser.parse_args()
    from pxr import Usd

    print(f"usd-core {Usd.GetVersion()}")
    results = {"search": run_search(), "did_you_mean": run_did_you_mean()}
    for row in results["search"]:
        print(
            f"\n## search: {row['query']}\nexpected ranks: {row['expected_ranks']}\n{row['output']}"
        )
    dym = results["did_you_mean"]
    print("\n## did you mean")
    for row in dym:
        mark = f"generic={'ok' if row['generic_ok'] else 'MISS'} shown={'ok' if row['shown_ok'] else 'MISS'}"
        first = (
            (row.get("generic_hint") or row.get("error", "")).splitlines()[0]
            if (row.get("generic_hint") or row.get("error"))
            else ""
        )
        print(f"- {row['name']}: {mark} | {first}")
    print(
        f"generic provider right: {sum(r['generic_ok'] for r in dym)}/{len(dym)}; "
        f"shown to agents (static hint or provider): {sum(r['shown_ok'] for r in dym)}/{len(dym)}"
    )
    if args.coverage_csv:
        cov = run_coverage(args.coverage_csv)
        results["coverage"] = cov
        n, u = cov["resolved_names"], cov["resolved_uses"]
        print(f"\n## coverage over {n} resolvable names ({u} uses)")
        print(
            f"before: {cov['before_names']} names ({100 * cov['before_names'] / n:.1f}%), "
            f"{cov['before_uses']} uses ({100 * cov['before_uses'] / u:.1f}%)"
        )
        print(
            f"now:    {cov['now_names']} names ({100 * cov['now_names'] / n:.1f}%), "
            f"{cov['now_uses']} uses ({100 * cov['now_uses'] / u:.1f}%)"
        )
        print(f"gained: {', '.join(cov['gained'])}")
        print(f"still without text: {', '.join(cov['still_without_text'])}")
    if args.json:
        json.dump(results, open(args.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
