#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark tracing one large string in a fresh process for each size."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from nooa.tracing._hooks_impl import OpenInferenceHooks

MIB = 1024 * 1024


def measure(size_mib: int) -> dict[str, float | int | bool]:
    """Measure serialization after allocating the source string."""
    value = {"args": ("x" * (size_mib * MIB),), "kwargs": {}}
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    serialized = OpenInferenceHooks._safe_json_value(value)
    elapsed = time.perf_counter() - started
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    parsed = json.loads(serialized)
    return {
        "size_mib": size_mib,
        "elapsed_seconds": elapsed,
        "output_chars": len(serialized),
        "rss_growth_mib": (rss_after - rss_before) / 1024,
        "truncated": parsed.get("$nooa", {}).get("kind") == "truncated-json",
    }


def parse_args() -> argparse.Namespace:
    """Parse benchmark sizes and the internal child-process mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes_mib", type=int, nargs="*", default=[1, 16, 64, 256])
    parser.add_argument("--child", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    """Run one measurement or coordinate isolated child measurements."""
    args = parse_args()
    if args.child is not None:
        print(json.dumps(measure(args.child)))
        return

    script = Path(__file__).resolve()
    print("size_mib elapsed_s output_chars rss_growth_mib truncated")
    for size_mib in args.sizes_mib:
        completed = subprocess.run(
            [sys.executable, str(script), "--child", str(size_mib)],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        print(
            f"{result['size_mib']:8d} {result['elapsed_seconds']:9.4f} "
            f"{result['output_chars']:12d} {result['rss_growth_mib']:14.1f} "
            f"{result['truncated']}"
        )


if __name__ == "__main__":
    main()
