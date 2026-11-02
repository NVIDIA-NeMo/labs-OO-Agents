# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure repeated-object formatting without model calls or private inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Annotated

from nooa.agentdoc import hidden, truncating_pformat


@dataclass
class Event:
    name: str
    inputs: dict[str, str]
    outputs: dict[str, str]
    duration: float
    internal: Annotated[str, hidden] = "private"


@dataclass
class Record:
    name: str
    events: list[Event]
    attributes: dict[str, str] = field(default_factory=dict)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    records = [
        Record(
            name=f"record-{index}",
            events=[
                Event(f"step-{step}", {"query": "example"}, {"result": "ok"}, 0.1)
                for step in range(20)
            ],
        )
        for index in range(args.records)
    ]
    timings = []
    digests = set()
    for _ in range(args.repeats):
        start = time.perf_counter()
        rendered = truncating_pformat(
            records, max_chars=100_000, max_depth=12, max_length=200, max_string=10_000
        )
        timings.append(time.perf_counter() - start)
        digests.add(hashlib.sha256(rendered.encode()).hexdigest())
    assert len(digests) == 1, "Formatting changed between repetitions"
    print(
        json.dumps(
            {
                "seconds": timings,
                "median_seconds": statistics.median(timings),
                "sha256": digests.pop(),
                "records": args.records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
