#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce synchronous tracing latency for large JSON-compatible arguments."""

from __future__ import annotations

import argparse
import asyncio
import gc
import itertools
import json
import resource
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opentelemetry.sdk.trace import TracerProvider
from pydantic import BaseModel

from nooa.agentdoc import truncating_pformat
from nooa.tracing._hooks_impl import OpenInferenceHooks
from nooa.tracing._trace_json import trace_fields

MIB = 1024 * 1024
CONTENT_BYTES = 1024
HEARTBEAT_SECONDS = 0.01
_CALL_IDS = itertools.count()


class _DummyAgent:
    pass


_HOOKS = OpenInferenceHooks(TracerProvider().get_tracer(__name__))
_AGENT = _DummyAgent()


@dataclass(frozen=True)
class Measurement:
    elapsed_seconds: float
    output_chars: int


class TrialPayload(BaseModel):
    """Minimal analogue of Experimentalist's TrialResult trace-heavy fields."""

    id: str
    outputs: dict[str, object]
    metadata: dict[str, object]


class CandidatePayload(BaseModel):
    """Minimal analogue of a Candidate carrying nested trial detail."""

    id: str
    description: str
    trials: list[TrialPayload]


def make_method_input(target_mib: int) -> dict[str, Any]:
    """Build a method input whose expanded JSON is approximately target_mib MiB."""
    content = "x" * CONTENT_BYTES
    # Each entry expands to slightly more than CONTENT_BYTES. An approximate count is
    # sufficient because the experiment reports the exact encoded size separately.
    count = max(1, target_mib * MIB // CONTENT_BYTES)
    records = [{"content": content, "index": index} for index in range(count)]
    return {"args": (records,), "kwargs": {}}


def make_entity_input(candidate_count: int) -> dict[str, Any]:
    """Build a compact in-memory graph with expensive repeated entity formatting."""
    trial = TrialPayload(
        id="trial-0",
        outputs={"artifact": "x" * 100_000},
        metadata={"log": "y" * 100_000},
    )
    candidate = CandidatePayload(
        id="candidate-0",
        description="z" * 10_000,
        trials=[trial] * 40,
    )
    return {"args": ([candidate] * candidate_count,), "kwargs": {}}


def current_serializer(value: Any) -> str:
    """Production path used by before_agent_call and method/tool invocation hooks."""
    return trace_fields(args=value["args"], kwargs=value["kwargs"]).text


def hook_before_call(value: dict[str, Any]) -> str:
    """Exercise the real synchronous hook that runs before an agent method."""
    call_id = f"reproduction-{next(_CALL_IDS)}"
    context = _HOOKS.before_agent_call(
        agent=_AGENT,
        method_name="run",
        args=tuple(value["args"]),
        kwargs=value["kwargs"],
        call_id=call_id,
        parent_call_id=None,
    )
    output = context["span"].attributes["input.value"]
    _HOOKS.after_agent_call(
        agent=_AGENT,
        method_name="run",
        result=None,
        exception=None,
        context=context,
    )
    return output


def legacy_serializer(value: dict[str, Any]) -> str:
    """Recreate the removed custom-object fallback that caused the long stall."""
    return json.dumps(
        value,
        default=lambda item: truncating_pformat(
            item,
            max_chars=50_000,
            max_depth=8,
            max_length=20,
            max_string=2_000,
        ),
    )


def measure(serializer: Callable[[Any], str], value: Any, *, collect: bool = True) -> Measurement:
    if collect:
        gc.collect()
    started = time.perf_counter()
    output = serializer(value)
    elapsed = time.perf_counter() - started
    return Measurement(elapsed_seconds=elapsed, output_chars=len(output))


def measure_child(payload_kind: str, amount: int, serializer_name: str) -> dict[str, float | int]:
    """Measure one serializer after payload allocation in an isolated process."""
    serializers = serializer_variants()
    if payload_kind == "native":
        value = make_method_input(amount)
    elif payload_kind == "entity":
        value = make_entity_input(amount)
    else:  # pragma: no cover - argparse constrains child invocations
        raise ValueError(f"unknown payload kind: {payload_kind}")

    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = measure(serializers[serializer_name], value, collect=False)
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "elapsed_seconds": result.elapsed_seconds,
        "output_chars": result.output_chars,
        "rss_growth_mib": (rss_after - rss_before) / 1024,
    }


def run_isolated(payload_kind: str, amount: int, serializer_name: str) -> dict[str, float | int]:
    """Run one payload/serializer pair in a fresh interpreter."""
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-payload",
            payload_kind,
            "--child-amount",
            str(amount),
            "--child-serializer",
            serializer_name,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def serializer_variants() -> dict[str, Callable[[Any], str]]:
    """Return the serializer variants compared by the experiment."""
    return {
        "current": current_serializer,
        "hook-before": hook_before_call,
        "legacy": legacy_serializer,
    }


async def measure_event_loop_stall(value: Any) -> tuple[Measurement, float]:
    """Return serialization metrics and the largest delayed heartbeat."""
    stop = asyncio.Event()
    heartbeat_delays: list[float] = []

    async def heartbeat() -> None:
        deadline = time.perf_counter() + HEARTBEAT_SECONDS
        while not stop.is_set():
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            now = time.perf_counter()
            heartbeat_delays.append(max(0.0, now - deadline))
            deadline += HEARTBEAT_SECONDS

    task = asyncio.create_task(heartbeat())
    # Let the heartbeat establish its first deadline before synchronous tracing work.
    gc.collect()
    await asyncio.sleep(HEARTBEAT_SECONDS * 2)
    result = measure(hook_before_call, value, collect=False)
    await asyncio.sleep(HEARTBEAT_SECONDS * 2)
    stop.set()
    await task
    return result, max(heartbeat_delays, default=0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes-mib",
        type=int,
        nargs="+",
        default=[8, 64, 256],
        help="Approximate expanded JSON sizes to benchmark (default: 8 64 256)",
    )
    parser.add_argument(
        "--event-loop-mib",
        type=int,
        default=256,
        help="Approximate expanded JSON size for the event-loop test (default: 256)",
    )
    parser.add_argument(
        "--entity-counts",
        type=int,
        nargs="+",
        default=[10, 100, 1000],
        help="Experimentalist-like candidate counts to benchmark (default: 10 100 1000)",
    )
    parser.add_argument(
        "--serializers",
        nargs="+",
        choices=("current", "hook-before", "legacy"),
        default=("current", "hook-before", "legacy"),
        help="Serializer variants to run (default: all)",
    )
    parser.add_argument("--child-payload", choices=("native", "entity"), help=argparse.SUPPRESS)
    parser.add_argument("--child-amount", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--child-serializer",
        choices=("current", "hook-before", "legacy"),
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    child_args = (args.child_payload, args.child_amount, args.child_serializer)
    if any(value is not None for value in child_args):
        if not all(value is not None for value in child_args):
            raise SystemExit("all internal child arguments are required together")
        print(json.dumps(measure_child(*child_args)))
        return

    selected_serializers = tuple(args.serializers)
    print("size_mib exact_json_mib serializer elapsed_s output_chars rss_growth_mib")
    retained: dict[int, dict[str, Any]] = {}
    for size_mib in args.sizes_mib:
        # Exact expansion is intentionally computed in the coordinator, never in
        # a measured serializer process where it would contaminate peak RSS.
        value = make_method_input(size_mib)
        exact_json_mib = len(json.dumps(value)) / MIB
        for name in selected_serializers:
            result = run_isolated("native", size_mib, name)
            print(
                f"{size_mib:8d} {exact_json_mib:14.2f} {name:13s} "
                f"{result['elapsed_seconds']:9.4f} {result['output_chars']:12d} "
                f"{result['rss_growth_mib']:14.1f}"
            )
        if size_mib == args.event_loop_mib:
            retained[size_mib] = value
        else:
            del value

    print("entity_count serializer elapsed_s output_chars rss_growth_mib")
    for entity_count in args.entity_counts:
        for name in selected_serializers:
            result = run_isolated("entity", entity_count, name)
            print(
                f"{entity_count:12d} {name:13s} {result['elapsed_seconds']:9.4f} "
                f"{result['output_chars']:12d} {result['rss_growth_mib']:14.1f}"
            )

    event_value = retained.get(args.event_loop_mib)
    if event_value is None:
        event_value = make_method_input(args.event_loop_mib)
    result, max_delay = asyncio.run(measure_event_loop_stall(event_value))
    print(
        "event_loop "
        f"size_mib={args.event_loop_mib} serialization_s={result.elapsed_seconds:.4f} "
        f"max_heartbeat_delay_s={max_delay:.4f} heartbeat_s={HEARTBEAT_SECONDS:.4f}"
    )


if __name__ == "__main__":
    main()
