# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run an unchanged quickstart and validate the values its main() prints."""

from __future__ import annotations

import ast
import math
import runpy
import sys
from pathlib import Path
from typing import Any


def validate_result(name: str, printed: list[tuple[Any, ...]], namespace: dict[str, Any]) -> None:
    """Check useful output contracts without requiring exact LLM wording."""
    if name == "01_first_generation_method.py":
        if (
            len(printed) != 1
            or len(printed[0]) != 1
            or not isinstance(printed[0][0], str)
            or not printed[0][0].strip()
        ):
            raise ValueError("Expected a nonempty feedback analysis")
    elif name == "02_structured_outputs.py":
        if len(printed) != 1 or len(printed[0]) != 1:
            raise ValueError("Expected one structured feedback analysis")
        result = printed[0][0]
        if not isinstance(result, namespace["FeedbackAnalysis"]):
            raise ValueError("Expected a validated FeedbackAnalysis instance")
        if not result.topics or any(not topic.strip() for topic in result.topics):
            raise ValueError("Expected nonempty feedback topics")
        if not result.summary.strip():
            raise ValueError("Expected a nonempty feedback summary")
    elif name == "03_codeact_tools.py":
        if len(printed) != 3 or any(
            len(values) != 1 or not isinstance(values[0], str) for values in printed
        ):
            raise ValueError("Expected the three inventory result fields")
        fields = dict(values[0].split(": ", 1) for values in printed)
        if fields.get("Can fulfill") != "False":
            raise ValueError("The out-of-stock orange must prevent fulfillment")
        if ast.literal_eval(fields.get("Unavailable items", "None")) != ["orange"]:
            raise ValueError("Expected orange to be the unavailable item")
        cost = float(fields.get("Total cost", "nan"))
        # The example does not specify whether unavailable items count toward cost.
        # Validate a finite cost within the supplied budget, without choosing a policy.
        if not math.isfinite(cost) or not 0 < cost <= 5.0:
            raise ValueError("Expected a positive finite inventory cost within budget")
    else:
        raise ValueError(f"No release validator for {name}")


def run_example(example: Path) -> None:
    """Execute an unchanged example and validate its captured application results."""
    printed: list[tuple[Any, ...]] = []

    def capture_print(*values: Any, **kwargs: Any) -> None:
        # Capture application results, excluding runtime diagnostics or REPL prints.
        """Capture result prints from the example entry point while forwarding output."""
        caller = sys._getframe(1)
        if caller.f_code.co_name == "main" and caller.f_code.co_filename == str(example):
            printed.append(values)
        print(*values, **kwargs)

    namespace = runpy.run_path(
        str(example), init_globals={"print": capture_print}, run_name="__main__"
    )
    if not printed:
        raise ValueError(
            f"Release harness captured no print output from main() in {example.name}; "
            "check whether the example's entry point or output mechanism changed"
        )
    validate_result(example.name, printed, namespace)
    print(f"Validated result: {example.name}")


if __name__ == "__main__":
    run_example(Path(sys.argv[1]).resolve())
