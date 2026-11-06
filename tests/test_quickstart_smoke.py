# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Result validation belongs to the release harness, outside the examples."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(scope="module")
def smoke():
    path = Path(__file__).resolve().parents[1] / "scripts" / "quickstart_smoke.py"
    spec = importlib.util.spec_from_file_location("_quickstart_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("value", [None, "", "   "])
def test_rejects_missing_feedback(smoke, value):
    with pytest.raises(ValueError, match="nonempty feedback"):
        smoke.validate_result("01_first_generation_method.py", [(value,)], {})


def test_accepts_feedback_without_exact_wording(smoke):
    smoke.validate_result("01_first_generation_method.py", [("Good product; slow delivery.",)], {})


@pytest.mark.parametrize(
    "result",
    [
        "not a typed result",
        SimpleNamespace(topics=[], summary="Broken feature"),
        SimpleNamespace(topics=[" "], summary="Broken feature"),
        SimpleNamespace(topics=["feature"], summary=" "),
    ],
)
def test_rejects_unpopulated_structured_feedback(smoke, result):
    with pytest.raises(ValueError):
        smoke.validate_result(
            "02_structured_outputs.py", [(result,)], {"FeedbackAnalysis": SimpleNamespace}
        )


def test_accepts_populated_structured_feedback(smoke):
    result = SimpleNamespace(topics=["feature"], summary="A feature is broken.")
    smoke.validate_result(
        "02_structured_outputs.py", [(result,)], {"FeedbackAnalysis": SimpleNamespace}
    )


@pytest.mark.parametrize(
    "fulfill,cost,unavailable",
    [
        ("True", "2.05", "['orange']"),
        ("False", "2.05", "[]"),
        ("False", "nan", "['orange']"),
        ("False", "0", "['orange']"),
        ("False", "6", "['orange']"),
    ],
)
def test_rejects_incorrect_inventory(smoke, fulfill, cost, unavailable):
    with pytest.raises(ValueError):
        smoke.validate_result(
            "03_codeact_tools.py",
            [
                (f"Can fulfill: {fulfill}",),
                (f"Total cost: {cost}",),
                (f"Unavailable items: {unavailable}",),
            ],
            {},
        )


@pytest.mark.parametrize("cost", [1.25, 2.05])
def test_accepts_valid_inventory_cost_policies(smoke, cost):
    smoke.validate_result(
        "03_codeact_tools.py",
        [
            ("Can fulfill: False",),
            (f"Total cost: {cost}",),
            ("Unavailable items: ['orange']",),
        ],
        {},
    )


@pytest.mark.parametrize("output", ["A useful feedback analysis", ""])
def test_harness_runs_source_unchanged_and_checks_main_output(smoke, tmp_path, capsys, output):
    example = tmp_path / "01_first_generation_method.py"
    source = f"print('startup diagnostic')\ndef main():\n    print({output!r})\nmain()\n"
    example.write_text(source)
    if output:
        smoke.run_example(example)
        assert "Validated result:" in capsys.readouterr().out
    else:
        with pytest.raises(ValueError, match="nonempty feedback"):
            smoke.run_example(example)
    assert example.read_text() == source
