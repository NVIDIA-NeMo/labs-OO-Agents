# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Known dependency regressions must not impose a ceiling on future fixes."""

import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement


@pytest.mark.parametrize(
    "version,allowed",
    [
        ("1.105.7", True),
        ("1.106.0.dev1", True),
        ("1.106.0.dev2", False),
        ("1.106.0", False),
        ("1.108.1", False),
        ("1.109.0rc1", True),
        # Future patch/minor releases stay eligible even within affected branches.
        ("1.106.8", True),
        ("1.108.2", True),
        ("1.109.0", True),
        ("2.0.0", True),
    ],
)
def test_litellm_excludes_known_regression_and_allows_future_fixes(version, allowed):
    """Exclude measured regressions without placing a ceiling on future fixes."""
    project = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text())
    requirements = [Requirement(value) for value in project["project"]["dependencies"]]
    requirement = next(value for value in requirements if value.name == "litellm")
    assert requirement.specifier.contains(version, prereleases=True) is allowed
