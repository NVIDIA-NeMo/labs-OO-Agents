# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static contracts for spec's annotation, decorator and imperative forms."""

import json
import subprocess
import sys
from pathlib import Path


def test_spec_preserves_decorated_types(tmp_path):
    source = tmp_path / "contract.py"
    source.write_text(
        """from typing import assert_type
from pydantic import BaseModel
from nooa.agentdoc._docs import SpecAnnotation, spec

assert_type(spec(), SpecAnnotation)
assert_type(spec(expand=False), SpecAnnotation)

@spec(expand=False)
class Model(BaseModel):
    value: str

assert_type(Model(value="test"), Model)
assert_type(Model(value="test").value, str)
assert_type(spec(Model, "value", expand=False), None)
assert_type(spec(Model, description="Example"), None)

@spec(hidden=True)
def length(value: str) -> int:
    return len(value)

assert_type(length("test"), int)
"""
    )
    config = tmp_path / "pyrightconfig.json"
    config.write_text(
        json.dumps(
            {
                "pythonVersion": "3.12",
                "extraPaths": [str(Path(__file__).resolve().parents[2] / "src")],
                "include": [source.name],
            }
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            "--pythonpath",
            sys.executable,
            "--project",
            str(config),
            str(source),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
