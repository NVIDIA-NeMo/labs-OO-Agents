# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Obsolete isolation requests must fail before cells or ARC processes start."""

import argparse
import ast
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from nooa.config import CodeActConfig

ARC = Path(__file__).resolve().parents[1] / "examples/arc_agi_3"


@pytest.mark.parametrize(
    "settings",
    [
        {"execution_backend": "sandbox"},
        {"sandbox": {}},
        {"sandbox": None},
        {"execution_backend": "inprocess", "sandbox": {"require": False}},
    ],
)
@pytest.mark.parametrize("serialized", [False, True])
def test_removed_codeact_settings_fail_with_migration(settings, serialized):
    with pytest.raises(ValidationError, match="OS sandbox backend.*removed") as error:
        if serialized:
            CodeActConfig.model_validate_json(json.dumps(settings))
        else:
            CodeActConfig(**settings)
    assert "cooperative" in str(error.value)
    assert "external OS isolation" in str(error.value)


def test_explicit_inprocess_and_unrelated_extra_policy_are_preserved():
    assert CodeActConfig(execution_backend="inprocess", prefill=None) == CodeActConfig(prefill=None)
    assert CodeActConfig(unrelated_extra="historically ignored", prefill=None) == CodeActConfig(
        prefill=None
    )


@pytest.fixture
def arc_module(monkeypatch):
    monkeypatch.syspath_prepend(str(ARC))
    for key in list(os.environ):
        if key.startswith("ARC_SANDBOX_"):
            monkeypatch.delenv(key)

    def load(name):
        spec = importlib.util.spec_from_file_location(f"_removal_{name}", ARC / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return load


@pytest.mark.parametrize(
    "flag",
    [
        "--sandbox-cells",
        "--sandbox-workspace=/tmp/work",
        "--sandbox-mem-mb=1024",
        "--sandbox-cpu-s=90",
        "--sandbox-require",
    ],
)
def test_arc_removed_flags_fail_early(arc_module, monkeypatch, capsys, flag):
    runner = arc_module("run_solver")
    monkeypatch.setattr(sys, "argv", ["run_solver", "--game", "test", flag])
    with pytest.raises(SystemExit) as error:
        runner.parse_args()
    assert error.value.code == 2
    assert "per-cell CodeAct OS sandbox was removed" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["drop", "full", "inproc", "off"])
def test_arc_whole_agent_modes_remain(arc_module, monkeypatch, mode):
    runner = arc_module("run_solver")
    monkeypatch.setattr(
        sys, "argv", ["run_solver", "--game", "test", "--variant", "memory", "--sandbox", mode]
    )
    assert runner.parse_args().sandbox == mode


@pytest.mark.parametrize("value", [None, False, {}, {"cells": False}, {"cells": True}])
def test_arc_obsolete_yaml_is_never_ignored(arc_module, tmp_path, value):
    runner = arc_module("run_multi")
    config = tmp_path / "fleet.yaml"
    config.write_text(json.dumps({"sandbox": value}))
    with pytest.raises(ValueError, match="Obsolete YAML sandbox: block"):
        runner.load_config(str(config), argparse.Namespace())


@pytest.mark.parametrize("key", ["CELLS", "WORKSPACE", "MEM_MB", "CPU_S", "REQUIRE", "UNKNOWN"])
@pytest.mark.parametrize("entry", ["run_solver", "run_multi", "solver_agent"])
def test_arc_legacy_environment_rejected_every_entrypoint(arc_module, monkeypatch, key, entry):
    # Even CELLS=0 must not silently acknowledge an obsolete isolation configuration.
    monkeypatch.setenv(f"ARC_SANDBOX_{key}", "0")
    if entry == "solver_agent":
        # Compile the actual config factory without importing the SDK or initializing an agent.
        factory = next(
            node
            for node in ast.parse((ARC / "solver_agent.py").read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_arc_cell_config"
        )
        from cell_config import reject_cell_sandbox_environment

        namespace = {
            "CodeActConfig": CodeActConfig,
            "reject_cell_sandbox_environment": reject_cell_sandbox_environment,
        }
        exec(
            compile(ast.Module(body=[factory], type_ignores=[]), "solver_agent.py", "exec"),
            namespace,
        )
        with pytest.raises(ValueError, match=f"ARC_SANDBOX_{key}"):
            namespace["_arc_cell_config"]()
    elif entry == "run_solver":
        runner = arc_module(entry)
        monkeypatch.setattr(sys, "argv", ["run_solver", "--game", "test"])
        with pytest.raises(SystemExit) as error:
            runner.parse_args()
        assert error.value.code == 2
    else:
        with pytest.raises(ValueError, match=f"ARC_SANDBOX_{key}"):
            arc_module(entry).load_config(None, argparse.Namespace())
