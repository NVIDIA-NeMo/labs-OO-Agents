# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-save imports do not treat another route's endpoint ceiling as current."""

import json
from copy import deepcopy

import pytest
import yaml
from click.testing import CliRunner
from nooa_cli.commands.connect import command

from nooa.unifiedllm import connect
from tests.unifiedllm.connect.test_connect_endpoint_scope import bounded_entry


@pytest.mark.parametrize(
    "patch,effective",
    [
        ({}, 64000),
        ({"api_base": "https://b.test/v1"}, 128000),
        ({"api_key_env": "B_KEY"}, 128000),
        ({"temperature": 0.5}, 64000),
    ],
)
@pytest.mark.parametrize("explicit_flag", [False, True])
def test_stage_save_checks_endpoint_scope_before_reusing_imported_facts(
    tmp_path, patch, effective, explicit_flag
):
    entry = deepcopy(bounded_entry())
    entry.update(patch, context_window=128000)
    source = tmp_path / "report.json"
    source.write_text(json.dumps({"alias": "local", "entry": entry}))
    target = tmp_path / "models.yaml"
    args = ["--stage", "save", "--input", str(source), "--output", str(target), "--yes"]
    if explicit_flag:
        args += ["--context-window", "128000"]
    result = CliRunner().invoke(command, args)
    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(target.read_text())["models"]["local"]
    assert saved["context_window"] == effective
    assert json.loads(result.stdout)["data"]["entry"]["context_window"] == effective
    assert connect.scoped_endpoint_limits(entry).get("max_input_tokens") == (
        64000 if effective == 64000 else None
    )
