# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Proposal layout stays readable with long values, narrow terminals and no color."""

import os
import re
from copy import deepcopy
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from nooa_cli.commands import _connect_view as view
from nooa_cli.commands._connect_wizard import WizardState, show_guided_proposal


def state(*, decision=False, tests=True, no_probe=False):
    entry = {
        "client_type": "decision" if decision else "completion",
        "media_inputs": ["audio"] if not decision else [],
        "context_window": 262144,
        "max_tokens": 32768,
        "reasoning_levels": {},
        "provenance": {"catalogue_limits": {"max_completion_tokens": 32768}},
    }
    return WizardState(
        model="nvidia/very-long-model-name-nemotron-3-ultra-550b-a55b",
        endpoint="https://integrate.api.nvidia.com/very-long-endpoint-path/v1",
        api_style="systemone" if decision else "chat",
        proposal=SimpleNamespace(entry=entry),
        budget_tokens=8192,
        test_media_inputs=["image"]
        if decision and tests
        else ["image", "audio", "video", "pdf"]
        if tests
        else None,
        no_probe=no_probe,
        decision_image_transport="typesafe_multipart" if decision else None,
        typesafe_danger_zone_images=decision,
    )


@pytest.mark.parametrize("columns", [24, 40, 80, 120])
@pytest.mark.parametrize("no_color", [False, True])
@pytest.mark.parametrize("decision", [False, True])
def test_proposal_layout_width_color_pending_tests_and_no_boilerplate(
    monkeypatch, columns, no_color, decision
):
    monkeypatch.setattr(
        view.shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((columns, 24))
    )
    if no_color:
        monkeypatch.setenv("NO_COLOR", "")
    else:
        monkeypatch.delenv("NO_COLOR", raising=False)
    current = state(decision=decision)
    before = deepcopy(current.proposal.entry)

    @click.command()
    def render():
        show_guided_proposal(current, deferred_style=True)

    result = CliRunner().invoke(render, color=True)
    assert result.exit_code == 0, result.output
    clean = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert all(len(row) <= columns for row in clean.splitlines())
    assert ("\x1b[" not in result.output) if no_color else ("\x1b[" in result.output)
    plain = " ".join(clean.split())
    for label in (
        "Model:",
        "Endpoint:",
        "Interface:",
        "Discovery:",
        "Media inputs",
        "Tests",
        "Media samples:",
    ):
        assert label in plain
    assert plain.count("Media samples:") == 1
    assert "Images: Test before enabling" in plain
    if decision:
        assert "Context window:" not in plain and "Reply limit:" not in plain
        assert "Audio:" not in plain and "PDFs:" not in plain
    else:
        assert "Context window: 262,144 tokens" in plain
        assert "Reply limit: 32,768 tokens" in plain
        assert "Audio: Enabled; test again" in plain
    for phrase in (
        "approved",
        "approves",
        "incur charges",
        "spending cap",
        "Selected media tests:",
    ):
        assert phrase not in plain
    assert current.proposal.entry == before


def test_no_tests_summary_and_redaction_before_wrapping(monkeypatch):
    monkeypatch.setattr(
        view.shutil, "get_terminal_size", lambda *a, **kw: os.terminal_size((40, 24))
    )
    current = state(tests=False, no_probe=True)
    current.api_key = "private-value"
    current.discovery_api_key = "catalogue-value"
    current.model += "/private-value"
    current.endpoint += "/catalogue-value"
    current.proposal.entry["reasoning_levels"] = {"private-value": {}}

    @click.command()
    def render():
        show_guided_proposal(current)

    result = CliRunner().invoke(render)
    assert result.exit_code == 0
    assert "private-value" not in result.output and "catalogue-value" not in result.output
    plain = " ".join(result.output.split())
    assert "Media samples:" not in plain
    assert "Status: Off" in plain
    assert "Images: Off" in plain
