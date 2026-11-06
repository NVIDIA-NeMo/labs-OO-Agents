# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The benchmark agent registry: built-in agents and ``nooa_bench.agents`` entry points."""

from __future__ import annotations

import pytest
from click.testing import CliRunner
from nooa_bench import agents, runner
from nooa_bench.bench_agent import BenchAgent
from nooa_bench.cli import main


class _HeadlessAgent:
    TURN_METHODS = ("handle_batch",)
    calls: list = []

    @classmethod
    async def run_headless(cls, task, **kwargs):
        cls.calls.append((task, kwargs))
        return {"status": "done", "task": task}


class _NotHeadless:
    pass


class _EntryPoint:
    """An installed entry point: ``load()`` is all the registry calls."""

    def __init__(self, target: type) -> None:
        self.target = target
        self.loads = 0

    def load(self) -> type:
        self.loads += 1
        return self.target


def _installed(monkeypatch, **targets: type) -> dict[str, _EntryPoint]:
    """Make the entry-point group hold exactly these agents."""
    points = {name: _EntryPoint(target) for name, target in targets.items()}
    monkeypatch.setattr(agents, "_entry_points", lambda: points)
    return points


def test_builtin_agents_load_without_an_entry_point(monkeypatch):
    _installed(monkeypatch)
    assert agents.load_agent("bench") is BenchAgent
    assert agents.available_agents() == ["bench", "rlm"]


def test_an_entry_point_agent_loads_lazily(monkeypatch):
    points = _installed(monkeypatch, fake=_HeadlessAgent)
    assert "fake" in agents.available_agents()
    assert points["fake"].loads == 0  # listing does not load
    assert agents.load_agent("fake") is _HeadlessAgent
    assert runner._import_agent_class("fake") is _HeadlessAgent


def test_a_missing_atom_agent_says_to_install_nooa_atom(monkeypatch):
    _installed(monkeypatch)
    with pytest.raises(agents.AgentNotInstalledError) as raised:
        agents.load_agent("atom")
    message = str(raised.value)
    assert "No benchmark agent named 'atom' is installed" in message
    assert "install nooa-atom" in message
    assert "bench, rlm" in message


def test_an_unknown_agent_lists_the_installed_ones(monkeypatch):
    _installed(monkeypatch)
    with pytest.raises(ValueError, match="installed: bench, rlm"):
        runner._import_agent_class("nope")


def test_the_real_entry_point_group_is_queried():
    # Whatever is installed, the group lookup itself works.
    points = agents._entry_points()
    assert all(point.group == agents.ENTRY_POINT_GROUP for point in points.values())


def test_nooa_bench_run_without_atom_is_a_clear_usage_error(monkeypatch, tmp_path):
    _installed(monkeypatch)
    outcome = CliRunner().invoke(main, ["run", "--workspace", str(tmp_path), "task"])
    assert outcome.exit_code == 2
    assert "install nooa-atom" in outcome.output


def test_nooa_bench_run_uses_the_registry(monkeypatch, tmp_path):
    _installed(monkeypatch, fake=_HeadlessAgent)
    _HeadlessAgent.calls.clear()
    outcome = CliRunner().invoke(
        main, ["run", "--agent", "fake", "--workspace", str(tmp_path), "do it"]
    )
    assert outcome.exit_code == 0, outcome.output
    assert '"task": "do it"' in outcome.stdout
    [(task, kwargs)] = _HeadlessAgent.calls
    assert task == "do it"
    assert kwargs["workspace"] == tmp_path.resolve()
    assert kwargs["max_turns"] is None
    assert kwargs["turn_method"] == "handle_batch"


def test_nooa_bench_run_checks_the_agents_turn_methods(monkeypatch, tmp_path):
    _installed(monkeypatch, fake=_HeadlessAgent)
    outcome = CliRunner().invoke(
        main,
        ["run", "--agent", "fake", "--workspace", str(tmp_path), "--turn-method", "x", "t"],
    )
    assert outcome.exit_code == 2
    assert "handle_batch" in outcome.output


def test_an_agent_without_run_headless_is_refused(monkeypatch, tmp_path):
    _installed(monkeypatch, plain=_NotHeadless)
    outcome = CliRunner().invoke(
        main, ["run", "--agent", "plain", "--workspace", str(tmp_path), "t"]
    )
    assert outcome.exit_code == 2
    assert "cannot run headless" in outcome.output
