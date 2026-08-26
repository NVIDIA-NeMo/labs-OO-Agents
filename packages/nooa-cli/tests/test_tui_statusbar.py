# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public API for extensible TUI statusbar items."""

from pathlib import Path

import pytest
from nooa_cli.tui.statusbar import StatusbarContext, StatusbarRegistry


def test_statusbar_renders_configured_items_in_order():
    registry = StatusbarRegistry(load_plugins=False)
    context = StatusbarContext(
        model="provider/claude-example",
        working_directory=Path("/work/repo"),
        context_usage="ctx 20%",
        session_id="12345678-abcd",
        session_title="migration",
    )
    assert registry.render(["model", "cwd", "context", "session"], context) == (
        "example · repo · ctx 20% · migration [12345678]"
    )


def test_statusbar_accepts_registered_extension():
    registry = StatusbarRegistry(load_plugins=False)
    registry.register("branch", lambda context: "dev/tui")
    assert registry.render(["branch"], StatusbarContext("model", Path("."), "")) == "dev/tui"


def test_statusbar_rejects_non_callable_provider():
    registry = StatusbarRegistry(load_plugins=False)
    with pytest.raises(TypeError, match="callable"):
        registry.register("broken", object())  # type: ignore[arg-type]


def test_statusbar_loads_entry_point_provider(monkeypatch):
    class Plugin:
        name = "branch"

        @staticmethod
        def load():
            return lambda context: context.working_directory.name

    monkeypatch.setattr("nooa_cli.tui.statusbar.entry_points", lambda **kwargs: [Plugin()])
    registry = StatusbarRegistry()

    assert "branch" in registry.names()
    assert registry.render(["branch"], StatusbarContext("model", Path("repo"), "")) == "repo"


def test_statusbar_loads_legacy_toolbar_entry_point_provider(monkeypatch):
    class Plugin:
        name = "legacy"

        @staticmethod
        def load():
            return lambda context: "toolbar"

    def fake_entry_points(*, group):
        if group == "nooa_cli.tui.toolbar_items":
            return [Plugin()]
        return []

    monkeypatch.setattr("nooa_cli.tui.statusbar.entry_points", fake_entry_points)
    registry = StatusbarRegistry()

    assert registry.render(["legacy"], StatusbarContext("model", Path("."), "")) == "toolbar"


def test_statusbar_entry_point_wins_over_legacy_duplicate(monkeypatch):
    class Plugin:
        name = "shared"

        def __init__(self, value):
            self.value = value

        def load(self):
            return lambda context: self.value

    def fake_entry_points(*, group):
        if group == "nooa_cli.tui.toolbar_items":
            return [Plugin("legacy")]
        if group == "nooa_cli.tui.statusbar_items":
            return [Plugin("current")]
        return []

    monkeypatch.setattr("nooa_cli.tui.statusbar.entry_points", fake_entry_points)
    registry = StatusbarRegistry()

    assert registry.render(["shared"], StatusbarContext("model", Path("."), "")) == "current"
