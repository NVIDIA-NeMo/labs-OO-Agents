# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for third-party subcommands contributed via the entry-point group.

Fake entry points (only ``.name`` and ``.load()`` are used) are injected by
rebinding the ``metadata`` global of ``nooa_cli.commands`` — no real
distribution needs to be installed, and the real ``importlib.metadata`` is
left alone for every other caller.
"""

import importlib
import logging
import sys
import types

import click
import pytest
from nooa_cli import commands as commands_pkg
from nooa_cli.commands import PLUGIN_ENTRY_POINT_GROUP, discover_commands

BUILTIN_NAMES = {"eval", "config", "start-dev", "tui"}


class FakeEntryPoint:
    """Minimal stand-in for ``importlib.metadata.EntryPoint``."""

    def __init__(self, name: str, value=None, error: Exception | None = None):
        self.name = name
        self._value = value
        self._error = error

    def load(self):
        if self._error is not None:
            raise self._error
        return self._value


def make_command(help_text: str = "A plugin command.") -> click.Command:
    @click.command(name="name-from-the-command-object")
    def _cmd():
        click.echo(help_text)

    return _cmd


def _install(monkeypatch, fake) -> None:
    """Rebind the module's ``metadata`` global rather than patching the real
    ``importlib.metadata``, so the fake is invisible to unrelated callers."""
    monkeypatch.setattr(commands_pkg, "metadata", types.SimpleNamespace(entry_points=fake))


def patch_entry_points(monkeypatch, eps) -> None:
    """Install a fake ``entry_points`` that returns ``eps``.

    The replacement asserts the group it is queried with. That assertion is
    load-bearing: the real call sits inside a ``try/except Exception``, so a
    fake with the wrong signature would raise ``TypeError``, get swallowed,
    and make the "plugin skipped" tests pass against a completely broken
    call path.
    """

    def _fake_entry_points(*, group):
        assert group == PLUGIN_ENTRY_POINT_GROUP
        return eps

    _install(monkeypatch, _fake_entry_points)


def patch_entry_points_raising(monkeypatch, error: Exception) -> None:
    def _fake_entry_points(*, group):
        assert group == PLUGIN_ENTRY_POINT_GROUP
        raise error

    _install(monkeypatch, _fake_entry_points)


def test_group_name_is_stable():
    """Third parties hard-code this string in pyproject.toml — pin it.

    Every other test fakes ``entry_points``, so a typo here would otherwise
    pass the whole suite while breaking the only thing consumers depend on.
    """
    assert PLUGIN_ENTRY_POINT_GROUP == "nooa_cli.commands"


def test_plugin_command_is_registered_alongside_builtins(monkeypatch):
    plugin = make_command()
    patch_entry_points(monkeypatch, [FakeEntryPoint("chat", plugin)])

    found = dict(discover_commands())

    assert found["chat"] is plugin
    assert BUILTIN_NAMES <= set(found)


def test_entry_point_name_becomes_the_subcommand_name(monkeypatch):
    plugin = make_command()
    patch_entry_points(monkeypatch, [FakeEntryPoint("term", plugin)])

    found = dict(discover_commands())

    assert "term" in found
    # The command object's own name is ignored, just like NAME/filename for
    # built-ins — the entry-point name wins.
    assert plugin.name == "name-from-the-command-object"
    assert "name-from-the-command-object" not in found


def test_plugins_are_sorted_by_entry_point_name(monkeypatch):
    patch_entry_points(
        monkeypatch,
        [
            FakeEntryPoint("zeta", make_command()),
            FakeEntryPoint("alpha", make_command()),
            FakeEntryPoint("mid", make_command()),
        ],
    )

    names = [name for name, _ in discover_commands()]
    plugin_order = [n for n in names if n in {"zeta", "alpha", "mid"}]

    assert plugin_order == ["alpha", "mid", "zeta"]


def test_plugin_cannot_shadow_a_builtin(monkeypatch, caplog):
    builtin_config = importlib.import_module("nooa_cli.commands.config").command
    patch_entry_points(monkeypatch, [FakeEntryPoint("config", make_command())])

    with caplog.at_level(logging.WARNING, logger="nooa_cli.commands"):
        pairs = list(discover_commands())

    names = [name for name, _ in pairs]
    assert names.count("config") == 1
    assert dict(pairs)["config"] is builtin_config
    assert any(
        "shadows" in record.getMessage() and "config" in record.getMessage()
        for record in caplog.records
    )


def test_two_plugins_with_the_same_name_yield_one_command(monkeypatch, caplog):
    """Two installed distributions can register the same entry-point name.

    Without deduping, ``add_command`` would be called twice and whichever
    distribution sorts later would silently win — the install-order
    instability that sorting exists to prevent.
    """
    first, second = make_command(), make_command()
    patch_entry_points(
        monkeypatch,
        [FakeEntryPoint("chat", first), FakeEntryPoint("chat", second)],
    )

    with caplog.at_level(logging.WARNING, logger="nooa_cli.commands"):
        pairs = list(discover_commands())

    names = [name for name, _ in pairs]
    assert names.count("chat") == 1
    assert dict(pairs)["chat"] is first
    assert any(
        "shadows" in record.getMessage() and "chat" in record.getMessage()
        for record in caplog.records
    )


def test_plugin_that_fails_to_load_is_skipped(monkeypatch, caplog):
    healthy = make_command()
    patch_entry_points(
        monkeypatch,
        [
            FakeEntryPoint("broken", error=ImportError("no such module")),
            FakeEntryPoint("healthy", healthy),
        ],
    )

    with caplog.at_level(logging.WARNING, logger="nooa_cli.commands"):
        found = dict(discover_commands())

    assert "broken" not in found
    assert found["healthy"] is healthy
    assert BUILTIN_NAMES <= set(found)
    assert any("broken" in record.getMessage() for record in caplog.records)


def test_plugin_that_is_not_a_click_command_is_skipped(monkeypatch, caplog):
    patch_entry_points(monkeypatch, [FakeEntryPoint("bogus", value="not a command")])

    with caplog.at_level(logging.WARNING, logger="nooa_cli.commands"):
        found = dict(discover_commands())

    assert "bogus" not in found
    assert BUILTIN_NAMES <= set(found)
    assert any("bogus" in record.getMessage() for record in caplog.records)


def test_broken_distribution_metadata_is_not_fatal(monkeypatch, caplog):
    patch_entry_points_raising(monkeypatch, RuntimeError("corrupt metadata"))

    with caplog.at_level(logging.WARNING, logger="nooa_cli.commands"):
        found = dict(discover_commands())

    assert BUILTIN_NAMES <= set(found)
    assert any("Failed to enumerate" in record.getMessage() for record in caplog.records)


def test_no_plugins_installed_yields_the_builtins(monkeypatch):
    patch_entry_points(monkeypatch, [])

    found = dict(discover_commands())

    assert BUILTIN_NAMES <= set(found)


def test_builtin_scan_still_raises_on_a_bad_command_attribute(monkeypatch):
    """The built-in path is deliberately *not* forgiving.

    A malformed module in this repo is a bug and must stay loud, unlike a
    malformed third-party plugin. Extracting the scan into
    ``_builtin_commands()`` is exactly the refactor that could silently lose
    this, so pin it.
    """
    # Seed the import cache instead of stubbing importlib.import_module, so
    # the fake is only visible to this one lookup and not to pytest itself.
    monkeypatch.setitem(
        sys.modules,
        "nooa_cli.commands.bogus_builtin",
        types.SimpleNamespace(command="not a click.Command"),
    )
    monkeypatch.setattr(
        commands_pkg,
        "pkgutil",
        types.SimpleNamespace(
            iter_modules=lambda *_args: [types.SimpleNamespace(name="bogus_builtin")]
        ),
    )

    with pytest.raises(TypeError, match="must be a click.Command"):
        list(discover_commands())
