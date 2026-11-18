# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Auto-discovered command modules for the nooa CLI.

╔══════════════════════════════════════════════════════════════════════╗
║                    HOW TO ADD A NEW COMMAND                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  1. Create a new .py file in this directory (commands/)              ║
║  2. Define a click command or group                                  ║
║  3. Assign it to a module-level variable named `command`             ║
║  4. That's it — it's automatically registered.                       ║
║                                                                      ║
║  See _template.py for a copy-paste starter.                          ║
║                                                                      ║
║  From *another* package? Register an entry point instead —           ║
║  see "Commands from other packages" below.                           ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝

Convention
----------

Each Python file in this directory that does NOT start with ``_`` is
auto-discovered and registered as a top-level subcommand of ``nooa``.

A command module **must** export a module-level variable named ``command``
that is a ``click.BaseCommand`` (either a ``@click.command()`` or a
``@click.group()``).

Optionally, you can set:
    NAME = "custom-name"      # Override the command name (default: filename)

The file name (minus ``.py``) becomes the subcommand name by default::

    commands/start_dev.py →  nooa start-dev ...

Files starting with ``_`` are ignored (private helpers, templates, etc.).

Minimal example (commands/hello.py)::

    import click

    @click.command()
    @click.argument("name", default="world")
    def command(name):
        \"\"\"Say hello.\"\"\"
        click.echo(f"Hello, {name}!")

Full example with a group (commands/things.py)::

    import click

    @click.group()
    def command():
        \"\"\"Manage things.\"\"\"

    @command.command()
    def list():
        \"\"\"List all things.\"\"\"
        click.echo("thing-1\\nthing-2")

    @command.command()
    @click.argument("name")
    def create(name):
        \"\"\"Create a new thing.\"\"\"
        click.echo(f"Created {name}")

Commands from other packages
----------------------------

A package installed alongside ``nooa-cli`` can contribute top-level
subcommands through the ``nooa_cli.commands`` entry-point group. The
entry-point name is the subcommand name; the object it points at must be a
``click.Command``::

    # pyproject.toml of the contributing package
    [project.entry-points."nooa_cli.commands"]
    hello   = "my_package.cli.hello:command"
    goodbye = "my_package.cli.goodbye:command"

Rules:

* **Built-ins win name collisions.** A plugin cannot shadow ``eval``,
  ``config`` or any other command shipped here — it is logged and skipped.
* **Broken plugins are skipped, not fatal.** An entry point that fails to
  import, or that resolves to something other than a ``click.Command``, logs a
  warning and is left out; ``nooa`` keeps working.
* Plugins are registered in entry-point-name order, so ``nooa --help`` is
  stable regardless of install order.
* **Keep imports lazy**, exactly as for in-repo commands: every registered
  entry point is loaded on *every* ``nooa`` invocation, including
  ``nooa --help``, so heavy imports belong inside the command handler.
"""

import importlib
import logging
import pkgutil
from collections.abc import Iterator
from importlib import metadata

import click

logger = logging.getLogger(__name__)

#: Entry-point group external packages use to contribute subcommands.
PLUGIN_ENTRY_POINT_GROUP = "nooa_cli.commands"


def _builtin_commands() -> Iterator[tuple[str, click.Command]]:
    """Yield (name, command) pairs from the command modules in this package.

    Scans this directory for Python modules, imports each one, and looks
    for a ``command`` attribute that is a ``click.Command`` or ``click.Group``.

    Modules starting with ``_`` are skipped (private / template files).

    Unlike the plugin pass, a malformed module here raises: it is a bug in
    this repository and should be loud.
    """
    package_path = __path__
    package_name = __name__

    for module_info in pkgutil.iter_modules(package_path):
        # Skip private modules (_template, _helpers, etc.)
        if module_info.name.startswith("_"):
            continue

        module = importlib.import_module(f"{package_name}.{module_info.name}")

        # The module must export `command`
        cmd = getattr(module, "command", None)
        if cmd is None:
            continue

        if not isinstance(cmd, click.Command):
            raise TypeError(
                f"commands/{module_info.name}.py: `command` must be a click.Command or click.Group, "
                f"got {type(cmd).__name__}. Use @click.command() or @click.group()."
            )

        # Allow overriding the command name
        name = getattr(module, "NAME", module_info.name)

        yield name, cmd


def _plugin_commands() -> Iterator[tuple[str, click.Command]]:
    """Yield (name, command) pairs contributed by other installed packages.

    Reads the ``nooa_cli.commands`` entry-point group. The entry-point name
    becomes the subcommand name, and ``ep.load()`` must return a
    ``click.Command``.

    Entry points are sorted by name so the command list does not depend on
    install order. Every failure is logged at WARNING and skipped: these are
    arbitrary third-party imports, and a broken one must never make ``nooa``
    unusable.
    """
    try:
        eps = metadata.entry_points(group=PLUGIN_ENTRY_POINT_GROUP)
    except Exception:  # noqa: BLE001 — importlib.metadata can raise on broken installs
        logger.warning(
            "Failed to enumerate %r entry points", PLUGIN_ENTRY_POINT_GROUP, exc_info=True
        )
        return

    for ep in sorted(eps, key=lambda e: e.name):
        try:
            cmd = ep.load()
        except Exception:  # noqa: BLE001 — third-party entry-point code, be defensive
            logger.warning("CLI plugin entry-point %r failed to load", ep.name, exc_info=True)
            continue

        if not isinstance(cmd, click.Command):
            logger.warning(
                "CLI plugin entry-point %r resolved to %s, not a click.Command; skipping",
                ep.name,
                type(cmd).__name__,
            )
            continue

        yield ep.name, cmd


def discover_commands() -> Iterator[tuple[str, click.Command]]:
    """Yield (name, command) pairs for every subcommand of ``nooa``.

    Built-in commands from this package come first, then commands contributed
    by other packages through the ``nooa_cli.commands`` entry-point group.
    A plugin whose name is already taken is logged and skipped, so built-ins
    always win and each name is yielded exactly once.
    """
    seen: set[str] = set()

    for name, cmd in _builtin_commands():
        seen.add(name)
        yield name, cmd

    for name, cmd in _plugin_commands():
        if name in seen:
            logger.warning(
                "CLI plugin entry-point %r shadows an existing command; ignoring the plugin",
                name,
            )
            continue
        seen.add(name)
        yield name, cmd
