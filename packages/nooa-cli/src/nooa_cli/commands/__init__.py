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
"""

import importlib
import pkgutil
from collections.abc import Iterator

import click


def discover_commands() -> Iterator[tuple[str, click.Command]]:
    """Yield (name, command) pairs from all command modules in this package.

    Scans this directory for Python modules, imports each one, and looks
    for a ``command`` attribute that is a ``click.Command`` or ``click.Group``.

    Modules starting with ``_`` are skipped (private / template files).
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
