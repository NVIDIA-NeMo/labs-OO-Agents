# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA Labs Object Oriented Agents (NOOA) CLI.

Usage:
    nooa eval --config config.yaml  # Run an eval-pipeline job
    nooa start-dev                  # Start the viewer
    nooa traces delete              # Delete traces
    nooa completion install         # Set up shell completions

Adding new commands:
    In this package: drop a .py file in nooa_cli/commands/ — see
    commands/_template.py
    From another package: register an entry point in the
    "nooa_cli.commands" group — see nooa_cli/commands/__init__.py

Shell completion:
    eval "$(_NOOA_COMPLETE=bash_source nooa)"    # bash
    eval "$(_NOOA_COMPLETE=zsh_source nooa)"     # zsh
"""

import click

from .commands import discover_commands
from .completion import completion

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


# `completion` just emits a shell script and doesn't need the ~1.5s core
# import cost or secrets preload.
_SKIP_SECRETS_PRELOAD = {"completion"}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(package_name="nooa-cli")
@click.pass_context
def oo(ctx):
    """NOOA — NVIDIA Labs Object Oriented Agents.

    Extensible CLI for running agents, evaluations, and trace management.
    Add new commands by dropping a .py file in nooa_cli/commands/, or from
    another package via the "nooa_cli.commands" entry-point group.
    """
    if ctx.invoked_subcommand in _SKIP_SECRETS_PRELOAD:
        return
    # Load secrets.yaml into the process env before any subcommand runs so
    # all commands (config show, eval, …) see the same API keys.
    # Non-clobbering (shell env wins) and best-effort: a broken or missing
    # secrets file must never block the CLI — but log the failure at debug
    # so it's recoverable rather than silently swallowed.
    try:
        from nooa.secrets import load_secrets_into_env

        load_secrets_into_env()
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Failed to preload secrets.yaml into the environment", exc_info=True
        )


# -- Auto-discover and register all commands (built-ins + entry-point plugins) --
for _name, _cmd in discover_commands():
    oo.add_command(_cmd, name=_name)

# -- Built-in infrastructure commands (not in commands/ because they're meta) -
# Registered *after* discovery on purpose: `completion` is not part of
# discover_commands(), so this ordering is what stops a third-party plugin
# named "completion" from taking the name.
oo.add_command(completion)


def main():
    """Entry point for console_scripts: nooa <subcommand>."""
    oo()
