# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NeMo OO Agents CLI — extensible command-line toolkit.

Usage:
    nooa eval --config config.yaml  # Run an eval-pipeline job
    nooa start-dev                  # Start the viewer
    nooa traces delete              # Delete traces
    nooa completion install         # Set up shell completions

Adding new commands:
    Drop a .py file in nooa_cli/commands/ — see commands/_template.py

Shell completion:
    eval "$(_NOOA_COMPLETE=bash_source nooa)"    # bash
    eval "$(_NOOA_COMPLETE=zsh_source nooa)"     # zsh
"""

import click

from .commands import discover_commands
from .completion import completion

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


# Thin launcher subcommands that don't touch API keys in *this* process:
# `term` runs a web/PTY server and re-spawns the CLI as a child (which loads
# secrets itself via bootstrap); `completion` just emits a shell script.
# Skipping the secrets preload for them avoids importing the heavy
# ``nooa`` core (~1.5s) on startup — keeping `nooa term` fast and
# its SIGINT shutdown prompt.
_SKIP_SECRETS_PRELOAD = {"term", "completion"}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.version_option(package_name="nemo-labs-oo-agents")
@click.pass_context
def oo(ctx):
    """OO Agents — agent toolkit.

    Extensible CLI for running agents, evaluations, and trace management.
    Add new commands by dropping a .py file in nooa_cli/commands/.
    """
    if ctx.invoked_subcommand in _SKIP_SECRETS_PRELOAD:
        return
    # Load secrets.yaml into the process env before any subcommand runs so
    # non-TUI commands (config show, eval, …) see the same API keys the TUI
    # does. Non-clobbering (shell env wins) and best-effort: a broken or
    # missing secrets file must never block the CLI — but log the failure at
    # debug so it's recoverable rather than silently swallowed.
    try:
        from nooa.secrets import load_secrets_into_env

        load_secrets_into_env()
    except Exception:
        import logging

        logging.getLogger(__name__).debug(
            "Failed to preload secrets.yaml into the environment", exc_info=True
        )


# -- Auto-discover and register all commands from commands/ -----------------
for _name, _cmd in discover_commands():
    oo.add_command(_cmd, name=_name)

# -- Built-in infrastructure commands (not in commands/ because they're meta) -
oo.add_command(completion)


def main():
    """Entry point for console_scripts: nooa <subcommand>."""
    oo()
