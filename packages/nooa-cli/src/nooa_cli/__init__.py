# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA OO Agents CLI — extensible command-line toolkit.

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

import os
import sys

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
    """OO Agents — agent toolkit.

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


# Same policy and wording as nooa._platform, checked here without importing
# nooa so the CLI stays lazy (tests/test_cli_platform_check.py keeps them equal).
_SKIP_PLATFORM_CHECK_ENV = "NOOA_SKIP_PLATFORM_CHECK"
_UNSUPPORTED_PLATFORM_MESSAGE = (
    "nooa does not support native Windows; use WSL 2 "
    "(https://learn.microsoft.com/windows/wsl/install). "
    f"Setting the {_SKIP_PLATFORM_CHECK_ENV} environment variable to 1 skips this check: "
    "only the pure-Python core and `nooa connect` are tested on native Windows, "
    "and other issues found only there will not be fixed."
)


def _refuse_unsupported_platform() -> None:
    """Exit with one clean line on native Windows unless the user opted in."""
    if sys.platform != "win32":
        return
    if os.environ.get(_SKIP_PLATFORM_CHECK_ENV, "").strip().lower() in {"1", "true", "yes"}:
        return
    raise SystemExit(f"Error: {_UNSUPPORTED_PLATFORM_MESSAGE}")


def _use_utf8_output() -> None:
    """Write UTF-8 when output is redirected on native Windows.

    Redirected stdout/stderr use the ANSI code page there (for example cp1252),
    which cannot encode the CLI's check marks and arrows and would crash the
    command midway. Consoles already use UTF-8, so this changes nothing there.
    """
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        encoding = (
            (getattr(stream, "encoding", None) or "").lower().replace("-", "").replace("_", "")
        )
        if encoding != "utf8" and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def main():
    """Entry point for console_scripts: nooa <subcommand>."""
    _refuse_unsupported_platform()
    _use_utf8_output()
    oo()
