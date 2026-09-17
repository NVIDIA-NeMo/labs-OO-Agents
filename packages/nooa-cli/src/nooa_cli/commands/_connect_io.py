# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Scoped input/output for embedding the unchanged ordered Connect wizard.

The console is the default. A native host can supply prompts, output, and an
async runner for one invocation without redirecting process streams or changing
module globals used by another session.
"""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar

import click

_host = ContextVar("connect_wizard_host", default=None)


def current_host():
    """Return the adapter for this invocation, or None for the console."""
    return _host.get()


@contextmanager
def use_host(host):
    """Bind an adapter to this invocation and restore the previous binding."""
    token = _host.set(host)
    try:
        yield
    finally:
        _host.reset(token)


def echo(message=None, **kwargs):
    """Route wizard output to its host without capturing unrelated logs."""
    host = current_host()
    if host is None:
        click.echo(message, **kwargs)
    else:
        host.echo("" if message is None else click.unstyle(str(message)), **kwargs)


def run_async(awaitable):
    """Run a library operation on the host loop, or a fresh console loop."""
    host = current_host()
    return asyncio.run(awaitable) if host is None else host.run(awaitable)
