# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native presentation for the same ordered wizard used by `nooa connect`.

The synchronous console workflow runs in a worker. Every prompt and async
library operation runs on the host loop; cancellation stops and joins both.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from pathlib import Path

import click

from nooa_cli.commands._connect_io import use_host


class _WizardHost:
    def __init__(self, frontend, registry_path: Path):
        self.frontend = frontend
        self.registry_path = registry_path
        self.secrets_path = registry_path.with_name("secrets.yaml")
        self.loop = asyncio.get_running_loop()
        self.cancelled = threading.Event()
        self.tasks: set[asyncio.Task] = set()
        self.render_error = None
        self.output = asyncio.Queue()
        self.renderer = asyncio.create_task(self._render())

    async def _render(self):
        from .commands import TextOutput

        while True:
            item = await self.output.get()
            try:
                if item is None:
                    return
                text, warning = item
                if self.render_error is None:
                    await self.frontend.render(TextOutput(text, "warning" if warning else "info"))
            except Exception as exc:
                self.render_error = exc
                self.cancel()
            finally:
                self.output.task_done()

    def echo(self, message, *, err=False, **kwargs):
        if message and not self.cancelled.is_set():
            self.loop.call_soon_threadsafe(self.output.put_nowait, (message, err))

    def run(self, awaitable):
        """Submit from the worker; return only after cancellation cleanup finishes."""
        waiter = concurrent.futures.Future()

        def complete(task):
            if task.cancelled():
                waiter.cancel()
            elif error := task.exception():
                waiter.set_exception(error)
            else:
                waiter.set_result(task.result())

        def start():
            if self.cancelled.is_set():
                awaitable.close()
                waiter.cancel()
                return
            task = self.loop.create_task(awaitable)
            self.tasks.add(task)
            task.add_done_callback(complete)

        self.loop.call_soon_threadsafe(start)
        return waiter.result()

    def prompt(self, text, **options):
        return self.run(self._prompt(text, options))

    async def _prompt(self, text, options):
        await self.output.join()
        while True:
            value = await self.frontend.prompt_connect(text, **options)
            if value is None:
                raise click.Abort()
            if value == "" and options.get("default") is not None:
                value = str(options["default"])
            choices = options.get("choices", ())
            if choices and value not in choices:
                self.echo("Choose an available value.", err=True)
                await asyncio.sleep(0)
                await self.output.join()
                continue
            if not value and options.get("default") != "":
                self.echo("Enter a value, or press Esc to cancel.", err=True)
                await asyncio.sleep(0)
                await self.output.join()
                continue
            return value

    def cancel(self):
        self.cancelled.set()
        for task in self.tasks:
            task.cancel()

    async def close(self):
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()
        await self.output.put(None)
        await self.renderer


def _invoke(host, args):
    from nooa_cli.commands.connect import command

    with use_host(host):
        # Keep Click's eager help and JSON stage output out of the terminal's
        # live screen. Staged session commands use ConnectControl instead.
        if args == ["--help"]:
            host.echo(command.get_help(click.Context(command, info_name="/connect")))
            return None
        with command.make_context("/connect", args, help_option_names=[]) as ctx:
            if ctx.params["stage"]:
                raise click.UsageError("Use nooa connect --stage outside the TUI.")
            return command.invoke(ctx)


async def run_native_wizard(frontend, args: list[str], registry_path: Path):
    """Run the CLI workflow with native completion, preserving its order and budget."""
    from nooa.unifiedllm import connect

    if args and args[0] in (*connect.PROVIDERS, "custom"):
        args = ["--provider", args[0], *args[1:]]
    elif args and args[0].startswith(("http://", "https://")):
        args = ["--endpoint", args[0], *args[1:]]
    host = _WizardHost(frontend, registry_path)
    worker = asyncio.create_task(asyncio.to_thread(_invoke, host, args))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        host.cancel()
        # Join the worker so a cancelled command cannot continue prompting or save.
        try:
            await asyncio.shield(worker)
        except (asyncio.CancelledError, concurrent.futures.CancelledError, click.Abort):
            pass
        if host.render_error is not None:
            raise RuntimeError("Could not render model setup") from host.render_error
        raise
    finally:
        await host.close()
