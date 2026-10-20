# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry points for the NOOA ACP agent."""

import asyncio
import os
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from nooa.runtime.sandbox.config import SandboxConfig
    from nooa.unifiedllm import UnifiedLLM


@click.command()
@click.option(
    "--model",
    envvar="NOOA_MODEL",
    required=True,
    help="LiteLLM model name or configured NOOA model alias. Or set NOOA_MODEL.",
)
@click.option(
    "--client-type",
    type=click.Choice(("completion", "responses")),
    default=None,
    help="Override the configured NOOA LLM client type.",
)
@click.option(
    "--sandbox",
    is_flag=True,
    default=False,
    help=(
        "Run generated Python cells in an OS sandbox (AppContainer on Windows, "
        "Landlock on Linux). Cells only: self.* tool calls, including shell "
        "commands, still execute in this process with its full environment and "
        "permissions. Requires a built runtime root on Windows; see docs/windows.md."
    ),
)
@click.option(
    "--sandbox-root",
    default=None,
    envvar="NOOA_SANDBOX_RUNTIME",
    help="Windows only: the private interpreter root the sandbox runs from.",
)
@click.option(
    "--sandbox-network",
    is_flag=True,
    default=False,
    help="Allow outbound network from sandboxed cells. Off by default.",
)
def command(
    model: str,
    client_type: str | None,
    sandbox: bool,
    sandbox_root: str | None,
    sandbox_network: bool,
) -> None:
    """Serve the NOOA coding agent over ACP on standard input/output."""
    from nooa.secrets import load_secrets_into_env
    from nooa.unifiedllm import get_llm_client
    from nooa_acp.server import serve

    load_secrets_into_env()
    nvidia_api_key = os.getenv("NVIDIA_API_KEY") if model.startswith("nvidia_nim/") else None

    def llm_factory() -> "UnifiedLLM":
        overrides = {"api_key": nvidia_api_key} if nvidia_api_key else {}
        return get_llm_client(model, client_type=client_type, **overrides)

    sandbox_config = _sandbox_config(sandbox, sandbox_root, sandbox_network)
    asyncio.run(serve(llm_factory, sandbox=sandbox_config))


def _sandbox_config(
    enabled: bool,
    runtime_root: str | None,
    network: bool,
) -> "SandboxConfig | None":
    """Build the sandbox config the server resolves per session, or None.

    ``start_method`` is not set here: the backend decides. Windows requires
    ``spawn`` (the launcher creates the process itself) and the executor
    enforces that; Linux keeps its default.
    """
    if not enabled:
        return None
    from nooa.runtime.sandbox.config import SandboxConfig

    return SandboxConfig(
        filesystem=True,
        network=network,
        runtime_root=runtime_root,
        # The workspace is filled in per session from the editor's root.
        workspace=None,
        require=True,
    )


def main() -> None:
    command()
