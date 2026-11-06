# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry points for the NOOA ACP agent.

Deprecated: nooa-acp is replaced by nooa-atom (``nooa atom``). This is its
last release; the command warns on standard error and keeps working.
"""

import asyncio
import os
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM

DEPRECATION_WARNING = (
    "Warning: nooa-acp is deprecated and this is its last release. "
    "It is replaced by nooa-atom: install nooa-atom and run `nooa atom` "
    "(see https://github.com/NVIDIA-NeMo/labs-OO-Agents/tree/main/packages/nooa-atom)."
)

_warned = False


def _warn_deprecated() -> None:
    """Print the deprecation notice once per process, on standard error.

    Standard output carries the ACP protocol, so the notice must not go there.
    """
    global _warned
    if not _warned:
        _warned = True
        click.echo(DEPRECATION_WARNING, err=True)


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
def command(model: str, client_type: str | None) -> None:
    """Serve the NOOA coding agent over ACP on standard input/output.

    Deprecated: use nooa-atom (`nooa atom`) instead.
    """
    _warn_deprecated()  # `nooa acp`; the `nooa-acp` script has warned already
    from nooa.secrets import load_secrets_into_env
    from nooa.unifiedllm import get_llm_client
    from nooa_acp.server import serve

    load_secrets_into_env()
    nvidia_api_key = os.getenv("NVIDIA_API_KEY") if model.startswith("nvidia_nim/") else None

    def llm_factory() -> "UnifiedLLM":
        overrides = {"api_key": nvidia_api_key} if nvidia_api_key else {}
        return get_llm_client(model, client_type=client_type, **overrides)

    asyncio.run(serve(llm_factory))


def main() -> None:
    """The ``nooa-acp`` script: warn that it is deprecated, then run as before."""
    _warn_deprecated()
    command()
