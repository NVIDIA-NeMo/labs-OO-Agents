# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command-line entry points for the NOOA ACP agent."""

import asyncio
import os
from typing import TYPE_CHECKING

import click

if TYPE_CHECKING:
    from nooa.unifiedllm import UnifiedLLM


@click.command()
@click.option(
    "--model",
    envvar="NOOA_MODEL",
    default="nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
    show_default=True,
    help="LiteLLM model name or configured NOOA model alias.",
)
@click.option(
    "--client-type",
    type=click.Choice(("completion", "responses")),
    default=None,
    help="Override the configured NOOA LLM client type.",
)
def command(model: str, client_type: str | None) -> None:
    """Serve the NOOA coding agent over ACP on standard input/output."""
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
    command()
