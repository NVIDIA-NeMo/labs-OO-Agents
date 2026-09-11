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
    "--agent", "agent_spec", help="Shared coding agent class (module:Class or file.py:Class)."
)
@click.option(
    "--legacy-agent", is_flag=True, help="Use the legacy multi-tool agent, as in nooa tui."
)
def command(
    model: str, client_type: str | None, agent_spec: str | None, legacy_agent: bool
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

    if agent_spec or legacy_agent:
        from nooa_cli.interactive.options import SessionOptions

        asyncio.run(
            serve(
                llm_factory,
                options_factory=lambda root: SessionOptions.load(
                    root, agent_spec=agent_spec, legacy_agent=legacy_agent
                ),
            )
        )
    else:
        asyncio.run(serve(llm_factory))


def main() -> None:
    command()
