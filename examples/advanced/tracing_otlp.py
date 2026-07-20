# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OTLP tracing examples — send agent traces to external observability platforms.

This module dispatches to backend-specific examples:

- **Langfuse**: ``tracing_langfuse.py`` — open-source LLM observability
- **Phoenix**: ``tracing_phoenix.py`` — Arize AI's open-source platform

Usage:
    uv run python examples/advanced/tracing_otlp.py --backend langfuse
    uv run python examples/advanced/tracing_otlp.py --backend phoenix

Or run the individual examples directly:
    uv run python examples/advanced/tracing_langfuse.py
    uv run python examples/advanced/tracing_phoenix.py

How it works:
    The ``exporters`` module provides factory functions that return standard
    OTel ``SpanExporter`` instances.  ``exporters.langfuse()`` reads credentials
    from env vars and sends OTLP/HTTP with Basic auth.  ``exporters.otlp()``
    sends to any OTLP-compatible endpoint (Phoenix, Jaeger, Grafana Tempo, etc.).
"""

import argparse
import asyncio


async def main(backend: str) -> None:
    if backend == "langfuse":
        from examples.advanced.tracing_langfuse import main as run

        await run()
    elif backend == "phoenix":
        from examples.advanced.tracing_phoenix import main as run  # type: ignore[assignment]

        await run()
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'langfuse' or 'phoenix'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OTLP tracing example")
    parser.add_argument(
        "--backend",
        choices=["langfuse", "phoenix"],
        default="langfuse",
        help="Tracing backend to use (default: langfuse)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.backend))
