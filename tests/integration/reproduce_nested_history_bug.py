#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone reproduction of nested agent history ordering bug.

This script reproduces the bug where nested agent calls cause tool_call_id
ordering issues. It runs against real LLM providers to trigger the actual
validation errors.

Usage:
    # Run with default provider (openai)
    python tests/integration/reproduce_nested_history_bug.py

    # Run with specific provider
    python tests/integration/reproduce_nested_history_bug.py --provider nvidia
    python tests/integration/reproduce_nested_history_bug.py --provider gemini

    # Run with specific model
    python tests/integration/reproduce_nested_history_bug.py --model gpt-4o-mini
    python tests/integration/reproduce_nested_history_bug.py --provider nvidia --model qwen/qwen3-next-80b-a3b-instruct

Providers:
    - openai: OpenAI API (requires OPENAI_API_KEY)
    - nvidia: NVIDIA NIM (requires NVIDIA_API_KEY)
    - gemini: Google Gemini (requires GOOGLE_API_KEY or through litellm routing)

Expected behavior: Should fail with "Missing corresponding tool call for tool response message"
when the bug is present.
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from nooa.config import CodeActConfig

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Load .env file if it exists
env_file = project_root / ".env"
if env_file.exists():
    from dotenv import load_dotenv

    load_dotenv(env_file)

from nooa import Agent, strategy  # noqa: E402
from nooa.strategies.codeact import CodeActStrategy  # noqa: E402


def get_llm_client(provider: str, model: str | None = None):
    """Get LLM client for the specified provider."""
    from nooa.unifiedllm import CompletionClient

    # Default models per provider (litellm format)
    default_models = {
        "openai": "gpt-4o-mini",
        "nvidia": "nvidia_nim/qwen/qwen3-next-80b-a3b-instruct",
        "nvidia_internal": "openai/azure/openai/gpt-4o",
        "gemini": "gemini/gemini-2.0-flash",
    }

    # Get model with proper prefix for litellm
    def get_model(provider_name: str, user_model: str | None) -> str:
        if user_model:
            # Add prefix if needed for certain providers
            if provider_name == "nvidia" and not user_model.startswith("nvidia_nim/"):
                return f"nvidia_nim/{user_model}"
            if provider_name == "nvidia_internal" and not user_model.startswith("openai/"):
                return f"openai/{user_model}"
            if provider_name == "gemini" and not user_model.startswith("gemini/"):
                return f"gemini/{user_model}"
            return user_model
        return default_models[provider_name]

    provider_configs = {
        "openai": {
            "model": get_model("openai", model),
            "api_key": os.getenv("OPENAI_API_KEY"),
        },
        "nvidia": {
            "model": get_model("nvidia", model),
            "api_key": os.getenv("NVIDIA_API_KEY"),
            "api_base": "https://integrate.api.nvidia.com/v1",
        },
        "nvidia_internal": {
            "model": get_model("nvidia_internal", model),
            "api_key": os.getenv("NVIDIA_INTERNAL_API_KEY"),
            "api_base": "https://inference-api.nvidia.com/v1",
        },
        "gemini": {
            "model": get_model("gemini", model),
            "api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        },
    }

    if provider not in provider_configs:
        raise ValueError(
            f"Unknown provider: {provider}. Choose from: {list(provider_configs.keys())}"
        )

    config = provider_configs[provider]
    if not config.get("api_key"):
        key_name = {
            "openai": "OPENAI_API_KEY",
            "nvidia": "NVIDIA_API_KEY",
            "nvidia_internal": "NVIDIA_INTERNAL_API_KEY",
            "gemini": "GOOGLE_API_KEY or GEMINI_API_KEY",
        }[provider]
        raise ValueError(f"Missing API key. Set {key_name} environment variable.")

    return CompletionClient(**config)


class NestedBugAgent(Agent):
    """Agent demonstrating the nested history bug.

    The bug occurs when outer_method's execute_python code calls inner_method.
    The tool_call_id ordering in history becomes invalid.
    """

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=5)))
    async def outer_method(self, value: int) -> str:
        """Process a value by DELEGATING to inner_method.

        CRITICAL: You MUST call self.inner_method(value) to get the result.
        Do NOT compute the result yourself - you MUST delegate to inner_method.

        Required steps:
        1. Call `result = await self.inner_method(value)` to get the processed value
        2. Print what you received: `print(f"Got from inner: {result}")`
        3. Return `f"outer_wrapped_{result}"`

        The final result MUST start with "outer_wrapped_" to prove delegation worked.
        """
        ...

    @strategy(CodeActStrategy(config=CodeActConfig(max_iterations=3)))
    async def inner_method(self, x: int) -> str:
        """Process a single value and return a formatted string.

        Return exactly: f"inner_{x}_done"
        """
        ...


async def run_reproduction(provider: str, model: str | None = None, verbose: bool = False):
    """Run the bug reproduction against a real provider."""
    print(f"\n{'=' * 60}")
    print("Running nested history bug reproduction")
    print(f"Provider: {provider}")
    print(f"Model: {model or 'default'}")
    print(f"{'=' * 60}\n")

    try:
        llm = get_llm_client(provider, model)
        print(f"Using model: {llm.model}")
    except ValueError as e:
        print(f"ERROR: {e}")
        return False

    agent = NestedBugAgent(llm=llm)

    print("Calling agent.outer_method(42)...")
    print("This should trigger a nested call to inner_method.\n")

    try:
        result = await agent.outer_method(42)
        print(f"\n✅ SUCCESS! Result: {result}")

        # Check if nested call actually happened
        if "outer_wrapped_" in result and "inner_" in result:
            print("\n✓ Nested call was executed (result format confirms delegation)")
            print("\nThe bug did NOT reproduce. Either:")
            print("1. The bug has been fixed")
            print("2. This provider doesn't validate tool_call_id ordering")
        else:
            print("\n⚠️ WARNING: Result doesn't show nested call pattern")
            print(f"Expected format: 'outer_wrapped_inner_42_done', got: '{result}'")
            print("The LLM may have bypassed the nested call.")
        return True

    except Exception as e:
        error_str = str(e)
        print(f"\n❌ ERROR: {error_str[:500]}...")

        # Check if this is the expected error
        if "tool_call" in error_str.lower() or "tool call" in error_str.lower():
            print("\n🐛 BUG REPRODUCED!")
            print("This is the expected error from the history ordering bug.")
            print("\nRoot cause: The outer execute_python tool result is added")
            print("AFTER the inner agent's events, causing tool_call_id mismatch.")
            return False  # False = bug reproduced
        else:
            print("\n⚠️ UNEXPECTED ERROR")
            print("This error is different from the expected bug.")
            if verbose:
                import traceback

                traceback.print_exc()
            return None  # None = unexpected error (not the bug)


async def run_all_providers():
    """Run reproduction against all available providers."""
    providers = []

    # Check which providers have API keys (use None for model to use defaults)
    if os.getenv("OPENAI_API_KEY"):
        providers.append(("openai", None))
    if os.getenv("NVIDIA_API_KEY"):
        providers.append(("nvidia", None))
    if os.getenv("NVIDIA_INTERNAL_API_KEY"):
        providers.append(("nvidia_internal", None))
    if os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
        providers.append(("gemini", None))

    if not providers:
        print("No API keys found. Set at least one of:")
        print("  - OPENAI_API_KEY")
        print("  - NVIDIA_API_KEY")
        print("  - NVIDIA_INTERNAL_API_KEY")
        print("  - GOOGLE_API_KEY or GEMINI_API_KEY")
        return

    print(f"Found {len(providers)} providers with API keys\n")

    results = {}
    for provider, model in providers:
        try:
            result = await run_reproduction(provider, model)
            if result is True:
                results[provider] = "PASSED"
            elif result is False:
                results[provider] = "BUG REPRODUCED"
            elif result is None:
                results[provider] = "UNEXPECTED ERROR"
            else:
                results[provider] = f"UNKNOWN: {result}"
        except Exception as e:
            results[provider] = f"ERROR: {str(e)[:50]}"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for provider, result in results.items():
        emoji = "✅" if result == "PASSED" else "🐛" if "BUG" in result else "❌"
        print(f"  {emoji} {provider}: {result}")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce nested agent history ordering bug",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "nvidia", "nvidia_internal", "gemini", "all"],
        default="all",
        help="LLM provider to test (default: all available)",
    )
    parser.add_argument(
        "--model",
        help="Specific model to use (overrides provider default)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show full stack traces on error",
    )

    args = parser.parse_args()

    if args.provider == "all":
        asyncio.run(run_all_providers())
    else:
        asyncio.run(run_reproduction(args.provider, args.model, args.verbose))


if __name__ == "__main__":
    main()
