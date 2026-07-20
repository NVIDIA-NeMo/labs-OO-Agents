# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay captured failed LLM requests for debugging.

This module provides tools to replay HTTP requests that were captured
by enable_http_request_logging() with errors_only=True.

Usage:
    python -m unifiedllm.replay_requests eval_errors/llm_errors.jsonl
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

# Load .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, rely on existing env vars


async def replay_request(
    error_entry: dict[str, Any],
    verbose: bool = True,
) -> tuple[bool, dict[str, Any]]:
    """Replay a single request from error log.

    Args:
        error_entry: Dictionary with 'request' and 'response' keys from JSONL
        verbose: Print details about each replay

    Returns:
        Tuple of (success: bool, response: dict)
        success is True if status < 400, False otherwise
    """
    try:
        import httpx
    except ImportError as e:
        raise ImportError(
            "httpx is required for request replay. Install it with: pip install httpx"
        ) from e

    request_data = error_entry["request"]
    original_response = error_entry["response"]

    # Extract API key from environment (don't use the redacted one from the log)
    headers = dict(request_data["headers"])
    if "authorization" in headers:
        # Detect which API key to use based on URL
        url = request_data["url"]
        api_key = None

        if "inference-api.nvidia.com" in url:
            api_key = os.getenv("NVIDIA_INFERENCE_API_KEY") or os.getenv("NVIDIA_INTERNAL_API_KEY")
        elif "integrate.api.nvidia.com" in url or "nvidia" in url:
            api_key = os.getenv("NVIDIA_API_KEY")
        elif "openai.com" in url or "api.openai.com" in url:
            api_key = os.getenv("OPENAI_API_KEY")
        elif "anthropic.com" in url:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        else:
            # Fallback: try common keys
            api_key = (
                os.getenv("OPENAI_API_KEY")
                or os.getenv("NVIDIA_API_KEY")
                or os.getenv("NVIDIA_INFERENCE_API_KEY")
                or os.getenv("NVIDIA_INTERNAL_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
            )

        if api_key:
            headers["authorization"] = f"Bearer {api_key}"
        else:
            if verbose:
                print(f"⚠️  Warning: No API key found for {url}. Replay may fail.")
                print(
                    "   Tried: NVIDIA_INFERENCE_API_KEY, NVIDIA_INTERNAL_API_KEY, "
                    "NVIDIA_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY"
                )

    if verbose:
        model = error_entry.get("model", "unknown")
        counter = error_entry.get("counter", "?")
        print(f"\n🔄 Replaying request #{counter} ({model})...")
        print(f"   Original status: {original_response['status_code']}")

    # Make the request
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.request(
                method=request_data["method"],
                url=request_data["url"],
                headers=headers,
                json=request_data["body"],
            )

            success = response.status_code < 400

            if verbose:
                if success:
                    print(f"   ✅ Now succeeds: {response.status_code}")
                else:
                    print(f"   ❌ Still fails: {response.status_code}")

            return success, {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.content else None,
            }

        except Exception as e:
            if verbose:
                print(f"   ❌ Exception: {type(e).__name__}: {e}")
            return False, {"error": str(e)}


async def replay_from_file(
    error_file: Path | str,
    max_requests: int | None = None,
    delay_seconds: float = 1.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Replay multiple requests from error log.

    Args:
        error_file: Path to llm_errors.jsonl file
        max_requests: Maximum number of requests to replay (None = all)
        delay_seconds: Delay between requests to avoid rate limits
        verbose: Print details about each replay

    Returns:
        Summary statistics dictionary with keys:
        - total: Total requests replayed
        - succeeded: Number that now succeed
        - failed: Number that still fail
        - success_rate: Percentage that now succeed
        - results: List of (entry, success, response) tuples
    """
    error_file = Path(error_file)

    if not error_file.exists():
        raise FileNotFoundError(f"Error file not found: {error_file}")

    # Read JSONL file
    entries = []
    with open(error_file) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    if max_requests:
        entries = entries[:max_requests]

    if verbose:
        print(f"📂 Loaded {len(entries)} error entries from {error_file}")
        print(f"{'=' * 60}")

    results = []
    succeeded = 0
    failed = 0

    for i, entry in enumerate(entries):
        if i > 0 and delay_seconds > 0:
            await asyncio.sleep(delay_seconds)

        success, response = await replay_request(entry, verbose=verbose)

        if success:
            succeeded += 1
        else:
            failed += 1

        results.append((entry, success, response))

    total = len(entries)
    success_rate = (succeeded / total * 100) if total > 0 else 0

    if verbose:
        print(f"\n{'=' * 60}")
        print("📊 Replay Summary:")
        print(f"   Total replayed: {total}")
        print(f"   Now succeed: {succeeded} ({success_rate:.1f}%)")
        print(f"   Still fail: {failed} ({100 - success_rate:.1f}%)")
        print("\n💡 Interpretation:")
        if success_rate > 80:
            print("   → Most errors were transient API issues")
        elif success_rate < 20:
            print("   → Most errors are likely payload-related")
        else:
            print("   → Mix of transient and payload issues")

    return {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "success_rate": success_rate,
        "results": results,
    }


async def main():
    """CLI entry point for replaying requests."""
    import argparse

    parser = argparse.ArgumentParser(description="Replay captured LLM error requests")
    parser.add_argument("error_file", type=Path, help="Path to llm_errors.jsonl file")
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Maximum number of requests to replay (default: all)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between requests (default: 1.0)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output",
    )

    args = parser.parse_args()

    stats = await replay_from_file(
        error_file=args.error_file,
        max_requests=args.max_requests,
        delay_seconds=args.delay,
        verbose=not args.quiet,
    )

    # Exit with non-zero if all requests still fail
    if stats["succeeded"] == 0 and stats["total"] > 0:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
