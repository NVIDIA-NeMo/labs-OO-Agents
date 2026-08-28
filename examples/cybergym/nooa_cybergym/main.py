# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NOOA CyberGym agent entry point.

Invoked inside the trial container as:
    python main.py --prompt "..." --model glm-5.2
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
from pathlib import Path

from nooa import hidden

with hidden:
    import argparse

    try:
        from .agent import DEFAULT_MODEL_NAME, CyberGymAgent
        from .util import (
            DEFAULT_API_BASE,
            _apply_reasoning_effort,
            _llm_client_kwargs,
            configure_tracing,
            install_summarizer,
            make_llm,
            shutdown_tracing,
        )
    except ImportError:  # pragma: no cover - script mode inside /app
        from agent import DEFAULT_MODEL_NAME, CyberGymAgent

        from util import (
            DEFAULT_API_BASE,  # noqa: F401 -- re-exported for runner tests
            _apply_reasoning_effort,  # noqa: F401 -- re-exported for runner tests
            _llm_client_kwargs,  # noqa: F401 -- re-exported for runner tests
            configure_tracing,
            install_summarizer,
            make_llm,
            shutdown_tracing,
        )

ARTIFACTS_DIR = Path("/app/artifacts")
LOG_PATH = Path("/logs/artifacts/log.txt")
MAX_OUTPUT_TOKENS = int(os.environ.get("NOOA_CYBERGYM_MAX_OUTPUT_TOKENS", "32768"))
SOFT_TIMEOUT_SEC = int(os.environ.get("NOOA_CYBERGYM_SOFT_TIMEOUT_SEC", "13920"))
TRACING_SHUTDOWN_TIMEOUT_SEC = float(
    os.environ.get("NOOA_CYBERGYM_TRACING_SHUTDOWN_TIMEOUT_SEC", "30")
)

logger = logging.getLogger("nooa_cybergym")


@hidden
def _setup_logging() -> None:
    """Configure dual-handler logging: file + stdout."""
    logger.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — live-mirrored by Harbor via /logs/artifacts/
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(LOG_PATH, mode="a")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except OSError:
        pass

    # Stream handler — keeps the CyberGym agent log populated.
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)


@hidden
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NOOA CyberGym agent")
    parser.add_argument("--prompt", required=True, help="Task instruction")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"LLM model or llm_config alias (default: {DEFAULT_MODEL_NAME}).",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("NOOA_CYBERGYM_REASONING_EFFORT"),
        help="Reasoning effort knob forwarded to AnyLLM.",
    )
    return parser.parse_args()


@hidden
def _write_output(result: str) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "output.txt").write_text(str(result) + "\n")
    logger.debug("wrote /app/artifacts/output.txt")


@hidden
def _shutdown_tracing_with_timeout(timeout_sec: float = TRACING_SHUTDOWN_TIMEOUT_SEC) -> bool:
    """Best-effort tracing shutdown that cannot consume the Harbor timeout gap."""
    done = threading.Event()

    def run_shutdown() -> None:
        try:
            shutdown_tracing()
        except Exception as exc:  # noqa: BLE001
            logger.warning("tracing shutdown failed (%s: %s)", type(exc).__name__, exc)
        finally:
            done.set()

    thread = threading.Thread(
        target=run_shutdown, name="nooa-cybergym-tracing-shutdown", daemon=True
    )
    thread.start()
    thread.join(timeout=max(0.0, timeout_sec))
    if done.is_set():
        return True

    logger.warning(
        "tracing shutdown did not finish within %.1fs; continuing without waiting for exporters",
        timeout_sec,
    )
    return False


@hidden
async def amain(prompt: str, model: str, reasoning_effort: str | None) -> str:
    llm = make_llm(model, max_tokens=MAX_OUTPUT_TOKENS, reasoning_effort=reasoning_effort)
    if llm.context_window is None:
        logger.warning(
            "no context_window for model=%r; summarizer will use the 100K fallback budget.",
            model,
        )

    agent = CyberGymAgent(llm=llm)
    configure_tracing(agent, model)
    install_summarizer(agent, llm)
    solve_task = asyncio.create_task(agent.solve(prompt))
    done, _ = await asyncio.wait({solve_task}, timeout=SOFT_TIMEOUT_SEC)
    if done:
        return solve_task.result()

    # Soft timeout reached
    logger.warning("soft timeout reached after %ds", SOFT_TIMEOUT_SEC)
    summary = agent.timeout_summary()
    logger.info("%s", summary)
    solve_task.cancel()
    # Write artifacts before tracing shutdown; shutdown may block on exporter threads.
    _write_output(summary)
    logger.info("solve() returned: %r", summary)
    _shutdown_tracing_with_timeout()
    # Force-exit before asyncio.run() tries to await pending tasks.
    os._exit(0)


def main() -> None:
    _setup_logging()
    args = _parse_args()

    logger.info(
        "starting; model=%s max_output_tokens=%d soft_timeout_sec=%d reasoning_effort=%r",
        args.model,
        MAX_OUTPUT_TOKENS,
        SOFT_TIMEOUT_SEC,
        args.reasoning_effort,
    )

    try:
        result = asyncio.run(amain(args.prompt, args.model, args.reasoning_effort))
    except Exception as exc:
        logger.error("solve() raised: %s: %s", type(exc).__name__, exc)
        if not _shutdown_tracing_with_timeout():
            os._exit(1)
        raise

    logger.info("solve() returned: %r", result)
    _write_output(result)
    if not _shutdown_tracing_with_timeout():
        os._exit(0)


if __name__ == "__main__":
    main()
