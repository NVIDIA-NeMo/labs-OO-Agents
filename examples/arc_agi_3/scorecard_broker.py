# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Competition shared-scorecard broker — runs in this venv (needs the `arc` extra).

Competition mode allows only ONE open scorecard per API key, so a parallel fleet
of games must share a single scorecard (one leaderboard entry). ``run_multi.py``
is stdlib-only (main venv) and can't hold an Arcade session, so it delegates the
scorecard's lifetime to this helper. This process:

1. mints ONE scorecard (Arcade COMPETITION) and prints a single JSON handshake
   line on stdout — ``{"scorecard_id": ..., "cookies": {...}}`` — which the
   launcher exports to every game subprocess (``ARC_SDK_SESSION_COOKIES`` env +
   ``--scorecard-id``) so they all land on the ALB replica that holds the card;
2. holds the minting session and pings the scorecard every ~10 min (the card has
   a documented ~15 min idle timeout) so a late, slow game can't idle it out;
3. on ``close`` (a line on stdin, or stdin EOF) closes the scorecard and prints a
   final summary line, then exits.

Usage (normally started by run_multi.py):

    python examples/arc_agi_3/scorecard_broker.py \
        --tags nemo_solver,competition
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading


def _emit(obj: dict) -> None:
    """One JSON object per line on stdout (the launcher reads these)."""
    sys.stdout.write(json.dumps(obj, default=str) + "\n")
    sys.stdout.flush()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tags", default="", help="comma-separated scorecard tags")
    p.add_argument(
        "--keepalive-seconds",
        type=float,
        default=600.0,
        help="scorecard ping interval (< the ~15 min idle timeout)",
    )
    args = p.parse_args()

    api_key = os.environ.get("ARC_API_KEY", "")
    if not api_key:
        _emit({"error": "ARC_API_KEY not set — competition mode needs an API key"})
        return 2

    try:
        from arc_agi import Arcade, OperationMode

        arcade = Arcade(operation_mode=OperationMode.COMPETITION, arc_api_key=api_key)
        tags = [t.strip() for t in args.tags.split(",") if t.strip()]
        scid = arcade.create_scorecard(tags=tags or None)
        if hasattr(arcade, "_default_scorecard_id"):
            arcade._default_scorecard_id = scid
    except Exception as e:  # setup failure must surface, not hang the fleet
        _emit({"error": f"{type(e).__name__}: {e}"})
        return 3

    # ALB stickiness: the create_scorecard response pinned this session to the
    # replica that holds the card. Hand those cookies to the launcher so every
    # game subprocess routes to the same replica (else RESET → "game not found").
    cookies = dict(arcade._session.cookies)
    _emit({"scorecard_id": scid, "cookies": cookies})

    # Keep-alive: reset the idle timer even when every game is mid-plan.
    ka_url = f"{arcade.arc_base_url}/api/scorecard/{scid}"
    ka_hdrs = {"X-API-Key": arcade.arc_api_key, "Accept": "application/json"}
    stop = threading.Event()

    def _keepalive() -> None:
        while not stop.wait(args.keepalive_seconds):
            try:
                lock = getattr(arcade, "_cookie_lock", None)
                if lock is None:
                    arcade._session.get(ka_url, headers=ka_hdrs, timeout=10)
                else:
                    with lock:
                        arcade._session.get(ka_url, headers=ka_hdrs, timeout=10)
            except Exception as e:  # noqa: BLE001 — a failed ping is non-fatal
                print(f"[broker] keep-alive ping failed: {e}", file=sys.stderr, flush=True)

    threading.Thread(target=_keepalive, daemon=True, name="scorecard-keepalive").start()

    # Block until the launcher signals close (a stdin line) or closes stdin (EOF).
    # KeyboardInterrupt included: a Ctrl-C on the fleet's process group must still
    # fall through to the bounded close below, or the scorecard is orphaned open
    # (the competition API allows one open scorecard per key).
    try:
        for line in sys.stdin:
            if line.strip().lower() in ("close", "stop", "quit"):
                break
    except (Exception, KeyboardInterrupt):
        pass
    stop.set()

    # Bounded close: the competition API can be slow/hang on scorecard ops (it even
    # returns 403 while a card is active). Run close in a watchdog thread so we
    # always emit a summary and exit rather than wedging the launcher's teardown.
    summary: dict = {"scorecard_id": scid}
    result: dict = {}

    def _do_close() -> None:
        try:
            result["scorecard"] = arcade.close_scorecard(scid)
            result["closed"] = True
        except Exception as e:  # noqa: BLE001
            result["closed"] = False
            result["close_error"] = f"{type(e).__name__}: {e}"

    ct = threading.Thread(target=_do_close, daemon=True)
    ct.start()
    ct.join(timeout=30)
    if ct.is_alive():
        summary["closed"] = False
        summary["close_error"] = "close_scorecard timed out after 30s"
    else:
        summary.update(result)
    _emit(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
