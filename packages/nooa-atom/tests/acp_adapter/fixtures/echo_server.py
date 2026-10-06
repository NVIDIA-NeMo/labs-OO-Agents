# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A stand-in agent server for the tee tests: answers each JSON-RPC line.

Options: ``--record PATH`` appends the raw bytes it reads; ``--exit-code N``
is the exit status after standard input ends; ``--term-marker PATH`` is
written when SIGTERM arrives (the process then exits with status 7).
"""

import argparse
import json
import signal
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", type=Path)
    parser.add_argument("--exit-code", type=int, default=0)
    parser.add_argument("--term-marker", type=Path)
    args = parser.parse_args()

    def on_term(signum, frame):
        if args.term_marker is not None:
            args.term_marker.write_text("terminated")
        sys.exit(7)

    signal.signal(signal.SIGTERM, on_term)
    sys.stderr.write("echo-server-stderr\n")
    sys.stderr.flush()
    print("ready", file=sys.stderr, flush=True)
    for line in sys.stdin.buffer:
        if args.record is not None:
            with args.record.open("ab") as record:
                record.write(line)
        request = json.loads(line)
        reply = {"jsonrpc": "2.0", "id": request.get("id"), "result": {"echo": request["method"]}}
        sys.stdout.buffer.write(json.dumps(reply).encode() + b"\n")
        sys.stdout.buffer.flush()
    sys.exit(args.exit_code)


if __name__ == "__main__":
    main()
