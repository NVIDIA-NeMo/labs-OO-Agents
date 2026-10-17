# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child-side broker probe: can a container process use an inherited pipe?

The whole sandbox design depends on this. ``self.*`` calls in a cell are
brokered to the parent agent over a pipe, so if handle inheritance into an
AppContainer does not produce a usable bidirectional channel, the backend needs
a different IPC mechanism before any of the isolation work matters.

The handles arrive as decimal integers: anonymous pipes inherited through
PROC_THREAD_ATTRIBUTE_HANDLE_LIST are already open in this process, so no name
or DACL lookup is involved.
"""

from __future__ import annotations

import argparse
import msvcrt
import os
import struct
import sys
import time


def _fd(handle: int) -> int:
    return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_handle", type=int, required=True)
    ap.add_argument("--out", dest="out_handle", type=int, required=True)
    ap.add_argument("--payload", default="HELLO")
    args = ap.parse_args()

    try:
        rfd = _fd(args.in_handle)
        wfd = _fd(args.out_handle)
    except OSError as exc:
        print(f"WRAP FAILED: {exc}", file=sys.stderr)
        return 3

    payload = args.payload.encode()
    try:
        # Length-prefixed frame, same shape as the real wire protocol.
        os.write(wfd, struct.pack("<I", len(payload)) + payload)
        os.close(wfd)
    except OSError as exc:
        print(f"WRITE FAILED: {exc}", file=sys.stderr)
        return 4

    try:
        header = os.read(rfd, 4)
        if len(header) != 4:
            print(f"SHORT HEADER: {len(header)}", file=sys.stderr)
            return 5
        (n,) = struct.unpack("<I", header)
        body = b""
        deadline = time.monotonic() + 10
        while len(body) < n and time.monotonic() < deadline:
            chunk = os.read(rfd, n - len(body))
            if not chunk:
                break
            body += chunk
        print(f"REPLY {body.decode(errors='replace')}", flush=True)
        os.close(rfd)
        return 0 if body.decode(errors="replace") == "WORLD" else 6
    except OSError as exc:
        print(f"READ FAILED: {exc}", file=sys.stderr)
        return 7


if __name__ == "__main__":
    sys.exit(main())
