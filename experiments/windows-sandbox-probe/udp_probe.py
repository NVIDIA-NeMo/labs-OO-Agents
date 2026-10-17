# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Does AppContainer actually stop UDP egress, or only connected TCP?

A bare ``sendto`` returning without error proves nothing: an unconnected UDP
send is asynchronous, so the failure (if any) surfaces later or never. This
probe completes a real request/response round trip — a DNS query to 8.8.8.8 —
so "no reply" and "packet never left" are distinguishable.

Run on the host first as a control; a blocked container must differ from it.
"""

from __future__ import annotations

import socket
import sys


def build_dns_query(name: str = "example.com") -> bytes:
    header = bytes.fromhex("abcd01000001000000000000")
    qname = b"".join(bytes([len(p)]) + p.encode() for p in name.split(".")) + b"\x00"
    return header + qname + bytes.fromhex("00010001")


def main() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(4)
    try:
        s.bind(("0.0.0.0", 0))
        print("bind OK", s.getsockname())
    except OSError as exc:
        print("bind FAILED", exc)
        return 0

    try:
        s.sendto(build_dns_query(), ("8.8.8.8", 53))
        print("sendto returned OK")
    except OSError as exc:
        print("sendto FAILED", exc)
        return 0

    try:
        data, addr = s.recvfrom(512)
        print(f"RECV {len(data)} bytes from {addr} -> UDP EGRESS WORKS")
    except TimeoutError:
        print("recvfrom TIMEOUT -> no reply; egress inconclusive")
    except OSError as exc:
        print(f"recvfrom FAILED {exc} -> egress blocked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
