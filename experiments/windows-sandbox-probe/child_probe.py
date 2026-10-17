# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Child-side probe. Runs INSIDE the AppContainer and reports what it can do.

Writes one JSON object to the file given by ``--out`` (which must be in a
granted directory) and exits 0. Any probe that cannot even run is recorded as
an error rather than crashing, so a partially-capable container still yields
data — the negative results are the whole point of this experiment.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import socket
import sys
import time


def _probe_identity() -> dict:
    """Confirm the process really is in a container and report how it sees itself."""
    from ctypes import wintypes

    out: dict = {}
    adv = ctypes.WinDLL("advapi32", use_last_error=True)
    token = wintypes.HANDLE()
    adv.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    adv.OpenProcessToken.restype = wintypes.BOOL
    adv.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    adv.GetTokenInformation.restype = wintypes.BOOL

    TOKEN_QUERY = 0x0008
    TokenIsAppContainer = 29
    TokenAppContainerSid = 31

    class _TokenAppContainerInformation(ctypes.Structure):
        _fields_ = [("TokenAppContainer", ctypes.c_void_p)]

    adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    adv.ConvertSidToStringSidW.restype = wintypes.BOOL

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    if not adv.OpenProcessToken(k32.GetCurrentProcess(), TOKEN_QUERY, ctypes.byref(token)):
        out["error"] = f"OpenProcessToken failed: {ctypes.get_last_error()}"
        return out
    try:
        val = wintypes.DWORD()
        need = wintypes.DWORD()
        if adv.GetTokenInformation(
            token, TokenIsAppContainer, ctypes.byref(val), ctypes.sizeof(val), ctypes.byref(need)
        ):
            out["is_app_container"] = bool(val.value)
        # TokenAppContainerSid copies the SID itself into the buffer (48 bytes
        # for a typical container SID), not just a pointer to one.
        buf = ctypes.create_string_buffer(256)
        if adv.GetTokenInformation(
            token, TokenAppContainerSid, buf, ctypes.sizeof(buf), ctypes.byref(need)
        ):
            psid = ctypes.c_void_p.from_buffer(buf).value
            text = wintypes.LPWSTR()
            if psid and adv.ConvertSidToStringSidW(psid, ctypes.byref(text)):
                out["container_sid"] = text.value
                k32.LocalFree(text)
    finally:
        k32.CloseHandle(token)
    out["username"] = os.environ.get("USERNAME", "")
    out["cwd"] = os.getcwd()
    out["prefix"] = sys.prefix
    out["executable"] = sys.executable
    return out


def _probe_read(path: str) -> dict:
    try:
        with open(path, "rb") as fh:
            fh.read(64)
        return {"ok": True}
    except OSError as exc:
        return {
            "ok": False,
            "errno": exc.errno,
            "winerror": getattr(exc, "winerror", None),
            "msg": exc.strerror,
        }


def _probe_write(path: str) -> dict:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("probe")
        os.remove(path)
        return {"ok": True}
    except OSError as exc:
        return {
            "ok": False,
            "errno": exc.errno,
            "winerror": getattr(exc, "winerror", None),
            "msg": exc.strerror,
        }


def _probe_network() -> dict:
    """Outbound TCP must fail when no capability SIDs are granted."""
    result: dict = {}
    for label, addr in (
        ("tcp_public", ("1.1.1.1", 443)),
        ("tcp_dns", ("8.8.8.8", 53)),
        ("udp_dns", None),
    ):
        if label == "udp_dns":
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.settimeout(3)
                s.sendto(b"\x00" * 12, ("8.8.8.8", 53))
                s.close()
                result[label] = {"ok": True, "note": "sendto returned (may be async)"}
            except OSError as exc:
                result[label] = {
                    "ok": False,
                    "errno": exc.errno,
                    "winerror": getattr(exc, "winerror", None),
                    "msg": exc.strerror,
                }
            continue
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(3)
            s.connect(addr)
            s.close()
            result[label] = {"ok": True, "note": "outbound connection established"}
        except OSError as exc:
            result[label] = {
                "ok": False,
                "errno": exc.errno,
                "winerror": getattr(exc, "winerror", None),
                "msg": exc.strerror,
            }
    # A loopback listener is the control: the broker channel must keep working.
    try:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        srv.close()
        result["loopback_listen"] = {"ok": True}
    except OSError as exc:
        result["loopback_listen"] = {
            "ok": False,
            "errno": exc.errno,
            "winerror": getattr(exc, "winerror", None),
            "msg": exc.strerror,
        }
    return result


def _probe_memory_cap(target_mb: int) -> dict:
    """Try to allocate past the job's ProcessMemoryLimit."""
    blocks = []
    chunk = 16 * 1024 * 1024
    try:
        for i in range(1, 64):
            blocks.append(bytearray(chunk))
            if i * 16 >= target_mb * 2:
                break
        return {
            "ok": True,
            "allocated_mb": len(blocks) * 16,
            "note": "allocation succeeded past expectation",
        }
    except MemoryError as exc:
        return {
            "ok": False,
            "allocated_mb": len(blocks) * 16,
            "kind": "MemoryError",
            "msg": str(exc),
        }
    except OSError as exc:
        return {
            "ok": False,
            "allocated_mb": len(blocks) * 16,
            "kind": type(exc).__name__,
            "msg": str(exc),
        }


def _probe_cpu_burn(seconds: int) -> dict:
    """Burn CPU until the job's per-job time limit kills us."""
    start = time.monotonic()
    total = 0
    while True:
        total += sum(range(1000))
        if time.monotonic() - start > seconds * 4:
            return {
                "ok": True,
                "note": "survived past 4x the job time limit",
                "elapsed_s": time.monotonic() - start,
            }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reads", nargs="*", default=[])
    ap.add_argument("--writes", nargs="*", default=[])
    ap.add_argument("--memory-mb", type=int, default=0)
    ap.add_argument("--cpu-seconds", type=int, default=0)
    args = ap.parse_args()

    report: dict = {"pid": os.getpid(), "identity": _probe_identity()}
    report["reads"] = {p: _probe_read(p) for p in args.reads}
    report["writes"] = {p: _probe_write(p) for p in args.writes}
    report["network"] = _probe_network()
    if args.memory_mb:
        report["memory"] = _probe_memory_cap(args.memory_mb)
    if args.cpu_seconds:
        report["cpu"] = _probe_cpu_burn(args.cpu_seconds)

    try:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except OSError:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
