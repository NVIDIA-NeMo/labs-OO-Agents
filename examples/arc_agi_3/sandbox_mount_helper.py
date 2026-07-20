# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-namespace mount-allowlist helper for sandbox.py's ``unshare`` fallback.

Runs as PID 1 inside an already-created user+mount+net namespace. Builds a fresh
tmpfs root containing ONLY the allowlisted binds, ``pivot_root``s into it, then
execs the inner command. Preferred backend is ``bwrap`` (see sandbox.py); this
exists for hosts without it. Only functional where the kernel permits namespace
mounts — inert/erroring elsewhere.

Not imported by the agent; invoked as a subprocess entrypoint.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import platform
import sys

# pivot_root syscall number is architecture-specific.
_PIVOT_ROOT_NR = {
    "x86_64": 155,
    "aarch64": 41,
    "armv7l": 218,
    "i686": 217,
    "i386": 217,
    "ppc64le": 203,
    "s390x": 217,
}.get(platform.machine(), 155)

libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6", use_errno=True)
MS_BIND, MS_REC, MS_RDONLY, MS_PRIVATE, MS_REMOUNT = 0x1000, 0x4000, 0x1, 0x40000, 0x20


def _mount(src: str, dst: str, flags: int, fstype: str = "", data: str = "") -> None:
    if (
        libc.mount(
            src.encode(), dst.encode(), fstype.encode() or None, flags, (data.encode() or None)
        )
        != 0
    ):
        e = ctypes.get_errno()
        raise OSError(e, f"mount({src} -> {dst}, flags={flags}): {os.strerror(e)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mounts", required=True, help="mode:src:dst;... allowlist")
    ap.add_argument("--tmpdir", required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    inner = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    if not inner:
        print("no inner command", file=sys.stderr)
        return 2

    newroot = "/tmp/.arcsbx_root"
    os.makedirs(newroot, exist_ok=True)
    # make the namespace's propagation private, then a tmpfs new root
    _mount("none", "/", MS_REC | MS_PRIVATE, "")
    _mount("tmpfs", newroot, 0, "tmpfs")

    # minimal OS (ro) + the allowlist
    base = [
        ("ro", p, p) for p in ("/usr", "/bin", "/lib", "/lib64", "/etc/ssl") if os.path.exists(p)
    ]
    entries = base + [tuple(m.split(":", 2)) for m in args.mounts.split(";") if m]
    for mode, src, dst in entries:
        if not os.path.exists(src):
            continue
        target = newroot + dst
        os.makedirs(target, exist_ok=True) if os.path.isdir(src) else os.makedirs(
            os.path.dirname(target), exist_ok=True
        )
        if not os.path.isdir(src) and not os.path.exists(target):
            open(target, "a").close()
        _mount(src, target, MS_BIND | MS_REC)
        if mode == "ro":
            _mount(src, target, MS_BIND | MS_REMOUNT | MS_RDONLY | MS_REC)

    for special in ("/proc", "/dev"):
        os.makedirs(newroot + special, exist_ok=True)
    _mount("proc", newroot + "/proc", 0, "proc")
    _mount("/dev", newroot + "/dev", MS_BIND | MS_REC)

    # pivot into the new root
    os.makedirs(newroot + "/.oldroot", exist_ok=True)
    if libc.syscall(_PIVOT_ROOT_NR, newroot.encode(), (newroot + "/.oldroot").encode()) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"pivot_root: {os.strerror(e)}")
    os.chdir("/")
    libc.umount2(b"/.oldroot", 2)  # MNT_DETACH

    os.environ["TMPDIR"] = args.tmpdir
    os.execvp(inner[0], inner)


if __name__ == "__main__":
    raise SystemExit(main())
