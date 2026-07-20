# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Irrevocable, OS-enforced sandbox guardrails (Linux, unprivileged).

Each primitive is a self-imposed kernel restriction the *child* installs on
itself after fork and before running any cell. Once installed none can be
lifted by subsequent code — that is the "no leakage by design" guarantee:

* ``apply_rlimits``       -> ``RLIMIT_AS`` / ``RLIMIT_CPU`` with ``soft == hard``
* ``apply_landlock``      -> Landlock path-beneath rules (default-deny filesystem)
* ``apply_seccomp_no_inet`` -> seccomp-BPF denying ``socket(AF_INET/AF_INET6)``

All are implemented with raw ``ctypes`` syscalls — no third-party dependency.
They target Linux; on other platforms the probes report the mechanism as
unavailable and the executor fails closed (see ``capabilities.py``).
"""

from __future__ import annotations

import ctypes
import os
import platform
import socket
import stat
import struct
from dataclasses import dataclass

from nooa.runtime.sandbox.config import LandlockRule, ResolvedSpec

# --- syscall numbers (Linux) ------------------------------------------------
# Landlock syscall numbers are identical across x86_64 and aarch64.
_NR_LANDLOCK_CREATE_RULESET = 444
_NR_LANDLOCK_ADD_RULE = 445
_NR_LANDLOCK_RESTRICT_SELF = 446
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1

_PR_SET_NO_NEW_PRIVS = 38
_SECCOMP_SET_MODE_FILTER = 1

_ARCH = platform.machine()
if _ARCH == "x86_64":
    _NR_SECCOMP = 317
    _AUDIT_ARCH = 0xC000003E
    _NR_SOCKET = 41
elif _ARCH in ("aarch64", "arm64"):
    _NR_SECCOMP = 277
    _AUDIT_ARCH = 0xC00000B7
    _NR_SOCKET = 198
else:  # pragma: no cover - unsupported arch; probes below report unavailable
    _NR_SECCOMP = -1
    _AUDIT_ARCH = 0
    _NR_SOCKET = -1

# --- Landlock filesystem access-right bits (linux/landlock.h) ---------------
_FS = {
    "EXECUTE": 1 << 0,
    "WRITE_FILE": 1 << 1,
    "READ_FILE": 1 << 2,
    "READ_DIR": 1 << 3,
    "REMOVE_DIR": 1 << 4,
    "REMOVE_FILE": 1 << 5,
    "MAKE_CHAR": 1 << 6,
    "MAKE_DIR": 1 << 7,
    "MAKE_REG": 1 << 8,
    "MAKE_SOCK": 1 << 9,
    "MAKE_FIFO": 1 << 10,
    "MAKE_BLOCK": 1 << 11,
    "MAKE_SYM": 1 << 12,
    "REFER": 1 << 13,  # ABI 2
    "TRUNCATE": 1 << 14,  # ABI 3
    "IOCTL_DEV": 1 << 15,  # ABI 5
}
_FS_READ = _FS["READ_FILE"] | _FS["READ_DIR"] | _FS["EXECUTE"]
_FS_WRITE = (
    _FS["WRITE_FILE"]
    | _FS["REMOVE_DIR"]
    | _FS["REMOVE_FILE"]
    | _FS["MAKE_CHAR"]
    | _FS["MAKE_DIR"]
    | _FS["MAKE_REG"]
    | _FS["MAKE_SOCK"]
    | _FS["MAKE_FIFO"]
    | _FS["MAKE_BLOCK"]
    | _FS["MAKE_SYM"]
    | _FS["REFER"]
    | _FS["TRUNCATE"]
)
# Rights that apply to a regular file (not a directory). Granting a directory-only
# right (READ_DIR, MAKE_*, REMOVE_*) on a non-directory makes landlock_add_rule
# fail with EINVAL, so file rules are masked to these bits.
_FS_FILE_MASK = _FS["EXECUTE"] | _FS["READ_FILE"] | _FS["WRITE_FILE"] | _FS["TRUNCATE"]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


class _RulesetAttr(ctypes.Structure):
    _fields_ = [
        ("handled_access_fs", ctypes.c_uint64),
        ("handled_access_net", ctypes.c_uint64),
    ]


class _PathBeneathAttr(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]


# ---------------------------------------------------------------------------
# capability probes
# ---------------------------------------------------------------------------
def landlock_abi() -> int:
    """Return the kernel's Landlock ABI version, or 0 if Landlock is absent."""
    if _ARCH not in ("x86_64", "aarch64", "arm64"):
        return 0
    try:
        libc = _libc()
        abi = libc.syscall(_NR_LANDLOCK_CREATE_RULESET, None, 0, _LANDLOCK_CREATE_RULESET_VERSION)
        return int(abi) if abi and abi > 0 else 0
    except OSError:
        return 0


def _abi_fs_mask(abi: int) -> int:
    """Mask of filesystem rights supported by ``abi`` (avoids EINVAL on old kernels)."""
    bits = (1 << 13) - 1  # ABI 1: bits 0..12
    if abi >= 2:
        bits |= _FS["REFER"]
    if abi >= 3:
        bits |= _FS["TRUNCATE"]
    if abi >= 5:
        bits |= _FS["IOCTL_DEV"]
    return bits


def seccomp_supported() -> bool:
    """True if this process can install a seccomp filter (probe in a child)."""
    if _NR_SECCOMP < 0:
        return False
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child branch
        try:
            _install_no_new_privs()
            _seccomp_install(_build_no_inet_filter())
            os._exit(0)
        except BaseException:
            os._exit(1)
    _, status = os.waitpid(pid, 0)
    return os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0


@dataclass(frozen=True)
class Capabilities:
    """What the running host can actually enforce."""

    linux: bool
    landlock_abi: int
    seccomp: bool
    rlimit: bool

    @property
    def filesystem(self) -> bool:
        return self.landlock_abi >= 1

    @property
    def network(self) -> bool:
        return self.seccomp


def probe_capabilities() -> Capabilities:
    """Probe the host once for every enforcement mechanism the sandbox uses."""
    import resource

    is_linux = platform.system() == "Linux"
    return Capabilities(
        linux=is_linux,
        landlock_abi=landlock_abi() if is_linux else 0,
        seccomp=seccomp_supported() if is_linux else False,
        rlimit=hasattr(resource, "RLIMIT_AS") and hasattr(resource, "RLIMIT_CPU"),
    )


# ---------------------------------------------------------------------------
# guardrail: memory + cpu (rlimits)
# ---------------------------------------------------------------------------
def _self_vmsize_bytes() -> int:
    """Current virtual address-space size of this process (0 if unreadable)."""
    try:
        with open("/proc/self/status", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("VmSize:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return 0
    return 0


def apply_rlimits(*, max_memory_mb: int = 0, max_cpu_seconds: int = 0) -> None:
    """Cap address space and/or CPU time with ``soft == hard`` (unraisable).

    ``max_memory_mb`` is interpreted as *headroom above the worker's current
    virtual size*: the forked worker already maps the interpreter and libraries,
    so an absolute cap below that baseline would break it instantly. The cap is
    ``current VmSize + max_memory_mb`` — i.e. how much more the cell may allocate.
    """
    import resource

    if max_memory_mb and max_memory_mb > 0:
        baseline = _self_vmsize_bytes()
        limit = baseline + int(max_memory_mb) * 1024 * 1024
        _, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard != resource.RLIM_INFINITY:
            limit = min(limit, hard)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    if max_cpu_seconds and max_cpu_seconds > 0:
        secs = int(max_cpu_seconds)
        _, hard = resource.getrlimit(resource.RLIMIT_CPU)
        if hard != resource.RLIM_INFINITY:
            secs = min(secs, hard)
        resource.setrlimit(resource.RLIMIT_CPU, (secs, secs))


# ---------------------------------------------------------------------------
# guardrail: filesystem (Landlock)
# ---------------------------------------------------------------------------
def _install_no_new_privs() -> None:
    if _libc().prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_NO_NEW_PRIVS) failed")


def _is_dir(fd: int) -> bool:
    """True if the O_PATH file descriptor refers to a directory."""
    try:
        return stat.S_ISDIR(os.fstat(fd).st_mode)
    except OSError:
        return False


def apply_landlock(rules: list[LandlockRule]) -> None:
    """Confine the filesystem to ``rules`` (default-deny) via Landlock.

    Any access to a path not covered by a rule granting that access is denied.
    A ``required`` rule whose path is missing raises ``OSError`` (a typo'd
    workspace must not silently become an unwritable sandbox); non-required
    (system) paths are skipped. Requires Landlock ABI >= 1. Irrevocable once
    ``landlock_restrict_self`` returns.
    """
    abi = landlock_abi()
    if abi < 1:
        raise OSError("Landlock is not available on this kernel")
    mask = _abi_fs_mask(abi)
    libc = _libc()

    attr = _RulesetAttr(handled_access_fs=mask, handled_access_net=0)
    rfd = libc.syscall(_NR_LANDLOCK_CREATE_RULESET, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if rfd < 0:
        raise OSError(ctypes.get_errno(), "landlock_create_ruleset failed")
    try:
        for rule in rules:
            try:
                pfd = os.open(rule.path, os.O_PATH | os.O_CLOEXEC)
            except OSError as exc:
                if rule.required:
                    raise OSError(
                        exc.errno, f"sandbox allow path does not exist: {rule.path}"
                    ) from exc
                continue  # optional (system) path missing: stays denied
            access = _FS_READ | (_FS_WRITE if rule.write else 0)
            # Directory-only rights on a plain file are rejected (EINVAL); mask them.
            if not _is_dir(pfd):
                access &= _FS_FILE_MASK
            pb = _PathBeneathAttr(allowed_access=access & mask, parent_fd=pfd)
            rc = libc.syscall(
                _NR_LANDLOCK_ADD_RULE,
                rfd,
                _LANDLOCK_RULE_PATH_BENEATH,
                ctypes.byref(pb),
                0,
            )
            err = ctypes.get_errno()
            os.close(pfd)
            if rc != 0:
                raise OSError(err, f"landlock_add_rule failed for {rule.path}")
        _install_no_new_privs()
        if libc.syscall(_NR_LANDLOCK_RESTRICT_SELF, rfd, 0) != 0:
            raise OSError(ctypes.get_errno(), "landlock_restrict_self failed")
    finally:
        os.close(rfd)


# ---------------------------------------------------------------------------
# guardrail: network (seccomp)
# ---------------------------------------------------------------------------
def _build_no_inet_filter() -> bytes:
    """BPF program: deny ``socket(AF_INET/AF_INET6)`` with EACCES, allow the rest.

    ``AF_UNIX`` (and every other syscall) is untouched, so the parent broker pipe
    and local IPC keep working while all internet sockets are refused at creation.
    """
    BPF_LD = 0x00
    BPF_JMP = 0x05
    BPF_RET = 0x06
    BPF_W = 0x00
    BPF_ABS = 0x20
    BPF_JEQ = 0x10
    BPF_K = 0x00
    RET_ALLOW = 0x7FFF0000
    RET_ERRNO = 0x00050000
    EACCES = 13

    def ins(code: int, jt: int, jf: int, k: int) -> bytes:
        return struct.pack("HBBI", code, jt, jf, k)

    # seccomp_data offsets: nr@0, arch@4, args[0]@16 (little-endian low word).
    return b"".join(
        [
            ins(BPF_LD | BPF_W | BPF_ABS, 0, 0, 4),  # A = arch
            ins(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, _AUDIT_ARCH),  # native? -> continue
            ins(BPF_RET | BPF_K, 0, 0, RET_ALLOW),  # foreign arch -> allow
            ins(BPF_LD | BPF_W | BPF_ABS, 0, 0, 0),  # A = nr
            ins(BPF_JMP | BPF_JEQ | BPF_K, 0, 4, _NR_SOCKET),  # nr!=socket -> allow(+4)
            ins(BPF_LD | BPF_W | BPF_ABS, 0, 0, 16),  # A = args[0] (domain)
            ins(BPF_JMP | BPF_JEQ | BPF_K, 1, 0, socket.AF_INET),  # INET -> errno
            ins(BPF_JMP | BPF_JEQ | BPF_K, 0, 1, socket.AF_INET6),  # INET6 -> errno else allow
            ins(BPF_RET | BPF_K, 0, 0, RET_ERRNO | EACCES),
            ins(BPF_RET | BPF_K, 0, 0, RET_ALLOW),
        ]
    )


def _seccomp_install(prog: bytes) -> None:
    buf = ctypes.create_string_buffer(prog, len(prog))
    fprog = _SockFprog(len=len(prog) // 8, filter=ctypes.cast(buf, ctypes.c_void_p).value)
    if _libc().syscall(_NR_SECCOMP, _SECCOMP_SET_MODE_FILTER, 0, ctypes.byref(fprog)) != 0:
        raise OSError(ctypes.get_errno(), "seccomp(SET_MODE_FILTER) failed")


def apply_seccomp_no_inet() -> None:
    """Block creation of any internet (AF_INET/AF_INET6) socket, irrevocably."""
    if _NR_SECCOMP < 0:
        raise OSError("seccomp is not available on this architecture")
    _install_no_new_privs()
    _seccomp_install(_build_no_inet_filter())


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def install_guards(spec: ResolvedSpec) -> None:
    """Apply every requested guardrail to the current process, in a safe order.

    Order matters: rlimits first (no dependency), then Landlock (needs to
    ``open()`` rule paths — must precede filesystem lock-down), then seccomp
    (does not affect ``open()``). All three are irrevocable once set.
    """
    apply_rlimits(max_memory_mb=spec.max_memory_mb, max_cpu_seconds=spec.max_cpu_seconds)
    if spec.filesystem:
        apply_landlock(list(spec.landlock_rules))
    if spec.block_network:
        apply_seccomp_no_inet()
