# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""L3 OS sandbox for the ARC-AGI-3 agent process (isolation plan §L3).

Wraps ONLY the agent/launcher command in a user + mount + network namespace so
that, by construction, the agent cannot:
  - reach the network (no interface; LLM/embeddings go through an out-of-namespace
    Unix-socket broker — see llm_broker.py),
  - read the game source (the vendored game wrapper + game data are never mounted),
  - read any other run (the results parent is never mounted; only THIS run's
    ipc/, workspace, own agent_logs, and traces are bound in).

Uses ``bwrap`` (bubblewrap) when present, else an ``unshare`` fallback. Both need
kernel support for user namespaces; ``preflight()`` verifies this and the wrapper
**fails closed** — if full isolation was requested but namespaces are unavailable,
it raises instead of silently running the agent unsandboxed.

This module builds and returns the wrapped argv; it does not itself require
privileges to import. Actual isolation is only exercised where the kernel permits
namespace creation (a host without ``kernel.apparmor_restrict_unprivileged_userns``
lockdown, or with a setuid ``bwrap``). It is intentionally inert on hosts that
forbid namespaces — there it refuses rather than pretends.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class SandboxUnavailable(RuntimeError):
    """Raised when full isolation is requested but the kernel forbids it."""


@dataclass(frozen=True)
class SandboxSpec:
    run_dir: Path  # <results>/<group>/<run>   (rw: ipc + workspace + own logs)
    repo_root: Path  # main repo (ro: venv + example code)
    llm_socket: Path  # UDS the broker listens on (bind-mounted in)
    tmp_dir: Path  # scratch (rw, becomes TMPDIR)

    def agent_mounts(self) -> list[tuple[str, Path, Path]]:
        """(mode, src, dst) allowlist. dst mirrors src so in-namespace paths are
        stable for the launcher/agent code."""
        ex = self.repo_root / "examples" / "arc_agi_3"
        venv = self.repo_root / ".venv"
        return [
            ("rw", self.run_dir / "ipc", self.run_dir / "ipc"),
            ("rw", self.run_dir / "team_nemo" / "shared", self.run_dir / "team_nemo" / "shared"),
            # The launcher writes its OWN logs + traces here, so these are rw (the
            # caller pre-creates both so bwrap's bind has an existing source).
            ("rw", self.run_dir / "agent_logs", self.run_dir / "agent_logs"),
            ("rw", self.run_dir / "traces", self.run_dir / "traces"),
            ("ro", venv, venv),
            ("ro", ex, ex),
            ("ro", self.repo_root / "src", self.repo_root / "src"),
            ("rw", self.tmp_dir, self.tmp_dir),
            ("rw", self.llm_socket.parent, self.llm_socket.parent),
        ]


def preflight() -> tuple[bool, str]:
    """Return (ok, reason). ok=True means a namespace sandbox can be created."""
    lock = Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns")
    if lock.exists():
        try:
            if lock.read_text().strip() == "1" and not _has_setuid_bwrap():
                return False, (
                    "unprivileged user namespaces are restricted by "
                    "AppArmor (kernel.apparmor_restrict_unprivileged_userns=1) "
                    "and no setuid bwrap is available"
                )
        except OSError:
            pass
    maxns = Path("/proc/sys/user/max_user_namespaces")
    if maxns.exists() and maxns.read_text().strip() == "0":
        return False, "user namespaces disabled (max_user_namespaces=0)"
    # Actually attempt a throwaway namespace — the only definitive check.
    tool = shutil.which("bwrap")
    if tool:
        probe = [tool, "--unshare-all", "--ro-bind", "/usr", "/usr", "true"]
    elif shutil.which("unshare"):
        probe = ["unshare", "--user", "--map-root-user", "--net", "true"]
    else:
        return False, "neither bwrap nor unshare is installed"
    try:
        r = subprocess.run(probe, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"namespace probe failed to launch: {e}"
    if r.returncode != 0:
        return False, (
            f"namespace probe rejected by kernel: "
            f"{r.stderr.decode().strip() or 'operation not permitted'}"
        )
    return True, f"namespaces available via {Path(tool or 'unshare').name}"


def _has_setuid_bwrap() -> bool:
    p = shutil.which("bwrap")
    if not p:
        return False
    try:
        return bool(Path(p).stat().st_mode & 0o4000)
    except OSError:
        return False


def wrap(inner_cmd: list[str], spec: SandboxSpec, *, require: bool = True) -> list[str]:
    """Return ``inner_cmd`` wrapped in the sandbox, or raise SandboxUnavailable.

    ``require=True`` (default) fails closed: if namespaces are unavailable the
    caller must NOT fall back to running the agent unsandboxed. Pass require=False
    only for explicit best-effort modes.
    """
    ok, reason = preflight()
    if not ok:
        if require:
            raise SandboxUnavailable(
                f"L3 sandbox requested but unavailable: {reason}. "
                f"Run on a host that permits user namespaces (or install a setuid "
                f"bwrap), or select --sandbox inproc/off explicitly."
            )
        return list(inner_cmd)

    if shutil.which("bwrap"):
        return _bwrap_argv(inner_cmd, spec)
    return _unshare_argv(inner_cmd, spec)


def _bwrap_argv(inner_cmd: list[str], spec: SandboxSpec) -> list[str]:
    argv = [
        "bwrap",
        "--unshare-all",  # user+mount+net+pid+ipc+uts+cgroup
        "--die-with-parent",
        "--new-session",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--setenv",
        "TMPDIR",
        str(spec.tmp_dir),
        # minimal read-only OS so the interpreter runs
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/bin",
        "/bin",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--ro-bind",
        "/etc/ssl",
        "/etc/ssl",
    ]
    for mode, src, dst in spec.agent_mounts():
        flag = "--bind" if mode == "rw" else "--ro-bind"
        argv += [flag, str(src), str(dst)]
    return argv + ["--"] + list(inner_cmd)


def _unshare_argv(inner_cmd: list[str], spec: SandboxSpec) -> list[str]:
    # Fallback: unshare cannot express an allowlist as tersely as bwrap. We
    # unshare user+mount+net+pid and rely on a helper that bind-mounts the
    # allowlist over a tmpfs root before exec. That helper is emitted inline.
    # (bwrap is strongly preferred; this path exists for hosts without it.)
    mounts = ";".join(f"{mode}:{src}:{dst}" for mode, src, dst in spec.agent_mounts())
    helper = Path(__file__).resolve().parent / "sandbox_mount_helper.py"
    return [
        "unshare",
        "--user",
        "--map-root-user",
        "--mount",
        "--net",
        "--pid",
        "--fork",
        "--kill-child",
        "python3",
        str(helper),
        "--mounts",
        mounts,
        "--tmpdir",
        str(spec.tmp_dir),
        "--",
        *inner_cmd,
    ]


if __name__ == "__main__":
    ok, reason = preflight()
    print(f"sandbox preflight: {'AVAILABLE' if ok else 'UNAVAILABLE'} — {reason}")
    raise SystemExit(0 if ok else 1)
