# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration surface for the sandboxed cell-execution backend.

``SandboxConfig`` declares *intent* — which guardrails to enforce and how. The
parent resolves it into a concrete :class:`ResolvedSpec` (expanding the
workspace and system paths into path grants, deciding the network/rlimit
values) which the worker applies to itself before any cell runs.

Every guardrail is independently configurable. Each maps to a kernel mechanism
chosen per platform (see ``guards.py``):

* timeout  -> parent hard-kill (reuses ``CodeActConfig.cell_timeout``)
* memory   -> ``RLIMIT_AS`` (Linux) / Job Object limit (Windows)
* cpu      -> ``RLIMIT_CPU`` (Linux) / Job Object limit (Windows)
* files    -> Landlock path rules (Linux) / AppContainer ACL (Windows)
* network  -> seccomp ``socket()`` (Linux) / capability omission (Windows)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FileAccess = Literal["read", "read_write"]
SandboxBackend = Literal["auto", "landlock", "appcontainer"]


class FileRule(BaseModel):
    """An explicit filesystem allow rule: grant ``access`` beneath ``path``."""

    model_config = ConfigDict(frozen=True)

    path: str
    access: FileAccess = "read"


class SandboxConfig(BaseModel):
    """Declarative configuration for the four sandbox guardrails.

    Defaults are the *safe, minimal* posture: filesystem confined (read-only
    system paths plus an optional read-write workspace), network off. Memory
    and CPU caps are opt-in (``0`` disables them). All fields are ignored unless
    ``CodeActConfig.execution_backend == "sandbox"``.
    """

    model_config = ConfigDict(frozen=True)

    # --- guardrail 3: filesystem ------------------------------------------
    filesystem: bool = Field(
        default=True,
        description="Confine the filesystem (Landlock on Linux, AppContainer on Windows).",
    )
    workspace: str | None = Field(
        default=None,
        description="A directory the cell gets read+write access to. None -> no writable dir.",
    )
    allow: tuple[FileRule, ...] = Field(
        default=(),
        description="Extra explicit allow rules (read or read_write) beyond the workspace.",
    )
    system_paths: bool = Field(
        default=True,
        description="Auto-allow read of interpreter/libs (/usr, /lib, ...) so Python keeps working.",
    )

    # --- guardrail 4: network ---------------------------------------------
    network: bool = Field(
        default=False,
        description="False -> internet sockets blocked (AF_INET/AF_INET6); True -> allowed.",
    )

    # --- guardrail 2: memory + cpu ----------------------------------------
    max_memory_mb: int = Field(
        default=0,
        ge=0,
        description=(
            "Address-space headroom in MiB the cell may allocate above the worker "
            "baseline (RLIMIT_AS). 0 disables."
        ),
    )
    max_cpu_seconds: int = Field(
        default=0, ge=0, description="RLIMIT_CPU cap in CPU-seconds. 0 disables."
    )
    rss_poll_s: float = Field(
        default=0.25, gt=0, description="Parent RSS-watchdog poll interval (seconds)."
    )

    # --- guardrail 1: timeout ---------------------------------------------
    # No timeout field of its own — reuses CodeActConfig.cell_timeout. This is
    # the extra grace the parent waits past cell_timeout before SIGKILL.
    timeout_grace_s: float = Field(
        default=2.0, ge=0, description="Grace past cell_timeout before the parent hard-kills."
    )
    # Brokered ``self.*`` calls run parent-side while the worker idles; their time
    # must NOT consume the cell deadline (the guardrail targets runaway CELL code).
    # They get their own generous bound instead. 0 -> unbounded.
    broker_timeout_s: float = Field(
        default=300.0,
        ge=0,
        description="Per-call bound for brokered self.* parent-side work; 0 = unbounded.",
    )

    # --- process / lifecycle ----------------------------------------------
    # "fork" inherits the live agent, the CurrentCall and the return_result
    # closure directly. "spawn" cannot inherit anything, so it rebuilds the
    # worker namespace from a WorkerBlueprint (see ``blueprint.py``) and brokers
    # every ``self.*`` access, including introspection.
    start_method: Literal["fork", "spawn"] = Field(
        default="fork",
        description=(
            "Process start method for the worker. 'fork' requires a Unix process "
            "model; 'spawn' rebuilds state from a blueprint and works on Windows."
        ),
    )
    backend: SandboxBackend = Field(
        default="auto",
        description=(
            "Enforcement backend. 'auto' picks the platform's mechanism; naming one "
            "explicitly fails closed if this host cannot provide it."
        ),
    )
    runtime_root: str | None = Field(
        default=None,
        description=(
            "Windows only: private interpreter root the AppContainer runs from. "
            "None -> $NOOA_SANDBOX_RUNTIME, else a per-user cache. Build it once with "
            "`python -m nooa.runtime.sandbox.win_runtime --build --root <dir>`."
        ),
    )
    container_name: str | None = Field(
        default=None,
        description=(
            "Windows only: the AppContainer profile this session runs under. The "
            "profile name determines the container SID, so two live sessions that "
            "share a name share an identity and can revoke each other's grants. "
            "None -> a unique per-session name. Set it only to make a session "
            "reuse a profile deliberately, never as a shared default."
        ),
    )
    env_passthrough: tuple[str, ...] = Field(
        default=(),
        description=(
            "Windows only: parent environment variables the AppContainer worker "
            "receives beyond the OS baseline (e.g. a proxy or an API key a cell "
            "needs with network=True). Everything else stays in the parent. Linux "
            "workers inherit the full environment."
        ),
    )
    recovery: Literal["restart_empty", "disabled"] = Field(
        default="restart_empty",
        description="After a kill: restart the worker fresh (empty namespace) or disable it.",
    )
    require: bool = Field(
        default=True,
        description="Fail closed: raise SandboxUnavailable if a requested guard is unenforceable.",
    )
    context_block: bool = Field(
        default=True, description="Inject a context block describing the active constraints."
    )

    def any_guard_requested(self) -> bool:
        """True if this config asks for at least one OS-enforced guardrail."""
        return bool(
            self.filesystem or not self.network or self.max_memory_mb or self.max_cpu_seconds
        )


# Access-right bundles are computed in guards.py from the live Landlock ABI;
# here we only carry the resolved, ABI-agnostic intent.
@dataclass(frozen=True)
class PathGrant:
    """A resolved path rule: grant ``read``/``write`` beneath ``path``.

    Platform-neutral by design: the Linux backend turns this into a Landlock
    rule, the Windows backend into an AppContainer ACL grant.

    ``required`` rules (the user's explicit workspace / allow paths) must exist at
    apply time — a missing one is an error, not a silent skip — whereas the
    system-path superset is best-effort.
    """

    path: str
    write: bool = False
    required: bool = False


# Backwards-compatible alias for the name this type had while only Landlock
# existed. Remove once external callers have migrated.
LandlockRule = PathGrant


@dataclass(frozen=True)
class ResolvedSpec:
    """Concrete, host-independent guard spec applied to the worker.

    Produced by ``resolve_spec()`` from a :class:`SandboxConfig`. Everything here
    is plain data so it pickles cleanly across a process boundary.
    """

    filesystem: bool = False
    path_grants: tuple[PathGrant, ...] = ()
    block_network: bool = False
    max_memory_mb: int = 0
    max_cpu_seconds: int = 0
    backend: SandboxBackend = "auto"

    def requests_anything(self) -> bool:
        return bool(
            self.filesystem or self.block_network or self.max_memory_mb or self.max_cpu_seconds
        )


# Default read-only system paths that must stay reachable for CPython to import
# modules and read its own standard library / installed packages. Missing paths
# are skipped at apply time, so listing a superset is safe.
DEFAULT_SYSTEM_READ_PATHS: tuple[str, ...] = (
    "/usr",
    "/bin",
    "/sbin",
    "/lib",
    "/lib64",
    "/etc",
    "/opt",
    "/proc",
    "/dev",
    "/sys",
)


def _interpreter_read_paths() -> list[str]:
    """Paths CPython needs to read: prefix, stdlib, site-packages, the venv."""
    import site
    import sys
    import sysconfig

    paths: list[str] = [sys.prefix, sys.base_prefix, sys.exec_prefix]
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        try:
            paths.append(sysconfig.get_path(key))
        except (KeyError, OSError):
            pass
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        paths.append(site.getusersitepackages())
    except Exception:
        pass
    return [p for p in paths if p]


def resolve_spec(config: SandboxConfig) -> ResolvedSpec:
    """Turn declarative :class:`SandboxConfig` into a concrete :class:`ResolvedSpec`.

    Deduplicates and normalizes path grants (read vs read-write), folds in the
    interpreter/system read paths, and records the network/resource intent.
    """
    paths: dict[str, tuple[bool, bool]] = {}  # path -> (write, required)

    def add(path: str, write: bool, required: bool = False) -> None:
        # Merge: a later read-write / required grant upgrades an earlier one.
        w, r = paths.get(path, (False, False))
        paths[path] = (w or write, r or required)

    grants: tuple[PathGrant, ...] = ()
    if config.filesystem:
        if config.system_paths:
            for p in DEFAULT_SYSTEM_READ_PATHS:
                add(p, False)
            for p in _interpreter_read_paths():
                add(p, False)
            # Writing /dev/null (subprocess.DEVNULL, output suppression) is common
            # and harmless; grant it explicitly since /dev is otherwise read-only.
            add("/dev/null", True)
        if config.workspace:
            add(config.workspace, True, required=True)
        for rule in config.allow:
            add(rule.path, rule.access == "read_write", required=True)
        grants = tuple(
            PathGrant(path=p, write=w, required=r) for p, (w, r) in sorted(paths.items())
        )

    return ResolvedSpec(
        filesystem=config.filesystem,
        path_grants=grants,
        block_network=not config.network,
        max_memory_mb=config.max_memory_mb,
        max_cpu_seconds=config.max_cpu_seconds,
        backend=config.backend,
    )
