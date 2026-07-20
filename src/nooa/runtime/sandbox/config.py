# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration surface for the sandboxed cell-execution backend.

``SandboxConfig`` declares *intent* — which guardrails to enforce and how. The
parent resolves it into a concrete :class:`ResolvedSpec` (expanding the
workspace and system paths into Landlock rules, deciding the seccomp/rlimit
values) which the worker applies to itself after fork, before any cell runs.

Every guardrail is independently configurable and each maps to an irrevocable
kernel mechanism (see ``guards.py``):

* timeout  -> parent hard-kill (reuses ``CodeActConfig.cell_timeout``)
* memory   -> ``RLIMIT_AS``          (``max_memory_mb``)
* cpu      -> ``RLIMIT_CPU``         (``max_cpu_seconds``)
* files    -> Landlock path rules    (``filesystem``/``workspace``/``allow``/``deny``)
* network  -> seccomp ``socket()``   (``network``)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

FileAccess = Literal["read", "read_write"]


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
        description="Enforce Landlock filesystem confinement (default-deny).",
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
    # Only "fork" is supported: the worker init inherits the live agent, the
    # CurrentCall and the return_result closure, none of which pickle for "spawn".
    start_method: Literal["fork"] = Field(
        default="fork", description="multiprocessing start method for the worker (fork only)."
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
class LandlockRule:
    """A resolved Landlock path rule: grant ``read``/``write`` beneath ``path``.

    ``required`` rules (the user's explicit workspace / allow paths) must exist at
    apply time — a missing one is an error, not a silent skip — whereas the
    system-path superset is best-effort.
    """

    path: str
    write: bool = False
    required: bool = False


@dataclass(frozen=True)
class ResolvedSpec:
    """Concrete, host-independent guard spec applied by the worker after fork.

    Produced by ``resolve_spec()`` from a :class:`SandboxConfig`. Everything here
    is plain data so it pickles cleanly across the ``spawn`` boundary.
    """

    filesystem: bool = False
    landlock_rules: tuple[LandlockRule, ...] = ()
    block_network: bool = False
    max_memory_mb: int = 0
    max_cpu_seconds: int = 0

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

    Deduplicates and normalizes Landlock rules (read vs read-write), folds in the
    interpreter/system read paths, and records the network/rlimit intent.
    """
    paths: dict[str, tuple[bool, bool]] = {}  # path -> (write, required)

    def add(path: str, write: bool, required: bool = False) -> None:
        # Merge: a later read-write / required grant upgrades an earlier one.
        w, r = paths.get(path, (False, False))
        paths[path] = (w or write, r or required)

    rules: tuple[LandlockRule, ...] = ()
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
        rules = tuple(
            LandlockRule(path=p, write=w, required=r) for p, (w, r) in sorted(paths.items())
        )

    return ResolvedSpec(
        filesystem=config.filesystem,
        landlock_rules=rules,
        block_network=not config.network,
        max_memory_mb=config.max_memory_mb,
        max_cpu_seconds=config.max_cpu_seconds,
    )
