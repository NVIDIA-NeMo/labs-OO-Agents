# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M0 feasibility runner: can a native Windows AppContainer host a CodeAct worker?

Five questions, in dependency order. Each prints a verdict; the whole run writes
``results.json``. Nothing here is production code — it is the measurement that
decides whether the AppContainer backend is worth building.

  1. reachable  — can the container's Python import and start at all?
  2. reversible — can the ACL grants be undone exactly?
  3. denied     — are reads/writes outside the grants actually refused?
  4. no-network — is outbound blocked without capability SIDs?
  5. job-limits — do memory/CPU caps and hard kill work?
  6. broker     — can an inherited pipe carry the broker protocol?
  7. nooa-import— does the package resolve inside the root, not the checkout?
  8. mp-pipe    — does the broker's real transport cross the boundary?

Run:  python runner.py            (all scenarios)
      python runner.py --only 1   (one scenario)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import win32 as w  # noqa: E402

HERE = Path(__file__).resolve().parent
RUNTIME_ROOT = HERE / ".rt"
WORKDIR = HERE / ".work"
RESULTS = HERE / "results.json"
CONTAINER_NAME = "nooa-m0-probe"

# Paths the container must NOT reach. These are read-only probes: nothing is
# modified, only the presence of an access denial is recorded.
#
# Every path here exists on this machine. A missing file would report errno 2
# ("not found") rather than errno 13 ("access denied"), which is not evidence of
# isolation — so the list is checked against the filesystem at startup.
DENY_READS = [
    r"C:\Windows\System32\config\SAM",
    r"C:\Windows\System32\config\SYSTEM",
    os.path.join(os.environ.get("USERPROFILE", r"C:\Users\default"), ".ssh"),
    os.path.join(os.environ.get("USERPROFILE", r"C:\Users\default"), ".claude", "CLAUDE.md"),
    # A real file in the repository, outside the granted work directory: the
    # checkout must not be reachable just because the interpreter copy is.
    str(HERE.parents[1] / "AGENTS.md"),
]
# A path with no grant at all, to separate "denied by ACL" from "denied by
# AppContainer default" — both are denials, but the error differs.
DENY_WRITES = [
    r"C:\Windows\System32\nooa_probe.txt",
    r"C:\Windows\Temp\nooa_probe.txt",
    os.path.join(os.environ.get("USERPROFILE", r"C:\Users\default"), "nooa_probe.txt"),
    r"E:\labs-OO-Agents\experiments\windows-sandbox-probe\_outside_grant.txt",
]

results: dict = {}


def note(section: str, key: str, value) -> None:
    results.setdefault(section, {})[key] = value
    print(f"    {key}: {value}")


def run_child(
    argv_extra: list[str],
    *,
    grant_write: bool = True,
    memory_mb: int = 0,
    cpu_seconds: int = 0,
    timeout_s: int = 30,
    capabilities: tuple[str, ...] = (),
) -> tuple[int, dict | None, float]:
    """Run child_probe.py inside the container; return (exit code, report, seconds).

    The process is created suspended, assigned to the job, then resumed — the
    window between creation and job membership is the escape surface, so it must
    be closed before any container code runs.
    """
    out = WORKDIR / "child_report.json"
    if out.exists():
        out.unlink()

    argv = [
        str(RUNTIME_ROOT / "python.exe"),
        "-I",  # isolated: ignore PYTHON* env, no user site-packages
        str(HERE / "child_probe.py"),
        "--out",
        str(out),
        "--reads",
        *DENY_READS,
        "--writes",
        str(WORKDIR / "granted_probe.txt"),
        *DENY_WRITES,
    ]
    if memory_mb:
        argv += ["--memory-mb", str(memory_mb)]
    if cpu_seconds:
        argv += ["--cpu-seconds", str(cpu_seconds)]
    argv += argv_extra

    app_sid = w.container_sid(CONTAINER_NAME)
    _grant_interpreter(app_sid)
    if grant_write:
        w.grant(str(WORKDIR), app_sid, w.FILE_ALL_ACCESS)

    job = w.Job()
    if memory_mb or cpu_seconds:
        job.set_limits(memory_mb=memory_mb, cpu_seconds=cpu_seconds)

    t0 = time.perf_counter()
    child = w.spawn_in_container(
        argv,
        container_name=CONTAINER_NAME,
        capabilities=capabilities,
        cwd=str(WORKDIR),
    )
    job.assign(child.pi.hProcess)
    child.resume()
    code = child.wait(timeout_ms=timeout_s * 1000)
    elapsed = time.perf_counter() - t0
    child.close()
    job.terminate(0)
    job.close()

    report = None
    if out.exists():
        try:
            report = json.loads(out.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            report = {"_parse_error": str(exc)}
    return code, report, elapsed


# --- scenario 1: reachable --------------------------------------------------
def scenario_reachable() -> None:
    print("\n[1] reachable — can the container start the interpreter?")
    app_sid = w.container_sid(CONTAINER_NAME)
    _clean(WORKDIR)

    grants = [
        (str(RUNTIME_ROOT), w.FILE_GENERIC_READ | w.FILE_GENERIC_EXECUTE, "runtime root"),
        (
            str(RUNTIME_ROOT / "Lib" / "site-packages"),
            w.FILE_GENERIC_READ | w.FILE_GENERIC_EXECUTE,
            "site-packages",
        ),
        # The probe script is a single read-only grant. The surrounding repo
        # directory stays denied — granting the directory would hand the
        # container the whole checkout, which is the leak this test exists to
        # detect in the first place.
        (str(HERE / "child_probe.py"), w.FILE_GENERIC_READ, "probe script"),
        (str(WORKDIR), w.FILE_ALL_ACCESS, "work dir"),
    ]
    backups: dict[str, str] = {}
    for path, access, label in grants:
        if not os.path.exists(path):
            note("reachable", f"skip_{label}", "missing")
            continue
        backups[path] = w.read_dacl_sddl(path)
        try:
            w.grant(path, app_sid, access)
            note("reachable", f"grant_{label}", "ok")
        except OSError as exc:
            note("reachable", f"grant_{label}", f"FAILED {exc}")

    code, report, elapsed = run_child([])
    note("reachable", "exit_code", code)
    note("reachable", "wall_clock_s", round(elapsed, 3))
    if report:
        ident = report.get("identity", {})
        note("reachable", "is_app_container", ident.get("is_app_container"))
        note("reachable", "container_sid", ident.get("container_sid"))
        note("reachable", "sys_prefix", ident.get("prefix"))
        note("reachable", "child_pid", report.get("pid"))
    else:
        note("reachable", "child_report", "MISSING (child could not run or write)")

    # Immediately measure the reversible-restore question on these same paths.
    restore_ok = True
    for path, sddl in backups.items():
        try:
            w.set_dacl_sddl(path, sddl)
        except OSError as exc:
            restore_ok = False
            note("reachable", f"restore_error_{Path(path).name}", str(exc))
    note("reachable", "restore_calls_ok", restore_ok)
    results["_reachable_report"] = report


# --- scenario 2: reversibility ---------------------------------------------
ISO = HERE / ".iso"
_PROTECTED_DACL = 0x80000000 | 0x00000004  # PROTECTED_DACL_SECURITY_INFORMATION


def _set_dacl_with_flags(path: str, sddl: str, *, protect: bool) -> None:
    """Restore a DACL from SDDL, optionally breaking inheritance.

    The auto-inherited (``AI``) bit is not writable as data: a directory that
    inherits from its parent gets it recomputed on every write. A reversibility
    test must therefore run on a directory whose DACL is protected, otherwise it
    measures the parent walk rather than the rollback.
    """
    import ctypes
    from ctypes import wintypes

    sd = ctypes.c_void_p()
    if not w._adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, 1, ctypes.byref(sd), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        dacl = ctypes.c_void_p()
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        if not w._adv.GetSecurityDescriptorDacl(
            sd, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        flags = 0x00000004 | (_PROTECTED_DACL if protect else 0)
        rc = w._adv.SetNamedSecurityInfoW(
            path, 1, flags, None, None, dacl if present.value else None, None
        )
        if rc != 0:
            raise OSError(rc, f"SetNamedSecurityInfoW failed for {path}")
    finally:
        w._k32.LocalFree(sd)


def scenario_reversibility() -> None:
    """Compare ACE sets, not SDDL strings: GRANT_ACCESS merges and reorders."""
    print("\n[2] reversible — can a grant be undone exactly?")
    app_sid = w.container_sid(CONTAINER_NAME)
    if ISO.exists():
        shutil.rmtree(ISO)
    ISO.mkdir(parents=True, exist_ok=True)
    target = str(ISO)

    # Start from a self-owned DACL with inheritance broken, so what we write is
    # what stays. Without this the AI bit is recomputed from the parent every
    # time and no restore can be byte-equal by construction.
    _set_dacl_with_flags(
        target, "D:P(A;OICI;FA;;;OW)(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)", protect=True
    )

    before_sddl = w.read_dacl_sddl(target)  # snapshot BEFORE any grant
    before_aces = _ace_set(target)
    w.grant(target, app_sid, w.FILE_GENERIC_READ)
    granted_sddl = w.read_dacl_sddl(target)
    after_grant = _ace_set(target)

    _set_dacl_with_flags(target, before_sddl, protect=True)
    restored_sddl = w.read_dacl_sddl(target)
    after_restore = _ace_set(target)

    note("reversibility", "sddl_changed_by_grant", before_sddl != granted_sddl)
    note("reversibility", "sddl_string_equal_after_restore", before_sddl == restored_sddl)
    note("reversibility", "ace_set_equal_after_restore", before_aces == after_restore)
    note("reversibility", "grant_added_aces", len(after_grant - before_aces))
    note("reversibility", "residual_aces", sorted(after_restore - before_aces))
    note("reversibility", "lost_aces", sorted(before_aces - after_restore))

    # The merge case: a second GRANT_ACCESS for the same SID must merge into the
    # existing ACE rather than appending a duplicate.
    w.grant(target, app_sid, w.FILE_GENERIC_READ)
    w.grant(target, app_sid, w.FILE_GENERIC_WRITE)
    merged = _ace_set(target)
    note("reversibility", "aces_added_by_two_grants", len(merged - before_aces))
    note("reversibility", "merged_ace", sorted(merged - before_aces))

    # Cleanup: revoke leaves the directory as it started, without relying on a
    # saved SDDL at all. This is the mechanism a crash-recovery path would use.
    w.revoke(target, app_sid)
    note("reversibility", "revoke_restores", _ace_set(target) == before_aces)
    note("reversibility", "revoke_sddl_equal", w.read_dacl_sddl(target) == before_sddl)
    shutil.rmtree(ISO, ignore_errors=True)


def _ace_set(path: str) -> set[str]:
    """Parsed ACE entries as a comparable set, order-insensitive.

    SDDL DACL form is ``D:<flags>(<ace>)(<ace>)...``. The trailing control flags
    (``AI``, ``AR``, ...) sit between ``D:`` and the first ``(`` and are not ACEs,
    so only parenthesised groups are collected.
    """
    sddl = w.read_dacl_sddl(path)
    entries: set[str] = set()
    depth = 0
    current: list[str] = []
    for ch in sddl:
        if ch == "(":
            depth += 1
            if depth == 1:
                current = []
                continue
        elif ch == ")":
            depth -= 1
            if depth == 0:
                entries.add("".join(current))
                continue
        if depth >= 1:
            current.append(ch)
    return entries


# --- scenario 3: denials ----------------------------------------------------
def scenario_denied() -> None:
    print("\n[3] denied — are reads/writes outside the grants refused?")
    app_sid = w.container_sid(CONTAINER_NAME)
    _clean(WORKDIR)
    _grant_interpreter(app_sid)

    code, report, _ = run_child([])
    note("denied", "exit_code", code)
    if not report:
        note("denied", "child_report", "MISSING")
        return
    for path, res in (report.get("reads") or {}).items():
        label = str(path)
        note("denied", f"read[{label}]", "DENIED" if not res.get("ok") else "ALLOWED")
        if not res.get("ok") and res.get("winerror"):
            note("denied", f"read[{label}].winerror", res["winerror"])
    for path, res in (report.get("writes") or {}).items():
        granted = str(path) == str(WORKDIR / "granted_probe.txt")
        verdict = "ALLOWED" if res.get("ok") else "DENIED"
        expect = "expect=ALLOWED" if granted else "expect=DENIED"
        note("denied", f"write[{path}]", f"{verdict} ({expect})")
        if not res.get("ok") and res.get("winerror"):
            note("denied", f"write[{path}].winerror", res["winerror"])
    results["_denied_report"] = report


# --- scenario 4: network ----------------------------------------------------
def scenario_network() -> None:
    print("\n[4] no-network — is outbound blocked without capability SIDs?")
    app_sid = w.container_sid(CONTAINER_NAME)
    _clean(WORKDIR)
    _grant_interpreter(app_sid)

    code, report, _ = run_child([], capabilities=())
    note("network", "no_capabilities_exit_code", code)
    if report:
        for label, res in (report.get("network") or {}).items():
            verdict = "ALLOWED" if res.get("ok") else "BLOCKED"
            note("network", f"nocap_{label}", verdict)
            if not res.get("ok") and res.get("winerror"):
                note("network", f"nocap_{label}_winerror", res["winerror"])
    else:
        note("network", "nocap_report", "MISSING")

    # Control: with the client capability granted, outbound should work.
    print("    (control) granting internetClient ...")
    code2, report2, _ = run_child([], capabilities=("internetClient",))
    note("network", "with_capability_exit_code", code2)
    if report2:
        tcp = (report2.get("network") or {}).get("tcp_public", {})
        note("network", "withcap_tcp_public", "ALLOWED" if tcp.get("ok") else "BLOCKED")
        if not tcp.get("ok"):
            note("network", "withcap_tcp_public_winerror", tcp.get("winerror"))
    else:
        note("network", "withcap_report", "MISSING")

    # ``sendto`` returning success proves nothing for unconnected UDP: the send
    # is asynchronous, so a blocked datagram fails silently. Only an external
    # observer can tell "blocked" from "sent but unanswered", and the two have
    # very different security meanings. Run both with and without the client
    # capability, otherwise a host firewall would be indistinguishable from the
    # container policy.
    print("    (control) UDP egress with internetClient ...")
    _udp_egress(app_sid, capabilities=())
    _udp_egress(app_sid, capabilities=("internetClient",))


def _udp_egress(app_sid, *, capabilities: tuple[str, ...]) -> None:
    """Send one datagram from the container to a host-side socket and observe it."""
    import socket as _socket

    lan_ip = _host_lan_ip()
    if lan_ip is None:
        note("network", "udp_egress", "SKIPPED (no LAN address)")
        return

    listener = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    listener.bind(("0.0.0.0", 0))
    listener.settimeout(8)
    port = listener.getsockname()[1]

    script = WORKDIR / "udp_egress.py"
    script.write_text(
        "import socket, sys\n"
        "host, port = sys.argv[1], int(sys.argv[2])\n"
        "s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "s.bind(('0.0.0.0', 0))\n"
        "print('sendto returned', s.sendto(b'NOOA-UDP-EGRESS', (host, port)), flush=True)\n",
        encoding="utf-8",
    )
    w.grant(str(script), app_sid, w.FILE_GENERIC_READ)

    argv = [str(RUNTIME_ROOT / "python.exe"), "-I", str(script), lan_ip, str(port)]
    job = w.Job()
    child = w.spawn_in_container(
        argv, container_name=CONTAINER_NAME, capabilities=capabilities, cwd=str(WORKDIR)
    )
    job.assign(child.pi.hProcess)
    child.resume()
    arrived = False
    payload = b""
    try:
        payload, _addr = listener.recvfrom(2048)
        arrived = True
    except TimeoutError:
        arrived = False
    finally:
        code = child.wait(timeout_ms=5000)
        child.close()
        job.terminate(0)
        job.close()
        listener.close()

    label = "withcap" if capabilities else "nocap"
    note("network", f"{label}_udp_egress_arrived", arrived)
    note("network", f"{label}_udp_egress_bytes", len(payload) if arrived else 0)
    note("network", f"{label}_udp_egress_child_exit", code)


def _host_lan_ip() -> str | None:
    import socket as _socket

    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


# --- scenario 5: job limits -------------------------------------------------
def scenario_job_limits() -> None:
    print("\n[5] job-limits — do memory/CPU caps and hard kill work?")
    app_sid = w.container_sid(CONTAINER_NAME)
    _clean(WORKDIR)
    _grant_interpreter(app_sid)

    # Baseline: no cap, to learn the container's own memory footprint.
    code, report, elapsed = run_child([], timeout_s=30)
    note("job_limits", "baseline_exit_code", code)
    note("job_limits", "baseline_wall_clock_s", round(elapsed, 3))

    # Memory: a cap low enough that the allocation probe must die.
    code, report, elapsed = run_child([], memory_mb=64, timeout_s=30)
    note("job_limits", "mem_capped_exit_code", code)
    note("job_limits", "mem_capped_wall_clock_s", round(elapsed, 3))
    if report:
        mem = report.get("memory") or {}
        note("job_limits", "mem_probe_kind", mem.get("kind"))
        note("job_limits", "mem_allocated_mb", mem.get("allocated_mb"))
        note("job_limits", "mem_probe_note", mem.get("note"))
    else:
        note(
            "job_limits",
            "mem_capped_report",
            "MISSING -> process died before reporting (consistent with a cap kill)",
        )

    # CPU: burn until the per-job time limit terminates the process.
    code, report, elapsed = run_child([], cpu_seconds=2, timeout_s=60)
    note("job_limits", "cpu_capped_exit_code", code)
    note("job_limits", "cpu_capped_wall_clock_s", round(elapsed, 3))
    if report:
        note("job_limits", "cpu_probe_note", (report.get("cpu") or {}).get("note"))
    else:
        note(
            "job_limits",
            "cpu_capped_report",
            "MISSING -> killed by the job time limit, as intended",
        )

    # Hard kill control: a long sleeper must die at TerminateJobObject.
    print("    (control) hard-kill a long-running child ...")
    argv = [str(RUNTIME_ROOT / "python.exe"), "-I", "-c", "import time; time.sleep(600)"]
    child = w.spawn_in_container(argv, container_name=CONTAINER_NAME, cwd=str(WORKDIR))
    job = w.Job()
    job.assign(child.pi.hProcess)
    child.resume()
    time.sleep(1.5)
    t0 = time.perf_counter()
    job.terminate(1)
    code = child.wait(timeout_ms=10000)
    note("job_limits", "hardkill_reap_s", round(time.perf_counter() - t0, 3))
    note("job_limits", "hardkill_exit_code", code)
    child.close()
    job.close()


# --- scenario 6: broker channel ---------------------------------------------
def scenario_pipe() -> None:
    """Can the container use an inherited anonymous pipe bidirectionally?

    Anonymous pipes carry no name and no DACL check at use time: inherit
    duplicates the handle with its existing access rights. That is precisely the
    property the broker needs, and it is also the only IPC shape that works
    without granting the container any filesystem or network access at all.
    """
    print("\n[6] broker pipe — is an inherited handle usable in the container?")
    app_sid = w.container_sid(CONTAINER_NAME)
    _clean(WORKDIR)
    _grant_interpreter(app_sid, scripts=("pipe_probe.py",))
    w.grant(str(WORKDIR), app_sid, w.FILE_ALL_ACCESS)
    # ``make_pipe`` returns (parent_end, child_end) and clears the inherit flag
    # on the parent end itself. Naming by owner instead of by direction keeps
    # the handle list correct: only child ends may appear in it.
    to_child_parent_w, to_child_child_r = w.make_pipe(read_inheritable=True)
    from_child_parent_r, from_child_child_w = w.make_pipe(read_inheritable=False)

    argv = [
        str(RUNTIME_ROOT / "python.exe"),
        "-I",
        str(HERE / "pipe_probe.py"),
        "--in",
        str(to_child_child_r),
        "--out",
        str(from_child_child_w),
        "--payload",
        "HELLO",
    ]
    job = w.Job()
    t0 = time.perf_counter()
    try:
        child = w.spawn_in_container(
            argv,
            container_name=CONTAINER_NAME,
            inherit_handles=(to_child_child_r, from_child_child_w),
            cwd=str(WORKDIR),
        )
    except OSError as exc:
        note("pipe", "spawn", f"FAILED {exc}")
        return
    job.assign(child.pi.hProcess)
    child.resume()

    parent_r = _fd(from_child_parent_r)
    parent_w = _fd(to_child_parent_w)
    received = b""
    try:
        reply = struct.pack("<I", len(b"WORLD")) + b"WORLD"
        os.write(parent_w, reply)
        header = os.read(parent_r, 4)
        if len(header) == 4:
            (n,) = struct.unpack("<I", header)
            while len(received) < n:
                chunk = os.read(parent_r, n - len(received))
                if not chunk:
                    break
                received += chunk
    except OSError as exc:
        note("pipe", "parent_io_error", str(exc))

    code = child.wait(timeout_ms=30000)
    elapsed = time.perf_counter() - t0
    note("pipe", "child_exit_code", code)
    note("pipe", "wall_clock_s", round(elapsed, 3))
    note("pipe", "child_to_parent_bytes", received.decode(errors="replace"))
    note("pipe", "bidirectional_ok", received == b"HELLO" and code == 0)
    child.close()
    job.terminate(0)
    job.close()
    for fd in (parent_r, parent_w):
        try:
            os.close(fd)
        except OSError:
            pass
    for h in (to_child_child_r, from_child_child_w):
        w._k32.CloseHandle(h)


def _fd(handle: int) -> int:
    import msvcrt

    return msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)


# --- scenario 7: importing the package from inside ---------------------------
def scenario_nooa_import() -> None:
    """Does ``import nooa`` resolve to the runtime-root copy, with the checkout denied?

    The runtime root's site-packages ships ``_editable_impl_nooa.pth`` pointing
    at the checkout. If that hook survives, a successful import would be reading
    the exact path the sandbox exists to deny, so the probe treats importability
    *and* checkout denial as one verdict: either alone is not evidence.

    Grants are deliberately minimal — runtime root, the probe script, the work
    directory. No grant on the checkout, the venv, or ``src``.
    """
    print("\n[7] nooa-import — does nooa resolve inside the root, not the checkout?")
    app_sid = w.container_sid(CONTAINER_NAME)
    _clean(WORKDIR)
    if not (RUNTIME_ROOT / "Lib" / "site-packages" / "nooa" / "__init__.py").exists():
        note("nooa_import", "package_present", "MISSING -> run _mkroot.py --sync-nooa")
        return
    note("nooa_import", "package_present", "yes")
    note(
        "nooa_import",
        "editable_hooks_remaining",
        len(list((RUNTIME_ROOT / "Lib" / "site-packages").glob("_editable_impl_*.pth"))),
    )

    _grant_interpreter(app_sid, scripts=("nooa_probe.py",))
    w.grant(str(WORKDIR), app_sid, w.FILE_ALL_ACCESS)

    out = WORKDIR / "nooa_report.json"
    if out.exists():
        out.unlink()

    argv = [
        str(RUNTIME_ROOT / "python.exe"),
        "-I",
        str(HERE / "nooa_probe.py"),
        str(out),
    ]
    job = w.Job()
    t0 = time.perf_counter()
    child = w.spawn_in_container(argv, container_name=CONTAINER_NAME, cwd=str(WORKDIR))
    job.assign(child.pi.hProcess)
    child.resume()
    code = child.wait(timeout_ms=120_000)
    elapsed = time.perf_counter() - t0
    child.close()
    job.terminate(0)
    job.close()

    note("nooa_import", "exit_code", code)
    note("nooa_import", "wall_clock_s", round(elapsed, 3))
    if not out.exists():
        note("nooa_import", "report", "MISSING (import chain or interpreter failed)")
        return
    try:
        report = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        note("nooa_import", "report", f"unparseable: {exc}")
        return

    for name, res in (report.get("imports") or {}).items():
        if res.get("ok"):
            note(
                "nooa_import", f"import[{name}]", f"OK inside_root={res.get('inside_runtime_root')}"
            )
        else:
            note("nooa_import", f"import[{name}]", f"FAILED {res.get('error')}")
    note("nooa_import", "version", report.get("version"))
    checkout = report.get("checkout") or {}
    note("nooa_import", "checkout_read_denied", checkout.get("denied"))
    if not checkout.get("denied"):
        note("nooa_import", "checkout_leak", checkout.get("error"))
    note("nooa_import", "sys_path", report.get("sys_path"))
    results["_nooa_import_report"] = report


# --- scenario 8: multiprocessing transport across the boundary ---------------
def scenario_mp_pipe() -> None:
    """Does a ``multiprocessing`` Connection survive into the container?

    The broker transport is ``multiprocessing.connection.Connection``. On
    Windows that is a named pipe whose security descriptor denies the container
    SID; the parent hands the child a *duplicated handle* instead of letting it
    open the name. Duplication does not consult the DACL, so this should work —
    but "should" is not a design input. If it fails, M3 needs an anonymous-pipe
    replacement for ``wire.py``/``worker.py``, which is a different project
    size, so the answer has to be measured before any of that code is written.
    """
    print("\n[8] mp-pipe — does multiprocessing's Connection cross the boundary?")
    app_sid = w.container_sid(CONTAINER_NAME)
    _clean(WORKDIR)
    _grant_interpreter(app_sid, scripts=("mp_pipe_probe.py",))

    import multiprocessing as mp

    parent_conn, child_conn = mp.Pipe(duplex=True)
    handle = child_conn.fileno()
    # PROC_THREAD_ATTRIBUTE_HANDLE_LIST accepts inheritable handles only; the
    # duplicated pipe end is not inheritable by default.
    w._k32.SetHandleInformation(wintypes_handle(handle), 0x00000001, 0x00000001)

    argv = [
        str(RUNTIME_ROOT / "python.exe"),
        "-I",
        str(HERE / "mp_pipe_probe.py"),
        str(handle),
        "hello",
    ]
    job = w.Job()
    t0 = time.perf_counter()
    try:
        child = w.spawn_in_container(
            argv,
            container_name=CONTAINER_NAME,
            inherit_handles=(handle,),
            cwd=str(WORKDIR),
        )
    except OSError as exc:
        note("mp_pipe", "spawn", f"FAILED {exc}")
        parent_conn.close()
        child_conn.close()
        job.close()
        return
    job.assign(child.pi.hProcess)
    child.resume()

    reply = None
    try:
        parent_conn.poll(60)
        if parent_conn.poll(0):
            reply = parent_conn.recv()
    except (EOFError, OSError) as exc:
        note("mp_pipe", "recv_error", f"{type(exc).__name__}: {exc}")
    code = child.wait(timeout_ms=90_000)
    elapsed = time.perf_counter() - t0

    note("mp_pipe", "child_exit_code", code)
    note("mp_pipe", "wall_clock_s", round(elapsed, 3))
    note("mp_pipe", "reply", reply)
    note("mp_pipe", "works", isinstance(reply, dict) and reply.get("reply") == "HELLO")

    child.close()
    job.terminate(0)
    job.close()
    parent_conn.close()
    child_conn.close()


def wintypes_handle(handle: int):
    from ctypes import wintypes

    return wintypes.HANDLE(handle)


# --- helpers ----------------------------------------------------------------
def _grant_interpreter(
    app_sid, *, include_script: bool = True, scripts: tuple[str, ...] = ("child_probe.py",)
) -> None:
    """Grant the minimum the container needs to start Python and read its script.

    Only these paths — never the repo directory. A denial for anything else is
    the measurement, so the grant list has to stay as narrow as possible.
    """
    paths = [
        (str(RUNTIME_ROOT), w.FILE_GENERIC_READ | w.FILE_GENERIC_EXECUTE),
        (str(RUNTIME_ROOT / "Lib" / "site-packages"), w.FILE_GENERIC_READ | w.FILE_GENERIC_EXECUTE),
    ]
    if include_script:
        for name in scripts:
            script = HERE / name
            if script.exists():
                paths.append((str(script), w.FILE_GENERIC_READ))
    for path in paths:
        if os.path.exists(path[0]):
            w.grant(path[0], app_sid, path[1])


def _clean(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", type=int, default=None)
    ap.add_argument("--skip-build-check", action="store_true")
    args = ap.parse_args()

    if not (RUNTIME_ROOT / "python.exe").exists():
        print(f"runtime root missing: {RUNTIME_ROOT}\nrun _mkroot.py first", file=sys.stderr)
        return 2

    print(f"runtime root : {RUNTIME_ROOT}")
    print(f"work dir     : {WORKDIR}")
    results["_meta"] = {
        "date": time.strftime("%Y-%m-%d"),
        "python": sys.version.split()[0],
        "windows_build": str(sys.getwindowsversion()),
        "runtime_root_mb": round(_dir_size(RUNTIME_ROOT) / 1e6, 1),
        "admin": _is_admin(),
    }
    print(f"admin        : {_is_admin()}")

    w.create_profile(CONTAINER_NAME)
    sid = w.container_sid(CONTAINER_NAME)
    _adv_sid = sid.value
    results["_meta"]["container_sid_handle"] = _adv_sid

    selected = args.only or [1, 2, 3, 4, 5, 6, 7, 8]
    try:
        if 1 in selected:
            scenario_reachable()
        if 2 in selected:
            scenario_reversibility()
        if 3 in selected:
            scenario_denied()
        if 4 in selected:
            scenario_network()
        if 5 in selected:
            scenario_job_limits()
        if 6 in selected:
            scenario_pipe()
        if 7 in selected:
            scenario_nooa_import()
        if 8 in selected:
            scenario_mp_pipe()
    finally:
        try:
            w.delete_profile(CONTAINER_NAME)
            results["_meta"]["profile_deleted"] = True
        except OSError as exc:
            results["_meta"]["profile_deleted"] = f"failed: {exc}"

    RESULTS.write_text(json.dumps(_redact(results), indent=2, default=str), encoding="utf-8")
    print(f"\nresults -> {RESULTS}")
    return 0


def _redact(value):
    """Replace this machine's home directory and user name before writing.

    ``results.json`` is committed, so it must not carry the path layout of the
    machine that produced it. The denials are the evidence; the exact home
    directory is not, and ``.ssh`` under a named user is not something to
    publish in a pull request.
    """
    home = os.environ.get("USERPROFILE") or str(Path.home())
    user = os.environ.get("USERNAME") or os.environ.get("USER") or ""
    if isinstance(value, dict):
        return {_redact(key): _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        redacted = value
        for variant in {home, home.replace("\\", "\\\\")}:
            redacted = redacted.replace(variant, "%USERPROFILE%")
        if user and redacted == user:
            return "%USERNAME%"
        return redacted
    return value


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _is_admin() -> bool:
    try:
        import ctypes

        return bool(ctypes.WinDLL("shell32", use_last_error=True).IsUserAnAdmin())
    except Exception:
        return False


if __name__ == "__main__":
    sys.exit(main())
