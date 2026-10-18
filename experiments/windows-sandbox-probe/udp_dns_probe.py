# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Does omitting capability SIDs block UDP egress, or only TCP?

M0's UDP check had no working positive control: no datagram arrived even
with ``internetClient``, so "nothing observed" could not be told apart from a
broken test. This probe completes real round trips (a DNS query to a public
resolver) and runs the same child three ways:

* on the host, unconfined — does this network allow the round trip at all?
* in a container with ``internetClient`` — the positive control
* in a container with no capability — the case ``network=False`` relies on

Only the no-capability case can show a leak, and only if the control
succeeded. ``getaddrinfo`` is reported separately: name resolution goes
through the system resolver, which is a different channel from a raw socket.

    uv run python experiments/windows-sandbox-probe/udp_dns_probe.py [--root DIR]
        [--case host|none|cap|all]

``--root`` defaults to $NOOA_SANDBOX_RUNTIME, then this directory's ``.rt``.
``--case`` runs one case alone, so a packet capture can bracket exactly it.
The workspace is a temp directory removed afterwards; nothing is written here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from nooa.runtime.sandbox import win_api as w
from nooa.runtime.sandbox.win_launcher import CONTAINER_ENV

HERE = Path(__file__).resolve().parent
CONTAINER = "nooa-udp-probe"
UDP_CHECKS = ("udp dns 1.1.1.1", "udp dns 8.8.8.8")

# Runs as the child, unconfined and contained alike.
CHILD = r"""
import json, socket, sys, time

def query():
    header = bytes.fromhex("abcd01000001000000000000")
    name = b"".join(bytes([len(p)]) + p.encode() for p in "example.com".split("."))
    return header + name + b"\x00" + bytes.fromhex("00010001")

def attempt(fn):
    start = time.perf_counter()
    try:
        out = fn()
    except Exception as exc:
        code = getattr(exc, "winerror", None) or getattr(exc, "errno", None)
        out = type(exc).__name__ + (f" {code}" if code else "")
    return {"result": out, "s": round(time.perf_counter() - start, 2)}

def tcp():
    with socket.create_connection(("1.1.1.1", 443), timeout=4):
        return "connected"

def udp(server):
    def run():
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(4)
            s.sendto(query(), (server, 53))
            data, _ = s.recvfrom(512)
            return "reply" if data[:2] == b"\xab\xcd" else "foreign reply"
    return run

def resolve():
    return "resolved" if socket.getaddrinfo("example.com", 443) else "empty"

report = {
    "tcp 1.1.1.1:443": attempt(tcp),
    "udp dns 1.1.1.1": attempt(udp("1.1.1.1")),
    "udp dns 8.8.8.8": attempt(udp("8.8.8.8")),
    "getaddrinfo": attempt(resolve),
}
with open(sys.argv[1], "w", encoding="utf-8") as fh:
    json.dump(report, fh)
"""


# name -> (label, capabilities); ``None`` means run on the host, unconfined.
CASES: dict[str, tuple[str, tuple[str, ...] | None]] = {
    "host": ("host, unconfined", None),
    "none": ("container, no capability", ()),
    "cap": ("container + internetClient", ("internetClient",)),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=os.environ.get("NOOA_SANDBOX_RUNTIME"))
    # One case per run lets a packet capture be bracketed around exactly it.
    parser.add_argument("--case", choices=[*CASES, "all"], default="all")
    return parser.parse_args()


def _firewall_state() -> str:
    """State of the Windows Defender Firewall service.

    AppContainer network isolation is enforced by WFP filters that this service
    maintains, so a result only describes a host where it is running.
    """
    out = subprocess.run(["sc.exe", "query", "mpssvc"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "STATE" in line:
            return line.split(":", 1)[1].strip()
    return "unknown"


def _read(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}


def _run_host(python: Path, script: Path, out: Path) -> dict:
    subprocess.run([str(python), "-I", str(script), str(out)], timeout=60, check=False)
    return _read(out)


def _run_container(python: Path, script: Path, out: Path, caps: tuple[str, ...]) -> dict:
    env = {name: os.environ[name] for name in CONTAINER_ENV if name in os.environ}
    job = w.Job()
    try:
        child = w.spawn_in_container(
            [str(python), "-I", str(script), str(out)],
            container_name=CONTAINER,
            capabilities=caps,
            cwd=str(script.parent),
            env=env,
        )
        try:
            try:
                job.assign(child.pi.hProcess)
            except OSError:
                # Still suspended and outside the job: nothing else would end it.
                w._k32.TerminateProcess(child.pi.hProcess, 1)
                raise
            child.resume()
            child.wait(timeout_ms=60_000)
        finally:
            child.close()
    finally:
        job.terminate(0)
        job.close()
    return _read(out)


def _completed(report: dict, check: str) -> bool:
    return report.get(check, {}).get("result") in ("reply", "connected", "resolved")


def _verdict(results: dict[str, dict]) -> str:
    control = results.get("container + internetClient", {})
    blocked = results.get("container, no capability", {})
    if not any(_completed(control, check) for check in UDP_CHECKS):
        return "INCONCLUSIVE: the positive control got no UDP reply, so no reply proves nothing"
    if any(_completed(blocked, check) for check in UDP_CHECKS):
        return "LEAK: a UDP round trip completed without any capability"
    return "BLOCKED: UDP replies arrive with internetClient and not without it"


def main() -> int:
    args = _args()
    root = Path(args.root).resolve() if args.root else HERE / ".rt"
    python = root / "python.exe"
    if not python.exists():
        print(f"no runtime root at {root}")
        return 2
    selected = list(CASES) if args.case == "all" else [args.case]

    work = Path(tempfile.mkdtemp(prefix="nooa-udp-probe-"))
    script = work / "child.py"
    script.write_text(CHILD, encoding="utf-8")
    w.create_profile(CONTAINER)
    sid = w.container_sid(CONTAINER)
    granted: list[str] = []
    results: dict[str, dict] = {}
    try:
        for path, access in ((str(root), w.READ_EXECUTE), (str(work), w.FILE_ALL_ACCESS)):
            w.grant(path, sid, access)
            granted.append(path)
        for name in selected:
            label, caps = CASES[name]
            out = work / f"{name}.json"
            if caps is None:
                results[label] = _run_host(python, script, out)
            else:
                results[label] = _run_container(python, script, out, caps)
    finally:
        # The runtime root is shared with the backend: every ACE added here
        # must go, or later containers would inherit this probe's grant.
        for path in reversed(granted):
            w.revoke(path, sid)
        w._k32.LocalFree(sid)
        w.delete_profile(CONTAINER)
        shutil.rmtree(work, ignore_errors=True)

    print(f"firewall service (mpssvc): {_firewall_state()}")
    for label, report in results.items():
        print(f"--- {label}")
        if not report:
            print("  (no report: the child did not finish)")
        for check, entry in report.items():
            print(f"  {check:<18} {entry['result']}  ({entry['s']} s)")
    if len(results) == len(CASES):
        print(_verdict(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
