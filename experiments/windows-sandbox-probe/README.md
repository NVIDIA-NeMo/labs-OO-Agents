# Windows native sandbox feasibility (M0)

## Research question

Can a native Windows **AppContainer** host the CodeAct worker process, with
filesystem, network, memory and CPU confinement strong enough to back a
`sandbox` execution backend — without WSL, Docker, Hyper-V, or administrator
rights?

This is the gating experiment for adding a Windows backend to
`src/nooa/runtime/sandbox/`, whose current implementation requires Linux
(`fork`, Landlock, seccomp, `RLIMIT_*`).

## Experiment design

`runner.py` drives six scenarios against a real AppContainer process. Every
child runs a **private copy** of the interpreter (`.rt/`, built by
`_mkroot.py`), because the base interpreter directory denies DACL writes to a
non-admin user.

| # | Scenario | Question |
|---|---|---|
| 1 | reachable | Does the container start Python and report itself as a container? |
| 2 | reversible | Can a grant be undone byte-exactly, and does `revoke` alone suffice? |
| 3 | denied | Are reads/writes outside the grants refused, while granted paths work? |
| 4 | no-network | Is TCP blocked without a capability, and allowed with one? Is UDP egress observable? |
| 5 | job-limits | Do memory/CPU caps and hard kill work? |
| 6 | broker pipe | Can an inherited anonymous pipe carry a bidirectional round trip? |

Process creation uses `CreateProcessW` with
`PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES` (the only way into an
AppContainer: it is a creation-time attribute) plus
`PROC_THREAD_ATTRIBUTE_HANDLE_LIST`, created **suspended**, assigned to a Job
Object, then resumed — closing the escape window between creation and job
membership.

All results come from one machine: Windows 11 build 26300, Python 3.13.12,
non-admin user.

## Key metrics

- Container startup latency (create → resume → exit)
- Denial verdicts per path, with `errno`/`WinError`
- TCP/UDP egress under capability combinations
- Memory-cap behaviour and CPU-cap termination timing
- Pipe round-trip success and latency
- Runtime-root copy size and build time

## How to run

```bash
# Build the private interpreter root (≈4.5 min, ~420 MB)
python _mkroot.py

# All six scenarios
python runner.py

# One scenario
python runner.py --only 4
```

Requires Windows 10 1809+ (AppContainer) and Windows 10 1803+ for named pipes.
No admin rights. Results are written to `results.json`.

## Results summary

One full run on 2026-10-17 (`results.json`).

### 1. Reachable — yes

| Metric | Value |
|---|---|
| `is_app_container` | `true` |
| Container SID | `S-1-15-2-606662641-…-4016704499` |
| `sys.prefix` inside | private root (`…/.rt`), not the system install |
| Startup (create→exit) | 0.19–0.56 s |
| Admin required | no |

The container cannot read its own script from the repository until that single
file is granted; granting the directory would expose the whole checkout. This
was observed directly: the first run failed with `Errno 13` on
`child_probe.py`.

### 2. Reversible — yes, with two preconditions

| Metric | Value |
|---|---|
| SDDL byte-equal after restore | **yes** |
| ACE set equal after restore | **yes** |
| ACEs added by grant | 1 |
| ACEs added by a second grant for the same SID | 1 (merged, not appended) |
| `revoke` alone restores | **yes** (byte-equal SDDL) |

Preconditions, both measured:

1. **The target must be user-owned.** DACL writes to the base interpreter
   (`D:\dev\uv\python\…`) and to `C:\Windows\System32` fail with `Errno 5`
   without elevation. This is why the interpreter is copied.
2. **Inheritance must be broken (`PROTECTED_DACL`).** On an inheriting
   directory the `AI` control bit is recomputed from the parent on every write,
   so no restore can be byte-equal by construction. The test therefore runs on a
   directory with `D:P…`.

`GRANT_ACCESS` merges into an existing ACE for the same SID
(`READ|EXECUTE|WRITE` → `0x12019f`), so a rollback journal that just replays
ACE additions will drift. **`REVOKE_ACCESS` is the reliable teardown path**;
store SDDL for audit, not as the undo mechanism.

### 3. Denied — all expected denials hold

| Probe | Result |
|---|---|
| read `C:\Windows\System32\config\SAM` | DENIED (EACCES) |
| read `C:\Windows\System32\config\SYSTEM` | DENIED (EACCES) |
| read `%USERPROFILE%\.ssh` | DENIED (EACCES) |
| read `%USERPROFILE%\.claude\CLAUDE.md` | DENIED (EACCES) |
| read `<repo>\AGENTS.md` | DENIED (EACCES) |
| write granted workspace | **ALLOWED** |
| write `System32`, `Windows\Temp`, `%USERPROFILE%` | DENIED (EACCES) |
| write next to the granted dir | DENIED (EACCES) |

Probe paths must exist: a missing file reports `ENOENT`, which is not evidence
of a denial. An earlier run used a nonexistent `.ssh/id_rsa` and was discarded.

### 4. Network — TCP solid, UDP unresolved

| Capability | TCP `1.1.1.1:443` | UDP to host socket | Loopback listen |
|---|---|---|---|
| none | **BLOCKED** (WSAEACCES 10013) | no datagram observed | ALLOWED |
| `internetClient` | **ALLOWED** | no datagram observed | — |
| `internetClient` + `privateNetworkClientServer` | — | no datagram observed | — |

TCP has a clean two-way control: blocked without the capability, allowed with
it. UDP is **not attributable**: `sendto` returns success in every case, but a
host-side socket observed no datagram even with the capability granted. The
only non-loopback address on this machine is `198.18.0.1`, a TUN adapter, so
container traffic to it may be dropped by the adapter rather than by the
AppContainer policy. Host-to-host UDP on that address does arrive, which rules
out a blanket host firewall but not the TUN path.

**Conclusion:** do not claim UDP isolation from this experiment. `network=False`
should be backed by an explicit WFP filter (or verified on a machine with a
real LAN adapter), not by capability omission alone.

### 5. Job limits — both work

| Metric | Value |
|---|---|
| Memory cap 64 MB → allocation probe | `MemoryError` at 48 MB allocated |
| CPU cap 2 s → process terminated | exit code `3221225540` (`0xC000013A`) |
| CPU-cap wall clock | 2.6–5.9 s (container overhead varies) |
| `TerminateJobObject` reap | 0.003–0.006 s |
| Hard-kill exit code | 1 |

Memory is enforced as a raised `MemoryError` inside Python, not a silent kill;
the worker can report it. CPU and wall-clock termination are hard kills.

### 6. Broker pipe — works

| Metric | Value |
|---|---|
| Child → parent payload | `HELLO` (length-prefixed frame) |
| Parent → child payload | `WORLD` |
| `bidirectional_ok` | `true` |
| Round trip latency | 0.19–0.28 s (includes interpreter start) |

`PROC_THREAD_ATTRIBUTE_HANDLE_LIST` accepts only **inheritable** handles; a
non-inheritable handle in the list makes `CreateProcessW` fail with
`ERROR_INVALID_PARAMETER` (87). `CreatePipe` + `SetHandleInformation` is
sufficient — Windows CPython has **no `AF_UNIX`** (`socket.AF_UNIX` does not
exist), so a socketpair design is not portable; anonymous pipes are the right
primitive.

## Consequences for the backend

1. **A private runtime root is mandatory, not an optimisation.** 22,598 files /
   419 MB / 274 s to build. Reuse it across processes; rebuilding per worker is
   not viable.
2. **The repo cannot be a dependency of the container by accident.** The
   editable installs in `.venv` are `.pth` files pointing at
   `E:\labs-OO-Agents\src`, so importing `nooa` inside the container would
   require granting the checkout. A Windows backend must either install the
   package non-editable into the runtime root or grant read on the source tree
   explicitly (and document that choice).
3. **Teardown uses `revoke`, not SDDL replay.** Plus a journal for crash
   recovery: a killed parent leaves ACEs behind, and they are persistent.
4. **`network=False` needs a non-AppContainer enforcement path** until UDP is
   verified on a normal adapter.
5. **The launcher must be a separate, unconfined process**: it needs to write
   ACLs and create the job, and `SECURITY_CAPABILITIES` cannot be applied to an
   existing process.
6. **`PROC_THREAD_ATTRIBUTE_HANDLE_LIST` + `SECURITY_CAPABILITIES` compose
   fine**, which is what makes the broker design work unchanged.

## Not measured

- Full `import nooa` inside the container (the runtime root carries
  site-packages, but the editable `.pth` targets are outside the grants).
- Long-running stability and startup cost under repeated worker restarts.
- Behaviour on a machine with a real LAN adapter (UDP attribution).
- Low-integrity / LPAC variants, and network enforcement via WFP.
