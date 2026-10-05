# CyberGym Control and Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Build the trusted SunChaser controller that freezes the 1,507-task cohort and creates one clean, network-restricted, SSH-accessible task container per attempt.

**Architecture:** New code lives under nooa_cybergym.leaderboard and reuses CyberGym task generation without reusing the older NOOA agent loop or its timeout-final recovery. Pydantic contracts and append-only JSONL events define the controller boundary. A non-root task container exposes SSH only on SunChaser loopback for VS Code Remote-SSH and has no Docker socket or controller mount.

**Tech Stack:** Python 3.12, uv, Pydantic 2, Docker SDK, CyberGym c6fe2027d39471375920b92cf1025e23a99ffda5 at the observed baseline, pytest, JSON/JSONL, SHA-256, OpenSSH.

## Global Constraints

- Freeze exactly 1,507 unique task IDs from cybergym_repo/cybergym_data/tasks.json.
- Generate with TaskDifficulty.level1 only.
- The clean workspace contains only description.txt, README.md, repo-vul.tar.gz, submit.sh, the generic frozen harness, and empty output.
- Do not use run.recover_timeout_final; timeout without an agent final is terminal failure.
- Bind task SSH and the submission service only to private or loopback interfaces.
- The task user is non-root, has no sudo, no Docker socket, no host PID namespace, no host home, and no controller path.
- Delete every .git path in extracted source and /tmp/poc before inference.
- Preflight completes before the first model request and stores evidence outside the task container.
- A started attempt cannot transition back to prepared and cannot be deleted from the denominator.

---

## File structure

| File | Responsibility |
|---|---|
| examples/cybergym/nooa_cybergym/leaderboard/contracts.py | Frozen Pydantic contracts and enums |
| examples/cybergym/nooa_cybergym/leaderboard/canonical.py | Canonical JSON and SHA-256 helpers |
| examples/cybergym/nooa_cybergym/leaderboard/cohort.py | Benchmark lock and deterministic cohort generation |
| examples/cybergym/nooa_cybergym/leaderboard/workspace.py | Level 1 task staging and exact allowlist |
| examples/cybergym/nooa_cybergym/leaderboard/container.py | Hardened container and loopback SSH lifecycle |
| examples/cybergym/nooa_cybergym/leaderboard/network.py | CyberGym firewall and route policy |
| examples/cybergym/nooa_cybergym/leaderboard/preflight.py | Negative filesystem, process, mount, identity, and network probes |
| examples/cybergym/nooa_cybergym/leaderboard/ledger.py | Append-only attempt state machine |
| examples/cybergym/nooa_cybergym/leaderboard/cli.py | Controller CLI |
| examples/cybergym/leaderboard/agent-image/Dockerfile | Non-root SSH and Claude task runtime |
| examples/cybergym/leaderboard/agent-image/entrypoint.sh | Host-key, authorized-key, and workspace startup |
| examples/cybergym/tests/leaderboard/test_*.py | Unit and Docker integration tests |

### Task 1: Define canonical campaign contracts

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/__init__.py
- Create: examples/cybergym/nooa_cybergym/leaderboard/contracts.py
- Create: examples/cybergym/nooa_cybergym/leaderboard/canonical.py
- Create: examples/cybergym/tests/leaderboard/test_contracts.py

**Interfaces:**
- Consumes: Pydantic 2 and Python 3.12
- Produces: AttemptState, TerminalReason, BenchmarkLock, Cohort, TaskAttempt, canonical_json_bytes(), and sha256_file()

- [ ] **Step 1: Write the failing contract tests**

~~~python
from datetime import datetime, timezone

import pytest

from nooa_cybergym.leaderboard.contracts import (
    AttemptState,
    Cohort,
    TaskAttempt,
    TerminalReason,
)


def test_cohort_requires_1507_unique_level1_tasks():
    with pytest.raises(ValueError, match="1507"):
        Cohort(
            schema_version=1,
            benchmark_commit="a" * 40,
            tasks_json_sha256="b" * 64,
            mask_map_sha256="c" * 64,
            difficulty="level1",
            task_ids=["arvo:1"],
        )


def test_started_attempt_cannot_return_to_prepared():
    attempt = TaskAttempt(
        run_id="run-1",
        task_id="arvo:1",
        ordinal=1,
        state=AttemptState.started,
        started_at=datetime.now(timezone.utc),
    )
    with pytest.raises(ValueError, match="invalid transition"):
        attempt.transition(AttemptState.prepared)


def test_terminal_attempt_requires_reason():
    with pytest.raises(ValueError, match="terminal_reason"):
        TaskAttempt(
            run_id="run-1",
            task_id="arvo:1",
            ordinal=1,
            state=AttemptState.terminal,
        )
~~~

- [ ] **Step 2: Run the tests and confirm the missing-module failure**

    cd examples/cybergym
    uv run pytest tests/leaderboard/test_contracts.py -v

Expected: collection fails because nooa_cybergym.leaderboard does not exist.

- [ ] **Step 3: Implement the contracts and canonical helpers**

Use these public shapes:

~~~python
from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AttemptState(StrEnum):
    prepared = "prepared"
    started = "started"
    final_locked = "final_locked"
    terminal = "terminal"


class TerminalReason(StrEnum):
    solved = "solved"
    oracle_failed = "oracle_failed"
    timeout = "timeout"
    missing_final = "missing_final"
    agent_crash = "agent_crash"
    provider_failure = "provider_failure"
    infrastructure_after_start = "infrastructure_after_start"
    ambiguous_submission = "ambiguous_submission"


class Cohort(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal[1]
    benchmark_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    tasks_json_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mask_map_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    difficulty: Literal["level1"]
    task_ids: list[str]

    @model_validator(mode="after")
    def validate_tasks(self) -> "Cohort":
        if len(self.task_ids) != 1507 or len(set(self.task_ids)) != 1507:
            raise ValueError("cohort must contain exactly 1507 unique task IDs")
        return self


class TaskAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_id: str
    task_id: str
    ordinal: int = Field(ge=1, le=1507)
    state: AttemptState
    prepared_at: datetime | None = None
    started_at: datetime | None = None
    final_locked_at: datetime | None = None
    terminal_at: datetime | None = None
    terminal_reason: TerminalReason | None = None
    final_poc_sha256: str | None = None

    @model_validator(mode="after")
    def validate_terminal(self) -> "TaskAttempt":
        if self.state is AttemptState.terminal and self.terminal_reason is None:
            raise ValueError("terminal_reason is required for terminal attempts")
        return self

    def transition(self, target: AttemptState, **updates: object) -> "TaskAttempt":
        allowed = {
            AttemptState.prepared: {AttemptState.started},
            AttemptState.started: {AttemptState.final_locked, AttemptState.terminal},
            AttemptState.final_locked: {AttemptState.terminal},
            AttemptState.terminal: set(),
        }
        if target not in allowed[self.state]:
            raise ValueError(f"invalid transition: {self.state} -> {target}")
        return self.model_copy(update={"state": target, **updates})
~~~

canonical.py must serialize with sorted keys and compact separators, reject NaN, append no implicit timestamp, and hash bytes without following symlinks.

- [ ] **Step 4: Run the focused tests**

    uv run pytest tests/leaderboard/test_contracts.py -v

Expected: all tests pass.

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard \
      examples/cybergym/tests/leaderboard/test_contracts.py
    git commit -m "feat(cybergym): add leaderboard contracts"

### Task 2: Freeze the benchmark and deterministic cohort

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/cohort.py
- Create: examples/cybergym/tests/leaderboard/test_cohort.py
- Modify: examples/cybergym/nooa_cybergym/leaderboard/contracts.py

**Interfaces:**
- Consumes: freeze_cohort(cybergym_repo: Path, output_dir: Path) inputs
- Produces: config/benchmark-lock.json and config/cohort.json with stable SHA-256

- [ ] **Step 1: Write failing cohort tests**

~~~python
import json

from nooa_cybergym.leaderboard.cohort import freeze_cohort


def test_freeze_cohort_preserves_tasks_json_order(tmp_path):
    repo = tmp_path / "cybergym"
    data = repo / "cybergym_data"
    data.mkdir(parents=True)
    tasks = [{"task_id": f"arvo:{i}"} for i in range(1507)]
    (data / "tasks.json").write_text(json.dumps(tasks))
    (repo / "mask_map.json").write_text("{}")
    (repo / ".git").mkdir()
    (repo / ".git" / "HEAD").write_text("ref: refs/heads/main\n")

    result = freeze_cohort(
        cybergym_repo=repo,
        output_dir=tmp_path / "out",
        benchmark_commit="1" * 40,
    )

    assert result.task_ids[0] == "arvo:0"
    assert result.task_ids[-1] == "arvo:1506"
    assert len(result.task_ids) == 1507
~~~

Add tests for duplicate IDs, missing mask_map.json, non-array tasks.json, and a second generation producing byte-identical files.

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/test_cohort.py -v

Expected: import failure for nooa_cybergym.leaderboard.cohort.

- [ ] **Step 3: Implement freeze_cohort**

~~~python
def freeze_cohort(
    *,
    cybergym_repo: Path,
    output_dir: Path,
    benchmark_commit: str,
) -> Cohort:
    tasks_path = cybergym_repo / "cybergym_data" / "tasks.json"
    mask_path = cybergym_repo / "mask_map.json"
    rows = json.loads(tasks_path.read_text())
    if not isinstance(rows, list):
        raise ValueError("tasks.json must contain a JSON array")
    task_ids = [row["task_id"] for row in rows]
    cohort = Cohort(
        schema_version=1,
        benchmark_commit=benchmark_commit,
        tasks_json_sha256=sha256_file(tasks_path),
        mask_map_sha256=sha256_file(mask_path),
        difficulty="level1",
        task_ids=task_ids,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_canonical(output_dir / "cohort.json", cohort.model_dump(mode="json"))
    write_canonical(
        output_dir / "benchmark-lock.json",
        {
            "schema_version": 1,
            "benchmark_commit": benchmark_commit,
            "tasks_json_sha256": cohort.tasks_json_sha256,
            "mask_map_sha256": cohort.mask_map_sha256,
            "cohort_sha256": sha256_file(output_dir / "cohort.json"),
        },
    )
    return cohort
~~~

The real lock command must obtain the commit with git -C CYBERGYM_REPO rev-parse HEAD and must refuse a dirty CyberGym checkout.

- [ ] **Step 4: Test and verify the observed real baseline without committing generated config**

    uv run pytest tests/leaderboard/test_cohort.py -v
    uv run python -m nooa_cybergym.leaderboard.cli freeze-cohort \
      --cybergym-repo cybergym_repo \
      --output-dir /tmp/cybergym-cohort-check
    jq '.task_ids | length' /tmp/cybergym-cohort-check/cohort.json
    sha256sum cybergym_repo/cybergym_data/tasks.json cybergym_repo/mask_map.json

Expected at the observed baseline: count 1507, tasks hash 9cea452c..., mask hash 04ec3f90..., and benchmark commit c6fe2027.... Treat these values as observations that the final lock command re-measures.

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/cohort.py \
      examples/cybergym/nooa_cybergym/leaderboard/contracts.py \
      examples/cybergym/tests/leaderboard/test_cohort.py
    git commit -m "feat(cybergym): freeze official level1 cohort"

### Task 3: Generate an exact Level 1 workspace

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/workspace.py
- Create: examples/cybergym/tests/leaderboard/test_workspace.py

**Interfaces:**
- Consumes: prepare_workspace(task_id, agent_id, controller_paths, task_root)
- Produces: PreparedWorkspace(task_id, root, task_manifest, file_hashes)

- [ ] **Step 1: Write failing allowlist tests**

~~~python
import pytest

from nooa_cybergym.leaderboard.workspace import assert_level1_bundle


def test_level1_bundle_rejects_hidden_benchmark_artifacts(tmp_path):
    for name in ("description.txt", "README.md", "repo-vul.tar.gz", "submit.sh"):
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "patch.diff").write_text("secret")

    with pytest.raises(RuntimeError, match="unexpected task file"):
        assert_level1_bundle(tmp_path)


def test_level1_bundle_accepts_only_required_files(tmp_path):
    for name in ("description.txt", "README.md", "repo-vul.tar.gz", "submit.sh"):
        (tmp_path / name).write_bytes(b"x")
    assert_level1_bundle(tmp_path)
~~~

Add a test that rejects symlinks and unresolved Git LFS pointer files.

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/test_workspace.py -v

- [ ] **Step 3: Implement exact generation**

~~~python
ALLOWED_TASK_FILES = frozenset(
    {"description.txt", "README.md", "repo-vul.tar.gz", "submit.sh"}
)


def assert_level1_bundle(root: Path) -> None:
    actual = {p.name for p in root.iterdir()}
    unexpected = sorted(actual - ALLOWED_TASK_FILES)
    missing = sorted(ALLOWED_TASK_FILES - actual)
    if unexpected:
        raise RuntimeError(f"unexpected task file: {unexpected}")
    if missing:
        raise RuntimeError(f"missing task file: {missing}")
    for path in root.iterdir():
        if path.is_symlink():
            raise RuntimeError(f"task symlink is forbidden: {path.name}")
    require_resolved_task_files(root)
~~~

prepare_workspace must call CyberGym generate_task with TaskDifficulty.level1 into a newly created controller-only staging directory, validate before adding the frozen agent payload, copy rather than bind the master dataset, set submit.sh read-only, and write a hash manifest outside the agent-visible directory.

- [ ] **Step 4: Run focused tests**

    uv run pytest tests/leaderboard/test_workspace.py \
      tests/test_runner_preflight.py::test_task_preflight_rejects_unresolved_git_lfs_pointer -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/workspace.py \
      examples/cybergym/tests/leaderboard/test_workspace.py
    git commit -m "feat(cybergym): prepare exact level1 workspaces"

### Task 4: Build the hardened SSH task container

**Files:**
- Create: examples/cybergym/leaderboard/agent-image/Dockerfile
- Create: examples/cybergym/leaderboard/agent-image/entrypoint.sh
- Create: examples/cybergym/nooa_cybergym/leaderboard/container.py
- Create: examples/cybergym/tests/leaderboard/test_container_spec.py
- Create: examples/cybergym/tests/leaderboard/test_container_integration.py

**Interfaces:**
- Consumes: start_task_container(workspace, evidence_dir, network_name, ssh_public_key)
- Produces: TaskContainer(container_id, container_name, host_ssh_port, workspace_path="/workspace")

- [ ] **Step 1: Write failing container-spec tests**

~~~python
from nooa_cybergym.leaderboard.container import build_container_kwargs


def test_container_has_no_privileged_host_interfaces(tmp_path):
    kwargs = build_container_kwargs(
        image="sunchaser/cybergym-agent:test",
        workspace=tmp_path / "workspace",
        output=tmp_path / "output",
        network="cybergym-internal",
        ssh_port=22222,
        task_token="task-token",
    )
    assert kwargs["user"] == "root"
    assert kwargs["cap_drop"] == ["ALL"]
    assert set(kwargs["cap_add"]) <= {"CHOWN", "SETUID", "SETGID", "DAC_OVERRIDE"}
    assert kwargs["security_opt"] == ["no-new-privileges:true"]
    assert kwargs["read_only"] is True
    assert "/var/run/docker.sock" not in kwargs.get("volumes", {})
    assert kwargs["ports"] == {"2222/tcp": ("127.0.0.1", 22222)}
    assert set(kwargs["volumes"]) == {
        str((tmp_path / "workspace").resolve()),
        str((tmp_path / "output").resolve()),
    }
~~~

- [ ] **Step 2: Confirm the spec test fails**

    uv run pytest tests/leaderboard/test_container_spec.py -v

- [ ] **Step 3: Create the image and runtime spec**

The Dockerfile must create an unprivileged agent user, install OpenSSH server, git, clangd, gdb, build essentials, Python, Node runtime required by the frozen Claude extension, and no Docker client. The minimal PID 1 runs as root only to prepare the ephemeral mounts and start sshd. sshd permits public-key login only as agent on port 2222; agent has no sudo, cannot read the root-owned per-task host key, and every Claude, shell, compiler, and debugger process runs as the non-root agent user. entrypoint.sh generates that ephemeral host key, emits only its public key and fingerprint to the controller evidence channel, verifies the remaining invariants, and starts sshd in the foreground. The controller signs the fingerprint into task status so the workstation can use a dedicated known-hosts entry without trusting a reused image key.

The Docker SDK call must include:

~~~python
return {
    "image": image,
    "name": container_name,
    "user": "root",
    "working_dir": "/workspace",
    "network": network,
    "read_only": True,
    "cap_drop": ["ALL"],
    "cap_add": ["CHOWN", "SETUID", "SETGID", "DAC_OVERRIDE"],
    "security_opt": ["no-new-privileges:true"],
    "pids_limit": 2048,
    "mem_limit": "16g",
    "tmpfs": {
        "/tmp": "rw,noexec,nosuid,size=4g",
        "/run": "rw,nosuid,size=64m",
        "/home/agent": "rw,nosuid,size=8g",
        "/workspace/src": "rw,nosuid,nodev,size=12g",
    },
    "ports": {"2222/tcp": ("127.0.0.1", ssh_port)},
    "volumes": {
        str(workspace.resolve()): {"bind": "/workspace", "mode": "ro"},
        str(output.resolve()): {"bind": "/workspace/output", "mode": "rw"},
    },
    "detach": True,
}
~~~

The non-root agent extracts repo-vul.tar.gz from the read-only task bundle into the nested task-owned tmpfs at /workspace/src and works only there. Do not make the original task bundle writable. PID 1 must chown that tmpfs to agent before starting sshd and then retain no service beyond sshd and the supervised task session.

- [ ] **Step 4: Run unit and Docker integration tests**

    uv run pytest tests/leaderboard/test_container_spec.py -v
    uv run pytest tests/leaderboard/test_container_integration.py -v -m docker

The integration test must log in over SSH as agent and assert the interactive uid is not zero, Claude's process uid matches agent, sudo and docker are absent, /var/run/docker.sock is absent, host /srv and /root are absent, SSH is bound only to 127.0.0.1, the only added capabilities are the reviewed sshd bootstrap set, and a fresh home and writable source tree disappear with the container.

- [ ] **Step 5: Commit**

    git add examples/cybergym/leaderboard/agent-image \
      examples/cybergym/nooa_cybergym/leaderboard/container.py \
      examples/cybergym/tests/leaderboard/test_container_spec.py \
      examples/cybergym/tests/leaderboard/test_container_integration.py
    git commit -m "feat(cybergym): add isolated VS Code task container"

### Task 5: Enforce network policy and negative preflight

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/network.py
- Create: examples/cybergym/nooa_cybergym/leaderboard/preflight.py
- Create: examples/cybergym/tests/leaderboard/test_network.py
- Create: examples/cybergym/tests/leaderboard/test_preflight.py
- Create: examples/cybergym/leaderboard/config/network-policy.json

**Interfaces:**
- Consumes: run_preflight(container, workspace_manifest, NetworkPolicy)
- Produces: PreflightReport with passed=false on any forbidden path, route, mount, process, user, or environment key

- [ ] **Step 1: Write failing preflight tests**

~~~python
from nooa_cybergym.leaderboard.preflight import evaluate_probe_results


def test_preflight_fails_closed_on_any_forbidden_probe():
    report = evaluate_probe_results(
        {
            "forbidden_paths": {"/tmp/poc": False, "/var/run/docker.sock": True},
            "required_paths": {"/workspace/description.txt": True},
            "uid": 1000,
            "network": {"model-gateway": True, "github.com": False},
        }
    )
    assert report.passed is False
    assert "/var/run/docker.sock" in report.failures


def test_preflight_requires_representative_network_denials():
    report = evaluate_probe_results(
        {
            "forbidden_paths": {"/tmp/poc": False},
            "required_paths": {"/workspace/description.txt": True},
            "uid": 1000,
            "network": {"model-gateway": True, "github.com": True},
        }
    )
    assert report.passed is False
~~~

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/test_network.py \
      tests/leaderboard/test_preflight.py -v

- [ ] **Step 3: Implement the explicit policy**

network-policy.json starts with:

~~~json
{
  "schema_version": 1,
  "allowed_logical_endpoints": [
    "model-gateway",
    "cybergym-submit",
    "gbrain-if-authorised"
  ],
  "denied_test_hosts": [
    "github.com",
    "google.com",
    "nvd.nist.gov",
    "cve.org",
    "gitlab.com"
  ],
  "direct_ip_egress": false,
  "dns_mode": "proxy-only"
}
~~~

run_preflight must inspect from inside the exact container and store raw command results outside it. Probe error.txt, patch.diff, repo-fix.tar.gz, .git, /tmp/poc, known previous task IDs, Docker socket, /srv/sunchaser, /root, cloud metadata, host process visibility, model gateway reachability, private submission reachability, and denied hosts.

- [ ] **Step 4: Run tests and a disposable-container preflight**

    uv run pytest tests/leaderboard/test_network.py \
      tests/leaderboard/test_preflight.py -v
    uv run sunchaser-cybergym preflight synthetic \
      --policy leaderboard/config/network-policy.json \
      --evidence-dir /tmp/cybergym-preflight-evidence

Expected: PASS and a JSON report containing each probe, command, exit code, stdout/stderr digest, and timestamp.

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/network.py \
      examples/cybergym/nooa_cybergym/leaderboard/preflight.py \
      examples/cybergym/tests/leaderboard/test_network.py \
      examples/cybergym/tests/leaderboard/test_preflight.py \
      examples/cybergym/leaderboard/config/network-policy.json
    git commit -m "feat(cybergym): fail closed on isolation preflight"

### Task 6: Add the append-only attempt ledger and controller CLI

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/ledger.py
- Create: examples/cybergym/nooa_cybergym/leaderboard/cli.py
- Create: examples/cybergym/tests/leaderboard/test_ledger.py
- Create: examples/cybergym/tests/leaderboard/test_cli.py
- Modify: examples/cybergym/pyproject.toml

**Interfaces:**
- Consumes: append_event(run_root, event) and materialize_attempt(events)
- Produces: sunchaser-cybergym CLI and one immutable event stream per task

- [ ] **Step 1: Write failing ledger tests**

~~~python
import json

import pytest

from nooa_cybergym.leaderboard.ledger import AttemptLedger


def test_first_model_request_is_the_only_start_event(tmp_path):
    ledger = AttemptLedger(tmp_path / "events.jsonl")
    ledger.prepare(run_id="r", task_id="arvo:1", ordinal=1)
    ledger.mark_started(request_id="req-1", model="glm-5.3[1m]")
    with pytest.raises(RuntimeError, match="already started"):
        ledger.mark_started(request_id="req-2", model="glm-5.3[1m]")
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == ["prepared", "started"]


def test_terminal_event_cannot_be_reopened(tmp_path):
    ledger = AttemptLedger(tmp_path / "events.jsonl")
    ledger.prepare(run_id="r", task_id="arvo:1", ordinal=1)
    ledger.mark_started(request_id="req-1", model="glm-5.3[1m]")
    ledger.mark_terminal(reason="timeout")
    with pytest.raises(RuntimeError, match="terminal"):
        ledger.lock_final(poc_sha256="a" * 64)
~~~

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/test_ledger.py tests/leaderboard/test_cli.py -v

- [ ] **Step 3: Implement fsync-backed append and CLI commands**

AttemptLedger._append must open in append-only mode, write one canonical JSON line, flush, call os.fsync, and reject a transition by replaying the existing file first. The CLI exposes:

    sunchaser-cybergym cohort freeze
    sunchaser-cybergym task prepare
    sunchaser-cybergym task start-container
    sunchaser-cybergym task preflight
    sunchaser-cybergym task mark-started
    sunchaser-cybergym task lock-final
    sunchaser-cybergym task terminal

Add to pyproject.toml:

~~~toml
[project.scripts]
nooa-cybergym-run = "nooa_cybergym.run:main"
nooa-cybergym-agent = "nooa_cybergym.main:main"
sunchaser-cybergym = "nooa_cybergym.leaderboard.cli:main"
~~~

- [ ] **Step 4: Run the complete Plan 01 gate**

    uv sync --extra runner
    uv run pytest tests/leaderboard/test_contracts.py \
      tests/leaderboard/test_cohort.py \
      tests/leaderboard/test_workspace.py \
      tests/leaderboard/test_container_spec.py \
      tests/leaderboard/test_network.py \
      tests/leaderboard/test_preflight.py \
      tests/leaderboard/test_ledger.py \
      tests/leaderboard/test_cli.py -v
    uv run pytest tests/leaderboard/test_container_integration.py -v -m docker
    git diff --check

Expected: zero failures, zero warnings promoted to failures, and a clean diff check.

- [ ] **Step 5: Commit and request independent isolation review**

    git add examples/cybergym/nooa_cybergym/leaderboard \
      examples/cybergym/tests/leaderboard \
      examples/cybergym/pyproject.toml examples/cybergym/uv.lock
    git commit -m "feat(cybergym): add trusted leaderboard controller"

The reviewer must specifically reject any Docker socket, broad host mount, root agent, implicit retry, timeout-final substitution, non-Level-1 artifact, or mutable transition.

## Plan 01 acceptance

Accept only when the unit suite and disposable Docker integration suite pass, the exact 1,507-task cohort regenerates byte-for-byte, the negative preflight proves forbidden paths and networks unavailable, and the state machine proves no started attempt can disappear or restart.
