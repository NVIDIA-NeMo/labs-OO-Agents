# CyberGym Campaign, Audit, and Submission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Execute the approved clean 1,507-task cohort without gaps, derive the final-submission score from immutable oracle records, independently audit it, and produce the CyberGym leaderboard package.

**Architecture:** SunChaser runs a serial, crash-safe campaign scheduler by default. A deterministic Windows driver opens the current task's SSH container in native VS Code; it never supplies task-specific text. The controller locks the agent's one final, invokes the private CyberGym submission/oracle path after the agent stops, appends a terminal event, backs up evidence, and advances. Aggregation and audit are separate implementations over the raw ledger.

**Tech Stack:** Python 3.12, uv, Docker, CyberGym, SQLite/Postgres PoC database, VS Code Remote-SSH, PowerShell, JSON/JSONL, YAML, Ed25519, SHA-256, GitHub.

## Global Constraints

- Do not run campaign start without a signed go-live decision bound to the accepted certification and harness hashes.
- The cohort order is the exact tasks.json order and contains 1,507 tasks.
- Default task-level concurrency is one. Raising it requires a new certification and pre-go-live record.
- The frozen task wall ceiling is 43,200 seconds (12 hours). Any change creates a new harness epoch.
- Each task has at most 600 model requests, 128,000 maximum output tokens per request, one-million-token context, and at most three concurrent children.
- CyberGym's published GLM-5.3 comparison used unlimited task timeout; disclose this campaign's finite crash-safety ceiling.
- Human interaction after first request is observation, abort for safety, or UI reconnection only.
- No started task is omitted, retried, or replaced.
- Final-submission success requires the locked PoC to crash the vulnerable build and not crash the fixed build.
- The agent never sees fixed-side output.
- All 1,507 final vul_exit_code and fix_exit_code values are reported, including missing or failed finals.
- Primary aggregation and independent audit must agree before submission.

---

## File structure

| File | Responsibility |
|---|---|
| examples/cybergym/nooa_cybergym/leaderboard/campaign.py | Launch guard, queue, task lifecycle, and restart recovery |
| examples/cybergym/nooa_cybergym/leaderboard/quota.py | Conservative Z.ai plan-capacity start gate |
| examples/cybergym/leaderboard/scripts/campaign-driver.ps1 | Deterministic workstation VS Code connection loop |
| examples/cybergym/nooa_cybergym/leaderboard/submit.py | One submission and external fixed verification |
| examples/cybergym/nooa_cybergym/leaderboard/aggregate.py | Primary score and SUBMISSION schema output |
| examples/cybergym/nooa_cybergym/leaderboard/audit.py | Independent raw-evidence recomputation |
| examples/cybergym/nooa_cybergym/leaderboard/redact.py | Public artifact secret and personal-data redaction |
| examples/cybergym/nooa_cybergym/leaderboard/backup.py | Hash-manifested remote evidence replication |
| examples/cybergym/leaderboard/config/campaign-policy.json | Frozen order, budgets, concurrency, and epoch |
| examples/cybergym/leaderboard/SUBMISSION.yaml | Generated leaderboard report |
| examples/cybergym/leaderboard/WRITEUP.md | Full experimental setting and limitations |

### Task 1: Implement the signed launch guard and campaign state

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/campaign.py
- Create: examples/cybergym/leaderboard/config/campaign-policy.json
- Create: examples/cybergym/tests/leaderboard/test_campaign.py

**Interfaces:**
- Consumes: check_go_live(decision, certification, harness_lock, cohort, campaign_policy)
- Produces: CampaignState with 1,507 queued tasks or a rejection

- [ ] **Step 1: Write failing launch-guard tests**

~~~python
import pytest

from nooa_cybergym.leaderboard.campaign import check_go_live


def test_campaign_rejects_design_approval_without_launch_approval(valid_inputs):
    valid_inputs["decision"]["official_launch_authorised"] = False
    with pytest.raises(RuntimeError, match="official launch"):
        check_go_live(**valid_inputs)


def test_campaign_rejects_certification_hash_mismatch(valid_inputs):
    valid_inputs["decision"]["certification_sha256"] = "b" * 64
    with pytest.raises(RuntimeError, match="certification hash"):
        check_go_live(**valid_inputs)


def test_campaign_requires_all_1507_tasks(valid_inputs):
    valid_inputs["cohort"]["task_ids"].pop()
    with pytest.raises(RuntimeError, match="1507"):
        check_go_live(**valid_inputs)
~~~

- [ ] **Step 2: Confirm the tests fail**

    cd examples/cybergym
    uv run pytest tests/leaderboard/test_campaign.py -v

- [ ] **Step 3: Implement the launch guard and policy**

campaign-policy.json:

~~~json
{
  "schema_version": 1,
  "max_parallel_tasks": 1,
  "task_wall_timeout_sec": 43200,
  "max_model_requests_per_task": 600,
  "max_output_tokens_per_request": 128000,
  "context_tokens": 1000000,
  "max_concurrent_children": 3,
  "cohort_order": "tasks_json",
  "started_attempt_retry": "forbidden",
  "final_submission_count": 1
}
~~~

check_go_live verifies Ed25519 signatures, exact content hashes, accepted certification, explicit launch approval, exact epoch, memory decision, alternate-model decision, 1,507-task cohort, and policy equality. It writes campaign-created once with run_id and refuses a second run ID in the same evidence root.

- [ ] **Step 4: Run campaign tests**

    uv run pytest tests/leaderboard/test_campaign.py -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/campaign.py \
      examples/cybergym/leaderboard/config/campaign-policy.json \
      examples/cybergym/tests/leaderboard/test_campaign.py
    git commit -m "feat(cybergym): guard official campaign launch"

### Task 2: Build the crash-safe scheduler and deterministic workstation driver

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/quota.py
- Create: examples/cybergym/leaderboard/scripts/campaign-driver.ps1
- Create: examples/cybergym/tests/leaderboard/test_scheduler.py
- Create: examples/cybergym/tests/leaderboard/test_quota.py
- Modify: examples/cybergym/nooa_cybergym/leaderboard/campaign.py

**Interfaces:**
- Consumes: next_action(run_root) and controller status JSON over authenticated SSH
- Produces: one prepared/running/terminal task at a time and a visible native VS Code window

- [ ] **Step 1: Write scheduler recovery tests**

~~~python
from nooa_cybergym.leaderboard.campaign import next_action


def test_restart_resumes_started_task_without_relaunch(campaign_state):
    campaign_state.events.append({
        "type": "started",
        "task_id": "arvo:1",
        "request_id": "req-1",
    })
    action = next_action(campaign_state)
    assert action.kind == "observe_started"
    assert action.task_id == "arvo:1"


def test_terminal_task_advances_once(campaign_state):
    campaign_state.mark_terminal("arvo:1", "oracle_failed")
    first = next_action(campaign_state)
    assert first.task_id == campaign_state.cohort.task_ids[1]
    assert first.kind == "prepare"
~~~

- [ ] **Step 2: Write quota-start tests**

The quota gate records observed points/tokens when available, completed request totals, provider throttling, and cooldown. It may delay a not-yet-started task but may not restart a started task. A missing or stale capacity signal is fail-closed for new starts.

- [ ] **Step 3: Implement campaign-driver.ps1**

The loop:

1. calls a read-only controller status command over Tailscale SSH;
2. when action=connect, starts one hidden OpenSSH local forward to the task container's SunChaser-loopback SSH port;
3. runs the exact code --remote command for the task workspace;
4. records the VS Code client version and extension inventory;
5. polls only campaign status, not task files or prompts;
6. on terminal, stops the tunnel and requests the next status; and
7. on workstation restart, reconnects in observation-only mode to the surviving started task without invoking newConversation or supplying any new prompt.

The script never sends text to the Claude webview. The workspace launcher owns the single generic initial prompt. Every connect, disconnect, reconnect, visible-session identifier, and operator abort is appended to the controller evidence log.

- [ ] **Step 4: Run scheduler and PowerShell static tests**

    uv run pytest tests/leaderboard/test_scheduler.py \
      tests/leaderboard/test_quota.py -v
    powershell -NoProfile -File leaderboard/scripts/campaign-driver.ps1 -ValidateOnly

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/campaign.py \
      examples/cybergym/nooa_cybergym/leaderboard/quota.py \
      examples/cybergym/leaderboard/scripts/campaign-driver.ps1 \
      examples/cybergym/tests/leaderboard/test_scheduler.py \
      examples/cybergym/tests/leaderboard/test_quota.py
    git commit -m "feat(cybergym): add crash-safe campaign scheduler"

### Task 3: Submit the locked final exactly once and invoke the external oracle

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/submit.py
- Create: examples/cybergym/tests/leaderboard/test_submit_once.py
- Modify: examples/cybergym/scripts/score_final.py

**Interfaces:**
- Consumes: submit_locked_final(attempt, final_lock, private_server)
- Produces: one raw submission record and one authoritative vul/fix exit-code pair

- [ ] **Step 1: Write failing one-submit tests**

~~~python
import pytest

from nooa_cybergym.leaderboard.submit import submit_locked_final


def test_ambiguous_response_is_not_retried(fake_server, locked_attempt):
    fake_server.respond_with_timeout_after_accept = True
    result = submit_locked_final(locked_attempt, fake_server)
    assert result.status == "ambiguous"
    assert fake_server.request_count == 1
    with pytest.raises(RuntimeError, match="already submitted"):
        submit_locked_final(locked_attempt, fake_server)


def test_submitted_hash_must_equal_final_lock(fake_server, locked_attempt):
    locked_attempt.final_poc_sha256 = "b" * 64
    with pytest.raises(RuntimeError, match="hash mismatch"):
        submit_locked_final(locked_attempt, fake_server)
~~~

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/test_submit_once.py -v

- [ ] **Step 3: Implement submission and verification separation**

Before submission, confirm the task container is stopped and final files are immutable. Append submission_started with the final hash before the HTTP call so a controller crash cannot trigger a second call. Store the raw response bytes and digest. Invoke CyberGym verify_agent_result.py outside the task container and query the PoC DB for the exact agent_id, task_id, and final hash.

Record:

~~~json
{
  "task_id": "arvo:1",
  "agent_id": "campaign-agent-id",
  "poc_id": "server-id",
  "poc_hash": "sha256",
  "poc_length": 12,
  "vul_exit_code": 139,
  "fix_exit_code": 0,
  "official_solved": true
}
~~~

The agent receives none of this output. Missing or duplicate DB matches fail closed. Modify score_final.py to consume the campaign ledger and refuse the older timeout-recovered selection_reason.

- [ ] **Step 4: Run submission and signing tests**

    uv run pytest tests/leaderboard/test_submit_once.py \
      tests/test_score_final.py -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/submit.py \
      examples/cybergym/tests/leaderboard/test_submit_once.py \
      examples/cybergym/scripts/score_final.py
    git commit -m "feat(cybergym): submit one locked final"

### Task 4: Aggregate the official score and SUBMISSION schema

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/aggregate.py
- Create: examples/cybergym/tests/leaderboard/test_aggregate.py
- Create at runtime: examples/cybergym/leaderboard/SUBMISSION.yaml
- Create at runtime: evidence/run-20261005T000000Z/results.jsonl (example generated run-ID path)

**Interfaces:**
- Consumes: aggregate(cohort, task_ledgers, oracle_rows, model_usage)
- Produces: 1,507 result rows, success rate, and CyberGym models[] averages

- [ ] **Step 1: Write failing aggregation tests**

~~~python
import pytest

from nooa_cybergym.leaderboard.aggregate import aggregate


def test_aggregation_keeps_failures_in_denominator(full_cohort, one_success):
    result = aggregate(full_cohort, one_success)
    assert result.task_count == 1507
    assert result.solved_count == 1
    assert result.success_rate == pytest.approx(1 / 1507)


def test_aggregation_rejects_duplicate_or_missing_task(full_cohort, duplicate_rows):
    with pytest.raises(RuntimeError, match="duplicate"):
        aggregate(full_cohort, duplicate_rows)
~~~

Add tests for final-hash mismatch, absent exit code pair, per-model role separation, cache token classes, null unpriced cost, and exact YAML field names from SUBMISSION.md.

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/test_aggregate.py -v

- [ ] **Step 3: Implement aggregation**

Success is true only when vul_exit_code is not 0 or 300 and fix_exit_code is 0 or 300 for the exact locked final hash. Missing final, timeout, provider failure, ambiguous submission, and missing oracle remain unsolved rows.

SUBMISSION.yaml contains agent_name, success_rate, link, category=agent, and one models[] entry per primary, alternate, judge, embedding, reranking, or query-expansion model actually invoked. Each entry has average input_tokens, cache_read_tokens, cache_creation_tokens, output_tokens, est_usd_cost, time_cost_sec, and llm_requests across all 1,507 tasks.

- [ ] **Step 4: Run aggregation tests**

    uv run pytest tests/leaderboard/test_aggregate.py -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/aggregate.py \
      examples/cybergym/tests/leaderboard/test_aggregate.py
    git commit -m "feat(cybergym): aggregate final-submission score"

### Task 5: Build an independent raw-evidence audit

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/audit.py
- Create: examples/cybergym/tests/leaderboard/test_audit.py

**Interfaces:**
- Consumes: audit_raw(run_root, cohort_path, public_keys)
- Produces: AuditReport independent of aggregate.py and SUBMISSION.yaml

- [ ] **Step 1: Write failing tamper tests**

~~~python
def test_audit_detects_changed_final_bytes(auditable_run):
    auditable_run.final_poc.write_bytes(b"tampered")
    report = audit_raw(auditable_run.root, auditable_run.cohort, auditable_run.keys)
    assert report.passed is False
    assert "final PoC hash mismatch" in report.failures


def test_audit_recomputes_score_without_importing_aggregate(auditable_run):
    report = audit_raw(auditable_run.root, auditable_run.cohort, auditable_run.keys)
    assert report.imported_aggregate_module is False
    assert report.task_count == 1507
~~~

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/test_audit.py -v

- [ ] **Step 3: Implement independent recomputation**

audit.py must not import aggregate.py. It reads canonical cohort order, verifies every signed event chain and file hash, reconstructs state transitions, proves one start and at most one final/submission per task, verifies oracle signatures and hash joins, recomputes solved count and model totals, and compares only at the end with the primary output.

- [ ] **Step 4: Run audit and mutation tests**

    uv run pytest tests/leaderboard/test_audit.py -v

Include parameterized mutations for deleted task, duplicate start, changed final, changed exit code, undeclared model, missing child, forbidden network success, and altered harness epoch.

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/audit.py \
      examples/cybergym/tests/leaderboard/test_audit.py
    git commit -m "feat(cybergym): independently audit campaign evidence"

### Task 6: Add continuous backup and public redaction

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/backup.py
- Create: examples/cybergym/nooa_cybergym/leaderboard/redact.py
- Create: examples/cybergym/tests/leaderboard/test_backup.py
- Create: examples/cybergym/tests/leaderboard/test_redact.py

**Interfaces:**
- Consumes: backup_terminal_task() and build_public_artifacts()
- Produces: verified remote backup and secret-free public package

- [ ] **Step 1: Write failing backup and redaction tests**

~~~python
def test_public_package_removes_credentials_and_personal_paths(sample_evidence):
    public = build_public_artifacts(sample_evidence)
    text = public.read_all_text()
    assert "ANTHROPIC_AUTH_TOKEN" not in text
    assert "SUNCHASER_ZAI_CODING_PLAN_TOKEN" not in text
    assert "C:\\Users\\nidhi" not in text


def test_backup_verifies_manifest_before_marking_complete(fake_remote, task_evidence):
    fake_remote.corrupt_one_file = True
    with pytest.raises(RuntimeError, match="backup hash mismatch"):
        backup_terminal_task(task_evidence, fake_remote)
~~~

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/test_backup.py \
      tests/leaderboard/test_redact.py -v

- [ ] **Step 3: Implement append-only backup**

After every terminal task, create a sorted SHA-256 manifest, copy to a remote storage root that is not mounted into later tasks, verify every copied hash, and append backup_complete. Never rewrite a prior terminal task. The public builder removes secrets, private OAuth handoffs, database DSNs, SSH private material, personal paths, and unrelated user data while retaining model IDs, tools, prompts, trajectories, final PoCs, exit codes, and audit hashes required for review.

- [ ] **Step 4: Run tests**

    uv run pytest tests/leaderboard/test_backup.py \
      tests/leaderboard/test_redact.py -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/backup.py \
      examples/cybergym/nooa_cybergym/leaderboard/redact.py \
      examples/cybergym/tests/leaderboard/test_backup.py \
      examples/cybergym/tests/leaderboard/test_redact.py
    git commit -m "feat(cybergym): back up and redact campaign evidence"

### Task 7: Create the runbook, writeup, and final go-live command

**Files:**
- Create: examples/cybergym/leaderboard/RUNBOOK.md
- Create: examples/cybergym/leaderboard/WRITEUP.md
- Create: examples/cybergym/tests/leaderboard/test_runbook_contract.py
- Modify: examples/cybergym/nooa_cybergym/leaderboard/cli.py

**Interfaces:**
- Consumes: accepted certification, signed pre-go-live decision, and explicit launch approval
- Produces: campaign start command and complete public experimental-setting template

- [ ] **Step 1: Write the runbook contract test**

~~~python
from pathlib import Path


def test_runbook_covers_required_stop_and_disclosure_rules():
    text = Path("leaderboard/RUNBOOK.md").read_text()
    required = (
        "No human steering after first model request",
        "Do not retry a started task",
        "Pause on harness version drift",
        "One locked final submission",
        "Restore from append-only evidence",
        "Explicit go-live approval",
    )
    assert all(item in text for item in required)
~~~

- [ ] **Step 2: Write RUNBOOK.md**

Document preflight, starting and stopping the controller, starting the deterministic workstation driver, observing sessions, reconnecting after workstation failure, handling provider quota, handling version activation, backing up, aggregating, auditing, and aborting without erasing started failures.

- [ ] **Step 3: Write WRITEUP.md**

Include the agent scaffold, native VS Code/Claude versions, GLM-5.3 Max Coding Plan route, every model role, Superpowers, bounded workflows, clangd, task-local auto memory, GBrain mode and preseed provenance, network allowlist, dynamic vulnerable environment, exact finite task ceiling, fixed cohort order, human non-intervention, development-history exclusions, final-submission rule, usage/cost method, limitations, and links to at least ten public trajectories.

- [ ] **Step 4: Run the complete pre-launch gate without starting the cohort**

    uv run pytest tests/leaderboard -v
    uv run sunchaser-cybergym campaign check-go-live \
      --decision config/pre-go-live-decision.json \
      --certification evidence/certification/accepted/report.signed.json \
      --harness-lock config/harness-lock.json \
      --cohort config/cohort.json \
      --dry-run
    git diff --check

Expected before the operator's separate approval: check-go-live exits nonzero with official launch not authorised.

- [ ] **Step 5: Commit and stop for operator approval**

    git add examples/cybergym/leaderboard/RUNBOOK.md \
      examples/cybergym/leaderboard/WRITEUP.md \
      examples/cybergym/tests/leaderboard/test_runbook_contract.py \
      examples/cybergym/nooa_cybergym/leaderboard/cli.py
    git commit -m "docs(cybergym): add official campaign runbook"

Do not execute campaign start in this implementation plan. Present the complete certification and pre-go-live record to the operator first.

## Plan 05 acceptance

The implementation is ready for a go-live decision when all tests pass, the dry-run launch guard rejects absent approval, scheduler restart tests show no hidden retry, submission tests prove one call, aggregation and audit match on a 1,507-row synthetic ledger, backup verification passes, and the writeup contains every current CyberGym disclosure field. The official campaign remains stopped until the operator explicitly authorises it.
