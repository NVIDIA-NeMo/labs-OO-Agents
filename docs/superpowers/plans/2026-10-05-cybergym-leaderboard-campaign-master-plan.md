# CyberGym Leaderboard Campaign Master Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Build, certify, and operate a clean 1,507-task CyberGym Level 1 leaderboard campaign around the native VS Code Claude Code extension and GLM-5.3 Max.

**Architecture:** SunChaser is the trusted controller and evidence authority. Each scored task runs in a new SSH-accessible container whose only writable task surface is /workspace, while the Windows VS Code client displays the native Claude Code session and a version-locked launcher supplies the initial generic prompt. Memory, model routing, certification, campaign scheduling, aggregation, and independent audit remain separate review gates.

**Tech Stack:** Python 3.12, uv, Pydantic 2, Docker, CyberGym, VS Code 1.140.0 or the recertified launch version, Claude Code extension 2.1.289 or the recertified launch version, Node.js built-in test runner, GLM-5.3 Max through Z.ai Coding Plan, GBrain MCP, JSON/JSONL, Ed25519, SHA-256.

## Global Constraints

- The official cohort is exactly the 1,507 Level 1 task IDs from the locked CyberGym tasks.json.
- C:\GLM is development-only and is never mounted into a scored container.
- The fixed image, repo-fix.tar.gz, patch.diff, error.txt, reference PoC, Git history, prior task state, controller source, Docker socket, and host homes are outside the agent boundary.
- The first primary-solver request irrevocably starts the single scored attempt.
- A started task receives one terminal result and is never rerun in the same official cohort.
- The agent selects exactly one final PoC; the controller never substitutes a timeout candidate.
- GLM-5.3 Max uses the Z.ai Coding Plan entitlement, not pay-as-you-go model billing.
- Alternate-model routes remain disabled until the separate pre-go-live decision selects an exact model and role or explicitly selects none.
- Claude native auto memory is enabled only inside one fresh task home and is archived without crossing tasks.
- The dedicated Xeus-CyberGym GBrain is the only candidate MCP. Strict recall/remember enforcement remains off until explicit approval.
- Web search, browsers, GitHub, issue trackers, changelogs, CodeRabbit, security guidance, general MCP servers, connectors, and human steering are unavailable during a scored attempt.
- Auto-updates remain enabled, but one 1,507-task headline cohort uses one certified harness epoch. A detected activation pauses scheduling.
- The official launch requires a separate approval after certification; executing these plans does not authorise it.

---

## Plan Set

| Order | Plan | Independent deliverable |
|---|---|---|
| 1 | 2026-10-05-cybergym-01-control-isolation.md | Locked cohort, clean workspaces, isolated task containers, network boundary, and one-attempt state machine |
| 2 | 2026-10-05-cybergym-02-vscode-claude-harness-telemetry.md | Native VS Code launch path, generic Claude harness, bounded workflows, model gateway, and immutable final-selection telemetry |
| 3 | 2026-10-05-cybergym-03-memory-multimodel.md | Task-local native memory, audited GBrain integration, preseed controls, failure-safe promotion, and disabled-by-default alternate-model gate |
| 4 | 2026-10-05-cybergym-04-synthetic-certification.md | Synthetic-only end-to-end certification with crash, reconnect, isolation, model, memory, and version-drift evidence |
| 5 | 2026-10-05-cybergym-05-campaign-audit-submission.md | Crash-safe scheduler, 1,507-task ledger, official oracle path, aggregation, independent audit, and leaderboard package |

## Dependency graph

    control and isolation
            |
            v
    VS Code harness and telemetry
            |
            v
    memory and multimodel policy
            |
            v
    synthetic certification
            |
            v
    explicit go-live approval
            |
            v
    campaign, audit, submission

No plan may skip its predecessor. Plan 5 may build and test its scheduler before approval, but the command that starts the official cohort must reject a missing signed go-live record.

### Task 1: Establish the implementation branch and execution record

**Files:**
- Create: docs/superpowers/plans/2026-10-05-cybergym-implementation-status.md
- Modify: none
- Test: repository status and plan hash checks

**Interfaces:**
- Consumes: this master plan and the approved campaign design
- Produces: a single branch name, baseline commit, and checkbox ledger used by all five plans

- [ ] **Step 1: Create an isolated implementation worktree**

Use the using-git-worktrees skill. Start from chore/sunchaser-official-preflight and create a codex-prefixed implementation branch.

    git fetch fork chore/sunchaser-official-preflight
    git worktree add ../labs-OO-Agents-cybergym-leaderboard \
      -b codex/cybergym-leaderboard-campaign \
      fork/chore/sunchaser-official-preflight

Expected: the new worktree starts at the pushed plan-suite commit on chore/sunchaser-official-preflight, and commit 5b9dbe37cae7abbd28b521db75a6d10f267d978e remains an ancestor as the approved-design commit.

- [ ] **Step 2: Write the execution record**

Create docs/superpowers/plans/2026-10-05-cybergym-implementation-status.md with:

~~~markdown
# CyberGym Leaderboard Implementation Status

- design_commit: 5b9dbe37cae7abbd28b521db75a6d10f267d978e
- implementation_branch: codex/cybergym-leaderboard-campaign
- official_launch_authorised: false
- alternate_model_policy: disabled
- strict_gbrain_hooks_authorised: false

| Plan | State | Review commit | Evidence |
|---|---|---|---|
| 01 control/isolation | not started | | |
| 02 harness/telemetry | blocked on 01 | | |
| 03 memory/multimodel | blocked on 02 | | |
| 04 synthetic certification | blocked on 03 | | |
| 05 campaign/audit/submission | blocked on certification and go-live approval | | |
~~~

- [ ] **Step 3: Verify the baseline and plan files**

Run:

    git rev-parse HEAD
    git merge-base --is-ancestor 5b9dbe37cae7abbd28b521db75a6d10f267d978e HEAD
    git status --short
    sha256sum docs/superpowers/specs/2026-10-05-cybergym-sunchaser-leaderboard-campaign-design.md
    find docs/superpowers/plans -maxdepth 1 -name '2026-10-05-cybergym-*.md' -print | sort

Expected: HEAD equals the pushed plan-suite commit, the approved-design ancestor check exits zero, the only worktree change is the new status file, and all six plan files are listed.

- [ ] **Step 4: Commit the execution record**

    git add docs/superpowers/plans/2026-10-05-cybergym-implementation-status.md
    git commit -m "chore: start CyberGym leaderboard implementation"

### Task 2: Execute and review Plans 01 through 03

**Files:**
- Modify: docs/superpowers/plans/2026-10-05-cybergym-implementation-status.md
- Test: each subplan's named unit and integration suites

**Interfaces:**
- Consumes: Tasks and acceptance gates in Plans 01, 02, and 03
- Produces: three independently reviewable commits and evidence links

- [ ] **Step 1: Execute Plan 01 task-by-task**

Use subagent-driven-development or executing-plans. Do not start Plan 02 until every Plan 01 test and the independent isolation review pass.

- [ ] **Step 2: Record Plan 01 acceptance**

Set its row to accepted only after recording the commit hash, pytest command, test count, and negative-preflight evidence path.

- [ ] **Step 3: Execute and review Plan 02**

Require a real VS Code extension smoke test in a synthetic workspace. A unit test of the launcher command is necessary but not sufficient.

- [ ] **Step 4: Execute and review Plan 03**

Keep config/alternate-model.json in disabled state and keep strict GBrain hooks false. Tests must prove disabled routes cannot be invoked.

- [ ] **Step 5: Commit the updated status ledger**

    git add docs/superpowers/plans/2026-10-05-cybergym-implementation-status.md
    git commit -m "docs: record core CyberGym harness acceptance"

### Task 3: Certify without touching the official cohort

**Files:**
- Modify: docs/superpowers/plans/2026-10-05-cybergym-implementation-status.md
- Create at runtime: evidence/certification/certification-20261005T000000Z/REPORT.md (example generated certification-ID path)
- Test: Plan 04's full synthetic certification suite

**Interfaces:**
- Consumes: accepted outputs of Plans 01 through 03
- Produces: a signed certification report and a frozen harness epoch

- [ ] **Step 1: Execute Plan 04 only against synthetic fixtures**

The certification runner must reject every task ID present in the locked 1,507-task cohort.

- [ ] **Step 2: Review all red gates**

Any failure in isolation, exact model identity, native extension launch, reconnection, final locking, memory separation, network denial, or evidence completeness leaves certification failed.

- [ ] **Step 3: Freeze the epoch**

Write the accepted extension, bundled Claude binary, launcher VSIX, agent image, prompt, skill, workflow, model-policy, network-policy, and controller commit hashes into harness-lock.json.

- [ ] **Step 4: Update the status record without authorising launch**

Set synthetic certification to accepted and leave official_launch_authorised false.

### Task 4: Hold the pre-go-live decision review

**Files:**
- Create: config/pre-go-live-decision.json
- Modify: docs/superpowers/plans/2026-10-05-cybergym-implementation-status.md
- Test: schema validation and certification-hash binding

**Interfaces:**
- Consumes: signed synthetic certification report
- Produces: the only record that may enable an official campaign

- [ ] **Step 1: Present the certification evidence to the operator**

The review explicitly covers the alternate-model decision, strict GBrain hooks, exact versions, auto-update epoch behavior, network allowlist, task budget, and campaign order.

- [ ] **Step 2: Record one alternate-model outcome**

The JSON must contain either:

~~~json
{"status":"disabled","models":[],"routes":[]}
~~~

or a fully populated model, role, trigger, budget, credential path, and telemetry policy that has passed a new synthetic certification.

- [ ] **Step 3: Record one GBrain enforcement outcome**

Set strict_gbrain_hooks to false with GBrain disabled for scored tasks, or true with the exact certified prologue/epilogue hashes. An explicit-MCP middle state is not allowed for the official run.

- [ ] **Step 4: Request separate official-launch approval**

Do not infer approval from design approval, plan approval, successful tests, or certification.

### Task 5: Execute Plan 05 after approval

**Files:**
- Modify: docs/superpowers/plans/2026-10-05-cybergym-implementation-status.md
- Test: campaign dry-run, aggregation parity, independent audit

**Interfaces:**
- Consumes: signed pre-go-live decision and explicit official-launch approval
- Produces: complete run evidence and the leaderboard submission package

- [ ] **Step 1: Verify the launch guard**

Run:

    uv run sunchaser-cybergym campaign check-go-live \
      --decision config/pre-go-live-decision.json \
      --harness-lock config/harness-lock.json

Expected: PASS only when approval and certification hashes match.

- [ ] **Step 2: Execute the fixed cohort**

Run the Plan 05 scheduler without changing cohort order, harness epoch, model policy, memory policy, or task budgets.

- [ ] **Step 3: Aggregate and independently audit**

The primary aggregation and clean-room audit must match on 1,507 task IDs, terminal-record count, solved count, success rate, every final PoC hash, both exit codes, and per-model usage.

- [ ] **Step 4: Produce the public package and submission**

Redact secrets, publish the reproducible non-secret artifacts, and submit only after the final audit is accepted.

## Master acceptance

The master plan is complete only when all five subplans are accepted, the 1,507-task ledger has one terminal row per task, aggregation and independent audit match, and the public package satisfies CyberGym SUBMISSION.md. Plan completion alone never implies a successful score or leaderboard acceptance.
