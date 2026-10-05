# CyberGym Memory and Multimodel Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Add auditable cross-task GBrain memory and a fail-closed multimodel policy without contaminating task-local Claude memory or enabling either feature before its go-live decision.

**Architecture:** Claude native auto memory lives only in each ephemeral task home. Cross-task memory uses the isolated Xeus-CyberGym GBrain 0.50.0 HTTP MCP service through its recall, capture, and get_stats operations, while controller-owned Postgres audit tables bind every retrieval to the final oracle outcome. Scored mode accepts only strict controller prologue/epilogue or fully disabled GBrain; development-only explicit MCP is rejected at go-live. Alternate models are represented by an empty disabled policy until a separate decision and recertification.

**Tech Stack:** Python 3.12, Pydantic 2, psycopg 3, HTTP MCP JSON-RPC, GBrain 0.50.0 observed service, Supabase Postgres, YAML frontmatter, SHA-256, pytest.

## Global Constraints

- Use the separate Xeus-CyberGym GBrain profile and database, never the personal brain.
- The private MCP endpoint remains behind Tailscale Serve and the service remains loopback-bound at 127.0.0.1:3132.
- Strict MCP enforcement, automatic recall, and automatic remember remain disabled until explicit operator approval.
- Development may use explicit audited calls. Official scored mode must be strict_hooks or disabled, never development_explicit.
- The controller is the only authority for oracle labels; model self-reports cannot populate task_outcome.
- Recall exposes principle and structurally matching procedural pages only. Raw episodic pages are not injected into later tasks.
- Failures may be stored as episodes but cannot become principles from one observation.
- Preseed contains general knowledge only and must have zero prohibited corpus matches.
- GBrain auxiliary embedding, reranking, and query-expansion models are disclosed separately from solver models.
- GEPA remains absent from this campaign epoch.
- Alternate model status begins disabled with empty models and routes. Any enabled policy requires a design revision, exact model choice, and complete synthetic recertification.

---

## File structure

| File | Responsibility |
|---|---|
| examples/cybergym/nooa_cybergym/leaderboard/memory/contracts.py | Memory tier, policy mode, retrieval, usage, and outcome contracts |
| examples/cybergym/nooa_cybergym/leaderboard/memory/migrations/001_audit.sql | memory_retrieval and task_outcome tables |
| examples/cybergym/nooa_cybergym/leaderboard/memory/mcp_client.py | OAuth HTTP MCP initialize, tools/list, recall, capture, get_stats |
| examples/cybergym/nooa_cybergym/leaderboard/memory/recall.py | Strict prologue retrieval, tier filter, injection artifact |
| examples/cybergym/nooa_cybergym/leaderboard/memory/remember.py | Oracle-labelled epilogue and promotion candidates |
| examples/cybergym/nooa_cybergym/leaderboard/memory/preseed.py | Provenance import and corpus-leak scan |
| examples/cybergym/nooa_cybergym/leaderboard/memory/snapshot.py | Database and configuration snapshot hashes |
| examples/cybergym/leaderboard/config/memory-policy.json | Development and scored memory posture |
| examples/cybergym/leaderboard/config/alternate-model.json | Disabled-by-default multimodel policy |
| examples/cybergym/leaderboard/memory/preseed/ | Frozen general-knowledge Markdown pages |

### Task 1: Define memory and model-policy contracts

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/__init__.py
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/contracts.py
- Create: examples/cybergym/leaderboard/config/memory-policy.json
- Create: examples/cybergym/leaderboard/config/alternate-model.json
- Create: examples/cybergym/tests/leaderboard/memory/test_contracts.py

**Interfaces:**
- Consumes: MemoryPolicy.model_validate_json() and AlternateModelPolicy.model_validate_json()
- Produces: MemoryTier, MemoryMode, OracleOutcome, MemoryPolicy, AlternateModelPolicy

- [ ] **Step 1: Write failing policy tests**

~~~python
import json

import pytest

from nooa_cybergym.leaderboard.memory.contracts import (
    AlternateModelPolicy,
    MemoryMode,
    MemoryPolicy,
)


def test_official_mode_rejects_development_explicit():
    policy = MemoryPolicy(
        schema_version=1,
        development_mode=MemoryMode.development_explicit,
        scored_mode=MemoryMode.development_explicit,
        source_id="xeus-cybergym-workspace",
    )
    with pytest.raises(ValueError, match="strict_hooks or disabled"):
        policy.assert_scored_ready()


def test_disabled_alternate_policy_has_no_routes():
    policy = AlternateModelPolicy(
        schema_version=1,
        status="disabled",
        models=[],
        routes=[],
    )
    policy.assert_ready()


def test_disabled_policy_rejects_hidden_model():
    with pytest.raises(ValueError, match="disabled policy"):
        AlternateModelPolicy(
            schema_version=1,
            status="disabled",
            models=[{"name": "some-model", "provider": "some-provider"}],
            routes=[],
        )
~~~

- [ ] **Step 2: Confirm the tests fail**

    cd examples/cybergym
    uv run pytest tests/leaderboard/memory/test_contracts.py -v

- [ ] **Step 3: Implement fail-closed contracts and initial configs**

Use:

~~~python
class MemoryTier(StrEnum):
    episodic = "episodic"
    semantic = "semantic"
    procedural = "procedural"
    principle = "principle"


class MemoryMode(StrEnum):
    disabled = "disabled"
    development_explicit = "development_explicit"
    strict_hooks = "strict_hooks"


class OracleOutcome(StrEnum):
    solved = "solved"
    failed = "failed"
    missing_final = "missing_final"
    ambiguous = "ambiguous"


class MemoryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1]
    development_mode: MemoryMode
    scored_mode: MemoryMode
    source_id: str
    recall_budget_tokens: int = Field(default=2000, ge=0, le=8000)
    recall_limit: int = Field(default=12, ge=0, le=50)

    def assert_scored_ready(self) -> None:
        if self.scored_mode not in {MemoryMode.strict_hooks, MemoryMode.disabled}:
            raise ValueError("scored mode must be strict_hooks or disabled")
~~~

Initial memory-policy.json:

~~~json
{
  "schema_version": 1,
  "development_mode": "development_explicit",
  "scored_mode": "disabled",
  "source_id": "xeus-cybergym-workspace",
  "recall_budget_tokens": 2000,
  "recall_limit": 12
}
~~~

Initial alternate-model.json:

~~~json
{
  "schema_version": 1,
  "status": "disabled",
  "models": [],
  "routes": []
}
~~~

- [ ] **Step 4: Run policy tests**

    uv run pytest tests/leaderboard/memory/test_contracts.py -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/memory \
      examples/cybergym/leaderboard/config/memory-policy.json \
      examples/cybergym/leaderboard/config/alternate-model.json \
      examples/cybergym/tests/leaderboard/memory/test_contracts.py
    git commit -m "feat(cybergym): define memory and model gates"

### Task 2: Add authoritative retrieval and outcome tables

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/migrations/001_audit.sql
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/audit_store.py
- Create: examples/cybergym/tests/leaderboard/memory/test_audit_store.py
- Modify: examples/cybergym/pyproject.toml
- Modify: examples/cybergym/uv.lock

**Interfaces:**
- Consumes: AuditStore.record_retrieval(), mark_used(), and record_outcome()
- Produces: immutable task_outcome rows and per-memory credit-assignment rows

- [ ] **Step 1: Write the SQL migration**

~~~sql
CREATE TABLE IF NOT EXISTS memory_retrieval (
    retrieval_id uuid PRIMARY KEY,
    run_id text NOT NULL,
    attempt_id text NOT NULL,
    task_id text NOT NULL,
    retrieval_batch_id uuid NOT NULL,
    memory_id text NOT NULL,
    tier text NOT NULL CHECK (tier IN ('episodic','semantic','procedural','principle')),
    rank integer NOT NULL CHECK (rank >= 1),
    score double precision,
    query_sha256 text NOT NULL CHECK (length(query_sha256) = 64),
    retrieved_at timestamptz NOT NULL,
    used boolean NOT NULL DEFAULT false,
    used_at timestamptz,
    use_reason text,
    UNIQUE (attempt_id, retrieval_batch_id, memory_id)
);

CREATE TABLE IF NOT EXISTS task_outcome (
    attempt_id text PRIMARY KEY,
    run_id text NOT NULL,
    task_id text NOT NULL,
    final_poc_sha256 text,
    oracle_result text NOT NULL CHECK (
        oracle_result IN ('solved','failed','missing_final','ambiguous')
    ),
    vul_exit_code integer,
    fix_exit_code integer,
    terminal_reason text NOT NULL,
    evidence_sha256 text NOT NULL CHECK (length(evidence_sha256) = 64),
    recorded_at timestamptz NOT NULL,
    UNIQUE (run_id, task_id)
);

CREATE INDEX IF NOT EXISTS memory_retrieval_task_idx
    ON memory_retrieval (run_id, task_id, retrieved_at);
CREATE INDEX IF NOT EXISTS memory_retrieval_credit_idx
    ON memory_retrieval (memory_id, used);
CREATE INDEX IF NOT EXISTS task_outcome_result_idx
    ON task_outcome (oracle_result, recorded_at);
~~~

- [ ] **Step 2: Write failing store tests against a temporary Postgres**

~~~python
import pytest

from nooa_cybergym.leaderboard.memory.audit_store import AuditStore


def test_outcome_is_immutable(postgres_dsn):
    store = AuditStore(postgres_dsn)
    store.apply_migrations()
    store.record_outcome(
        attempt_id="r:1",
        run_id="r",
        task_id="arvo:1",
        oracle_result="failed",
        terminal_reason="oracle_failed",
        evidence_sha256="a" * 64,
    )
    with pytest.raises(Exception):
        store.record_outcome(
            attempt_id="r:1",
            run_id="r",
            task_id="arvo:1",
            oracle_result="solved",
            terminal_reason="solved",
            evidence_sha256="b" * 64,
        )
~~~

Add a test that mark_used cannot reference another attempt and a test that outcome insertion happens only after an attempt is terminal.

- [ ] **Step 3: Implement AuditStore**

Add psycopg[binary] to the runner extras with uv. Use parameterized SQL only. Transactions must insert an outcome and bind all retrieval rows for that attempt without updating an existing outcome. The database DSN comes from a root-owned controller environment file and never enters the task container.

- [ ] **Step 4: Run migration and store tests**

    uv run pytest tests/leaderboard/memory/test_audit_store.py -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/memory \
      examples/cybergym/tests/leaderboard/memory/test_audit_store.py \
      examples/cybergym/pyproject.toml examples/cybergym/uv.lock
    git commit -m "feat(cybergym): add memory credit assignment store"

### Task 3: Implement the scoped GBrain MCP client and strict recall prologue

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/mcp_client.py
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/recall.py
- Create: examples/cybergym/tests/leaderboard/memory/test_mcp_client.py
- Create: examples/cybergym/tests/leaderboard/memory/test_recall.py

**Interfaces:**
- Consumes: GBrainMcpClient.recall(query, budget_tokens, limit) and get_stats()
- Produces: /workspace/.sunchaser/recall.md and memory_retrieval rows

- [ ] **Step 1: Write failing MCP and tier-filter tests**

~~~python
from nooa_cybergym.leaderboard.memory.recall import select_recall


def test_recall_injects_principles_and_matching_procedures_only():
    rows = [
        {"id": "p1", "slug": "cybergym/principle/parser-state-map", "text": "Prefer parser state maps.", "score": 0.2},
        {"id": "p2", "slug": "cybergym/procedural/riff-chunk-length", "text": "Trace RIFF chunk lengths.", "score": 0.9},
        {"id": "e1", "slug": "cybergym/episodic/run-1/0001-deadbeef", "text": "Task arvo:1 failed.", "score": 1.0},
        {"id": "s1", "slug": "cybergym/semantic/pdf-xref", "text": "Unrelated PDF fact.", "score": 0.8},
    ]
    selected = select_recall(rows, structural_terms={"riff", "chunk"})
    assert [row["id"] for row in selected] == ["p1", "p2"]
~~~

The MCP tests use a fake JSON-RPC server and assert initialize, tools/list, tools/call recall, OAuth bearer handling, bounded timeout, retry-free failure, and response ID correlation.

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/memory/test_mcp_client.py \
      tests/leaderboard/memory/test_recall.py -v

- [ ] **Step 3: Implement exact GBrain calls**

The HTTP MCP client calls:

~~~json
{
  "jsonrpc": "2.0",
  "id": "request-id",
  "method": "tools/call",
  "params": {
    "name": "recall",
    "arguments": {
      "query": "source-derived structural terms",
      "budget_tokens": 2000,
      "limit": 12
    }
  }
}
~~~

On startup it requires tools/list to contain recall, capture, and get_stats for the scoped client. It sends Authorization: Bearer from a root-owned controller credential file, records request/response hashes and latency, and never copies the bearer into evidence.

build_recall_query may use only the current Level 1 description and source-derived structural terms. Tier is encoded in the frozen slug namespace because recall results are not guaranteed to include complete YAML frontmatter. select_recall accepts only cybergym/principle/ slugs and structurally matching cybergym/procedural/ slugs, cross-checks each accepted content hash against the signed preseed or promotion manifest, and rejects episodic or semantic slugs, task IDs, final PoC bytes, fixed-side fields, and records without provenance. Frontmatter is validated when content is imported or captured. The resulting recall.md is read-only and includes memory IDs for credit assignment.

- [ ] **Step 4: Run tests**

    uv run pytest tests/leaderboard/memory/test_mcp_client.py \
      tests/leaderboard/memory/test_recall.py -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/memory/mcp_client.py \
      examples/cybergym/nooa_cybergym/leaderboard/memory/recall.py \
      examples/cybergym/tests/leaderboard/memory/test_mcp_client.py \
      examples/cybergym/tests/leaderboard/memory/test_recall.py
    git commit -m "feat(cybergym): add audited GBrain recall"

### Task 4: Add the oracle-labelled remember epilogue and conservative promotion

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/remember.py
- Create: examples/cybergym/tests/leaderboard/memory/test_remember.py

**Interfaces:**
- Consumes: remember_episode(attempt, oracle_record, task_summary) after official verification
- Produces: a GBrain capture page plus immutable task_outcome

- [ ] **Step 1: Write failing remember tests**

~~~python
from nooa_cybergym.leaderboard.memory.remember import build_episode_page, promotable


def test_failure_is_labelled_by_oracle_and_not_promoted():
    page = build_episode_page(
        run_id="r",
        task_id="arvo:1",
        oracle_result="failed",
        observation="Length arithmetic hypothesis did not produce a final success.",
        evidence_sha256="a" * 64,
    )
    assert "tier: episodic" in page
    assert "oracle_result: failed" in page
    assert promotable([page]) is False


def test_promotion_requires_three_distinct_tasks_and_repeated_outcome():
    episodes = [
        {"task_id": "arvo:1", "oracle_result": "solved", "structure": "riff-chunk"},
        {"task_id": "arvo:2", "oracle_result": "solved", "structure": "riff-chunk"},
        {"task_id": "arvo:3", "oracle_result": "solved", "structure": "riff-chunk"},
    ]
    assert promotable(episodes) is True
~~~

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/memory/test_remember.py -v

- [ ] **Step 3: Implement capture pages and promotion rules**

Capture pages use a stable slug; for example, run run-20261005T000000Z, ordinal 1, and content hash beginning a1b2c3d4 produce cybergym/episodic/run-20261005T000000Z/0001-a1b2c3d4. The content is:

~~~markdown
---
type: note
tier: episodic
run_id: run-1
task_id: arvo:1
oracle_result: failed
evidence_sha256: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
source: sunchaser-controller
---

## Observation

Length arithmetic hypothesis did not produce a final success.

## Transfer boundary

This is one task-local observation, not a general rule.
~~~

Call GBrain capture with the explicit slug and the declared GBrain page type note; tier remains a separate frontmatter field. The controller records task_outcome in the same epilogue. Do not include fixed source, patch text, reference PoC, hidden crash trace, final PoC bytes, or the fixed verifier output beyond vul_exit_code and fix_exit_code in the separate outcome row.

Promotion requires at least three distinct tasks, matching structural tags, and consistent oracle-labelled evidence. A promoted procedure or principle is a new reviewed page; raw episodes are never rewritten. Principle promotion requires explicit human or independent-review approval and remains rare.

- [ ] **Step 4: Run remember tests**

    uv run pytest tests/leaderboard/memory/test_remember.py -v

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/memory/remember.py \
      examples/cybergym/tests/leaderboard/memory/test_remember.py
    git commit -m "feat(cybergym): remember oracle-labelled episodes"

### Task 5: Freeze and scan the general-knowledge preseed

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/preseed.py
- Create: examples/cybergym/leaderboard/memory/preseed/README.md
- Create: examples/cybergym/leaderboard/memory/preseed/manifest.json
- Create: examples/cybergym/tests/leaderboard/memory/test_preseed.py

**Interfaces:**
- Consumes: scan_preseed(preseed_dir, tasks_json, cybergym_data_root)
- Produces: a signed zero-match report and import-ready principle, procedural, and semantic pages

- [ ] **Step 1: Write failing leak-scanner tests**

~~~python
import pytest

from nooa_cybergym.leaderboard.memory.preseed import scan_text


def test_scanner_rejects_task_id_and_patch_derived_text():
    signatures = {
        "task_ids": {"arvo:3848"},
        "forbidden_phrases": {"known patch function"},
        "forbidden_sha256": set(),
    }
    with pytest.raises(RuntimeError, match="task id"):
        scan_text("lesson mentions arvo:3848", signatures)
    with pytest.raises(RuntimeError, match="forbidden phrase"):
        scan_text("the known patch function is unsafe", signatures)
~~~

Add tests for missing provenance, absent tier, duplicate content hashes, fixed-source filenames, reference PoC encodings, and a clean generic parser principle.

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/memory/test_preseed.py -v

- [ ] **Step 3: Implement provenance and corpus scanning**

Each page must have frontmatter fields type=note, tier, title, source_url or source_document, source_sha256, license, captured_at, and reviewed=true. Allowed tiers are semantic, procedural, and principle; preseed cannot contain episodic. The slug namespace carries the same tier and the manifest rejects any slug/frontmatter mismatch.

The scanner builds prohibited signatures from all task IDs, descriptions, error.txt, patch.diff, repo-fix metadata, reference PoCs, locally known candidates, and fixed-source fragments. Exact IDs and hashes are always rejected. Fuzzy phrase matching is used for patch/crash/reference-derived sentences, with a human review queue for uncertain matches. No uncertain item enters the frozen manifest.

- [ ] **Step 4: Run the real zero-match scan**

    uv run pytest tests/leaderboard/memory/test_preseed.py -v
    uv run sunchaser-cybergym memory scan-preseed \
      --preseed-dir leaderboard/memory/preseed \
      --tasks-json cybergym_repo/cybergym_data/tasks.json \
      --data-root cybergym_repo/cybergym_data/data \
      --report /tmp/preseed-scan-report.json

Expected: zero prohibited matches and zero unreviewed items.

- [ ] **Step 5: Commit**

    git add examples/cybergym/nooa_cybergym/leaderboard/memory/preseed.py \
      examples/cybergym/leaderboard/memory/preseed \
      examples/cybergym/tests/leaderboard/memory/test_preseed.py
    git commit -m "feat(cybergym): add compliant memory preseed gate"

### Task 6: Add memory snapshots and the pre-go-live enforcement gate

**Files:**
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/snapshot.py
- Create: examples/cybergym/nooa_cybergym/leaderboard/memory/gate.py
- Create: examples/cybergym/tests/leaderboard/memory/test_snapshot.py
- Create: examples/cybergym/tests/leaderboard/memory/test_gate.py

**Interfaces:**
- Consumes: snapshot_memory(), validate_scored_memory_policy(), validate_model_policy()
- Produces: immutable snapshot manifest or a pre-go-live rejection

- [ ] **Step 1: Write failing gate tests**

~~~python
import pytest

from nooa_cybergym.leaderboard.memory.gate import validate_go_live_policies


def test_go_live_rejects_explicit_mcp_and_unselected_alternate():
    with pytest.raises(RuntimeError, match="memory policy"):
        validate_go_live_policies(
            memory={"scored_mode": "development_explicit"},
            alternate={"status": "disabled", "models": [], "routes": []},
        )


def test_go_live_accepts_disabled_memory_and_disabled_alternate():
    validate_go_live_policies(
        memory={"scored_mode": "disabled"},
        alternate={"status": "disabled", "models": [], "routes": []},
    )
~~~

- [ ] **Step 2: Confirm the tests fail**

    uv run pytest tests/leaderboard/memory/test_snapshot.py \
      tests/leaderboard/memory/test_gate.py -v

- [ ] **Step 3: Implement snapshots and gate**

snapshot_memory records GBrain health/version, source ID, page/fact counts by tier, all page IDs and content hashes, audit-table row counts, schema migration hash, retrieval-model identifiers, configuration hash, and encrypted database-backup hash. It stores no credential values.

validate_go_live_policies accepts only:

1. scored memory disabled plus no task MCP credential; or
2. scored memory strict_hooks plus certified prologue/epilogue hashes and a scoped MCP credential.

Alternate status disabled requires empty models and routes. Enabled status requires at least one exact model, provider, role, trigger, max_calls, max_tokens, max_seconds, credential_path, and certification_hash. No automatic provider fallback field is permitted.

- [ ] **Step 4: Run the complete Plan 03 gate**

    uv run pytest tests/leaderboard/memory -v
    uv run sunchaser-cybergym memory verify-service \
      --endpoint https://sunchaser-20260905.tailfd2212.ts.net/mcp \
      --source xeus-cybergym-workspace \
      --required-tools recall,capture,get_stats
    git diff --check

Expected before operator authorisation: development explicit calls pass, scored_mode remains disabled, strict hooks remain off, and alternate models remain disabled.

- [ ] **Step 5: Commit and request memory-policy review**

    git add examples/cybergym/nooa_cybergym/leaderboard/memory \
      examples/cybergym/tests/leaderboard/memory \
      examples/cybergym/leaderboard/config \
      examples/cybergym/leaderboard/memory \
      examples/cybergym/pyproject.toml examples/cybergym/uv.lock
    git commit -m "feat(cybergym): gate audited memory and multimodel policy"

The reviewer must verify the personal brain is unreachable, episodic records are not recalled, outcomes come only from the official oracle, failures are not over-promoted, auxiliary models are itemized, strict hooks remain unauthorised, and alternate routing is impossible.

## Plan 03 acceptance

Accept when the isolated GBrain service passes scoped OAuth calls, the audit tables bind retrievals to immutable oracle outcomes, task-local Claude memory cannot cross containers, preseed scan has zero prohibited matches, scored strict hooks remain disabled pending approval, and every alternate-model invocation path is rejected by the model gateway.
