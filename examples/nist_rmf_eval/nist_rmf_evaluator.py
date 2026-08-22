# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Contributed by Council of AI (CSOAI Ltd, UK 16939677) — https://councilof.ai
"""NIST AI RMF evaluator — deterministic core, LLM-narrated verdicts.

Companion to the GSPC provision-anchored evaluator (PR #75), applying the same
Council of AI pattern to a voluntary framework instead of statute:

    Python decides. The LLM only narrates.

A scenario description (e.g. "the team documented the model's intended purpose
and deployment context, ran adversarial tests, and maintains a rollback plan")
is evaluated against a small built-in corpus of real NIST AI RMF 1.0
subcategories across the four functions (GOVERN, MAP, MEASURE, MANAGE).
Matching, verdict construction, and signing are ordinary deterministic Python —
no model in the loop, no fabricated precision: anything not matched is
UNMEASURED. An optional generation method (`explain_verdict`) lets an LLM
narrate the signed verdict in plain English; the deterministic layer never
depends on it.

Run (deterministic only, no API key needed):

    uv run python examples/nist_rmf_eval/nist_rmf_evaluator.py

With OPENAI_API_KEY or NVIDIA_API_KEY set, the example additionally calls
`explain_verdict` for a live LLM narration of the signed verdict.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from nooa import Agent, hidden

with hidden:
    import hashlib
    import hmac
    import json
    import os
    import secrets

    from nooa.unifiedllm.fake import FakeLLMClient

# ---------------------------------------------------------------------------
# NIST AI RMF 1.0 subcategory corpus
#
# Eight real subcategories (two per function) chosen for demonstration. This
# is NOT the full 72-subcategory AI RMF crosswalk; see README.md. Each anchor
# carries the literal keyword phrases the deterministic matcher looks for and
# a faithful plain-language restatement of the subcategory text (NIST AI
# 100-1, Appendix A). Nothing here is inferred by a model — a subcategory is
# "matched" only when its anchor terms appear in the scenario text, or when
# the framework itself makes it a prerequisite of a matched subcategory (the
# AI RMF is voluntary guidance, not statute, so there is exactly one such
# documented edge: MANAGE-1.3's own text says risk responses address the
# risks "as identified by the MAP function", so a matched MANAGE-1.3 attaches
# MAP-5.1, the MAP subcategory that identifies and documents those impacts).
# ---------------------------------------------------------------------------


class Subcategory(BaseModel):
    """A NIST AI RMF subcategory and the deterministic anchors used to match it."""

    id: str = Field(description="Stable identifier, e.g. MEASURE-2.7.")
    function: Literal["govern", "map", "measure", "manage"] = Field(
        description="The AI RMF Core function the subcategory belongs to."
    )
    title: str = Field(description="Faithful plain-language restatement of the subcategory.")
    keywords: list[str] = Field(
        default_factory=list,
        description="Lowercase anchor phrases; any substring hit counts as a match.",
    )
    attached_by: list[str] = Field(
        default_factory=list,
        description=(
            "Subcategory IDs whose match automatically attaches this subcategory "
            "(framework-declared dependency, not keyword inference)."
        ),
    )


SUBCATEGORY_CORPUS: list[Subcategory] = [
    Subcategory(
        id="GOVERN-1.1",
        function="govern",
        title="Legal and regulatory requirements involving AI are understood, managed, and documented",
        keywords=[
            "legal and regulatory requirements",
            "applicable laws",
            "regulatory register",
            "legal requirements",
        ],
    ),
    Subcategory(
        id="GOVERN-2.1",
        function="govern",
        title="Roles, responsibilities, and lines of communication for AI risk are documented and clear",
        keywords=[
            "roles and responsibilities",
            "accountable owner",
            "raci",
            "lines of communication",
        ],
    ),
    Subcategory(
        id="MAP-1.1",
        function="map",
        title="Intended purposes, context-specific laws and norms, and deployment settings are documented",
        keywords=[
            "intended purpose",
            "deployment context",
            "context of use",
        ],
    ),
    Subcategory(
        id="MAP-5.1",
        function="map",
        title="Likelihood and magnitude of each identified impact, beneficial and harmful, are documented",
        keywords=[
            "likelihood and magnitude",
            "impact assessment",
            "potential impacts",
            "societal impact",
        ],
        attached_by=["MANAGE-1.3"],
    ),
    Subcategory(
        id="MEASURE-1.1",
        function="measure",
        title="Appropriate metrics for AI measurement and assessment of AI risks are identified and documented",
        keywords=[
            "evaluation metrics",
            "performance metrics",
            "documented metrics",
        ],
    ),
    Subcategory(
        id="MEASURE-2.7",
        function="measure",
        title="AI system security and resilience are evaluated and documented",
        keywords=[
            "red team",
            "red-team",
            "adversarial test",
            "security evaluation",
            "resilience test",
        ],
    ),
    Subcategory(
        id="MANAGE-1.3",
        function="manage",
        title="Responses to high-priority AI risks identified by MAP are developed, planned, and documented",
        keywords=[
            "risk response",
            "risk treatment",
            "mitigation plan",
        ],
    ),
    Subcategory(
        id="MANAGE-2.4",
        function="manage",
        title="Mechanisms and assigned responsibilities to supersede, disengage, or deactivate AI systems",
        keywords=[
            "deactivate",
            "disengage",
            "kill switch",
            "rollback",
            "supersede",
        ],
    ),
]

VERDICT_FULL = "FULL_FUNCTION_COVERAGE"
VERDICT_PARTIAL = "PARTIAL_FUNCTION_COVERAGE"
VERDICT_UNMEASURED = "UNMEASURED"

Verdict = Literal["FULL_FUNCTION_COVERAGE", "PARTIAL_FUNCTION_COVERAGE", "UNMEASURED"]

ALL_FUNCTIONS = ("govern", "map", "measure", "manage")


class VerdictRecord(BaseModel):
    """A signed, tamper-evident evaluation verdict.

    `signature` is HMAC-SHA256 over the canonical JSON of every other field,
    keyed with a local key file. Recomputing it detects any tampering with the
    scenario hash, the matched subcategories, the verdict, or the evidence.
    """

    scenario_hash: str = Field(description="SHA-256 of the normalized scenario text.")
    subcategories_matched: list[str] = Field(description="IDs of matched subcategories, sorted.")
    verdict: Verdict
    evidence: dict[str, list[str]] = Field(
        description="Subcategory ID -> anchor terms / attachment chain that matched."
    )
    signature: str = Field(description="HMAC-SHA256 over canonical JSON of the fields above.")


def _canonical_payload(record_fields: dict) -> bytes:
    """Canonical JSON (sorted keys, compact separators) for signing/verifying."""
    return json.dumps(record_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


class NistRmfEvaluator(Agent):
    """You are an audit-grade NIST AI RMF evaluator for AI deployment scenarios.

    Your deterministic methods match scenario text against a corpus of real
    NIST AI RMF 1.0 subcategories and emit signed verdict records. You never
    guess: if no subcategory anchor matches, the verdict is UNMEASURED.
    """

    signing_key_path: Annotated[str, hidden] = "nist_signing_key.hex"
    strict_narration: Annotated[bool, hidden] = False

    def __init__(
        self,
        *args,
        signing_key_path: str | Path | None = None,
        strict_narration: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if signing_key_path is not None:
            self.signing_key_path = str(signing_key_path)
        # `strict_narration` rejects LLM narration when the verdict is
        # UNMEASURED — i.e. when the deterministic core has nothing to anchor
        # on, the model is forbidden from filling the gap. Off by default to
        # preserve the existing call signature.
        self.strict_narration = strict_narration

    # ------------------------------------------------------------------
    # Deterministic core — ordinary Python, no LLM in the loop.
    # ------------------------------------------------------------------

    def _load_or_create_key(self) -> bytes:
        """Load the local HMAC signing key, creating it (0600) on first use."""
        path = Path(self.signing_key_path)
        if path.exists():
            return bytes.fromhex(path.read_text().strip())
        key = secrets.token_bytes(32)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(key.hex())
        return key

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase and collapse whitespace so matching is stable."""
        return " ".join(text.lower().split())

    def match_subcategories(self, scenario: str) -> dict[str, list[str]]:
        """Match a scenario against the NIST AI RMF subcategory corpus.

        Returns {subcategory_id: matched_terms}. A subcategory is included
        only when at least one anchor keyword appears in the normalized
        scenario, or when the framework declares it a prerequisite of a
        matched subcategory.
        """
        text = self._normalize(scenario)
        evidence: dict[str, list[str]] = {}
        for subcategory in SUBCATEGORY_CORPUS:
            hits = [kw for kw in subcategory.keywords if kw in text]
            if hits:
                evidence[subcategory.id] = hits
        # Framework-declared attachment (deterministic rule, not inference):
        # MANAGE-1.3's own text defines its responses as addressing the risks
        # "as identified by the MAP function", so a matched MANAGE-1.3
        # attaches MAP-5.1.
        changed = True
        while changed:
            changed = False
            for subcategory in SUBCATEGORY_CORPUS:
                if subcategory.id in evidence:
                    continue
                for trigger in subcategory.attached_by:
                    if trigger in evidence:
                        evidence[subcategory.id] = [f"attached via {trigger}"]
                        changed = True
                        break
        return dict(sorted(evidence.items()))

    def build_verdict(self, scenario: str) -> VerdictRecord:
        """Evaluate a scenario and emit a signed verdict record.

        Verdict rules (no fabricated precision):
        - matches span all four AI RMF functions -> FULL_FUNCTION_COVERAGE
        - otherwise any subcategory matched       -> PARTIAL_FUNCTION_COVERAGE
        - nothing matched                         -> UNMEASURED
        """
        evidence = self.match_subcategories(scenario)
        functions = {s.id: s.function for s in SUBCATEGORY_CORPUS}
        covered = {functions[sid] for sid in evidence}
        if covered == set(ALL_FUNCTIONS):
            verdict: Verdict = VERDICT_FULL
        elif evidence:
            verdict = VERDICT_PARTIAL
        else:
            verdict = VERDICT_UNMEASURED

        scenario_hash = hashlib.sha256(self._normalize(scenario).encode("utf-8")).hexdigest()
        unsigned = {
            "scenario_hash": scenario_hash,
            "subcategories_matched": sorted(evidence),
            "verdict": verdict,
            "evidence": evidence,
        }
        signature = hmac.new(
            self._load_or_create_key(), _canonical_payload(unsigned), hashlib.sha256
        ).hexdigest()
        return VerdictRecord(**unsigned, signature=signature)

    def verify_verdict(self, record: VerdictRecord) -> bool:
        """Recompute the HMAC over a verdict record; False means tampered."""
        unsigned = {
            "scenario_hash": record.scenario_hash,
            "subcategories_matched": record.subcategories_matched,
            "verdict": record.verdict,
            "evidence": record.evidence,
        }
        expected = hmac.new(
            self._load_or_create_key(), _canonical_payload(unsigned), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, record.signature)

    # ------------------------------------------------------------------
    # LLM-optional layer — the model narrates; it never decides.
    # ------------------------------------------------------------------

    async def explain_verdict(self, verdict: VerdictRecord) -> str:
        """Narrate this signed deterministic verdict in plain English for an auditor.

        State the verdict, the subcategories matched, and the evidence terms.
        Do NOT change the verdict, add subcategories, or speculate about
        subcategories that are not in the record — if a subcategory is absent,
        it was not measured.

        Strict mode: when `strict_narration=True` was passed to the
        constructor, this method refuses to generate any narration for an
        UNMEASURED verdict (raises `ValueError` rather than letting the LLM
        fill the gap the deterministic core explicitly left empty). Off by
        default; opt-in for audit pipelines that prefer a hard "no
        narration without anchor" rule over a permissive "summarise what we
        don't know" rule.
        """
        if self.strict_narration and verdict.verdict == VERDICT_UNMEASURED:
            raise ValueError(
                "strict_narration=True: refusing to narrate an UNMEASURED "
                "verdict (no subcategory matched; deterministic core returned "
                "nothing to anchor on)."
            )
        # Permissive mode (or a measured verdict in strict mode) delegates to
        # the LLM generation body below. NOOA runs `...` as LLM generation;
        # by splitting strict-mode enforcement out of the body, we ensure
        # the model is never invoked on UNMEASURED in strict mode.
        return await self._narrate_verdict(verdict)

    async def _narrate_verdict(self, verdict: VerdictRecord) -> str:
        """LLM generation: narrate the signed verdict (deterministic record unchanged).

        Receives only verified fields — scenario_hash, subcategories_matched,
        verdict, evidence, signature — so the model cannot introduce new
        subcategories or change the verdict. Used by `explain_verdict`; do not
        call directly (callers wanting strict-mode behaviour should go
        through `explain_verdict`).
        """
        ...


def _demo_llm():
    """Build an LLM client from env keys, or None to run narration offline."""
    from nooa.unifiedllm import get_llm_client

    if os.getenv("NVIDIA_API_KEY"):
        return get_llm_client(
            "nvidia_nim/nvidia/nemotron-3-super-120b-a12b",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
    if os.getenv("OPENAI_API_KEY"):
        return get_llm_client("gpt-5-mini")
    return None


def main() -> None:
    agent = NistRmfEvaluator(
        llm=_demo_llm() or FakeLLMClient(),
        signing_key_path=Path(__file__).parent / "nist_signing_key.hex",
    )

    scenario = (
        "The team documented the model's intended purpose and deployment "
        "context, maintains performance metrics on a public dashboard, ran "
        "red-team adversarial tests before launch, and keeps a rollback plan "
        "to deactivate the system if it drifts from intended use."
    )
    record = agent.build_verdict(scenario)
    print(record.model_dump_json(indent=2))
    print("signature verifies:", agent.verify_verdict(record))

    if not isinstance(agent._llm, FakeLLMClient):
        import asyncio

        narration = asyncio.run(agent.explain_verdict(record))
        print("\nLLM narration (deterministic verdict unchanged):\n", narration)
    else:
        print(
            "\n(no API key set — deterministic verdict above is complete; "
            "set OPENAI_API_KEY or NVIDIA_API_KEY for an LLM narration)"
        )


if __name__ == "__main__":
    main()
