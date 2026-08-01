# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Contributed by CSOAI (csoai.org) — Council for the Safety of Artificial Intelligence.
"""GSPC provision-anchored evaluator — deterministic core, LLM-narrated verdicts.

Demonstrates the pattern CSOAI contributes to the Open Secure AI Alliance's
"evaluations and benchmarks" lane:

    Python decides. The LLM only narrates.

A scenario description (e.g. "a hospital deploys an AI triage model in the EU
that ranks ER patients by urgency") is evaluated against a small built-in
corpus of real statute provisions. Matching, verdict construction, and signing
are ordinary deterministic Python — no model in the loop, no fabricated
precision: anything not matched is UNMEASURED. An optional generation method
(`explain_verdict`) lets an LLM narrate the signed verdict in plain English;
the deterministic layer never depends on it.

Run (deterministic only, no API key needed):

    uv run python examples/gspc_provision_eval/provision_evaluator.py

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
# Provision corpus
#
# Six real provisions for demonstration. This is NOT the full CSOAI corpus
# (417 provisions); see README.md. Each provision carries the literal keyword
# anchors the deterministic matcher looks for. Nothing here is inferred by a
# model — a provision is "matched" only when its anchor terms appear in the
# scenario text, or when it is statutorily attached to a matched provision
# (e.g. an Annex III high-risk classification attaches the Art 9 / Art 14
# obligations that the Regulation imposes on high-risk systems).
# ---------------------------------------------------------------------------


class Provision(BaseModel):
    """A statute provision and the deterministic anchors used to match it."""

    id: str = Field(description="Stable identifier, e.g. EU-AI-ACT-ART-9.")
    instrument: str = Field(description="Statute the provision belongs to.")
    title: str = Field(description="Short human-readable title.")
    effect: Literal["prohibited", "high_risk", "obligation", "transparency"] = Field(
        description="Regulatory effect when matched."
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="Lowercase anchor phrases; any substring hit counts as a match.",
    )
    attached_by: list[str] = Field(
        default_factory=list,
        description=(
            "Provision IDs whose match automatically attaches this provision "
            "(statutory consequence, not keyword inference)."
        ),
    )


PROVISION_CORPUS: list[Provision] = [
    Provision(
        id="EU-AI-ACT-ART-5",
        instrument="EU AI Act, Article 5",
        title="Prohibited AI practices (incl. social scoring)",
        effect="prohibited",
        keywords=[
            "social scor",
            "social credit",
            "trustworthiness score",
            "scores citizens",
            "scoring citizens",
        ],
    ),
    Provision(
        id="EU-AI-ACT-ANNEX-III-5A",
        instrument="EU AI Act, Annex III 5(a)",
        title="Emergency triage — high-risk classification",
        effect="high_risk",
        keywords=[
            "triage",
            "emergency healthcare",
            "emergency dispatch",
            "er patient",
            "by urgency",
        ],
    ),
    Provision(
        id="EU-AI-ACT-ART-9",
        instrument="EU AI Act, Article 9",
        title="Risk management system for high-risk AI",
        effect="obligation",
        keywords=["risk management system"],
        attached_by=["EU-AI-ACT-ANNEX-III-5A"],
    ),
    Provision(
        id="EU-AI-ACT-ART-14",
        instrument="EU AI Act, Article 14",
        title="Human oversight of high-risk AI",
        effect="obligation",
        keywords=["human oversight"],
        attached_by=["EU-AI-ACT-ANNEX-III-5A"],
    ),
    Provision(
        id="EU-AI-ACT-ART-50",
        instrument="EU AI Act, Article 50",
        title="Transparency for AI interaction and synthetic content",
        effect="transparency",
        keywords=[
            "chatbot",
            "conversational ai",
            "interacts with humans",
            "ai interaction",
            "synthetic content",
            "deepfake",
        ],
    ),
    Provision(
        id="GDPR-ART-22",
        instrument="GDPR, Article 22",
        title="Automated individual decision-making",
        effect="obligation",
        keywords=[
            "automated decision",
            "solely automated",
            "automated processing",
            "no human in the loop",
            "without human intervention",
        ],
    ),
]

VERDICT_PERMITTED = "PERMITTED_WITH_CONDITIONS"
VERDICT_PROHIBITED = "PROHIBITED_RISK"
VERDICT_UNMEASURED = "UNMEASURED"

Verdict = Literal["PERMITTED_WITH_CONDITIONS", "PROHIBITED_RISK", "UNMEASURED"]


class VerdictRecord(BaseModel):
    """A signed, tamper-evident evaluation verdict.

    `signature` is HMAC-SHA256 over the canonical JSON of every other field,
    keyed with a local key file. Recomputing it detects any tampering with the
    scenario hash, the matched provisions, the verdict, or the evidence.
    """

    scenario_hash: str = Field(description="SHA-256 of the normalized scenario text.")
    provisions_matched: list[str] = Field(description="IDs of matched provisions, sorted.")
    verdict: Verdict
    evidence: dict[str, list[str]] = Field(
        description="Provision ID -> anchor terms / attachment chain that matched."
    )
    signature: str = Field(description="HMAC-SHA256 over canonical JSON of the fields above.")


def _canonical_payload(record_fields: dict) -> bytes:
    """Canonical JSON (sorted keys, compact separators) for signing/verifying."""
    return json.dumps(record_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")


class ProvisionEvaluator(Agent):
    """You are an audit-grade provision evaluator for AI deployment scenarios.

    Your deterministic methods match scenario text against a statute provision
    corpus and emit signed verdict records. You never guess: if no provision
    anchor matches, the verdict is UNMEASURED.
    """

    signing_key_path: Annotated[str, hidden] = "gspc_signing_key.hex"

    def __init__(self, *args, signing_key_path: str | Path | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if signing_key_path is not None:
            self.signing_key_path = str(signing_key_path)

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

    def match_provisions(self, scenario: str) -> dict[str, list[str]]:
        """Match a scenario against the provision corpus.

        Returns {provision_id: matched_terms}. A provision is included only
        when at least one anchor keyword appears in the normalized scenario,
        or when it is statutorily attached to a matched provision.
        """
        text = self._normalize(scenario)
        evidence: dict[str, list[str]] = {}
        for provision in PROVISION_CORPUS:
            hits = [kw for kw in provision.keywords if kw in text]
            if hits:
                evidence[provision.id] = hits
        # Statutory attachment (deterministic rule, not inference): a matched
        # high-risk classification attaches the obligations the Regulation
        # imposes on high-risk systems.
        changed = True
        while changed:
            changed = False
            for provision in PROVISION_CORPUS:
                if provision.id in evidence:
                    continue
                for trigger in provision.attached_by:
                    if trigger in evidence:
                        evidence[provision.id] = [f"attached via {trigger}"]
                        changed = True
                        break
        return dict(sorted(evidence.items()))

    def build_verdict(self, scenario: str) -> VerdictRecord:
        """Evaluate a scenario and emit a signed verdict record.

        Verdict rules (no fabricated precision):
        - any 'prohibited' provision matched      -> PROHIBITED_RISK
        - otherwise any provision matched          -> PERMITTED_WITH_CONDITIONS
        - nothing matched                          -> UNMEASURED
        """
        evidence = self.match_provisions(scenario)
        effects = {p.id: p.effect for p in PROVISION_CORPUS}
        if any(effects[pid] == "prohibited" for pid in evidence):
            verdict: Verdict = VERDICT_PROHIBITED
        elif evidence:
            verdict = VERDICT_PERMITTED
        else:
            verdict = VERDICT_UNMEASURED

        scenario_hash = hashlib.sha256(self._normalize(scenario).encode("utf-8")).hexdigest()
        unsigned = {
            "scenario_hash": scenario_hash,
            "provisions_matched": sorted(evidence),
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
            "provisions_matched": record.provisions_matched,
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

        State the verdict, the provisions matched, and the evidence terms.
        Do NOT change the verdict, add provisions, or speculate about
        provisions that are not in the record — if a provision is absent,
        it was not measured.
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
    agent = ProvisionEvaluator(
        llm=_demo_llm() or FakeLLMClient(),
        signing_key_path=Path(__file__).parent / "gspc_signing_key.hex",
    )

    scenario = (
        "A hospital deploys an AI triage model in the EU that ranks ER patients "
        "by urgency to prioritise treatment order."
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
