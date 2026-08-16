<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Contributed by CSOAI (csoai.org) — Council of AI (CSOAI LTD, UK #16939677). -->

# GSPC Provision-Anchored Evaluator

**Python decides. The LLM only narrates.**

`ProvisionEvaluator` is a NOOA agent that evaluates a natural-language AI
deployment scenario against a built-in corpus of statute provisions and emits
a **signed, tamper-evident verdict record** — entirely deterministically. An
optional generation method (`explain_verdict`) lets an LLM narrate the verdict
in plain English, but the verdict itself is produced by ordinary Python and
never depends on a model.

Contributed by [CSOAI](https://csoai.org) (Council of AI (CSOAI LTD, UK #16939677)) to the Open Secure AI Alliance's **evaluations and
benchmarks** lane, as a minimal reference for the deterministic-core /
LLM-narrated split that audit-grade evaluation requires.

## What it demonstrates

- **Provision-anchored.** A provision is "matched" only when its literal
  anchor terms appear in the scenario, or when it is statutorily attached to a
  matched provision (an Annex III high-risk classification attaches the
  Art 9 / Art 14 obligations the Regulation imposes on high-risk systems).
  No vibes, no model inference.
- **No fabricated precision.** Anything not matched is `UNMEASURED` — the
  evaluator says "I don't know" instead of guessing.
- **Signed and tamper-evident.** Each verdict record carries
  `scenario_hash` (SHA-256) and `signature` (HMAC-SHA256 over the canonical
  JSON of the record, keyed with a local `0600` key file). Flip one field and
  `verify_verdict` returns `False`; change one character of the scenario and
  the hash changes.
- **Deterministic core / LLM-narrated split.** `match_provisions`,
  `build_verdict`, and `verify_verdict` are ordinary Python (SW1 in NOOA
  terms — tools the LLM can call but not influence). Only `explain_verdict`
  has an ellipsis body: the docstring instructs the model to narrate the
  signed record without altering it. An auditor can replay the deterministic
  layer byte-for-byte; the narration is presentation, not evidence.

## The verdict record

```json
{
  "scenario_hash": "sha256 of normalized scenario text",
  "provisions_matched": ["EU-AI-ACT-ANNEX-III-5A", "EU-AI-ACT-ART-14", "EU-AI-ACT-ART-9"],
  "verdict": "PERMITTED_WITH_CONDITIONS",
  "evidence": {"EU-AI-ACT-ANNEX-III-5A": ["triage", "er patient", "by urgency"],
               "EU-AI-ACT-ART-9": ["attached via EU-AI-ACT-ANNEX-III-5A"], "...": ["..."]},
  "signature": "HMAC-SHA256 over canonical JSON of the fields above"
}
```

Verdict rules: any `prohibited` match → `PROHIBITED_RISK`; otherwise any
match → `PERMITTED_WITH_CONDITIONS`; nothing matched → `UNMEASURED`.

## Strict narration (audit-pipeline mode)

By default, `explain_verdict` calls the LLM regardless of the verdict — if
the deterministic core returned `UNMEASURED`, the narration simply says
"no provision matched." For pipelines that want a hard refusal instead,
pass `strict_narration=True` to the constructor:

```python
agent = ProvisionEvaluator(
    llm=...,
    signing_key_path="gspc_signing_key.hex",
    strict_narration=True,
)
# ...
asyncio.run(agent.explain_verdict(record))  # raises ValueError on UNMEASURED
```

In strict mode, `explain_verdict` raises `ValueError` for an `UNMEASURED`
verdict rather than letting the LLM fill the gap the deterministic core
explicitly left empty. Permissive (default) and strict modes are covered by
`test_strict_narration_refuses_unmeasured`.

## Run it

Deterministic core — no API key needed:

```bash
uv run python examples/gspc_provision_eval/provision_evaluator.py
```

With `OPENAI_API_KEY` or `NVIDIA_API_KEY` set, the script additionally calls
`explain_verdict` for a live LLM narration of the signed verdict (the verdict
itself is unchanged either way).

Run the tests (hermetic — they use `FakeLLMClient`, no network):

```bash
uv run pytest examples/gspc_provision_eval/test_provision_evaluator.py
```

## Honest scope note

The built-in corpus is **6 provisions** chosen for demonstration (EU AI Act
Art 5, Art 9, Art 14, Art 50, Annex III 5(a); GDPR Art 22). It is **not** the
full 417-provision CSOAI corpus, and keyword anchoring is a demonstration
matcher, not a legal classifier. The point of the example is the architecture
— deterministic, signed, provision-anchored evaluation with LLM narration on
top — which scales to a larger corpus without design changes.

## Files

| File | Purpose |
|---|---|
| `provision_evaluator.py` | The `ProvisionEvaluator` agent (corpus, deterministic core, `explain_verdict`) |
| `test_provision_evaluator.py` | Hermetic pytest suite for the deterministic core |
| `conftest.py` | Makes the example importable when pytest runs this directory |
| `verify_offline.py` | Public-key offline verifier for signed measurement cards (zero network, zero secrets) |
| `sample_card.json` | A real signed fleet measurement card to verify (id af0a3e88..., 2026-08-16) |

## Verify a real signed measurement card offline

 is a real Ed25519-signed measurement card from the Council
of AI live fleet board (EU AI Act Art-5 prohibited-practice screening across
13 open-weight models, 2026-08-16). The verifier is public-key only — zero
network, zero secrets:

```bash
uv run python examples/gspc_provision_eval/verify_offline.py --card examples/gspc_provision_eval/sample_card.json
# VALID fleet-art5-sov6 id=af0a3e88a649ea0e...
```

Card format: `id = SHA-256(canonical JSON of body)` (sorted keys, compact
separators); `signature = Ed25519(id)` under the card's `pubkey`; `prev`
links each card to the previous one, forming a tamper-evident chain. Flip any
byte of the body and verification fails.

## What this example does NOT measure

Honesty block, kept in code-review form:

- The 6-provision demo corpus is a teaching corpus. It does not determine
  legal compliance, and a `PERMITTED_WITH_CONDITIONS` verdict is not legal
  advice.
- Keyword anchoring is a demonstration matcher; production measurement uses
  per-axis method hashes so a reader can tell exactly which method produced
  a number. Anything not measured is disclosed as `UNMEASURED` — never
  interpolated, never averaged over.
- The fleet-extent claim is bounded and live: the evaluator's provision bank
  is measured nightly across 22 open models x 13 axes, with a public
  scoreboard and signed ledger at https://csoai.org — check the numbers there
  rather than trusting this sentence.

## The containment framing

**deterministic core decides; LLM narrates; the harness never trusts the
narrator.** The signed record is the evidence; the narration is presentation.
An audit pipeline can strip the narration entirely and lose nothing.
