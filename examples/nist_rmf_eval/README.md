<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- Contributed by CSOAI (csoai.org) — Council for the Safety of Artificial Intelligence. -->

# NIST AI RMF Evaluator

**Python decides. The LLM only narrates.**

`NistRmfEvaluator` is a NOOA agent that evaluates a natural-language AI
deployment scenario against a built-in corpus of real NIST AI RMF 1.0
subcategories across the four Core functions (GOVERN, MAP, MEASURE, MANAGE)
and emits a **signed, tamper-evident verdict record** — entirely
deterministically. An optional generation method (`explain_verdict`) lets an
LLM narrate the verdict in plain English, but the verdict itself is produced
by ordinary Python and never depends on a model.

Contributed by [CSOAI](https://csoai.org) (Council for the Safety of Artificial
Intelligence, UK) to the Open Secure AI Alliance's **evaluations and
benchmarks** lane. This is the voluntary-framework companion to the
statute-anchored [`gspc_provision_eval/`](../gspc_provision_eval/) evaluator
(PR #75) and follows the same architecture exactly.

## What it demonstrates

- **Subcategory-anchored.** A subcategory is "matched" only when its literal
  anchor terms appear in the scenario, or when the framework itself declares
  it a prerequisite of a matched subcategory (MANAGE-1.3's own text addresses
  risks "as identified by the MAP function", so a matched MANAGE-1.3 attaches
  MAP-5.1). No vibes, no model inference.
- **No fabricated precision.** Anything not matched is `UNMEASURED` — the
  evaluator says "I don't know" instead of guessing.
- **Signed and tamper-evident.** Each verdict record carries
  `scenario_hash` (SHA-256) and `signature` (HMAC-SHA256 over the canonical
  JSON of the record, keyed with a local `0600` key file). Flip one field and
  `verify_verdict` returns `False`; change one character of the scenario and
  the hash changes.
- **Deterministic core / LLM-narrated split.** `match_subcategories`,
  `build_verdict`, and `verify_verdict` are ordinary Python (SW1 in NOOA
  terms — tools the LLM can call but not influence). Only `explain_verdict`
  has an ellipsis body: the docstring instructs the model to narrate the
  signed record without altering it. An auditor can replay the deterministic
  layer byte-for-byte; the narration is presentation, not evidence.

## The verdict record

```json
{
  "scenario_hash": "sha256 of normalized scenario text",
  "subcategories_matched": ["MANAGE-2.4", "MAP-1.1", "MEASURE-1.1", "MEASURE-2.7"],
  "verdict": "PARTIAL_FUNCTION_COVERAGE",
  "evidence": {"MAP-1.1": ["intended purpose", "deployment context"],
               "MEASURE-2.7": ["red-team", "adversarial test"], "...": ["..."]},
  "signature": "HMAC-SHA256 over canonical JSON of the fields above"
}
```

Verdict rules: matches spanning all four AI RMF functions →
`FULL_FUNCTION_COVERAGE`; otherwise any match → `PARTIAL_FUNCTION_COVERAGE`;
nothing matched → `UNMEASURED`. Coverage of a function is evidence that a
scenario mentions the practice, not that the practice is adequate — the
evaluator measures *anchored evidence*, not quality.

## Strict narration (audit-pipeline mode)

By default, `explain_verdict` calls the LLM regardless of the verdict — if
the deterministic core returned `UNMEASURED`, the narration simply says
"no subcategory matched." For pipelines that want a hard refusal instead,
pass `strict_narration=True` to the constructor:

```python
agent = NistRmfEvaluator(
    llm=...,
    signing_key_path="nist_signing_key.hex",
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
uv run python examples/nist_rmf_eval/nist_rmf_evaluator.py
```

With `OPENAI_API_KEY` or `NVIDIA_API_KEY` set, the script additionally calls
`explain_verdict` for a live LLM narration of the signed verdict (the verdict
itself is unchanged either way).

Run the tests (hermetic — they use `FakeLLMClient`, no network):

```bash
uv run pytest examples/nist_rmf_eval/test_nist_rmf_evaluator.py
```

## Honest scope note

The built-in corpus is **8 subcategories** (two per function) chosen for
demonstration — GOVERN-1.1, GOVERN-2.1, MAP-1.1, MAP-5.1, MEASURE-1.1,
MEASURE-2.7, MANAGE-1.3, MANAGE-2.4. Every ID is a real NIST AI RMF 1.0
subcategory (NIST AI 100-1, Appendix A) and each title is a faithful
plain-language restatement of the published text. It is **not** the full
72-subcategory AI RMF crosswalk, and keyword anchoring is a demonstration
matcher, not a conformity assessment — a `FULL_FUNCTION_COVERAGE` verdict
means the scenario *mentions* practices in all four functions, not that the
system is AI RMF-conformant. The point of the example is the architecture —
deterministic, signed, anchor-based evaluation with LLM narration on top —
which scales to the full crosswalk without design changes.

## Files

| File | Purpose |
|---|---|
| `nist_rmf_evaluator.py` | The `NistRmfEvaluator` agent (corpus, deterministic core, `explain_verdict`) |
| `test_nist_rmf_evaluator.py` | Hermetic pytest suite for the deterministic core |
| `conftest.py` | Makes the example importable when pytest runs this directory |
