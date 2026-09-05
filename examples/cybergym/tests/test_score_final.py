from __future__ import annotations

import base64
import hashlib
import json
import os

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cybergym.server.pocdb import PoCRecord, Session, init_engine
from scripts.score_final import score_run
from xeus_cybergym.canonical import canonical_json
from xeus_cybergym.ledger import Ed25519Verifier, SignedEnvelope


def test_score_run_matches_only_frozen_poc_and_writes_verifiable_evidence(tmp_path, monkeypatch):
    seed = os.urandom(32)
    monkeypatch.setenv("SUNCHASER_EVIDENCE_SIGNING_SEED", base64.b64encode(seed).decode())
    monkeypatch.setenv("SUNCHASER_EVIDENCE_KEY_ID", "test-key")

    run_dir = tmp_path / "run"
    log_dir = run_dir / "logs" / "task"
    final_dir = log_dir / "artifacts" / "final_submission"
    final_dir.mkdir(parents=True)
    poc = b"official-final"
    digest = hashlib.sha256(poc).hexdigest()
    (final_dir / "poc").write_bytes(poc)
    (final_dir / "selection.json").write_text(
        json.dumps({"submission_number": 3, "sha256": digest, "byte_length": len(poc)})
    )
    (log_dir / "args.json").write_text(
        json.dumps({"agent_id": "agent-1", "task": {"task_id": "arvo:1"}})
    )

    db_path = tmp_path / "poc.db"
    engine = init_engine(db_path)
    with Session(engine) as session:
        session.add(
            PoCRecord(
                agent_id="agent-1",
                task_id="arvo:1",
                poc_id="poc-1",
                poc_hash=digest,
                poc_length=len(poc),
                vul_exit_code=139,
                fix_exit_code=0,
            )
        )
        session.commit()

    output_dir = run_dir / "official_evidence"
    summary = score_run(run_dir, db_path, output_dir)

    assert summary["official_solved"] == 1
    envelope = SignedEnvelope.model_validate_json((output_dir / "arvo_1.signed.json").read_bytes())
    keys = json.loads((output_dir / "verifiers.json").read_text())
    public = Ed25519PublicKey.from_public_bytes(base64.b64decode(keys["test-key"]))
    payload = Ed25519Verifier({"test-key": public}).verify(envelope)
    assert json.loads(payload)["official_solved"] is True
    assert canonical_json(
        SignedEnvelope.model_validate_json(canonical_json(envelope))
    ) == canonical_json(envelope)
