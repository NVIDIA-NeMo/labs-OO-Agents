#!/usr/bin/env python3
"""Score and sign each frozen SunChaser final PoC with the Xeus authority code."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cybergym.server.pocdb import PoCRecord, Session, init_engine
from xeus_cybergym.canonical import canonical_json
from xeus_cybergym.integrations.sunchaser import sign_sunchaser_final_evidence
from xeus_cybergym.ledger import Ed25519Signer


def _private_key() -> tuple[Ed25519PrivateKey, str]:
    encoded = os.environ.get("SUNCHASER_EVIDENCE_SIGNING_SEED")
    key_id = os.environ.get("SUNCHASER_EVIDENCE_KEY_ID", "sunchaser-evaluator-v1")
    if not encoded:
        raise RuntimeError("SUNCHASER_EVIDENCE_SIGNING_SEED is not configured")
    raw = base64.b64decode(encoded, validate=True)
    if len(raw) != 32:
        raise RuntimeError("SUNCHASER_EVIDENCE_SIGNING_SEED must encode exactly 32 bytes")
    return Ed25519PrivateKey.from_private_bytes(raw), key_id


def _matches_hash(record_hash: str, digest: str) -> bool:
    return record_hash == digest or record_hash == f"sha256:{digest}"


def score_run(run_dir: Path, poc_db: Path, output_dir: Path) -> dict[str, object]:
    private, key_id = _private_key()
    signer = Ed25519Signer(private_key=private, key_id=key_id)
    output_dir.mkdir(parents=True, exist_ok=False)

    public_raw = private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    (output_dir / "verifiers.json").write_text(
        json.dumps({key_id: base64.b64encode(public_raw).decode("ascii")}, sort_keys=True) + "\n"
    )

    args_files = sorted(run_dir.rglob("args.json"))
    if not args_files:
        raise RuntimeError(f"no args.json files found under {run_dir}")
    engine = init_engine(poc_db)
    solved = 0
    results: list[dict[str, object]] = []
    with Session(engine) as session:
        for args_path in args_files:
            run = json.loads(args_path.read_text())
            agent_id = run["agent_id"]
            task_id = run["task"]["task_id"]
            final_dir = args_path.parent / "artifacts" / "final_submission"
            selection = json.loads((final_dir / "selection.json").read_text())
            poc_bytes = (final_dir / "poc").read_bytes()
            digest = hashlib.sha256(poc_bytes).hexdigest()
            records = (
                session.query(PoCRecord)
                .filter(PoCRecord.agent_id == agent_id, PoCRecord.task_id == task_id)
                .all()
            )
            matches = [r for r in records if _matches_hash(str(r.poc_hash), digest)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"expected one official DB record for final PoC {task_id}, got {len(matches)}"
                )
            record = matches[0]
            if record.vul_exit_code is None or record.fix_exit_code is None:
                raise RuntimeError(f"official verification is incomplete for final PoC {task_id}")
            evidence, envelope = sign_sunchaser_final_evidence(
                selection=selection,
                poc_bytes=poc_bytes,
                task_id=task_id,
                agent_id=agent_id,
                vul_exit_code=int(record.vul_exit_code),
                fix_exit_code=int(record.fix_exit_code),
                signer=signer,
            )
            name = task_id.replace(":", "_") + ".signed.json"
            (output_dir / name).write_bytes(canonical_json(envelope))
            solved += int(evidence.official_solved)
            results.append(evidence.model_dump(mode="json"))

    summary: dict[str, object] = {
        "schema_version": 1,
        "task_count": len(results),
        "official_solved": solved,
        "official_score": solved / len(results),
        "results": results,
    }
    (output_dir / "summary.json").write_bytes(canonical_json(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--poc-db", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = score_run(args.run_dir.resolve(), args.poc_db.resolve(), args.output_dir.resolve())
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
