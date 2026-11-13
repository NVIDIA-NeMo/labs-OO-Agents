# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``nooa import-atif``: discovery, session IDs, skipping, and posting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from nooa_cli.commands import import_atif


def _trajectory(trajectory_id: str | None) -> dict:
    value = {
        "schema_version": "ATIF-v1.7",
        "agent": {"name": "a", "version": "1"},
        "steps": [
            {"step_id": 1, "source": "user", "message": "hi"},
            {"step_id": 2, "source": "agent", "message": "hello"},
        ],
    }
    if trajectory_id:
        value["trajectory_id"] = trajectory_id
    return value


@pytest.fixture
def viewer(monkeypatch):
    posted: list[dict] = []
    existing: set[str] = set()
    monkeypatch.setattr(import_atif, "check_endpoint_reachable", lambda _e: True)
    monkeypatch.setattr(import_atif, "session_exists", lambda _e, sid: sid in existing)
    monkeypatch.setattr(
        import_atif, "post_traces_batch_with_retry", lambda _e, bodies, **_k: posted.extend(bodies)
    )
    monkeypatch.setattr(import_atif, "sync_ingest", lambda _e: None)
    return posted, existing


def _session_ids(bodies: list[dict]) -> list[str]:
    out = []
    for body in bodies:
        attrs = body["resourceSpans"][0]["resource"]["attributes"]
        out.append(next(a["value"]["stringValue"] for a in attrs if a["key"] == "session.id"))
    return out


def test_imports_json_and_jsonl_from_a_directory(tmp_path: Path, viewer):
    posted, existing = viewer
    (tmp_path / "trial").mkdir()
    (tmp_path / "trial" / "trajectory.json").write_text(json.dumps(_trajectory(None)))
    (tmp_path / "trial" / "result.json").write_text(json.dumps({"reward": 1}))
    (tmp_path / "all.jsonl").write_text(
        "\n".join(json.dumps(_trajectory(t)) for t in ("A/r0", "A/r0", "B/r1")) + "\n"
    )
    existing.add("p-B-r1")
    result = CliRunner().invoke(
        import_atif.command,
        [str(tmp_path), "--session-prefix", "p-", "--batch-id", "b", "--experiment", "e"],
    )
    assert result.exit_code == 0, result.output
    assert sorted(_session_ids(posted)) == ["p-A-r0", "p-A-r0-2", "p-trial-trajectory"]
    assert "Imported 3" in result.output and "1 already present" in result.output
    resource = posted[0]["resourceSpans"][0]["resource"]["attributes"]
    assert {"key": "batch_id", "value": {"stringValue": "b"}} in resource


def test_rejects_a_file_that_is_not_atif(tmp_path: Path, viewer):
    path = tmp_path / "x.json"
    path.write_text(json.dumps({"resourceSpans": []}))
    result = CliRunner().invoke(import_atif.command, [str(path)])
    assert result.exit_code == 1
    assert "not an ATIF trajectory" in result.output
