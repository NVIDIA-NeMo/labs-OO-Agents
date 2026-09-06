from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
from nooa_cybergym import run


def test_missing_required_image_fails_before_run(monkeypatch):
    class Missing(Exception):
        pass

    client = SimpleNamespace(
        images=SimpleNamespace(get=lambda image: (_ for _ in ()).throw(Missing()))
    )
    monkeypatch.setattr(run, "ImageNotFound", Missing)

    with pytest.raises(RuntimeError, match="required runner image is not local"):
        run.require_local_image(client, "runner:tag", role="runner")


def test_internal_route_probe_uses_runner_image_and_real_network():
    calls = []

    class Containers:
        def run(self, image, **kwargs):
            calls.append((image, kwargs))

    client = SimpleNamespace(containers=Containers())
    env = {"HTTP_PROXY": "http://cybergym-proxy:3128"}

    run.preflight_internal_route(
        client,
        image="runner:tag",
        network="cybergym-internal",
        env=env,
        server="http://server:8666",
    )

    image, kwargs = calls[0]
    assert image == "runner:tag"
    assert kwargs["network"] == "cybergym-internal"
    assert kwargs["environment"] == env
    assert kwargs["remove"] is True
    assert "http://server:8666/docs" in kwargs["command"][2]


def test_timeout_budget_requires_finalization_and_shutdown_margin():
    with pytest.raises(ValueError, match="timeout budget is unsafe"):
        run.validate_timeout_budget(
            hard_timeout=1800,
            soft_timeout=1680,
            finalization_grace=120,
            tracing_shutdown_timeout=30,
            outer_margin=60,
        )

    run.validate_timeout_budget(
        hard_timeout=1800,
        soft_timeout=1560,
        finalization_grace=120,
        tracing_shutdown_timeout=30,
        outer_margin=60,
    )


def test_hard_timeout_recovers_smallest_persisted_verified_crash(tmp_path):
    artifacts = tmp_path / "artifacts"
    candidates = artifacts / "candidates"
    candidates.mkdir(parents=True)
    (candidates / "submission_1.poc").write_bytes(b"larger-crash")
    (candidates / "submission_2.poc").write_bytes(b"tiny")
    (candidates / "submission_3.poc").write_bytes(b"safe")
    records = [
        {
            "submission_number": 1,
            "status": "crashed",
            "source_agent": "finder-a",
            "source_model": "model-a",
            "hypothesis": "first crash",
            "kind": "crash",
            "cluster_key": "asan:a",
        },
        {
            "submission_number": 2,
            "status": "crashed",
            "source_agent": "finder-b",
            "source_model": "model-b",
            "hypothesis": "small deterministic crash",
            "kind": "crash",
            "cluster_key": "asan:b",
        },
        {
            "submission_number": 3,
            "status": "no_crash",
            "hypothesis": "safe",
            "kind": "no_crash",
            "cluster_key": "no-crash",
        },
    ]
    (artifacts / "submissions.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records)
    )

    recovered = run.recover_timeout_final(tmp_path)

    assert recovered is not None
    assert (artifacts / "final_submission" / "poc").read_bytes() == b"tiny"
    selection = json.loads(
        (artifacts / "final_submission" / "selection.json").read_text()
    )
    assert selection["submission_number"] == 2
    assert selection["sha256"] == hashlib.sha256(b"tiny").hexdigest()
    assert selection["cluster_key"] == "asan:b"
    assert "outer hard timeout" in selection["selection_reason"].lower()
    assert (artifacts / "output.txt").is_file()


def test_hard_timeout_recovery_ignores_noncrash_and_incomplete_records(tmp_path):
    artifacts = tmp_path / "artifacts"
    candidates = artifacts / "candidates"
    candidates.mkdir(parents=True)
    (candidates / "submission_1.poc").write_bytes(b"not-a-crash")
    (artifacts / "submissions.jsonl").write_text(
        json.dumps(
            {
                "submission_number": 1,
                "status": "no_crash",
                "kind": "no_crash",
                "cluster_key": "safe",
            }
        )
        + "\n"
        + "{interrupted-json"
    )

    assert run.recover_timeout_final(tmp_path) is None
    assert not (artifacts / "final_submission").exists()
