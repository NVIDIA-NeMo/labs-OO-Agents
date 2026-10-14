# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run selection persists atomically, updates live readers, and requires auth."""

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from starlette.testclient import TestClient

from nooa.viewer import main, sources


class RunSource:
    version = "1"

    def __init__(self, *, name, options):
        self.name, self.options = name, options

    def prepare_run(self, run_id):
        if run_id == "missing":
            raise KeyError(run_id)
        if run_id == "invalid":
            raise ValueError("Invalid run")
        return {**self.options, "runs": list(dict.fromkeys([*self.options["runs"], run_id]))}

    def list_sessions(self):
        if "unavailable" in self.options["runs"]:
            raise RuntimeError("Unavailable")
        return [{"id": f"{self.name}-{run}", "experiment": run} for run in self.options["runs"]]


@pytest.fixture
def selection(tmp_path, monkeypatch):
    path = tmp_path / "sources.json"
    config = {
        "sources": [
            {
                "name": "example",
                "plugin": "example-source",
                "options": {"runs": ["first"], "token_env": "PRIVATE_TOKEN"},
            }
        ],
        "unrelated": {"keep": True},
    }
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    registry = {"example": RunSource(name="example", options=config["sources"][0]["options"])}
    monkeypatch.setenv("NOOA_VIEWER_SOURCES_CONFIG", str(path))
    monkeypatch.setattr(sources, "configured_sources", lambda: registry)
    entry = SimpleNamespace(name="example-source", load=lambda: RunSource)
    monkeypatch.setattr(sources, "entry_points", lambda **kwargs: [entry])
    return path, registry


def test_selection_updates_without_restart(selection):
    path, registry = selection
    client = TestClient(main.app)
    response = client.post("/api/sources/example/runs", json={"run_id": "second"})
    assert response.status_code == 200
    assert response.json()["added"] is True
    assert response.json()["experiments"] == ["first", "second"]
    assert registry["example"].options["runs"] == ["first", "second"]
    saved = json.loads(path.read_text())
    assert saved["sources"][0]["options"]["runs"] == ["first", "second"]
    assert saved["sources"][0]["options"]["token_env"] == "PRIVATE_TOKEN"
    assert saved["unrelated"] == {"keep": True}
    assert path.stat().st_mode & 0o777 == 0o600
    stamp = path.stat().st_mtime_ns
    assert (
        client.post("/api/sources/example/runs", json={"run_id": "second"}).json()["added"] is False
    )
    assert path.stat().st_mtime_ns == stamp


def test_failed_validation_preserves_config_and_active_source(selection):
    path, registry = selection
    before = path.read_text()
    active = registry["example"]
    client = TestClient(main.app)
    for run, status in [("missing", 404), ("invalid", 400), ("unavailable", 502)]:
        response = client.post("/api/sources/example/runs", json={"run_id": run})
        assert response.status_code == status
        assert path.read_text() == before
        assert registry["example"] is active
    assert client.post("/api/sources/unknown/runs", json={"run_id": "second"}).status_code == 404
    assert client.post("/api/sources/example/runs", json={"run_id": ""}).status_code == 422


def test_write_failure_does_not_activate_source(selection, monkeypatch):
    path, registry = selection
    before, active = path.read_text(), registry["example"]

    def fail(*args):
        raise PermissionError("Not writable")

    monkeypatch.setattr(sources.os, "replace", fail)
    response = TestClient(main.app).post("/api/sources/example/runs", json={"run_id": "second"})
    assert response.status_code == 503
    assert path.read_text() == before
    assert registry["example"] is active
    assert list(path.parent.iterdir()) == [path]


def test_read_only_source_does_not_offer_run_selection(selection, monkeypatch):
    path, _ = selection
    before = path.read_text()
    monkeypatch.setattr(RunSource, "prepare_run", None)
    response = TestClient(main.app).post("/api/sources/example/runs", json={"run_id": "second"})
    assert response.status_code == 405
    assert path.read_text() == before


def test_concurrent_additions_do_not_lose_runs(selection):
    path, registry = selection
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda run: sources.add_run("example", run), ["a", "b", "c", "d"]))
    assert set(registry["example"].options["runs"]) == {"first", "a", "b", "c", "d"}
    assert set(json.loads(path.read_text())["sources"][0]["options"]["runs"]) == {
        "first",
        "a",
        "b",
        "c",
        "d",
    }


def test_remote_run_selection_requires_authorization(selection, monkeypatch):
    path, _ = selection
    before = path.read_text()
    monkeypatch.setenv("NOOA_VIEWER_AUTH_TOKEN", "test-token")
    client = TestClient(main.app, client=("203.0.113.1", 54321))
    assert client.post("/api/sources/example/runs", json={"run_id": "second"}).status_code == 401
    assert path.read_text() == before
    response = client.post(
        "/api/sources/example/runs",
        json={"run_id": "second"},
        headers={"Authorization": "Bearer test-token"},
    )
    assert response.status_code == 200


def test_selection_collision_with_another_source_preserves_config(selection):
    path, registry = selection
    config = json.loads(path.read_text())
    config["sources"].append(
        {"name": "example-b", "plugin": "example-source", "options": {"runs": ["task"]}}
    )
    path.write_text(json.dumps(config))
    registry["example-b"] = RunSource(name="example-b", options={"runs": ["task"]})
    before, active = path.read_text(), registry["example"]
    response = TestClient(main.app).post("/api/sources/example/runs", json={"run_id": "b-task"})
    assert response.status_code == 400
    assert path.read_text() == before
    assert registry["example"] is active
    assert len(sources._catalog()) == 2
