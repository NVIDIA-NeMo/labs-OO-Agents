# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "prepare_nooa_bench_bundle.py"
TEMPLATE = ROOT / "packages" / "nooa-bench" / "agent-bundle"


def _load_prepare_module():
    spec = importlib.util.spec_from_file_location("prepare_nooa_bench_bundle", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bundle_uses_glibc_231_compatible_bullseye_images() -> None:
    dockerfile = (TEMPLATE / "Dockerfile").read_text()
    assert "python:3.12-slim-bullseye@sha256:" in dockerfile
    assert "debian:bullseye-slim@sha256:" in dockerfile
    assert "bookworm" not in dockerfile


def test_prepare_context_locks_revision_and_version(tmp_path: Path) -> None:
    output = tmp_path / "context"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--repo-root",
            str(ROOT),
            "--version",
            "9.8.7",
            "--revision",
            "HEAD",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads(completed.stdout)
    descriptor = json.loads((output / "descriptor.json").read_text())
    source_lock = json.loads((output / "source-lock.json").read_text())

    assert descriptor == summary["descriptor"]
    assert source_lock == summary["source_lock"]
    assert descriptor["agent_name"] == "nemotronooagent"
    assert descriptor["agent_version"] == "9.8.7"
    assert descriptor["builder_profile"].endswith("bullseye-v1")
    assert (
        source_lock["source_revision"]
        == subprocess.run(
            ["git", "rev-parse", "HEAD^{commit}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    assert source_lock["uv_lock_digest"].startswith("sha256:")
    assert descriptor["source_lock_digest"] == source_lock["uv_lock_digest"]
    module = _load_prepare_module()
    assert descriptor["fingerprint"] == module._fingerprint(
        output, tuple(descriptor["fingerprint_inputs"])
    )
    assert (output / "source" / "packages" / "nooa-bench" / "pyproject.toml").is_file()
    assert "BUNDLE_AGENT_VERSION=9.8.7" in (output / "bundle.env").read_text()


def test_prepare_context_rejects_noncanonical_version(tmp_path: Path) -> None:
    module = _load_prepare_module()
    with pytest.raises(ValueError, match="canonical X.Y.Z"):
        module.prepare_context(
            repo_root=ROOT,
            output=tmp_path / "context",
            version="0.0.10.dev1",
            revision="HEAD",
            source_ref=None,
        )
