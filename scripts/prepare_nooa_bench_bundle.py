# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare an immutable Scaled Evals Docker context for ``nooa-bench``."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from hashlib import sha256
from pathlib import Path
from typing import Any

AGENT_NAME = "nemotronooagent"
PLATFORM = "linux/amd64"
RUNTIME_ABI = "glibc"
BUNDLE_LAYOUT_VERSION = 1
BUILDER_PROFILE = "python-public-source-uv-lock-bullseye-v1"
SOURCE_REPOSITORY = "https://github.com/NVIDIA-NeMo/labs-OO-Agents.git"
TEMPLATE_FILES = (
    "Dockerfile",
    "collect-python-libs",
    "copy-agent",
    "nemo-harbor",
    "validate-installed",
)
EXECUTABLE_FILES = frozenset(
    {"collect-python-libs", "copy-agent", "nemo-harbor", "validate-installed"}
)
VERSION_PATTERN = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


def _run_git(repo_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _directory_digest(root: Path) -> str:
    digest = sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def _fingerprint(context: Path, inputs: tuple[str, ...]) -> str:
    digest = sha256()
    for relative in inputs:
        path = context / relative
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def _archive_source(repo_root: Path, revision: str, destination: Path) -> None:
    destination.mkdir(parents=True)
    with tempfile.NamedTemporaryFile(suffix=".tar") as archive:
        subprocess.run(
            ["git", "archive", "--format=tar", "--output", archive.name, revision],
            cwd=repo_root,
            check=True,
        )
        with tarfile.open(archive.name) as stream:
            stream.extractall(destination, filter="data")


def _write_bundle_env(path: Path, descriptor: dict[str, Any]) -> None:
    values = {
        "BUNDLE_AGENT_NAME": descriptor["agent_name"],
        "BUNDLE_AGENT_VERSION": descriptor["agent_version"],
        "BUNDLE_AGENT_PLATFORM": descriptor["platform"],
        "BUNDLE_AGENT_RUNTIME_ABI": descriptor["runtime_abi"],
        "BUNDLE_LAYOUT_VERSION": str(descriptor["bundle_layout_version"]),
        "BUNDLE_BUILDER_PROFILE": descriptor["builder_profile"],
        "BUNDLE_SOURCE_LOCK_DIGEST": descriptor["source_lock_digest"],
        "BUNDLE_FINGERPRINT": descriptor["fingerprint"],
    }
    lines = [f"{name}={shlex.quote(value)}" for name, value in values.items()]
    path.write_text("\n".join(lines) + "\n")


def prepare_context(
    *,
    repo_root: Path,
    output: Path,
    version: str,
    revision: str,
    source_ref: str | None,
) -> dict[str, Any]:
    if VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"version must be canonical X.Y.Z, got {version!r}")

    repo_root = repo_root.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"output path already exists: {output}")

    source_revision = _run_git(repo_root, "rev-parse", f"{revision}^{{commit}}")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise ValueError(f"could not resolve a full source revision: {source_revision!r}")
    if source_ref is not None:
        if source_ref != f"v{version}":
            raise ValueError(f"source ref {source_ref!r} does not match v{version}")
        ref_revision = _run_git(repo_root, "rev-parse", f"{source_ref}^{{commit}}")
        if ref_revision != source_revision:
            raise ValueError(
                f"source ref {source_ref} resolves to {ref_revision}, not {source_revision}"
            )

    template = repo_root / "packages" / "nooa-bench" / "agent-bundle"
    output.mkdir(parents=True)
    for name in TEMPLATE_FILES:
        destination = output / name
        shutil.copy2(template / name, destination)
        destination.chmod(0o755 if name in EXECUTABLE_FILES else 0o644)

    source = output / "source"
    _archive_source(repo_root, source_revision, source)
    lock_digest = f"sha256:{sha256((source / 'uv.lock').read_bytes()).hexdigest()}"
    source_lock = {
        "schema_version": 1,
        "source_repository": SOURCE_REPOSITORY,
        "source_tag": source_ref or source_revision,
        "source_revision": source_revision,
        "source_tree_digest": _directory_digest(source),
        "uv_lock_digest": lock_digest,
    }
    _write_json(output / "source-lock.json", source_lock)
    fingerprint_inputs = ("source-lock.json", *TEMPLATE_FILES)
    descriptor = {
        "schema_version": 1,
        "agent_name": AGENT_NAME,
        "agent_version": version,
        "entrypoint": "bin/nemo-harbor",
        "platform": PLATFORM,
        "runtime_abi": RUNTIME_ABI,
        "bundle_layout_version": BUNDLE_LAYOUT_VERSION,
        "builder_profile": BUILDER_PROFILE,
        # The existing nemotronooagent Scaled Evals contract uses the locked
        # uv.lock digest as its source-lock identity.
        "source_lock_digest": lock_digest,
        "fingerprint": _fingerprint(output, fingerprint_inputs),
        "fingerprint_inputs": list(fingerprint_inputs),
    }
    _write_json(output / "descriptor.json", descriptor)
    _write_bundle_env(output / "bundle.env", descriptor)
    return {"descriptor": descriptor, "source_lock": source_lock}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="canonical X.Y.Z version")
    parser.add_argument("--revision", default="HEAD", help="Git commit to archive")
    parser.add_argument(
        "--source-ref",
        help="optional canonical vX.Y.Z tag; must resolve to --revision",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
    )
    return parser.parse_args()


def main() -> None:
    arguments = _parse_args()
    result = prepare_context(
        repo_root=arguments.repo_root,
        output=arguments.output,
        version=arguments.version,
        revision=arguments.revision,
        source_ref=arguments.source_ref,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
