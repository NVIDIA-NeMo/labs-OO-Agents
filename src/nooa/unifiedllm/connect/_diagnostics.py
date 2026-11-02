# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local installation references and bounded, non-secret failure observations."""

import json
import os
import re
from importlib import metadata
from pathlib import Path


def scrub_report(value, *, api_key=None, api_key_env=None):
    """Remove active credential values throughout a report, including mapping keys."""
    secrets = tuple(s for s in (api_key, os.environ.get(api_key_env or "")) if s)

    def scrub(item):
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, "[redacted]")
            return item
        if isinstance(item, dict):
            return {scrub(k): scrub(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [scrub(v) for v in item]
        return item

    return scrub(value)


def installation_context():
    from nooa import _version

    package = Path(_version.__file__).resolve().parent
    relative = (
        "skills/nooa-model-configuration/SKILL.md",
        "docs/model-connect.md",
        "docs/model-configuration.md",
    )
    root = next(
        (p for p in package.parents if all((p / name).is_file() for name in relative)), None
    )
    result = {"package_location": str(package), "source_root": str(root) if root else None}
    if root:
        result["reference_paths"] = [str(root / name) for name in relative]
        result["reference_guidance"] = "Read these absolute paths, local to the running machine."
        return result
    result["reference_guidance"] = (
        "This installation does not include the skill and companion docs."
    )
    revision = None
    try:
        direct = json.loads(metadata.distribution("nooa").read_text("direct_url.json") or "{}")
        commit = direct.get("vcs_info", {}).get("commit_id", "")
        if re.fullmatch(r"[0-9a-f]{40,64}", commit):
            revision = commit
    except (metadata.PackageNotFoundError, ValueError, TypeError, AttributeError):
        pass
    if revision is None and re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", _version.__version__):
        revision = "v" + _version.__version__
    if revision:
        result["source_revision"] = revision
        result["reference_command"] = (
            "git clone --no-checkout https://github.com/NVIDIA-NeMo/labs-OO-Agents.git nooa-reference"
            f" && git -C nooa-reference checkout --detach {revision}"
        )
        result["reference_guidance"] += (
            " Use the recorded commit or matching release tag below, then read "
            + ", ".join(relative)
            + " under that checkout. If the tag is unavailable, locate the release commit; do not substitute main."
        )
    else:
        result["reference_guidance"] += (
            " The matching source revision is unknown; obtain it from the installer before cloning. Do not use main."
        )
    return result


def timeout_details(exc, *, deadline_expired=False):
    """Inspect exception types, not provider messages (which may contain secrets)."""
    chain = []
    current = exc
    while current is not None and len(chain) < 6 and type(current).__name__ not in chain:
        chain.append(type(current).__name__)
        current = current.__cause__ or current.__context__
    if deadline_expired or any("timeout" in name.lower() for name in chain):
        return {
            "outcome": "not_confirmed",
            "error_chain": chain,
            "timeout_kind": "probe_deadline"
            if deadline_expired
            else next(name for name in reversed(chain) if "timeout" in name.lower()),
            "reason": "Timed out; server receipt and completion are unknown",
        }
    return {}
