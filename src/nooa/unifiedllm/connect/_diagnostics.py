# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic helpers for Connect reports.

- ``scrub_report`` removes API key values from a report before it is shown.
- ``installation_context`` tells the user (or an assisting agent) where the
  Connect documentation for this installation can be found.
- ``timeout_details`` describes a timed-out test call using exception class
  names only, never exception messages, which may contain secrets.
"""

import json
import os
import re
from importlib import metadata
from pathlib import Path


def scrub_report(value, *, api_key=None, api_key_env=None):
    """Return ``value`` with every occurrence of the API key replaced by ``[redacted]``.

    The key is ``api_key`` and/or the current value of the environment variable
    named ``api_key_env``. Strings are searched recursively through dicts
    (keys and values), lists and tuples; tuples come back as lists.
    """
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
    """Describe where to find the Connect skill and docs for this installation.

    If the running package sits inside a source checkout that contains the
    skill and docs files, their absolute paths are returned. Otherwise the
    result names the exact source revision (the installed commit, or the
    release tag for a release version) and a ``git`` command to fetch it, so
    readers never fall back to documentation from ``main`` that may not match
    the installed code.
    """
    from nooa import _version

    package = Path(_version.__file__).resolve().parent
    relative = (
        "skills/nooa-connect/SKILL.md",
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
    """Return timeout details for ``exc``, or ``{}`` if it was not a timeout.

    A failure counts as a timeout if Connect's own deadline expired
    (``deadline_expired``) or any exception class name in the cause/context
    chain contains "timeout". Only class names are inspected, because provider
    messages may contain secrets. The chain walk stops after 6 exceptions or
    when a class name repeats.
    """
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
