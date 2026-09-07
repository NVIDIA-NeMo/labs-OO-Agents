# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Emit a bounded, parseable envelope for a submit.sh response artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

MAX_ENVELOPE_CHARS = 24_000
INITIAL_OUTPUT_CHARS = 16_000


def _last_json_object_line(text: str) -> dict | None:
    for raw_line in reversed(text.splitlines()):
        line = raw_line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    marker = f"\n... <raw verifier output truncated from {len(text)} chars> ...\n"
    head = max(0, (limit - len(marker)) * 3 // 4)
    tail = max(0, limit - len(marker) - head)
    return text[:head] + marker + text[-tail:]


def capture_response(path: Path, shell_exit_code: int) -> str:
    """Read the exact response artifact and return a shell-safe JSON envelope."""
    raw = path.read_text(errors="replace")
    payload = _last_json_object_line(raw)
    if payload is None:
        envelope = {
            "_capture_error": "submit.sh stdout contained no complete JSON object",
            "shell_exit_code": shell_exit_code,
            "raw_response_path": str(path),
            "raw_response_length": len(raw),
            "raw_response_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "output": _head_tail(raw, INITIAL_OUTPUT_CHARS),
        }
    else:
        full_output = str(payload.get("output", ""))
        envelope = dict(payload)
        envelope["output"] = _head_tail(full_output, INITIAL_OUTPUT_CHARS)
        envelope["raw_response_path"] = str(path)
        envelope["raw_response_length"] = len(raw)
        envelope["raw_response_sha256"] = hashlib.sha256(raw.encode()).hexdigest()
        envelope["raw_output_length"] = len(full_output)
        envelope["raw_output_truncated"] = len(full_output) > len(envelope["output"])

    encoded = json.dumps(envelope, ensure_ascii=True, separators=(",", ":"))
    while len(encoded) > MAX_ENVELOPE_CHARS and envelope.get("output"):
        envelope["output"] = _head_tail(str(envelope["output"]), len(str(envelope["output"])) // 2)
        encoded = json.dumps(envelope, ensure_ascii=True, separators=(",", ":"))
    return encoded


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: capture_submit_response RESPONSE_PATH SHELL_EXIT_CODE")
    print(capture_response(Path(sys.argv[1]), int(sys.argv[2])))


if __name__ == "__main__":
    main()
