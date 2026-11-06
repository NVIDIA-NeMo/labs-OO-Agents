# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise a running AionCore with real NOOA tools and a scripted provider."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = Path(__file__).with_name("launch.sh")


def request(base: str, method: str, route: str, body: Any = None) -> Any:
    data = None if body is None else json.dumps(body).encode()
    req = Request(
        base + route,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=45) as response:
            result = json.load(response)
    except HTTPError as error:
        raise RuntimeError(
            f"{method} {route}: HTTP {error.code}: {error.read().decode()}"
        ) from error
    if result.get("success") is not True:
        raise RuntimeError(f"{method} {route}: {result}")
    return result["data"]


def messages(base: str, conversation_id: str) -> list[dict[str, Any]]:
    result = request(
        base, "GET", f"/api/conversations/{conversation_id}/messages?limit=100&content_mode=full"
    )
    assert not result["has_more_before"] and not result["has_more_after"], result
    return result["items"]


def verify_turn(rows: list[dict[str, Any]]) -> dict[str, int]:
    failures = [row for row in rows if row.get("status") == "error"]
    assert not failures, failures
    cards = [row for row in rows if row["type"] == "acp_tool_call"]
    kinds = Counter(row["content"]["update"]["kind"] for row in cards)
    # AionCore maps ACP's "other" kind (NOOA's Python execution) to "execute".
    assert kinds == {"edit": 1, "execute": 2}, kinds
    for card in cards:
        assert card["status"] == "finish", card
        assert card["content"]["update"]["status"] == "completed", card
    edit = next(
        card["content"]["update"] for card in cards if card["content"]["update"]["kind"] == "edit"
    )
    assert any(
        item["type"] == "diff" and "NOOA AionUi" in item["new_text"] for item in edit["content"]
    ), edit
    assert any(card["content"]["update"]["title"] == "Ran Python" for card in cards)
    terminal = next(
        card for card in cards if "command" in card["content"]["update"].get("raw_input", {})
    )
    assert "NOOA_AION_ASSERTION_PASSED" in json.dumps(terminal), terminal
    assert terminal["content"]["update"]["raw_output"]["exit_code"] == 0, terminal
    texts = [row for row in rows if row["type"] == "text" and row["position"] == "left"]
    assert len(texts) == 2, texts
    assert all(row["status"] == "finish" for row in texts), texts
    assert any("Verified:" in row["content"]["content"] for row in texts), texts
    return {
        "assistant_text_blocks": len(texts),
        "completed_tool_cards": len(cards),
        "file_diffs": 1,
    }


def run_turn(
    base: str, conversation_id: str, content: str, run_dir: Path, index: int
) -> dict[str, Any]:
    previous_ids = {row["id"] for row in messages(base, conversation_id)}
    started = time.monotonic()
    accepted = request(
        base, "POST", f"/api/conversations/{conversation_id}/messages", {"content": content}
    )
    deadline = started + 60
    while time.monotonic() < deadline:
        detail = request(base, "GET", f"/api/conversations/{conversation_id}")
        rows = messages(base, conversation_id)
        current = [row for row in rows if row["id"] not in previous_ids]
        errors = [row for row in current if row.get("status") == "error"]
        if errors:
            (run_dir / f"turn-{index}-messages.json").write_text(json.dumps(rows, indent=2) + "\n")
            raise AssertionError(errors)
        if current and detail["runtime"]["state"] == "idle":
            (run_dir / f"turn-{index}-messages.json").write_text(json.dumps(rows, indent=2) + "\n")
            counts = verify_turn(current)
            return {
                **counts,
                "turn_id": accepted["turn_id"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
        time.sleep(0.5)
    raise TimeoutError(f"Turn {accepted['turn_id']} did not finish within 60 seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="http://127.0.0.1:25812")
    parser.add_argument("--ui", default="http://127.0.0.1:25811")
    args = parser.parse_args()
    started = time.monotonic()
    run_dir = ROOT / "tmp" / "aion-ui-spike" / f"aion-run-{uuid4().hex[:12]}"
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True)
    env = {"NEMO_OO_USER_DIR": str(run_dir / "user-config"), "TMPDIR": str(run_dir)}
    probe = request(
        args.backend,
        "POST",
        "/api/agents/custom/try-connect",
        {
            "command": str(LAUNCHER),
            "acp_args": ["--demo"],
            "env": env,
        },
    )
    assert probe["step"] == "success", probe
    agent = request(
        args.backend,
        "POST",
        "/api/agents/custom",
        {
            "name": "NOOA scripted spike",
            "command": str(LAUNCHER),
            "icon": "🧪",
            "args": ["--demo"],
            "env": [{"name": name, "value": value} for name, value in env.items()],
        },
    )
    conversation = request(
        args.backend,
        "POST",
        "/api/conversations",
        {
            "type": "acp",
            "name": "NOOA × AionUi — scripted tool demo",
            "extra": {
                "agent_id": agent["id"],
                "backend": "custom",
                "agent_name": "NOOA scripted spike",
                "workspace": str(workspace),
            },
        },
    )
    conversation_id = conversation["id"]
    url = f"{args.ui}/#/conversation/{conversation_id}"
    (run_dir / "conversation.json").write_text(json.dumps(conversation, indent=2) + "\n")
    print(
        json.dumps({"conversation_id": conversation_id, "url": url, "run_directory": str(run_dir)}),
        flush=True,
    )
    turns = [run_turn(args.backend, conversation_id, "Run the scripted NOOA demo.", run_dir, 1)]
    artifact = workspace / "nooa_aion_demo.py"
    assert "DEMO_TURN = 1" in artifact.read_text()
    turns.append(run_turn(args.backend, conversation_id, "Run it a second time.", run_dir, 2))
    assert "DEMO_TURN = 2" in artifact.read_text()
    before_restart = request(args.backend, "GET", f"/api/conversations/{conversation_id}")
    restarted = request(
        args.backend, "POST", f"/api/conversations/{conversation_id}/runtime/restart"
    )
    turns.append(
        run_turn(args.backend, conversation_id, "Continue after runtime restart.", run_dir, 3)
    )
    assert "DEMO_TURN = 1" in artifact.read_text(), (
        "Scripted provider counter should reset after process restart"
    )
    after_restart = request(args.backend, "GET", f"/api/conversations/{conversation_id}")
    persisted = messages(args.backend, conversation_id)
    session_ids = {
        row["content"]["session_id"] for row in persisted if row["type"] == "acp_tool_call"
    }
    assert len(session_ids) == 1, "Aion must resume the original NOOA session after restart"
    summary = {
        "result": "passed",
        "scope": "AionCore REST → NOOA ACP → real CodeAct/file/shell tools; scripted provider",
        "live_llm_calls": 0,
        "agent_id": agent["id"],
        "conversation_id": conversation_id,
        "conversation_url": url,
        "probe": probe,
        "completed_turns": len(turns),
        "turns": turns,
        "runtime_restarts": 1,
        "acp_session_id": next(iter(session_ids)),
        "restart_response": restarted,
        "artifact": str(artifact),
        "before_restart_extra": before_restart["extra"],
        "after_restart_extra": after_restart["extra"],
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "run_directory": str(run_dir),
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
