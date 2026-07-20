# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real-time trace exporter that writes the REAL LLM messages to the run logs.

Added to the launcher's tracing exporters, it receives each LLM-call span as it
completes (~5s batch flush) and writes the actual prompt + response into
``agent_logs/nemo/team_leader/messages/step_NNN_round_NN_{user,assistant}.md`` —
the exact path/format the arc_league web viewer (``analyze/messages``) and the
``viewer`` read. So the real conversation is visible LIVE in both
viewers and stored in the logs, replacing the synthetic reconstruction.

Each call is mapped to the env-step in progress when it started, using the
harness-written ``env_step`` event timestamps (also live).
"""

from __future__ import annotations

import json
from pathlib import Path

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

LLM_SPANS = {"aresponses", "responses", "acompletion", "completion"}


def _indexed(at: dict, kind: str) -> list[tuple[str, str]]:
    """(role, content[+tool_call]) for llm.<kind>_messages.*, in index order."""
    prefix = f"llm.{kind}_messages."
    grp: dict[int, dict] = {}
    for k, v in at.items():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        i = int(rest.split(".", 1)[0])
        grp.setdefault(i, {})[rest.split(".", 1)[1]] = v
    out = []
    for i in sorted(grp):
        m = grp[i]
        content = m.get("message.role", "?"), str(m.get("message.content", "") or "")
        role, body = content
        tcs = [
            (("name" if k.endswith("function.name") else "args"), str(v))
            for k, v in m.items()
            if "tool_calls" in k
            and (k.endswith("function.name") or k.endswith("function.arguments"))
        ]
        if tcs:
            body = (body + "\n[tool_call] " + " ".join(f"{a}={b}" for a, b in tcs)).strip()
        out.append((role, body))
    return out


class ViewerMessageExporter(SpanExporter):
    """Writes real LLM prompts/responses to the viewer's messages/ dir, live."""

    def __init__(self, run_dir: str | Path, team: str = "nemo", role: str = "team_leader"):
        self.msgs_dir = Path(run_dir) / "agent_logs" / team / role / "messages"
        self.events_path = Path(run_dir) / "agent_logs" / team / role / "events.jsonl"
        self.msgs_dir.mkdir(parents=True, exist_ok=True)
        self._round_by_step: dict[int, int] = {}
        self._cleared = False

    def _step_for(self, call_start_s: float) -> int:
        cur = 0
        if not self.events_path.exists():
            return cur
        try:
            for line in self.events_path.read_text().splitlines():
                e = json.loads(line)
                if e.get("event") == "env_step":
                    t = float(e.get("unix_time_s", 0))
                    if t <= call_start_s:
                        cur = int(e.get("step", 0))
                    else:
                        break
        except (OSError, json.JSONDecodeError, ValueError):
            pass
        return cur

    def export(self, spans) -> SpanExportResult:
        # On first real batch, clear any synthetic placeholders once.
        if not self._cleared:
            for f in self.msgs_dir.glob("step_*.md"):
                f.unlink(missing_ok=True)
            self._cleared = True
        for sp in spans:
            if sp.name not in LLM_SPANS:
                continue
            at = dict(sp.attributes or {})
            if not any(k.startswith("llm.input_messages") for k in at):
                continue
            step = self._step_for((sp.start_time or 0) / 1e9)
            rnd = self._round_by_step.get(step, 0)
            self._round_by_step[step] = rnd + 1
            hdr = (
                f"<!-- REAL LLM I/O (live) step={step} round={rnd} "
                f"prompt_tok={at.get('llm.token_count.prompt', '?')} "
                f"out_tok={at.get('llm.token_count.completion', '?')} "
                f"reasoning_tok={at.get('llm.token_count.completion_details.reasoning', '?')} -->\n\n"
            )
            user = "\n\n".join(f"### {r}\n\n{c}" for r, c in _indexed(at, "input")) or "(no prompt)"
            asst = (
                "\n\n".join(f"### {r}\n\n{c}" for r, c in _indexed(at, "output"))
                or "(tool_call / no text)"
            )
            base = self.msgs_dir / f"step_{step:03d}_round_{rnd:02d}"
            try:
                (base.with_name(base.name + "_user.md")).write_text(hdr + user)
                (base.with_name(base.name + "_assistant.md")).write_text(hdr + asst)
            except OSError:
                return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True
