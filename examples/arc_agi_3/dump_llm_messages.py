# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump the REAL LLM messages (prompts, responses, tool calls) for a run.

The agent_logs/.../messages/*.md files are a synthetic reconstruction for the
viewer's analyze tab — NOT the real LLM I/O. The real prompts and
responses live in the OTLP trace files (<run>/traces/*.jsonl): each LLM call is
an `aresponses`/`responses` span carrying `llm.input_messages.*` (the exact
prompt sent) and `llm.output_messages.*` (the model's reply + tool calls).

This reads those spans, in call order, and writes a readable transcript:
one file per LLM call under <run>/llm_transcript/, plus a combined file.

    python examples/arc_agi_3/dump_llm_messages.py <run_dir> [--full]

--full keeps the entire prompt (default truncates each input message to 4000
chars so the transcript stays browsable; tool-call code is never truncated).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _attrs(span: dict) -> dict:
    return {a["key"]: list(a["value"].values())[0] for a in span.get("attributes", [])}


def _messages(at: dict, kind: str) -> list[tuple[str, str]]:
    """Collect (role, content) for input/output messages, in index order."""
    out: dict[int, dict] = {}
    prefix = f"llm.{kind}_messages."
    for k, v in at.items():
        if not k.startswith(prefix):
            continue
        rest = k[len(prefix) :]
        idx = int(rest.split(".", 1)[0])
        field = rest.split(".", 1)[1]
        out.setdefault(idx, {})[field] = v
    msgs = []
    for i in sorted(out):
        m = out[i]
        role = m.get("message.role", "?")
        content = m.get("message.content", "")
        # tool calls (the model's generated code / tool invocation)
        tcs = []
        for kk, vv in m.items():
            if "tool_calls" in kk and kk.endswith("function.name"):
                tcs.append(("name", vv))
            elif "tool_calls" in kk and kk.endswith("function.arguments"):
                tcs.append(("args", vv))
        if tcs:
            content = (content or "") + "\n[tool_call] " + " ".join(f"{a}={b}" for a, b in tcs)
        msgs.append((role, str(content)))
    return msgs


def _fmt_msg(role: str, content: str) -> str:
    return f"### {role}\n\n{content}\n"


def write_viewer_messages(run_dir: Path) -> int:
    """Replace the synthetic analyze-tab messages with the REAL LLM I/O.

    Maps each LLM call to the env-step in progress when it started (via env_step
    event timestamps), and writes real prompts/responses as
    ``agent_logs/nemo/team_leader/messages/step_NNN_round_NN_{user,assistant}.md``
    — the format the arc_league viewer's analyze tab reads. Returns files written.
    """
    events_path = run_dir / "agent_logs" / "nemo" / "team_leader" / "events.jsonl"
    msgs_dir = events_path.parent / "messages"
    calls = _spans(run_dir)
    if not calls or not events_path.exists():
        return 0
    # env_step event times → step boundaries
    steps: list[tuple[float, int]] = []
    for line in events_path.read_text().splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "env_step":
            steps.append((float(e.get("unix_time_s", 0)), int(e.get("step", 0))))
    steps.sort()

    def step_for(call_start_s: float) -> int:
        cur = 0
        for t, s in steps:
            if t <= call_start_s:
                cur = s
            else:
                break
        return cur

    # clear synthetic placeholders, then write real
    if msgs_dir.exists():
        for f in msgs_dir.glob("step_*.md"):
            f.unlink()
    msgs_dir.mkdir(parents=True, exist_ok=True)
    round_by_step: dict[int, int] = {}
    written = 0
    for _, sp, at in calls:
        start_s = int(sp.get("startTimeUnixNano", 0)) / 1e9
        step = step_for(start_s)
        rnd = round_by_step.get(step, 0)
        round_by_step[step] = rnd + 1
        ins = _messages(at, "input")
        outs = _messages(at, "output")
        user = "\n".join(_fmt_msg(r, c) for r, c in ins) or "(no prompt captured)"
        asst = "\n".join(_fmt_msg(r, c) for r, c in outs) or "(tool_call / no text)"
        hdr = (
            f"<!-- REAL LLM I/O from trace; step={step} round={rnd} "
            f"prompt_tok={at.get('llm.token_count.prompt', '?')} "
            f"out_tok={at.get('llm.token_count.completion', '?')} -->\n\n"
        )
        (msgs_dir / f"step_{step:03d}_round_{rnd:02d}_user.md").write_text(hdr + user)
        (msgs_dir / f"step_{step:03d}_round_{rnd:02d}_assistant.md").write_text(hdr + asst)
        written += 2
    return written


def _spans(run_dir: Path) -> list[tuple[float, dict, dict]]:
    calls = []
    for t in sorted(run_dir.glob("traces/*.jsonl")):
        for line in t.open():
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            for rs in doc.get("resourceSpans", []):
                for ss in rs.get("scopeSpans", []):
                    for sp in ss.get("spans", []):
                        if sp.get("name") in ("aresponses", "responses"):
                            calls.append((int(sp.get("startTimeUnixNano", 0)), sp, _attrs(sp)))
    calls.sort(key=lambda c: c[0])
    return calls


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--full", action="store_true", help="don't truncate input messages")
    args = p.parse_args()
    run_dir = Path(args.run_dir)
    cap = None if args.full else 4000

    calls = _spans(run_dir)
    if not calls:
        print(f"no LLM-call spans in {run_dir}/traces/ — was tracing enabled?")
        return

    out_dir = run_dir / "llm_transcript"
    out_dir.mkdir(exist_ok=True)
    combined = [f"REAL LLM transcript for {run_dir.name}", f"{len(calls)} LLM calls\n"]

    for n, (_, sp, at) in enumerate(calls):
        lines = [
            f"{'=' * 70}",
            f"LLM CALL {n}  (span={sp.get('name')})",
            f"prompt_tokens={at.get('llm.token_count.prompt', '?')} "
            f"completion={at.get('llm.token_count.completion', '?')} "
            f"reasoning={at.get('llm.token_count.completion_details.reasoning', '?')}",
            f"{'-' * 70}",
            "PROMPT (input messages):",
        ]
        for role, content in _messages(at, "input"):
            body = (
                content
                if (cap is None or len(content) <= cap)
                else content[:cap] + f"\n…[+{len(content) - cap} chars truncated; --full for all]"
            )
            lines.append(f"\n[{role}]\n{body}")
        lines.append(f"\n{'-' * 70}\nRESPONSE (output):")
        outs = _messages(at, "output")
        lines.append("\n".join(f"[{r}]\n{c}" for r, c in outs) if outs else "(no output content)")
        text = "\n".join(lines)
        (out_dir / f"call_{n:04d}.txt").write_text(text)
        combined.append(text)

    (run_dir / "llm_transcript.txt").write_text("\n".join(combined))
    print(f"wrote {len(calls)} calls to {out_dir}/call_*.txt")
    print(f"combined transcript: {run_dir}/llm_transcript.txt")


if __name__ == "__main__":
    main()
