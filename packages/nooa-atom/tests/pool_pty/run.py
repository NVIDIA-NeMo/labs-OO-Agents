# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run actual mapper schemas through an isolated installed Pool PTY (Linux only)."""

import argparse
import codecs
import fcntl
import hashlib
import json
import os
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path

from nooa_atom.acp.need_input import pool_form_schema
from pydantic import ValidationError
from screen import Screen

from nooa.interactive import (
    FormChoice,
    FormResponse,
    NeedInputForm,
    PickOneOrTextQuestion,
    PickOneQuestion,
    TextQuestion,
)

DOWN = "\x1b[B"


def questions():
    def choices(*pairs):
        return [FormChoice(value=value, title=title) for value, title in pairs]

    # Only the diagnostic probes bypass the new construction guard, deliberately
    # reproducing the formerly accepted descriptors through the ACTUAL mapper.
    return [
        TextQuestion(id="nickname", label="Nickname?", help="Enter a nickname"),
        TextQuestion(id="notes", label="Notes?", help="Optional notes", required=False),
        PickOneQuestion(
            id="color",
            label="Color?",
            help="Pick a color",
            choices=choices(("red", "Ruby red"), ("blue", "Ocean blue")),
        ),
        PickOneQuestion(
            id="optional_color",
            label="Optional color?",
            help="May skip",
            choices=choices(("green", "Forest green"), ("gold", "Golden yellow")),
        ).model_copy(update={"required": False}),
        PickOneOrTextQuestion(
            id="snack",
            label="Snack?",
            help="Choose or type",
            choices=choices(("apple", "An apple"), ("nuts", "Mixed nuts")),
        ),
        PickOneOrTextQuestion(
            id="optional_snack",
            label="Optional snack?",
            help="May skip or type",
            choices=choices(("toast", "Toast")),
        ).model_copy(update={"required": False}),
        TextQuestion(
            id="count",
            label="Count?",
            help="Enter anything, including nonnumeric text; domain validation is later",
        ),
    ]


def cases():
    q = questions()
    original = NeedInputForm.model_construct(
        heading="Details", reason="Synthetic reproduction", questions=q
    )
    for kind in (PickOneQuestion, PickOneOrTextQuestion):
        try:
            kind(id="x", label="X?", required=False, choices=[FormChoice(value="a", title="A")])
        except ValidationError as exc:
            assert "Optional picker questions are unsupported" in str(exc)
        else:
            raise AssertionError("optional picker construction did not fail")
    alternative = list(q)
    alternative[3] = TextQuestion(
        id="optional_color",
        label="Optional color?",
        required=False,
        help="Explicit text alternative: Forest green: green; Golden yellow: gold. Leave blank to skip.",
    )
    alternative[5] = TextQuestion(
        id="optional_snack",
        label="Optional snack?",
        required=False,
        help="Explicit text alternative: Toast: toast, or other text. Leave blank to skip.",
    )
    mixed = NeedInputForm(heading="Details", reason="Synthetic reproduction", questions=alternative)

    def single(question):
        return NeedInputForm.model_construct(
            heading="Details", reason="Synthetic reproduction", questions=[question]
        )

    return [
        (
            "strict-required",
            single(q[2]),
            [DOWN, "\r"],
            {"color": "blue"},
            ["Ruby red", "Ocean blue"],
        ),
        (
            "flex-select",
            single(q[4]),
            [DOWN, "\r"],
            {"snack": "nuts"},
            ["An apple", "Mixed nuts", "Type your own answer"],
        ),
        (
            "flex-custom",
            single(q[4]),
            [DOWN + DOWN, "purple", "\r"],
            {"snack": "purple"},
            ["Type your own answer", "purple"],
        ),
        ("intentional-decline", single(q[2]), ["\x1b"], None, ["Ruby red"]),
        ("strict-optional-unsupported", single(q[3]), [], None, []),
        ("flex-optional-unsupported", single(q[5]), [], None, []),
        ("original-mixed-unsupported", original, [], None, []),
        (
            "mixed-explicit-text-alternative",
            mixed,
            [
                "hello\r",
                "\r",
                DOWN,
                "\r",
                "\r",
                DOWN + DOWN,
                "pretzels\r",
                "\r",
                "not-a-number\r",
                "\r",
            ],
            {
                "nickname": "hello",
                "notes": "",
                "color": "blue",
                "optional_color": "",
                "snack": "pretzels",
                "optional_snack": "",
                "count": "not-a-number",
            },
            [
                "Question 1/7",
                "Question 7/7",
                "Ruby red",
                "Ocean blue",
                "An apple",
                "Mixed nuts",
                "Review answers",
            ],
        ),
        (
            "optional-text-values",
            NeedInputForm(heading="Optional text", questions=[alternative[3], alternative[5]]),
            ["gold\r", "custom-toast\r", "\r"],
            {"optional_color": "gold", "optional_snack": "custom-toast"},
            ["Forest green", "Golden yellow", "Toast"],
        ),
    ]


def run_case(binary, root, case):
    name, need, actions, content, markers = case
    root = root / name
    root.mkdir()
    for directory in ("home", "config/poolside", "cache", "data", "state"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(__file__).with_name("agent.py"), root / "agent.py")
    params = {
        "sessionId": "synthetic-session",
        "mode": "form",
        "message": need.heading + "\n\n" + (need.reason or ""),
        "requestedSchema": pool_form_schema(need),
    }
    if len(need.questions) > 1:
        params["_meta"] = {"poolside/field_order": [q.id for q in need.questions]}
    visible = (
        need.heading
        + "\n\n"
        + (need.reason or "")
        + "\nExplicit synthetic form request retained as text. Client-reported decline supplies no reason.\n"
        + json.dumps([q.model_dump() for q in need.questions], indent=2)
    )
    (root / "case.json").write_text(
        json.dumps({"params": params, "visible_request": visible}, indent=2)
    )
    (root / "config/poolside/settings.yaml").write_text(
        "agent_servers:\n  synthetic:\n    command: "
        + json.dumps(sys.executable)
        + "\n    args: ["
        + json.dumps(str(root / "agent.py"))
        + "]\n"
    )
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(root / "home"),
        "TERM": "xterm-256color",
        "POOLSIDE_API_URL": "http://127.0.0.1:1",
    }
    env.update(
        {
            "XDG_" + key + "_HOME": str(root / directory)
            for key, directory in [
                ("CONFIG", "config"),
                ("CACHE", "cache"),
                ("DATA", "data"),
                ("STATE", "state"),
            ]
        }
    )
    master, slave = pty.openpty()

    def setup():
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, 120, 0, 0))
        process = subprocess.Popen(
            [str(binary), "--agent-server", "synthetic", "--directory", str(root)],
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=root,
            preexec_fn=setup,
        )
    except BaseException:
        os.close(master)
        raise
    finally:
        os.close(slave)
    captured = bytearray()
    screen = Screen()
    decoder = codecs.getincrementaldecoder("utf8")("replace")
    screens = []
    stages = []

    def wire():
        path = root / "wire.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def responses():
        return [
            x["rpc"] for x in wire() if x["direction"] == "in" and x["rpc"].get("id") == "elicit-1"
        ]

    def collect(label, duration, gate=None):
        data = bytearray()
        deadline = time.monotonic() + duration
        met = gate is None
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [master], [], [], min(0.1, max(0, deadline - time.monotonic()))
            )
            if ready:
                try:
                    chunk = os.read(master, 65536)
                except OSError:
                    break
                data.extend(chunk)
                screen.feed(decoder.decode(chunk))
                if b"\x1b[6n" in chunk:
                    os.write(master, b"\x1b[1;1R")
                if b"\x1b]11;?" in chunk:
                    os.write(master, b"\x1b]11;rgb:0000/0000/0000\x07")
            if gate and gate():
                met = True
                gate = None
                deadline = min(deadline, time.monotonic() + 0.5)
        captured.extend(data)
        rendered = screen.text()
        screens.append(rendered)
        (root / (label + ".screen.txt")).write_text(rendered)
        (root / (label + ".ansi")).write_bytes(data)
        stages.append({"label": label, "gate_met": met, "t": time.monotonic()})
        assert met, f"{name}: timeout at {label}"

    try:
        collect("00-start", 5, lambda: any(x["rpc"].get("method") == "session/new" for x in wire()))
        os.write(master, b"probe\r")
        collect(
            "01-form",
            5,
            lambda: any(
                x["direction"] == "out" and x["rpc"].get("id") == "elicit-1" for x in wire()
            ),
        )
        if actions:
            assert not responses(), f"{name}: client auto-answered before input"
            assert "Question 1/" in screens[-1], f"{name}: no pane before input"
            if need.questions[0].kind != "text":
                pane = screens[-1].split("Question 1/", 1)[1]
                assert "● " + need.questions[0].choices[0].title in pane
                if len(need.questions[0].choices) > 1:
                    assert "○ " + need.questions[0].choices[1].title in pane
                if need.questions[0].kind == "pick_one_or_text":
                    assert "○ Type your own answer" in pane
        for i, action in enumerate(actions):
            assert not responses(), f"{name}: answered before planned action {i}"
            stages.append({"input_hex": action.encode().hex(), "t": time.monotonic()})
            os.write(master, action.encode())
            collect(f"{i + 2:02d}-input", 0.65)
        collect("final", 3, lambda: bool(responses()))
        [response] = responses()
        expected = (
            {"action": "decline"} if content is None else {"action": "accept", "content": content}
        )
        assert response["result"] == expected, (name, response)
        text = "\n".join(screens)
        assert not screen.unknown, screen.unknown
        for marker in markers:
            assert marker in text, (name, marker)
        if "unsupported" in name:
            logs = "\n".join(p.read_text() for p in root.glob("state/**/*.log.jsonl"))
            assert "optional non-text property" in logs
            assert not any("input_hex" in s for s in stages)
        if content is not None:
            assert (
                need.validate_response(FormResponse(action="accept", content=content)).content
                == content
            )
        return name
    finally:
        cleanup_error = None
        try:
            stop_group(process)
        except Exception as exc:
            cleanup_error = repr(exc)
        finally:
            os.close(master)
            (root / "all.ansi").write_bytes(captured)
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "stages": stages,
                        "responses": responses(),
                        "pool_returncode": process.returncode,
                        "process_group_stopped": cleanup_error is None,
                        "cleanup_error": cleanup_error,
                    },
                    indent=2,
                )
            )
        if cleanup_error is not None and sys.exception() is None:
            raise RuntimeError(cleanup_error)


def live_group_members(pgid):
    """Linux /proc check; zombies cannot run and are left to their parent to reap."""
    members = []
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pgid and fields[0] != "Z":
                members.append(int(path.parent.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return members


def stop_group(process):
    """Signal descendants even if the leader exits first; tolerate an absent group."""

    def signal_group(sig):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    signal_group(signal.SIGTERM)
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        pass
    signal_group(signal.SIGKILL)
    process.wait(timeout=2)
    deadline = time.monotonic() + 2
    while live_group_members(process.pid) and time.monotonic() < deadline:
        select.select([], [], [], 0.05)
    assert not live_group_members(process.pid), "synthetic process group survived cleanup"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", default=shutil.which("pool"))
    args = parser.parse_args()
    if not args.pool or not Path(args.pool).is_file():
        print("SKIP: installed Pool absent; pass --pool /path/to/pool")
        return
    binary = Path(args.pool).resolve()
    root = Path(tempfile.mkdtemp(prefix="pool-final-mapper."))
    version = subprocess.check_output([str(binary), "--version"], timeout=5).decode().strip()
    (root / "runtime.json").write_text(
        json.dumps(
            {
                "binary": str(binary),
                "version": version,
                "sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            },
            indent=2,
        )
    )
    print(f"Artifacts: {root}", flush=True)
    for case in cases():
        print("PASS: " + run_case(binary, root, case), flush=True)
    print("PASS: 9 isolated real-UI cases; optional-picker constructors rejected", flush=True)


if __name__ == "__main__":
    main()
