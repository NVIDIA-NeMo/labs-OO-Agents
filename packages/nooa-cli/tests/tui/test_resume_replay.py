# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for session resume replay truncation and batch rendering."""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from nooa_cli.tui.output import HistoryReplay, HistoryTurn
from nooa_cli.tui.session_manager import (
    RESUME_MAX_TURNS,
    build_resume_outputs,
)


def _make_session_db(turns: list[tuple[str, str]], rich_events: list[dict] | None = None) -> Path:
    """Create a temp session DB with the given turns and optional rich events."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE events (event_type TEXT, data TEXT, insertion_order INTEGER)")
    order = 0
    for role, content in turns:
        if role == "user":
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?)",
                ("TUIUserInput", json.dumps({"text": content}), order),
            )
        elif role == "agent":
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?)",
                ("TUIAgentMessage", json.dumps({"content": content}), order),
            )
        elif role == "rich":
            conn.execute(
                "INSERT INTO events VALUES (?, ?, ?)",
                ("RichOutput", json.dumps({"payload": {"kind": "plot", "data": content}}), order),
            )
        order += 1
    conn.commit()
    conn.close()
    return db_path


class TestTruncation:
    """Tests for turn truncation in build_resume_outputs."""

    def test_no_truncation_when_under_limit(self):
        """Sessions with fewer turns than max_turns are not truncated."""
        turns = [("user", f"msg {i}") for i in range(5)]
        db = _make_session_db(turns)
        outputs = build_resume_outputs(db, "abc12345", max_turns=10)
        replays = [o for o in outputs if isinstance(o, HistoryReplay)]
        total = sum(len(r.turns) for r in replays)
        assert total == 5
        assert replays[0].omitted_count == 0

    def test_truncation_keeps_last_n_turns(self):
        """Only the last max_turns turns are kept."""
        turns = [("user", f"msg {i}") for i in range(50)]
        db = _make_session_db(turns)
        outputs = build_resume_outputs(db, "abc12345", max_turns=10)
        replays = [o for o in outputs if isinstance(o, HistoryReplay)]
        total = sum(len(r.turns) for r in replays)
        assert total == 10
        # Should show last 10 messages
        all_turns = [t for r in replays for t in r.turns]
        assert all_turns[0].content == "msg 40"
        assert all_turns[-1].content == "msg 49"

    def test_omitted_count_in_header(self):
        """The first HistoryReplay reports how many turns were omitted."""
        turns = [("user", f"msg {i}") for i in range(30)]
        db = _make_session_db(turns)
        outputs = build_resume_outputs(db, "abc12345", max_turns=10)
        replays = [o for o in outputs if isinstance(o, HistoryReplay)]
        assert replays[0].omitted_count == 20

    def test_truncation_disabled_with_zero(self):
        """max_turns=0 disables truncation."""
        turns = [("user", f"msg {i}") for i in range(50)]
        db = _make_session_db(turns)
        outputs = build_resume_outputs(db, "abc12345", max_turns=0)
        replays = [o for o in outputs if isinstance(o, HistoryReplay)]
        total = sum(len(r.turns) for r in replays)
        assert total == 50

    def test_default_max_turns(self):
        """Default truncation uses RESUME_MAX_TURNS."""
        turns = [("user", f"msg {i}") for i in range(50)]
        db = _make_session_db(turns)
        outputs = build_resume_outputs(db, "abc12345")
        replays = [o for o in outputs if isinstance(o, HistoryReplay)]
        total = sum(len(r.turns) for r in replays)
        assert total == RESUME_MAX_TURNS

    def test_rich_items_after_truncated_turns_are_dropped(self):
        """Rich items interleaved with truncated turns are not kept."""
        # Create: 10 user turns, then a rich event, then 5 more user turns
        turns = (
            [("user", f"early {i}") for i in range(10)]
            + [("rich", "plot_data")]
            + [("user", f"late {i}") for i in range(5)]
        )
        db = _make_session_db(turns)
        # Keep only 5 turns — all "early" turns + the rich event should be dropped
        outputs = build_resume_outputs(db, "abc12345", max_turns=5, in_nemo_term=True)
        replays = [o for o in outputs if isinstance(o, HistoryReplay)]
        total = sum(len(r.turns) for r in replays)
        assert total == 5
        # No rich events should survive since they precede kept turns
        from nooa_cli.tui.output import _RichReplayPayload

        rich_items = [o for o in outputs if isinstance(o, _RichReplayPayload)]
        assert len(rich_items) == 0

    def test_rich_items_after_kept_turns_are_preserved(self):
        """Rich items interleaved with kept turns survive truncation."""
        # Create: 5 user turns, then 5 more + a rich event at the end
        turns = (
            [("user", f"early {i}") for i in range(5)]
            + [("user", f"late {i}") for i in range(4)]
            + [("rich", "kept_plot")]
            + [("user", "final")]
        )
        db = _make_session_db(turns)
        # Keep 5 turns — last 5 turns include "late 3", "late 4" area + rich + final
        outputs = build_resume_outputs(db, "abc12345", max_turns=5, in_nemo_term=True)
        from nooa_cli.tui.output import _RichReplayPayload

        rich_items = [o for o in outputs if isinstance(o, _RichReplayPayload)]
        assert len(rich_items) == 1


class TestBatchRendering:
    """Tests for batch rendering in TerminalFrontend._render_history_replay."""

    def test_single_write_to_terminal(self):
        """History replay writes to terminal file in one call, not per-turn."""
        from nooa_cli.tui.frontend import TerminalFrontend

        # Create a mock console
        mock_file = MagicMock()
        mock_console = MagicMock()
        mock_console.console.width = 80
        mock_console.console.file = mock_file

        frontend = TerminalFrontend.__new__(TerminalFrontend)
        frontend._console = mock_console

        replay = HistoryReplay(
            turns=[
                HistoryTurn(role="user", content="hello"),
                HistoryTurn(role="agent", content="world"),
                HistoryTurn(role="user", content="bye"),
            ],
            session_id="abc123",
            show_header=True,
            show_footer=True,
        )

        frontend._render_history_replay(replay)

        # Should write once (batch) + flush once
        assert mock_file.write.call_count == 1
        assert mock_file.flush.call_count == 1
        # The single write should contain content from all turns
        written = mock_file.write.call_args[0][0]
        assert "hello" in written
        assert "world" in written
        assert "bye" in written

    def test_history_replay_on_emit_stream_keeps_semantic_replay_callback(self):
        """Live TUI rendering stores resumed HistoryReplay as a reflowable block."""
        from nooa_cli.tui.frontend import TerminalFrontend
        from nooa_cli.tui.session import _EmitStream

        emitted = []
        current_width = 80

        def emit(text: str, replay=None):
            emitted.append((text, replay))

        stream = _EmitStream(emit, replay_width=lambda: current_width)
        mock_console = MagicMock()
        mock_console.console.width = 120
        mock_console.console.file = stream

        frontend = TerminalFrontend.__new__(TerminalFrontend)
        frontend._console = mock_console

        replay = HistoryReplay(
            turns=[
                HistoryTurn(
                    role="agent",
                    content="This is a resumed markdown paragraph that should wrap differently.",
                )
            ],
            session_id="abc123",
            show_header=True,
            show_footer=True,
        )

        frontend._render_history_replay(replay)

        assert len(emitted) == 1
        rendered, replay_callback = emitted[0]
        assert "abc123" in rendered
        assert callable(replay_callback)
        current_width = 24
        rerendered = replay_callback()
        assert rerendered != rendered
        assert "abc123" in rerendered
        assert len(rerendered.splitlines()[0]) < len(rendered.splitlines()[0])

    def test_emit_stream_preserves_order_when_semantic_block_appears_inside_hold(self):
        """A semantic HistoryReplay inside batch_render must not jump ahead of buffered text."""
        from nooa_cli.tui.session import _EmitStream

        emitted = []

        def emit(text: str, replay=None):
            emitted.append((text, replay))

        stream = _EmitStream(emit)
        with stream.hold():
            stream.write("before")
            stream.emit_with_replay("history", lambda: "history-reflowed")
            stream.write("after")

        assert [text for text, _ in emitted] == ["before", "history", "after"]
        assert emitted[1][1]() == "history-reflowed"
