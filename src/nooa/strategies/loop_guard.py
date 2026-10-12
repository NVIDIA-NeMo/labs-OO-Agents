# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic detection of CodeAct loops that repeat one tool call.

The guard compares each tool call with the most recent ones in a fixed window.
It never interprets an error or output; it only notices that the same action
produced the same outcome again, warns once with a fixed template, and stops
generation if the model repeats the action after that warning.
"""

from __future__ import annotations

import ast
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from typing import Literal

from nooa.config.strategy_config import LoopGuardConfig

Verdict = Literal["block", "nudge", "stop"]


@dataclass(frozen=True)
class _Entry:
    action: str
    outcome: str
    failed: bool


def normalize_code(code: str) -> str:
    """Canonical form of Python source, ignoring comments and formatting."""
    try:
        return ast.dump(ast.parse(code))
    except (SyntaxError, ValueError):
        return "\n".join(line.strip() for line in code.strip().splitlines() if line.strip())


def action_fingerprint(tool_name: str, arguments: object, python_tool: str) -> str:
    """Hash a tool call's name and normalized arguments."""
    if tool_name == python_tool and isinstance(arguments, dict):
        code = arguments.get("code")
        rest = {k: v for k, v in arguments.items() if k != "code"}
        normalized = json.dumps(
            [normalize_code(code) if isinstance(code, str) else repr(code), _stable(rest)]
        )
    else:
        normalized = _stable(arguments)
    return hashlib.sha256(f"{tool_name}\0{normalized}".encode()).hexdigest()[:16]


def _stable(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(value)


@dataclass
class LoopGuard:
    """Per-generation repetition state for one CodeAct session."""

    config: LoopGuardConfig
    _history: deque[_Entry] = field(init=False)
    _calls: int = 0
    # Action fingerprint -> call index of the warning, for blocked failing actions.
    _warned_failing: dict[str, int] = field(default_factory=dict)
    # (action, outcome) -> call index of the warning, for succeeding repeats.
    _warned_repeats: dict[tuple[str, str], int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._history = deque(maxlen=self.config.window)

    def _recent(self, warned_at: int | None) -> bool:
        return warned_at is not None and self._calls - warned_at <= self.config.window

    def before(self, action: str) -> tuple[Verdict, int, str] | None:
        """Check a call before it runs.

        Returns ``("stop", ...)`` when a blocked failing action is proposed
        again soon after its warning, ``("block", ...)`` when its last
        ``repeat_threshold - 1`` runs in the window failed identically, and
        ``None`` otherwise. The tuple carries the repeat count and the previous
        outcome text.
        """
        self._calls += 1
        matches = [e for e in self._history if e.action == action]
        if self._recent(self._warned_failing.get(action)):
            last = matches[-1].outcome if matches else ""
            return ("stop", len(matches) + 1, last)
        self._warned_failing.pop(action, None)
        needed = self.config.repeat_threshold - 1
        tail = matches[-needed:]
        if (
            len(tail) == needed
            and all(e.failed for e in tail)
            and len({e.outcome for e in tail}) == 1
        ):
            self._warned_failing[action] = self._calls
            self._history.append(_Entry(action, tail[-1].outcome, True))
            return ("block", len(matches) + 1, tail[-1].outcome)
        return None

    def after(self, action: str, outcome: str, failed: bool) -> tuple[Verdict, int] | None:
        """Record a call's outcome; warn on, or stop at, identical successful repeats."""
        self._history.append(_Entry(action, outcome, failed))
        if failed:
            return None
        key = (action, outcome)
        repeats = sum(1 for e in self._history if (e.action, e.outcome) == key and not e.failed)
        if self._recent(self._warned_repeats.get(key)):
            return ("stop", repeats)
        self._warned_repeats.pop(key, None)
        if repeats >= self.config.repeat_threshold:
            self._warned_repeats[key] = self._calls
            return ("nudge", repeats)
        return None

    def message(
        self,
        verdict: Verdict,
        tool_name: str,
        repeats: int,
        outcome: str,
        *,
        completion: bool,
    ) -> str:
        """Render the fixed-template message for a verdict."""
        window = self.config.window
        limit = self.config.max_outcome_chars
        shown = outcome if len(outcome) <= limit else outcome[:limit] + "..."
        if verdict == "stop":
            return (
                f"Loop guard stopped generation: the {tool_name} call warned about earlier "
                f"was repeated ({repeats} identical calls in the last {window} tool calls)."
            )
        if verdict == "block":
            text = (
                f"Loop guard: this {tool_name} call was not executed. The identical call "
                f"already failed {repeats - 1} times in the last {window} tool calls with "
                f"the same result:\n{shown}\n"
                "Do not submit it again; repeating it stops the run. "
                "Submit a materially different call."
            )
        else:
            text = (
                f"Loop guard: this {tool_name} call ran {repeats} times in the last {window} "
                "tool calls with identical output, so repeating it gives no new information. "
                "Repeating it again with the same output stops the run. Submit a materially "
                "different call; if you are waiting for something to change, wait inside "
                "the cell before checking again."
            )
        if completion:
            text += (
                "\nTo finish, build the return value in a variable first, then call "
                "return_result(variable)."
            )
        return text


def mentions_return_result(arguments: object) -> bool:
    """Whether a Python-cell call's code syntactically calls ``return_result``."""
    code = arguments.get("code") if isinstance(arguments, dict) else None
    return isinstance(code, str) and "return_result" in code
