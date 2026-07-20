# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ARC-AGI-3 single-agent solver built on the normal nemo-oo TUI agent.

One agent (no team) plays an ARC-AGI-3 game through a file-IPC harness:

- the harness appends game states to ``ipc/states.jsonl``; the agent tails that
  file into a ``game_states`` queue, so each new state wakes ``handle()``;
- the agent analyzes the state in CodeAct (numpy on the grid, persistent helper
  modules) and answers by calling ``submit_actions([...], rationale)``, which
  appends to ``ipc/actions.jsonl`` for the harness to execute.

Two knowledge variants share this base:

- :class:`MemArcSolverAgent` — the nemo-oo **memory system** (``self.memory.*``,
  installed as a skill by the launcher) holds accumulated knowledge, hypotheses
  and reflections;
- :class:`MdArcSolverAgent` — the knowledge-file baseline: knowledge lives in
  ``knowledge/*.md`` files via ``read/append/write_notes``.
"""

from __future__ import annotations

# Module-level imports are visible to LLM-generated CodeAct cells.
import asyncio  # noqa: F401
import collections  # noqa: F401
import itertools  # noqa: F401
import json  # noqa: F401
import math  # noqa: F401
import re  # noqa: F401
from pathlib import Path  # noqa: F401
from types import SimpleNamespace

import numpy as np  # noqa: F401

from nooa import hidden, strategy
from nooa.interactive import InteractiveAgent, RespondReason, RespondResult  # noqa: F401
from nooa.media import Image  # visual grid input (show()n to the LLM)
from nooa.runtime import show  # CodeAct builtin — attach an image to the turn

with hidden:
    import importlib.util
    import io
    import os  # module-level only (hidden from cells; cells' os access stays blocked)
    import time
    from typing import Annotated

    from PIL import Image as _PILImage  # PNG rendering only

    from nooa.agentdoc import pformat, spec
    from nooa.config import CodeActConfig
    from nooa.interactive import SummarizationConfig, install_summarizer
    from nooa.runtime.channels import Channel, _ChannelReader
    from nooa.runtime.restrictions import (
        DEFAULT_BLOCKED_MODULES,
        RestrictionsConfig,
    )
    from nooa.runtime.sandbox import SandboxConfig
    from nooa.skill_registry import SkillRegistry
    from nooa.storage.markers import nosnapshot
    from nooa.strategies import CodeActStrategy

    # _ReturnResultSignal is the internal signal raised by return_result(); no public
    # re-export exists yet. If nooa ever exposes a public path, update this import.
    from nooa.strategies.codeact import _ReturnResultSignal
    from nooa.strategy_validation import InvariantError

# L2 guard (isolation plan §L2): deny file/network/process modules in generated
# CodeAct cells. The agent's own tools (which use pathlib internally) are
# unaffected — restrictions apply only to LLM-generated code. `open()` on a
# forbidden path is neutralised structurally by L3 (the path is absent from the
# sandbox mount); L2 closes the import-based vectors below.
_ARC_RESTRICTIONS = RestrictionsConfig(
    blocked_modules=DEFAULT_BLOCKED_MODULES
    | frozenset(
        {
            # file/dir access (incl. alternate open() entry points io/codecs/linecache)
            "os",
            "shutil",
            "tempfile",
            "glob",
            "fileinput",
            "pathlib",
            "io",
            "codecs",
            "linecache",
            # C-level file access that bypasses the open() jail
            "sqlite3",
            "dbm",
            "shelve",
            # network
            "urllib",
            "http",
            "requests",
            "httpx",
            "aiohttp",
            "socketserver",
            # process / low-level escape (multiprocessing 'spawn' starts a fresh
            # interpreter WITHOUT this denylist — a hard escape, so block it)
            "ctypes",
            "mmap",
            "pty",
            "fcntl",
            "multiprocessing",
            "concurrent",
            # dynamic import / exec-by-path (partial bypass of the name denylist)
            "importlib",
            "runpy",
            "pickle",
            "marshal",
            "webbrowser",
            # `sys` closes the sys.modules['os'] cached-module bypass (see _scan_cell)
            "sys",
        }
    ),
    # `asyncio` stays allowed (cells use `await self...` / gather); its network
    # ENTRYPOINTS are blocked at the AST layer by _scan_cell below (open_connection
    # / create_connection / …). NB: a module denylist is best-effort — the hard
    # boundary for the filesystem is the external uid-drop (uid_sandbox / run_solver
    # --sandbox drop), which holds regardless of any in-process import trick.
)


def _wait_requires_submission(agent, result, call) -> None:
    """Postcondition on ``handle``: a WAIT turn-result must be backed by a
    submission for the current turn (method-local; not LLM-visible).

    In the 20260716 fleet the agent 13 times ended a turn claiming "submitted N
    actions" while nothing reached ``ipc/actions.jsonl`` (implicit-return bug,
    worker restarts) — and both sides idled the full 900s nudge timer. Raising
    ``InvariantError`` routes the unbacked WAIT through the standard
    validation-retry channel: an immediate same-turn resubmit instead of a stall.
    """
    kind = getattr(result, "kind", None)
    if kind is None and isinstance(result, dict):
        kind = result.get("kind")
    if kind is None or str(kind).upper() != "WAIT":
        return
    state = agent._latest_state()
    turn = state.get("turn") if state else None
    if turn is None or agent._last_submitted_turn() == turn:
        return
    raise InvariantError(
        f"result kind WAIT, but no actions were submitted for turn {turn} — "
        "nothing reached the harness. Call self.submit_actions([...], rationale) now; "
        "a successful submit ends the turn by itself."
    )


def _arc_cell_config() -> CodeActConfig:
    """Build the per-cell CodeAct config, optionally with the OS sandbox backend.

    Default keeps the historical in-process backend: ``cell_timeout`` is advisory
    and the module denylist plus the external uid-drop are the guardrails. Enable
    the per-cell OS sandbox — where ``cell_timeout`` becomes a *hard* kill and the
    filesystem / network / memory / CPU limits are kernel-enforced (Landlock +
    seccomp + rlimits) — via ``run_solver.py --sandbox-cells`` (or a run_multi
    ``sandbox:`` config block), which set the ``ARC_SANDBOX_*`` env this reads. The
    cell's ``self.*`` tool calls broker back to this live agent, so it needs no
    network of its own.
    """
    base = {
        "restrictions": _ARC_RESTRICTIONS,
        "cell_timeout": 60.0,
        "postconditions": (_wait_requires_submission,),
    }
    if os.environ.get("ARC_SANDBOX_CELLS") != "1":
        return CodeActConfig(**base)
    workspace = os.environ.get("ARC_SANDBOX_WORKSPACE") or os.getcwd()
    return CodeActConfig(
        **base,
        execution_backend="sandbox",
        sandbox=SandboxConfig(
            workspace=workspace,
            network=False,
            max_memory_mb=int(os.environ.get("ARC_SANDBOX_MEM_MB", "4096")),
            max_cpu_seconds=int(os.environ.get("ARC_SANDBOX_CPU_S", "90")),
            # Fail open in the example so a host lacking Landlock/seccomp still runs;
            # production callers should keep the default require=True.
            require=os.environ.get("ARC_SANDBOX_REQUIRE", "0") == "1",
        ),
    )


_ARC_CELL_CONFIG = _arc_cell_config()

# ── Cell guard (B-lite): AST-level block of the escape / network / denylist-bypass
# primitives a generated cell could otherwise use, which a module denylist alone
# cannot stop (__import__('os'), sys.modules['os'], subclass gadgets all reach
# blocked modules). Best-effort defense-in-depth ON TOP of the external uid-drop —
# a determined model can still build names dynamically, so this is a speed bump,
# not a boundary. Applied to every cell via an execute_python guard (see __init__).
_CELL_BANNED_ATTRS = frozenset(
    {
        # host escape tools (network + host FS; no legitimate use for a grid solver)
        "shell",
        "repo",
        "pyp",
        "web",
        "mcp",
        "libwriting",
        "tui_config",
        # denylist-bypass gadgets
        "__subclasses__",
        "__bases__",
        "__base__",
        "__mro__",
        "__globals__",
        "__builtins__",
        "__loader__",
        "__spec__",
        "__code__",
        # asyncio network entrypoints (the path the import denylist misses)
        "open_connection",
        "create_connection",
        "create_server",
        "getaddrinfo",
        "sock_connect",
        "sock_accept",
        "start_server",
        "open_unix_connection",
    }
)
_CELL_BANNED_NAMES = frozenset({"__import__", "eval", "exec", "compile", "breakpoint"})


@hidden
def _scan_cell(code: str) -> str | None:
    """Return the first banned primitive a cell references, or None. AST-based, so
    it ignores matches inside strings/comments (e.g. helper source passed to
    write_helper). Syntax errors fall through to the real executor to report."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in _CELL_BANNED_ATTRS:
            return (
                f"self.{node.attr}"
                if node.attr in ("shell", "repo", "pyp", "web", "mcp")
                else node.attr
            )
        if isinstance(node, ast.Name) and node.id in _CELL_BANNED_NAMES:
            return node.id
    return None


_NOTE_NAME = re.compile(r"^[a-z0-9_\-]+\.md$")
_HELPER_NAME = re.compile(r"^[a-z0-9_]+\.py$")
_CLICK = re.compile(r"^CLICK\s+(\d{1,2})\s+(\d{1,2})$")

MAX_ACTIONS_PER_TURN = 20

# ── L1.5 in-process path jail (inspired by beam/agent/repl.py `_safe_open`) ────
# Generated CodeAct cells inherit THIS module's globals, so a module-level `open`
# shadows the builtin for agent-generated code. We route it through a workspace
# jail: reads/writes are allowed ONLY under the registered roots (the agent's own
# workspace + IPC). This closes the vector L2 cannot — a bare `open()` on an
# absolute path to the game source or another run — WITHOUT needing OS namespaces
# (which the L3 sandbox provides where available). io/codecs/linecache (alternate
# open() entry points) are blocked by L2 above so this is the only file gateway.
_REAL_OPEN = open
_JAIL_ROOTS: list[Path] = []


def _register_jail_roots(*roots: Path) -> None:
    """Set the allowlisted roots for the shadowed open (one agent per process)."""
    _JAIL_ROOTS.clear()
    _JAIL_ROOTS.extend(r.resolve() for r in roots)


def _jailed_open(file, mode="r", *args, **kwargs):
    """Path-jailed replacement for builtin open. Only paths under a registered
    root are permitted; anything else raises PermissionError. Falls back to the
    real open when no jail is set (framework internals before agent __init__)."""
    if not _JAIL_ROOTS:
        return _REAL_OPEN(file, mode, *args, **kwargs)
    try:
        resolved = Path(file).resolve()
    except (TypeError, ValueError):
        raise PermissionError(f"open({file!r}) — invalid path") from None
    for root in _JAIL_ROOTS:
        try:
            resolved.relative_to(root)
            return _REAL_OPEN(file, mode, *args, **kwargs)
        except ValueError:
            continue
    raise PermissionError(
        f"open({file!r}) denied — outside the agent workspace. Use "
        f"self.read_notes/read_helper/latest_state/trajectory for data."
    )


open = _jailed_open  # noqa: A001  shadow builtin in this module's exec namespace


def _build_redactions(run_dir: Path) -> list[tuple[str, str]]:
    """Identity substrings to scrub from cell output (isolation §anonymization).

    Cell stdout/stderr can echo absolute paths (tracebacks, helper-import errors,
    reprs) that embed `results/arc_agi_3/…/<game>_<variant>/…` — leaking the
    benchmark + game id into the next prompt. Replace the run-dir path prefixes
    and identity tokens with neutral placeholders (longest-first)."""
    reps: list[tuple[str, str]] = []
    p = run_dir.resolve()
    for _ in range(5):
        reps.append((str(p), "/workspace"))
        if p.name == "arc_agi_3" or p.parent == p:
            break
        p = p.parent
    reps.append(("arc_agi_3", "env"))
    # game/variant tokens from the run-dir name: <ts>_<game>_<variant>[_tag]
    parts = run_dir.name.split("_")
    if len(parts) >= 4:
        game = parts[2]
        reps.append((f"{game}_", "game_"))
        reps.append((game, "game"))
    reps.sort(key=lambda kv: -len(kv[0]))
    return reps


def _redact(text: str, reps: list[tuple[str, str]]) -> str:
    for find, rep in reps:
        if find and find in text:
            text = text.replace(find, rep)
    return text


# L1 guard (isolation plan §L1): helper modules are importlib-loaded, so their
# module-level code runs OUTSIDE the CodeAct cell sandbox — the one in-process
# path that could `import urllib`/`open(...)` a game-source file. Restrict helper
# source to pure computation: only these imports, no file/network/process/dunder.
_HELPER_IMPORT_ALLOW = frozenset(
    {
        "numpy",
        "np",
        "math",
        "json",
        "re",
        "collections",
        "itertools",
        "statistics",
        "functools",
        "operator",
        "heapq",
        "bisect",
        "dataclasses",
        "typing",
        "enum",
        "fractions",
        "copy",
    }
)
_HELPER_BANNED_NAMES = frozenset(
    {
        "open",
        "eval",
        "exec",
        "compile",
        "__import__",
        "input",
        "breakpoint",
        "globals",
        "locals",
        "vars",
        "memoryview",
        "getattr",
        "setattr",
        "delattr",
    }
)
_HELPER_BANNED_DUNDERS = frozenset(
    {
        "__globals__",
        "__builtins__",
        "__subclasses__",
        "__bases__",
        "__mro__",
        "__class__",
        "__dict__",
        "__getattribute__",
        "__import__",
        "__loader__",
        "__code__",
        "__closure__",
        "__self__",
        "__module__",
        "__base__",
    }
)


def _check_helper_ast(source: str) -> str | None:
    """Return an error string if the helper source uses anything beyond pure
    computation (disallowed import / open / dunder / banned builtin), else None."""
    import ast
    import textwrap

    # Models routinely paste uniformly indented source (e.g. lifted out of a
    # docstring or an if-block); that's valid code — dedent before parsing
    # instead of bouncing it with a spurious IndentationError (~220 wasted
    # rounds across the 20260716 fleet).
    source = textwrap.dedent(source)

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return f"syntax error — {e}"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                top = a.name.split(".")[0]
                if top not in _HELPER_IMPORT_ALLOW:
                    return f"import {a.name!r} not allowed (helpers are pure-compute only)"
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or "").split(".")[0]
            if top not in _HELPER_IMPORT_ALLOW:
                return f"import from {node.module!r} not allowed (helpers are pure-compute only)"
        elif isinstance(node, ast.Name) and node.id in _HELPER_BANNED_NAMES:
            return f"use of {node.id!r} not allowed in a helper module"
        elif isinstance(node, ast.Attribute) and node.attr in _HELPER_BANNED_DUNDERS:
            return f"attribute {node.attr!r} not allowed in a helper module"
    return None


@hidden
async def _tail_from_start(path: str, poll_interval: float = 0.5):
    """Like runtime.producers.tail() but starts at position 0, not EOF.

    The harness may write state 0 between the agent_ready marker and the tail
    task's first run — seeking to EOF (as producers.tail does) would silently
    drop it and deadlock the turn loop. Reading from the start makes the
    attach timing irrelevant (the file is always fresh per run).
    """
    fh = open(path)
    try:
        while True:
            pos = fh.tell()
            line = fh.readline()
            if line.endswith("\n"):
                if line.strip():
                    yield line.rstrip("\n")
            else:
                fh.seek(pos)  # partial or no line — wait for the writer
                await asyncio.sleep(poll_interval)
    finally:
        fh.close()


@hidden
def _effort_for_elapsed(elapsed: float, ladder: list[tuple[float, str]]) -> str | None:
    """Pick the reasoning effort for a turn that has been running ``elapsed`` seconds.

    ``ladder`` is ascending ``[(after_seconds, effort), …]`` (rung 0 is the
    primary effort the client was built with). Returns the effort of the last
    rung whose threshold ``elapsed`` has crossed — so a turn spends its later
    minutes at a cheaper effort. Mirrors beam's ``REASONING_EFFORT_LEVELS``
    ladder, but stepped by wall-clock rather than by retry count.
    """
    effort = ladder[0][1] if ladder else None
    for after_s, level in ladder:
        if elapsed >= after_s:
            effort = level
        else:
            break
    return effort


# The environment's canonical 16-color palette (index -> RGB), the same colors the
# reference renders grid PNGs with. Hidden from exec_globals: the name is only used
# by the @hidden renderer below and must not surface on the agent's tool surface.
with hidden:
    _PALETTE_16 = (
        (255, 255, 255),
        (204, 204, 204),
        (153, 153, 153),
        (102, 102, 102),
        (51, 51, 51),
        (0, 0, 0),
        (229, 58, 163),
        (255, 123, 204),
        (249, 60, 49),
        (30, 147, 255),
        (136, 216, 241),
        (255, 220, 0),
        (255, 133, 27),
        (146, 18, 49),
        (79, 204, 48),
        (163, 86, 214),
    )


@hidden
def _grid_png_bytes(grid, scale: int) -> bytes:
    """Render a 2D palette-index grid to PNG bytes at ``scale``x — a
    (scale*H)x(scale*W) RGB image (out-of-range colors -> black, index 5),
    matching image_artifacts.grid_to_rgb_array. For a 64x64 grid at scale s the
    PNG is (s*64)x(s*64)."""
    s = max(int(scale or 1), 1)
    arr = np.asarray(grid, dtype=np.int16)
    palette = np.asarray(_PALETTE_16, dtype=np.uint8)
    safe = np.where((0 <= arr) & (arr < len(palette)), arr, 5).astype(np.intp)
    rgb = palette[safe]
    if s > 1:
        rgb = np.repeat(np.repeat(rgb, s, axis=0), s, axis=1)
    buf = io.BytesIO()
    _PILImage.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


class ArcSolverBase(InteractiveAgent):
    """You are an interactive grid-game-solving agent. You discover the unknown rules of an
    interactive 64x64 grid game by experimenting, then solve all its levels.
    Follow the <arc_skill> context block: observe with code (never squint), one
    experiment per weak hypothesis, keep a world model in helper files, record
    knowledge after every turn, and always end a turn by submitting actions.
    """

    # Producer-side channel (hidden) + LLM-facing reader for harness states.
    _game_states_in: Annotated[Channel, hidden, nosnapshot]
    game_states: Annotated[_ChannelReader, nosnapshot]
    # Namespace of loaded helper modules: after load_helpers(), call e.g.
    # self.h.world_model.predict(...). HIDDEN from the rendered <state> block:
    # a loaded module's repr is "<module 'x' from '/abs/run_dir/.../x.py'>", and
    # the run-dir path embeds the game code + benchmark name — an identity leak
    # into the agent's prompt every turn once helpers are loaded. `nosnapshot`
    # alone did NOT hide it from state; the agent still learns about self.h.<name>
    # from load_helpers()'s report (which prints no path) and the method docs.
    h: Annotated[SimpleNamespace, hidden, nosnapshot]
    skills: Annotated[SkillRegistry, nosnapshot]
    # Filesystem paths are hidden from the LLM surface: the run dir embeds
    # `results/arc_agi_3/…/<game>_<variant>/`, which would leak the benchmark
    # name and the real game id into the rendered <self>/<state> blocks.
    run_dir: Annotated[Path, hidden]
    workspace: Annotated[Path, hidden]
    helpers_dir: Annotated[Path, hidden]
    ipc_dir: Annotated[Path, hidden]

    def __init__(
        self,
        llm=None,
        *,
        run_dir: str | Path,
        game_id: str = "",
        alias: str = "",
        reflect_every: int = 8,
        skill_path: str | Path | None = None,
        effort_ladder: list[tuple[float, str]] | None = None,
        visual: str = "off",
        png_scale: int = 8,
        max_actions_per_turn: int = MAX_ACTIONS_PER_TURN,
        **kwargs,
    ):
        super().__init__(llm=llm, **kwargs)
        # Per-turn action batch cap enforced by submit_actions. <=0 disables the
        # cap entirely (the harness must be launched with the same setting so its
        # backstop truncation is off too — run_multi forwards one config key to both).
        self._max_actions_per_turn = int(max_actions_per_turn)
        # Visual input mode (off|only|additive): 'only'/'additive' auto-show the
        # current settled grid to the LLM each turn as a color PNG
        # ((png_scale*64)x(png_scale*64), 16-color palette) — the picture the
        # reference solver sends, last grid only (no event frames). 'only' also drops the hex
        # grid_rows from the state (harness side) so the image REPLACES the grid;
        # 'additive' keeps both.
        self._visual = str(visual or "off")
        self._png_scale = int(png_scale)
        # Effort ladder: a turn's reasoning_effort steps DOWN this ascending
        # [(after_seconds, effort), …] ladder (rung 0 = the client's primary
        # effort) as its wall-clock grows. The clock is "time since we last
        # submitted an action batch" — the same silence the harness times for its
        # nudge/kill — so a slow multi-call turn spends its later minutes cheaper.
        self._turn_started_at = time.monotonic()
        self._effort_ladder: list[tuple[float, str]] = sorted(effort_ladder or [])
        self._ladder_active_effort: str | None = (
            self._effort_ladder[0][1] if self._effort_ladder else None
        )
        self._ladder_downshifts = 0
        self.run_dir = Path(run_dir)
        # game_id is the real id (kept off the agent surface). The agent only
        # ever sees `alias` — an opaque per-run handle carrying no game identity
        # — so it cannot recognise the game and recall training-data knowledge.
        self.game_id = game_id
        self.alias = alias or "the game"
        self.reflect_every = reflect_every
        self.workspace = self.run_dir / "team_nemo" / "shared"
        self.helpers_dir = self.workspace / "helpers"
        self.ipc_dir = self.run_dir / "ipc"
        self._states_path = self.ipc_dir / "states.jsonl"
        self._actions_path = self.ipc_dir / "actions.jsonl"
        for d in (self.workspace, self.helpers_dir, self.ipc_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._states_path.touch()
        self._actions_path.touch()

        # Jail the shadowed open() to this agent's own workspace + IPC, so a
        # generated cell cannot open the game source or another run (L1.5).
        _register_jail_roots(self.workspace, self.ipc_dir)

        # Scrub identity-bearing paths from cell output before the agent sees it:
        # tracebacks/reprs echo the run-dir absolute path, which embeds
        # arc_agi_3 + the game id. Redact them in an execute_python middleware.
        self._redactions = _build_redactions(self.run_dir)

        async def _redact_cell_output(ctx, nxt):
            ctx = await nxt(ctx)
            r = getattr(ctx, "result", None)
            if r is not None:
                upd = {}
                for field in ("stdout", "stderr", "error"):
                    val = getattr(r, field, None)
                    if isinstance(val, str) and val:
                        red = _redact(val, self._redactions)
                        if red != val:
                            upd[field] = red
                if upd:
                    ctx.result = r.model_copy(update=upd)
            return ctx

        self.event_manager.intercept("execute_python", _redact_cell_output)

        # Cell guard (B-lite): if a cell reaches for an escape tool, a network
        # entrypoint, or a denylist-bypass gadget, replace the cell with a refusal
        # so the model gets feedback and nothing runs. No runtime network patching
        # (that would also block concurrent background reflection LLM calls) — the
        # block is purely at the cell-code layer. Defense-in-depth over the uid-drop.
        async def _guard_cell(ctx, nxt):
            code = getattr(ctx, "code", None)
            if isinstance(code, str):
                bad = _scan_cell(code)
                if bad:
                    ctx.code = (
                        f"print('BLOCKED: {bad} is not permitted in agent cells "
                        "(host tool / network / bypass primitive). Use self.memory, "
                        "the grid methods and numpy only.')"
                    )
            return await nxt(ctx)

        self.event_manager.intercept("execute_python", _guard_cell)

        self.h = SimpleNamespace()

        # Defense in depth: the framework renders the <state> block as
        # pformat(self, ...), which bypasses the cell-output redaction above — a
        # value repr (e.g. a loaded module's path, a stray Path) can re-expose the
        # run-dir path (game code + benchmark). Override the protected block with a
        # redacted renderer so ANY such leak is scrubbed, not just today's known one.
        self.context_manager.set_dynamic_protected("state", "self._render_state()")

        # Harness states arrive on their own queue: tail the JSONL file the
        # harness appends to, so every new state wakes the dispatcher.
        self._game_states_in = self.queue_manager.queue("game_states")
        self.game_states = self._game_states_in.reader
        self.queue_manager.spawn(
            _tail_from_start(str(self._states_path)),
            channel="game_states",
            label="game-state tail",
        )

        # Skill registry (bootstrap and the memory skill require it) + LLM
        # access to context/events, mirroring TUIAgent.
        self.skills = SkillRegistry(self)
        spec(self, "context", hidden=False)
        spec(self, "events", hidden=False)
        # Long games overflow the context window — compress older history.
        install_summarizer(SummarizationConfig(), agent=self)

        if skill_path is not None:
            self.context["arc_skill"] = Path(skill_path).read_text()
        self.context.set_dynamic("arc_game", "self.format_game_context()")

        # Step reasoning_effort down the ladder as a turn runs long (wraps
        # self._llm.acall — must come after super().__init__ set self._llm).
        self._install_effort_ladder()

        # tail() starts at end-of-file — the harness must not write state 0
        # before the tail producer above is attached. This marker releases it.
        (self.ipc_dir / "agent_ready").write_text(str(time.time()))

    @hidden
    def _install_effort_ladder(self) -> None:
        """Wrap ``self._llm.acall`` so each call's ``reasoning_effort`` follows the
        ladder for how long the current turn has run.

        A single already-running call can't be downshifted mid-flight — only the
        NEXT call picks a lower effort — so this bounds slow *multi-call* turns,
        not a single hung request (for that, see the harness kill / retry bounds).
        Injects ``reasoning_effort`` as a per-call kwarg, which overrides the
        client's built-in effort (``api_params = {**self.config, **kwargs}``)
        without building or mutating a second client.
        """
        ladder = self._effort_ladder
        if len(ladder) < 2:
            return  # only the primary rung — nothing to step down to
        llm = self._llm
        orig_acall = getattr(llm, "acall", None)
        if orig_acall is None:
            return
        primary = ladder[0][1]

        async def _laddered_acall(*args, **kwargs):
            if "reasoning_effort" not in kwargs:
                elapsed = time.monotonic() - self._turn_started_at
                effort = _effort_for_elapsed(elapsed, ladder)
                if effort is not None and effort != primary:
                    kwargs["reasoning_effort"] = effort
                    if effort != self._ladder_active_effort:
                        self._ladder_active_effort = effort
                        self._ladder_downshifts += 1
            return await orig_acall(*args, **kwargs)

        llm.acall = _laddered_acall

    # ------------------------------------------------------------------ state

    @hidden
    def _render_state(self) -> str:
        """Redacted replacement for the framework's default ``<state>`` block
        (``pformat(self, …)``). Scrubs run-dir path prefixes + the game/benchmark
        tokens so no value repr leaks the game identity into the prompt."""
        return _redact(
            pformat(self, max_length=50, max_string=500, max_depth=4),
            self._redactions,
        )

    @hidden
    def _latest_state(self) -> dict | None:
        lines = self._states_path.read_text().strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return json.loads(line)
        return None

    @hidden
    def _last_submitted_turn(self) -> int | None:
        lines = self._actions_path.read_text().strip().splitlines()
        for line in reversed(lines):
            line = line.strip()
            if line:
                return json.loads(line).get("turn")
        return None

    def trajectory(
        self,
        per_step: bool = True,
        last_n: int | None = None,
        include_grids: bool = True,
        include_events: bool = False,
    ) -> list[dict]:
        """Programmatic access to the FULL game history (like the reference's
        trajectory variable) — use it to diff distant states, detect
        periodicity, replay what an action did, or count effects.

        per_step=True (default): one entry per executed env action —
        step, turn, action, levels_completed, diff_pixels, event_frame_count
        (how many animation frames the action produced) and grid_rows (the 64
        hex rows AFTER the action; parse with self.grid_array). per_step=False:
        one entry per harness turn (the raw state lines: turn, step, state,
        action_results, diff_summary, grid_rows, ...). last_n keeps only the
        most recent N entries; include_grids=False drops grids (cheap for long
        histories). include_events=True (opt-in — frames are large) adds
        event_frames: the per-action ANIMATION, a list of up to ~5 evenly-spaced
        intermediate 64-hex-row grids between the pre- and post-action states
        (empty when the action didn't animate); parse each with self.grid_array.
        Assign to a variable and compute — avoid printing whole entries.
        """
        if per_step:
            events_path = self.run_dir / "agent_logs" / "nemo" / "team_leader" / "events.jsonl"
            entries: list[dict] = []
            turn = None

            def _hex_rows(g):
                return [
                    "".join("0123456789abcdef"[v] if 0 <= v <= 15 else "?" for v in row)
                    for row in g
                ]

            if events_path.exists():
                for line in events_path.read_text().splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if e.get("event") == "agent_turn":
                        turn = e.get("turn")
                    elif e.get("event") == "env_step":
                        entry = {
                            "step": e.get("step"),
                            "turn": turn,
                            "action": e.get("action_name"),
                            "levels_completed": e.get("levels_completed"),
                            "diff_pixels": e.get("diff_pixels"),
                            "event_frame_count": e.get("event_frame_count", 0),
                        }
                        if include_grids and e.get("grid_data") is not None:
                            entry["grid_rows"] = _hex_rows(e["grid_data"])
                        if include_events and e.get("event_frames"):
                            entry["event_frames"] = [_hex_rows(f) for f in e["event_frames"]]
                        entries.append(entry)
        else:
            entries = []
            for line in self._states_path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    s = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not include_grids:
                    s = {k: v for k, v in s.items() if k != "grid_rows"}
                entries.append(s)
        if last_n is not None:
            entries = entries[-last_n:]
        return entries

    def game_status(self) -> dict:
        """Current game status: latest state header (without the grid) plus whether
        an action batch was already submitted for this turn."""
        state = self._latest_state()
        if state is None:
            return {"status": "no game state received yet — wait for the harness"}
        header = {k: v for k, v in state.items() if k not in ("grid_rows",)}
        header["submitted_this_turn"] = self._last_submitted_turn() == state.get("turn")
        return header

    def latest_state(self) -> dict:
        """Return the FULL latest game state (including grid_rows), re-read from
        the harness state file. Use whenever the game_states notification is
        missing or truncated — never wait for a state you can fetch yourself."""
        state = self._latest_state()
        return state if state is not None else {"status": "no game state yet"}

    @hidden
    def format_game_context(self) -> str:
        state = self._latest_state()
        if state is None:
            return f"game={self.alias} — waiting for the first state from the harness."
        submitted = self._last_submitted_turn() == state.get("turn")
        parts = [
            f"game={self.alias}",
            f"turn={state.get('turn')}",
            f"env_step={state.get('step')}",
            f"level={state.get('level')}",
            f"state={state.get('state')}",
            f"levels_completed={state.get('levels_completed')}",
            f"available_actions={state.get('available_actions')}",
            f"submitted_this_turn={submitted}",
        ]
        line = " | ".join(str(p) for p in parts)
        hint = self._knowledge_status_line()
        return line + ("\n" + hint if hint else "")

    @hidden
    def _knowledge_status_line(self) -> str:
        return ""

    # ------------------------------------------------------------------ grid

    def grid_array(self, grid_rows: list[str]) -> np.ndarray:
        """Parse the state's ``grid_rows`` (64 hex strings) into a 64x64 numpy int
        array. Index as arr[row, col]; row 0 is the top of the screen."""
        return np.array([[int(c, 16) for c in row] for row in grid_rows], dtype=int)

    def render_grid(self, grid_rows: list[str], every: int = 4) -> str:
        """Human-readable grid text with row/column coordinate rulers every
        ``every`` cells. Colors are hex digits 0-f."""
        header = "    " + "".join(f"{c:<{every}}" for c in range(0, len(grid_rows[0]), every))
        lines = [header]
        for r, row in enumerate(grid_rows):
            lines.append(f"{r:>3} {row}")
        return "\n".join(lines)

    def grid_image(self, grid_rows: list[str] | None = None) -> Image:
        """The current settled grid as a color PNG Image (standard 16-color palette,
        each cell an N×N block so a 64×64 grid is (N·64)×(N·64) px). Pass grid_rows to render
        a specific grid; default is the latest state's grid. Look at it with
        show(self.grid_image()); when visual input is enabled it is auto-shown each
        turn. Colors match the hex digits in grid_rows (parse those with
        self.grid_array for exact values)."""
        if grid_rows is None:
            grid_rows = (self._latest_state() or {}).get("grid_rows") or []
            if not grid_rows:  # 'only' mode drops grid_rows from state — use trajectory
                traj = self.trajectory(last_n=1, include_grids=True)
                grid_rows = traj[-1].get("grid_rows") if traj else []
        arr = self.grid_array(grid_rows) if grid_rows else np.zeros((64, 64), dtype=int)
        return Image.from_bytes(_grid_png_bytes(arr, self._png_scale), media_type="image/png")

    # ---------------------------------------------------------------- actions

    def submit_actions(self, actions: list[str], rationale: str) -> str:
        """Submit the action sequence for the CURRENT turn — this is how you play.

        ``actions``: one or more of the game's available actions (subject to the
        run's per-turn cap, if one is configured — an oversized batch executes
        only its first ``cap`` actions and the yield explanation + next state's
        note tell you what was NOT executed), e.g.
        ["UP", "UP", "RIGHT", "USE"] or ["CLICK 32 15"] (CLICK is x y = col row).
        The harness executes them in order (stopping early on level completion,
        WIN or GAME_OVER) and then publishes the next state on ``game_states``.
        ``rationale``: one or two lines — your prediction of what this sequence
        will do; you will check it against the next state's action_results.

        A successful submit ENDS your turn (it yields until the next state arrives),
        so do any knowledge/helper writes BEFORE calling this — code after it in the
        same cell will not run. On invalid input it instead returns a ``"REJECTED: …"``
        string (no yield) so you can fix and resubmit in the same turn.
        """
        state = self._latest_state()
        if state is None:
            return "REJECTED: no game state received yet."
        if state.get("state") == "WIN":
            return "REJECTED: game already won — no more actions needed."
        if not actions:
            return "REJECTED: empty action list."
        available = set(state.get("available_actions", []))
        for a in actions:
            base = a.split()[0] if a.strip() else ""
            if base not in available:
                return (
                    f"REJECTED: {a!r} — base action {base!r} not in available {sorted(available)}."
                )
            if base == "CLICK":
                m = _CLICK.match(a.strip())
                if not m or not all(0 <= int(g) <= 63 for g in m.groups()):
                    return f"REJECTED: {a!r} — CLICK needs 'CLICK x y' with 0<=x,y<=63."
        turn = state.get("turn")
        if self._last_submitted_turn() == turn:
            return f"REJECTED: already submitted for turn {turn} — wait for the next state."
        # Over-cap batches execute their first ``cap`` actions instead of bouncing:
        # a REJECTED round-trip costs a full model call, while the useful prefix
        # can play now. The drop is LOUD — flagged here in the yield explanation
        # and relayed by the harness as a note on the next state (truncated_from).
        cap = getattr(self, "_max_actions_per_turn", MAX_ACTIONS_PER_TURN)
        requested = len(actions)
        truncation_warning = ""
        if cap > 0 and requested > cap:
            actions = actions[:cap]
            truncation_warning = (
                f"; WARNING: batch had {requested} actions — only the first {cap} were "
                f"submitted, actions {cap + 1}..{requested} were NOT executed"
            )
        entry = {"turn": turn, "actions": actions, "rationale": rationale[:2000]}
        if truncation_warning:
            entry["truncated_from"] = requested
        with self._actions_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        # New turn starts now: reset the effort-ladder clock (mirrors the harness
        # resetting its silence timer when it receives this batch) and rewind to
        # the primary effort.
        self._turn_started_at = time.monotonic()
        self._ladder_active_effort = self._effort_ladder[0][1] if self._effort_ladder else None
        raise _ReturnResultSignal(
            result={
                "kind": RespondReason.WAIT,
                "explanation": f"submitted {len(actions)} action(s) for turn {turn}"
                f"{truncation_warning}; waiting for the next state",
            }
        )

    # ---------------------------------------------------------------- helpers

    def write_helper(self, filename: str, source: str) -> str:
        """Create or overwrite a persistent python helper module in the workspace
        (``helpers/<filename>``). Use for your evolving world model: parsing,
        entity extraction, predict(state, action) functions. Syntax-checked on
        write; call load_helpers() afterwards to (re)load into ``self.h``."""
        import textwrap

        if not _HELPER_NAME.match(filename):
            return f"REJECTED: filename {filename!r} must match [a-z0-9_]+.py"
        source = textwrap.dedent(source)  # keep the written file importable (see _check_helper_ast)
        err = _check_helper_ast(source)
        if err is not None:
            return f"REJECTED: {err}"
        (self.helpers_dir / filename).write_text(source)
        return f"wrote helpers/{filename} ({len(source)} chars). Call load_helpers() to use it."

    def read_helper(self, filename: str) -> str:
        """Read back a helper module's source from the workspace."""
        p = self.helpers_dir / filename
        if not _HELPER_NAME.match(filename) or not p.exists():
            return f"no such helper: {filename!r} (have: {[q.name for q in self.helpers_dir.glob('*.py')]})"
        return p.read_text()

    def load_helpers(self) -> str:
        """(Re)load every ``helpers/*.py`` module into ``self.h.<module_name>``.
        Returns the loaded modules and their public functions. Call at the start
        of a turn to reuse the world model you built in earlier turns."""
        report = []
        for f in sorted(self.helpers_dir.glob("*.py")):
            # Re-check before executing: a seeded/older helper could predate the
            # write-time guard, so importing it unchecked would reopen the hole.
            err = _check_helper_ast(f.read_text())
            if err is not None:
                report.append(f"{f.name}: BLOCKED — {err}")
                continue
            try:
                spec = importlib.util.spec_from_file_location(f"arc_helper_{f.stem}", f)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
            except Exception as e:
                report.append(f"{f.name}: FAILED to load — {e!r}")
                continue
            setattr(self.h, f.stem, mod)
            fns = [n for n in dir(mod) if not n.startswith("_") and callable(getattr(mod, n))]
            report.append(f"{f.name} -> self.h.{f.stem} ({', '.join(fns) or 'no functions'})")
        return "\n".join(report) or "no helper modules yet — write_helper() to create one."

    # ----------------------------------------------------------------- turn

    @hidden
    @strategy(CodeActStrategy(config=_ARC_CELL_CONFIG))
    async def handle(self, notification: dict[str, list]) -> RespondResult:
        """One interactive-grid-game solving turn. Follow the <arc_skill> context block.

        ``notification`` maps channel name to new items:

        - "game_states": raw JSON lines from the harness — parse with
          json.loads(line). Fields: turn, step, level, state, levels_completed,
          available_actions, action_results (what each action of your previous
          submission actually did: action, reward, diff_pixels, level_completed,
          state, event_frame_count), grid_rows (64 strings of 64 hex digits,
          row 0 = top; parse with self.grid_array), diff_summary (which cells
          changed since the previous state), optionally note (harness warnings),
          and optionally event_summary — present only when an action ANIMATED:
          {frames_by_action, total_frames}. The settled grid_rows hide the
          motion between states; call self.trajectory(include_events=True) to
          inspect the intermediate animation frames of any step. Multiple
          queued states mean your view was stale — act on the LAST one. If the
          notification content is missing or truncated, call
          self.latest_state() — it always returns the full current state.
        - "user_messages": operational guidance injected into the run.
          It OVERRIDES your current plan — acknowledge it with message() and
          follow it.

        Per-turn workflow:
        1. Parse the newest state; check your previous rationale/prediction
           against action_results and diff_summary. A wrong prediction is
           information — record it.
        2. Consult your knowledge store (see <knowledge_api>) BEFORE deciding.
        3. Analyze the grid in code (self.grid_array + numpy; self.load_helpers()
           to reuse your world model; self.trajectory() for the full history of
           executed actions and their grids — diff any two points in time).
        4. Record new observations / hypotheses / action semantics per
           <knowledge_api>; reflect at the cadence the skill prescribes.
        5. Decide the next experiment or plan step, then END the turn by calling
           self.submit_actions([...], rationale="prediction...") with at least one
           action — a successful submit yields until the next state arrives (no
           separate return_result call is needed; do steps 1-4 before submitting).

        If state == "WIN": write your final level reflection to the knowledge
        store, message() a short victory summary, then
        return_result(RespondReason.DONE, explanation="game solved").
        If the game is stuck/over (harness note says so), summarize what you
        learned and return_result(RespondReason.DONE, ...).
        Never end a turn without either submitting actions or (only when the
        game is finished) reporting the outcome.
        """
        # Prefill: when visual input is on (only|additive), show the current grid
        # PNG so the LLM sees the colored picture this turn. In 'only' the harness
        # has dropped the hex grid_rows from the state (image replaces the grid); in
        # 'additive' both are present.
        if self._visual != "off":
            show(self.grid_image())
        ...


class MemArcSolverAgent(ArcSolverBase):
    """You are an interactive grid-game-solving agent. You discover the unknown rules of an
    interactive 64x64 grid game by experimenting, then solve all its levels.
    Follow the <arc_skill> context block. Your accumulated knowledge lives in
    your long-term MEMORY (self.memory.*) — recall before deciding, remember
    after observing, reflect to consolidate.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The launcher registers MemorySkill (self.memory) after bootstrap —
        # this block documents the contract the skill expects the LLM to follow.
        self.context["knowledge_api"] = (
            "KNOWLEDGE API (memory variant):\n"
            "- self.memory.recall(query, k=5) — associative recall (also happens\n"
            "  spontaneously each turn: see <recalled_memories>).\n"
            "- self.memory.search(query, k=5) — term-focused lookup.\n"
            "- self.memory.remember(content, type=..., importance=..., tags=[...]) — write one\n"
            "  durable fact. Types: 'info' (observations, action semantics),\n"
            "  'skill' (verified reusable procedure/strategy: include applicability),\n"
            "  'todo' (open experiment to run; close with update_memory(id, status='DONE')),\n"
            "  'episode' (what happened in a level attempt).\n"
            "- self.memory.update_memory(id, ...) — revise/close; use for hypothesis\n"
            "  status changes (confirmed/contradicted — say which evidence).\n"
            "- self.memory.associate(a_id, b_id, relation) — link related facts.\n"
            f"- self.memory.reflect() — consolidate. Call after each level completion\n"
            f"  and roughly every {self.reflect_every} turns; set self.v.last_reflect_turn.\n"
            "Store CURATED knowledge, not raw grids: one fact per memory, with the\n"
            "evidence. Tag with the game id and level."
        )

    @hidden
    def _knowledge_status_line(self) -> str:
        last = self.vars.get("last_reflect_turn") if hasattr(self.vars, "get") else None
        state = self._latest_state() or {}
        turn = state.get("turn", 0) or 0
        if last is None:
            return f"memory: no reflection yet (reflect every ~{self.reflect_every} turns)"
        return f"memory: last reflection at turn {last} (now {turn}; reflect every ~{self.reflect_every})"


class MdArcSolverAgent(ArcSolverBase):
    """You are an interactive grid-game-solving agent. You discover the unknown rules of an
    interactive 64x64 grid game by experimenting, then solve all its levels.
    Follow the <arc_skill> context block. Your accumulated knowledge lives in
    markdown files (read_notes/append_notes/write_notes) — read them before
    deciding, append after observing, curate them periodically.
    """

    knowledge_dir: Annotated[Path, hidden]  # path embeds the run dir — hide it

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge_dir = self.workspace / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.context["knowledge_api"] = (
            "KNOWLEDGE API (markdown variant):\n"
            "- self.list_notes() / self.read_notes(name) — read before deciding.\n"
            "- self.append_notes(name, content) — add entries after observing.\n"
            "- self.write_notes(name, content) — rewrite a file when curating.\n"
            "Keep these files (create on demand):\n"
            "- step_log.md — per-turn record: state, hypothesis checked, result,\n"
            "  and a Carry-Forward section with everything worth preserving.\n"
            "- hypotheses.md — hypothesis | status (untested/confirmed/contradicted)\n"
            "  | evidence table; update statuses as evidence arrives.\n"
            "- level_log.md — on each level completion: mechanic, winning strategy,\n"
            f"  what generalizes. Curate/rewrite files roughly every {self.reflect_every}\n"
            "  turns (this is your reflection); set self.v.last_reflect_turn."
        )

    @hidden
    def _knowledge_status_line(self) -> str:
        names = [p.name for p in sorted(self.knowledge_dir.glob("*.md"))]
        last = self.vars.get("last_reflect_turn") if hasattr(self.vars, "get") else None
        return f"notes: {names or 'none yet'} | last curation turn: {last}"

    def list_notes(self) -> list[str]:
        """List the knowledge markdown files in the workspace."""
        return [p.name for p in sorted(self.knowledge_dir.glob("*.md"))]

    def read_notes(self, name: str) -> str:
        """Read one knowledge file (e.g. 'hypotheses.md')."""
        if not _NOTE_NAME.match(name):
            return f"REJECTED: name {name!r} must match [a-z0-9_-]+.md"
        p = self.knowledge_dir / name
        return p.read_text() if p.exists() else f"(no file {name} yet — have {self.list_notes()})"

    def append_notes(self, name: str, content: str) -> str:
        """Append an entry to a knowledge file (created if missing)."""
        if not _NOTE_NAME.match(name):
            return f"REJECTED: name {name!r} must match [a-z0-9_-]+.md"
        p = self.knowledge_dir / name
        with p.open("a") as f:
            f.write(content.rstrip() + "\n\n")
        return f"appended {len(content)} chars to {name}"

    def write_notes(self, name: str, content: str) -> str:
        """Overwrite a knowledge file — use when curating/consolidating."""
        if not _NOTE_NAME.match(name):
            return f"REJECTED: name {name!r} must match [a-z0-9_-]+.md"
        (self.knowledge_dir / name).write_text(content)
        return f"wrote {name} ({len(content)} chars)"
