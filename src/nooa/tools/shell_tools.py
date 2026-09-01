# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ShellTools — persistent shell + file ops with grep that hands you editable Match objects.

Identical surface to v4 (run / read / replace / write_file). The one addition:
when ``run()`` executes a *pure search* (a bare grep/rg/egrep with no
output-mangling pipe or anchor-dropping flag), the result still prints as the
exact bytes the agent's command produced — but it ALSO carries a parsed
``.matches`` list of ``Match`` objects, ready to hand straight to ``replace()``.

The agent sees no new method and no changed output. Under the hood we run the
equivalent ``rg --json`` purely to harvest anchors, and attach the matches ONLY
when we can prove the anchor set is trustworthy. Any divergence, any unhandled
flag, any pipe -> ``.matches`` is ``None`` (fail-closed). An incomplete gate can
only ever *miss* an opportunity to help; it can never produce a wrong anchor.

Motivation: in the SWE-bench bake-off the agent issued ~12 search calls/session
and used the structured ``.matches()`` path 0 times — it greps and eyeballs
text. Making every safe grep an on-ramp to a Match-based edit attacks the #1
error class (string-escaping in inline edits) for free.

Attach to an agent::

    class MyAgent(Agent, llm=llm):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.shell = ShellTools(cwd="/path/to/repo")
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

from nooa.agentdoc import hidden, spec
from nooa.skill import Skill
from nooa.tools._bash_session import BashSession
from nooa.tools._results import StreamDone, StreamEvent


class FileWrite:
    """Result of a write/replace operation.

    ``new_text`` is the text as it was actually written, after any
    normalization the write applied. Observers report from it so what they
    record matches what landed on disk. It is deliberately left out of
    ``__str__`` — the agent already knows the text it asked for.
    """

    def __init__(self, path: str, message: str, diff: str = "", new_text: str = ""):
        self.path = path
        self.message = message
        self.diff = diff
        self.new_text = new_text

    def __str__(self) -> str:
        parts = [self.message]
        if self.diff:
            parts.append(self.diff)
        return "\n".join(parts)

    def __repr__(self) -> str:
        return str(self)


class Match:
    """Editable file anchor returned by file/search tools.

    Pass any ``Match`` to ``self.shell.replace(match, new_text)``. Print it to
    view numbered lines. Slice with ``match[start:end]`` (1-indexed inclusive
    line numbers) to narrow the region before replacing. ``resolved_path`` is
    required so the anchor remains bound to that file if the shell cwd changes.
    """

    def __init__(
        self,
        path: str,
        start: int,
        end: int,
        text: str,
        *,
        resolved_path: str | Path,
    ):
        self._path = path
        anchor = Path(resolved_path)
        if not anchor.is_absolute():
            # resolve() on a relative path resolves against the *process* cwd,
            # which is not the shell cwd — so a relative anchor would silently
            # point at a different file. Fail loudly instead.
            raise ValueError(f"Match.resolved_path must be absolute, got {str(resolved_path)!r}")
        self._resolved_path = str(anchor.resolve())
        self._start = start
        self._end = end
        self._text = text

    @property
    def path(self) -> str:
        """File path."""
        return self._path

    @property
    def resolved_path(self) -> str:
        """Canonical file path captured when this match was created."""
        return self._resolved_path

    @property
    def start(self) -> int:
        """First line number (1-indexed)."""
        return self._start

    @property
    def end(self) -> int:
        """Last line number (1-indexed, inclusive)."""
        return self._end

    @property
    def text(self) -> str:
        """Raw file content (no line numbers)."""
        return self._text

    @property
    def numbered(self) -> str:
        """Content with line-number gutter."""
        lines = self._text.splitlines(keepends=True)
        width = len(str(self._end))
        numbered = []
        for i, line in enumerate(lines, self._start):
            numbered.append(f"{i:>{width}}| {line.rstrip(chr(10))}")
        return "\n".join(numbered)

    def __str__(self) -> str:
        return self.numbered

    def __repr__(self) -> str:
        return f"Match({self._path!r}, lines {self._start}-{self._end})"

    def __getitem__(self, key: Any) -> Match:
        lines = self._text.splitlines(keepends=True)
        if isinstance(key, slice):
            start = key.start if key.start is not None else self._start
            stop = key.stop if key.stop is not None else self._end
            if start < self._start:
                start = self._start
            if stop > self._end:
                stop = self._end
            idx_start = start - self._start
            idx_end = stop - self._start + 1
            text = "".join(lines[idx_start:idx_end])
            return Match(
                self._path,
                start,
                stop,
                text,
                resolved_path=self._resolved_path,
            )

        raise TypeError(f"indices must be int or slice, not {type(key).__name__}")


class ShellResult(str):
    """Result of run() — a str subclass whose VALUE is stdout.

    String operations (``"x" in r``, ``r.splitlines()``, ``r.strip()``) act on
    stdout, so existing code keeps working. But ``repr(r)`` / ``print(r)`` show
    a structured ``BashOutput(...)`` view that surfaces stderr and a non-zero
    return code as named fields — so failures can't be missed (a crashing
    command with empty stdout no longer prints as blank).

    ``.matches`` is a ``list[Match]`` when the command was a pure search and the
    anchors are trustworthy; otherwise ``None``.
    """

    stdout: str
    stderr: str
    returncode: int
    timed_out: bool
    success: bool
    matches: list[Match] | None

    def __new__(
        cls,
        stdout: str,
        stderr: str = "",
        returncode: int = 0,
        matches: list[Match] | None = None,
        timed_out: bool = False,
    ):
        obj = super().__new__(cls, stdout)
        obj.stdout = stdout
        obj.stderr = stderr
        obj.returncode = returncode
        obj.timed_out = timed_out
        obj.success = returncode == 0
        obj.matches = matches
        return obj

    def __repr__(self) -> str:
        parts = [f"stdout={self.stdout!r}"]
        if self.stderr:
            parts.append(f"stderr={self.stderr!r}")
        if self.returncode != 0:
            parts.append(f"return_code={self.returncode}")
        return f"BashOutput({', '.join(parts)})"

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def text(self) -> str:
        """The structured display text (i.e. ``str(self)``)."""
        return self.__repr__()


# Flag letters that drop the per-line/span anchor or change match semantics so
# that grep and rg can disagree. If a short-flag cluster contains any of these,
# we refuse to attach matches.
#   o = only-matching, c = count, l/L = files-with/without, A/B/C = context,
#   z/Z = NUL data / multiline, P = PCRE (semantics differ from rg default).
_ANCHOR_BREAKING_FLAGS = set("oclLABCDzZP")

# Pipe targets that rewrite columns/lines, breaking the file:line mapping.
_MANGLING_PIPE = re.compile(r"\|\s*(sed|awk|cut|sort|uniq|tr|head|tail|wc|xargs|rev)\b")

_SEARCH_HEAD = re.compile(r"^\s*(grep|egrep|rg)\b")
_LONG_ANCHOR_BREAKING = (
    "--pcre2",
    "--null-data",
    "--count",
    "--files-with-matches",
    "--files-without-match",
    "--only-matching",
    "--multiline",
    "--context",
    "--after-context",
    "--before-context",
)


def _has_unquoted_pipe(cmd: str) -> bool:
    """True if cmd contains a shell pipe ``|`` outside of quotes."""
    in_single = in_double = False
    for ch in cmd:
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "|" and not in_single and not in_double:
            return True
    return False


# Pipes that merely truncate output without mangling line structure.
_SAFE_TAIL_PIPE = re.compile(r"\|\s*(head|tail)(\s+-[0-9n]+)?\s*$")


# Pattern: find ... | xargs grep ... (content search via xargs)
_XARGS_GREP = re.compile(r"find\s+.+\|\s*xargs\s+grep")


def is_pure_search_command(cmd: str) -> bool:
    """True iff ``cmd`` is a bare grep/rg/egrep whose matches map 1:1 to lines.

    Also detects `find ... | xargs grep -n ...` which produces the same
    filename:lineno:content format as grep -rn.

    Conservative by design: anything that could make grep and rg disagree, or
    that mangles the output columns, returns False so the caller attaches no
    matches. This is the gate that keeps the feature fail-closed.
    """
    c = cmd.strip()
    # Strip shell prefix commands: "cd /path &&", "pushd /x &&", etc.
    if "&&" in c:
        c = c.split("&&")[-1].strip()
    # Strip safe tail pipes (| head -N, | tail -N) before checking.
    c_no_tail = _SAFE_TAIL_PIPE.sub("", c).strip()
    # Detect "find ... | xargs grep ..." — extract the grep portion
    if _XARGS_GREP.search(c_no_tail):
        xargs_idx = c_no_tail.find("xargs grep")
        grep_part = "grep" + c_no_tail[xargs_idx + len("xargs grep") :]
        # Strip any trailing | head/tail from the grep part
        grep_part = _SAFE_TAIL_PIPE.sub("", grep_part).strip()
        # Check for anchor-breaking flags on the grep portion
        for cluster in re.findall(r"(?:^|\s)-([A-Za-z]+)", grep_part):
            if set(cluster) & _ANCHOR_BREAKING_FLAGS:
                return False
        # Must have -n somewhere for line numbers
        has_n = any("n" in cl for cl in re.findall(r"(?:^|\s)-([A-Za-z]+)", grep_part))
        return has_n
    if not _SEARCH_HEAD.match(c_no_tail):
        return False
    # A pipe to anything that reshapes lines/columns invalidates anchors.
    if _MANGLING_PIPE.search(c_no_tail):
        return False
    # A real shell pipe (outside quotes) that isn't head/tail — refuse.
    if _has_unquoted_pipe(c_no_tail):
        return False
    # Long-form anchor-breaking flags.
    if any(flag in c_no_tail for flag in _LONG_ANCHOR_BREAKING):
        return False
    # Short-flag clusters: -rnP, -o, -A2, etc. Inspect each cluster's letters.
    for cluster in re.findall(r"(?:^|\s)-([A-Za-z]+)", c_no_tail):
        if set(cluster) & _ANCHOR_BREAKING_FLAGS:
            return False
    return True


class ShellTools(Skill):
    """
    Persistent shell + file ops, with grep that hands you editable Match objects.

    Four methods — no new tools to learn:
        run(command, stdin=, timeout=)  — shell command (cd/env/cwd persist)
        read(path, lines=)             — view a file/region -> Match
        replace(match_or_path, ...)    — edit at a Match anchor, or by unique string
        write_file(path, content)      — create/overwrite a file

    Grep that you can edit from directly. When run() executes a plain search
    (grep/rg/egrep), the result still prints the EXACT bytes your command
    produced — and it also carries ``.matches``, a list of Match objects you can
    pass straight to replace(). No re-grep, no parsing the text yourself::

        r = await shell.run("grep -rn 'def foo' src/")
        print(r)                          # byte-accurate grep output, unchanged
        await shell.replace(r.matches[0], new_code)   # edit the first hit

    ``r.matches`` is ``None`` (not an error — just "no structured anchors") when
    the command isn't a verifiable plain search: anything piped into
    sed/awk/cut/sort/head, context/count/only-matching/files-only flags
    (-A/-B/-C/-c/-o/-l), PCRE (-P), or grep without -n. In those cases use the
    text in ``r``/``r.stdout`` as usual. The matches are attached ONLY when they
    provably equal what your own grep reported — so they are never wrong, only
    sometimes absent.

    Editing without a search — view a region, then replace it (no copy-paste of
    the old text, so no quoting/escaping mistakes)::

        region = await shell.read("f.py", lines=(10, 25))  # -> Match
        print(region)                                       # numbered lines
        await shell.replace(region, new_code)               # exact-anchor edit

    Running scripts — pass the payload as ``stdin=`` instead of embedding quotes
    in an inline ``python -c "..."`` (which is the #1 source of syntax errors)::

    """

    def __init__(self, cwd: str = ".", init_command: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.cwd = Path(cwd).resolve()
        # Construct the session eagerly (it starts lazily on first run) so a
        # consumer wired at construction time — e.g. RepoTools(session=shell.session)
        # in the TUI — shares this shell's bash session instead of capturing None.
        # ``init_command`` (if given) runs once on session start, before any user
        # command, to set up the environment (e.g. activate a conda env).
        self._session: BashSession = BashSession(cwd=str(self.cwd), init_command=init_command)

    def __repr__(self) -> str:
        return f"ShellTools(cwd={self.cwd!s})"

    @property
    @hidden
    def session(self) -> BashSession:
        """The underlying persistent bash session (shared with e.g. RepoTools)."""
        return self._session

    async def _get_session(self) -> BashSession:
        if not self._session._started:
            await self._session.start()
        return self._session

    @hidden
    async def close(self) -> None:
        """Terminate the underlying bash session owned by this shell."""
        await self._session.close()

    def _resolve_path(self, path: str) -> Path:
        """Resolve a file-operation path relative to cwd when not absolute."""
        return (self.cwd / path).resolve()

    async def run(
        self,
        command: Annotated[str, spec(description="Shell command to execute")],
        *,
        stdin: Annotated[
            str | None, spec(description="Text piped to stdin (replaces heredocs)")
        ] = None,
        timeout: Annotated[float, spec(description="Max seconds")] = 30.0,
    ) -> ShellResult:
        """
        Run a shell command in the persistent session (cd/env/cwd survive).

        Pass a payload as stdin= instead of heredocs. Result is a str subclass
        with .stdout / .stderr / .returncode / .success.

        If the command is a pure search (grep/rg/egrep, no mangling pipe or
        anchor-dropping flag), the result also carries .matches — a list of
        Match objects you can pass straight to replace(). The printed output is
        always the exact bytes your command produced.

        Args:
            command: Shell command to execute.
            stdin: Text piped to stdin (no quoting needed).
            timeout: Max seconds before timeout.
        """
        session = await self._get_session()
        run_cmd = self._with_stdin(command, stdin)
        stdout, stderr, code, timed_out = await session.run_with_timeout_flag(
            run_cmd, timeout=timeout
        )
        # Track cwd changes for read/replace/write_file path resolution
        pwd_out, _, _, _ = await session.run_with_timeout_flag("pwd", timeout=5.0)
        if pwd_out.strip():
            self.cwd = Path(pwd_out.strip())

        matches: list[Match] | None = None
        _is_search = stdin is None and is_pure_search_command(command)
        if _is_search:
            matches = await self._harvest_matches(command, stdout)

        if matches:
            print(
                f"# self.shell.run({command!r:.60}) found {len(matches)} match(es).\n"
                "# Edit directly: m = <result>.matches[0]; "
                "await self.shell.replace(m, new_text)"
            )

        return ShellResult(
            stdout=stdout,
            stderr=stderr,
            returncode=code,
            timed_out=timed_out,
            matches=matches,
        )

    async def run_stream(
        self,
        command: Annotated[str, spec(description="Shell command to execute")],
        timeout: Annotated[float, spec(description="Max seconds to wait before timeout")] = 30.0,
    ) -> AsyncIterator[StreamEvent | StreamDone]:
        """Stream command output line-by-line as it arrives, ending with a done event.

        Yields ``StreamEvent`` chunks (``.kind`` is "stdout"/"stderr", ``.text``
        the chunk) incrementally, then a final ``StreamDone`` (``.returncode``,
        ``.timed_out``) once the command completes. Runs in the persistent
        session, like ``run``.

        This is what ``pyp.arun(self.shell, ...)`` consumes to stream output::

            fails = await self.pyp.arun(self.shell, "make test").grep("FAIL").collect()
        """
        session = await self._get_session()
        timed_out = False
        exit_code = 0
        async for stream_name, chunk in session.run_stream(command, timeout=timeout):
            if stream_name == "__done__":
                parts = chunk.split(",")
                exit_code = int(parts[0])
                timed_out = bool(int(parts[1])) if len(parts) > 1 else False
                break
            yield StreamEvent(kind=stream_name, text=chunk)
        yield StreamDone(kind="done", returncode=exit_code, timed_out=timed_out)

    @staticmethod
    def _with_stdin(command: str, stdin: str | None) -> str:
        """Wrap a command so ``stdin`` is fed via a base64'd tempfile (no quoting)."""
        if stdin is None:
            return command
        import base64

        b64 = base64.b64encode(stdin.encode()).decode()
        return (
            f"__nemo_in=$(mktemp); base64 -d <<<{b64} > $__nemo_in; "
            f"({command}) < $__nemo_in; __nemo_rc=$?; rm -f $__nemo_in; "
            f"( exit $__nemo_rc )"
        )

    async def _harvest_matches(self, command: str, displayed_stdout: str) -> list[Match] | None:
        """Run the rg --json equivalent and parse anchors — fail-closed.

        Returns a list of Match (possibly empty) only if the rg run succeeds and
        the set of (path, line) anchors it reports is consistent with the lines
        the agent's own command printed. On any doubt, returns None.
        """
        # Strip shell prefix (cd ... &&) and safe tail pipes (| head/tail)
        cmd = command.strip()
        if "&&" in cmd:
            cmd = cmd.split("&&")[-1].strip()
        cmd = _SAFE_TAIL_PIPE.sub("", cmd).strip()
        # For "find ... | xargs grep ...", extract the grep portion
        if _XARGS_GREP.search(cmd):
            xargs_idx = cmd.find("xargs grep")
            cmd = "grep" + cmd[xargs_idx + len("xargs grep") :]
        try:
            pattern, paths, ignore_case, fixed = self._parse_search(cmd)
        except Exception:
            return None
        if pattern is None:
            return None

        args = ["rg", "--json", "-n", "--no-ignore"]
        if ignore_case:
            args.append("-i")
        if fixed:
            args.append("-F")
        args += ["--", pattern, *(paths or ["."])]
        rg_cmd = " ".join(shlex.quote(a) for a in args)
        # Find rg: try PATH, then common overlay/venv locations
        rg_cmd = "PATH=/opt/harbor/cpython312/bin:/opt/harbor/bin:$PATH " + rg_cmd

        session = await self._get_session()
        try:
            rg_out, _rg_err, rg_code, _timed = await session.run_with_timeout_flag(
                rg_cmd, timeout=30.0
            )
        except Exception:
            return None
        # rg: 0 = matches, 1 = no matches. Anything else (bad regex, no rg) -> bail.
        if rg_code not in (0, 1):
            return None

        anchors: list[tuple[str, int]] = []
        file_cache: dict[str, tuple[Path, list[str]]] = {}
        for raw in rg_out.splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError:
                return None  # malformed JSON stream — don't guess
            if d.get("type") != "match":
                continue
            data = d["data"]
            mpath = data["path"]["text"]
            if mpath.startswith("./"):
                mpath = mpath[2:]
            line_no = data["line_number"]
            anchors.append((mpath, line_no))

        # Cross-check: every anchor line must correspond to a line the agent's
        # displayed output reported (when that output is the standard file:line:
        # format). If the displayed output isn't in that shape, we can't verify
        # -> attach nothing.
        #
        # Single explicit file: grep omits the filename, so its output is
        # ``line:...`` (no path). The path is known a priori from the parsed
        # command, so pass it in to let line-only output reconcile against the
        # one file we searched.
        single_file = paths[0] if len(paths) == 1 and not paths[0].endswith("/") else None
        if single_file and single_file.startswith("./"):
            single_file = single_file[2:]
        displayed = self._displayed_anchor_lines(displayed_stdout, single_file=single_file)
        if displayed is None:
            # The command's own output isn't in verifiable file:line: form
            # (e.g. grep without -n). Can't cross-check -> attach nothing.
            return None

        # Reconciliation. Normally the displayed anchors must EQUAL the rg
        # anchors. But a safe ``| head -N`` / ``| tail -N`` truncates the
        # *display* to a prefix/suffix of the full result set, so the agent saw
        # fewer lines than rg reports. When such a tail pipe was present, accept
        # displayed ⊆ rg-anchors (subset) and attach only the lines the agent
        # actually saw — every one of which is still a real, verified rg match,
        # so the anchors are never wrong, only fewer. Without a tail pipe we keep
        # strict equality (any mismatch means the output was reshaped -> bail).
        anchor_set = {(p, n) for p, n in anchors}
        truncated = bool(_SAFE_TAIL_PIPE.search(command.strip()))
        if truncated:
            if not displayed <= anchor_set:
                # Displayed lines that rg didn't report -> output was reshaped,
                # not merely truncated. Can't trust it -> bail.
                return None
            # Attach only what the agent saw, in the order rg reported them.
            keep = [(p, n) for (p, n) in anchors if (p, n) in displayed]
        elif anchor_set != displayed:
            return None
        else:
            keep = anchors

        out: list[Match] = []
        for mpath, line_no in keep:
            if mpath not in file_cache:
                try:
                    resolved = self._resolve_path(mpath)
                    lines = resolved.read_text().splitlines(keepends=True)
                    file_cache[mpath] = (resolved, lines)
                except (OSError, ValueError):
                    return None
            resolved, lines = file_cache[mpath]
            if not (1 <= line_no <= len(lines)):
                return None
            out.append(
                Match(
                    mpath,
                    line_no,
                    line_no,
                    lines[line_no - 1],
                    resolved_path=resolved,
                )
            )
        return out

    @staticmethod
    def _displayed_anchor_lines(
        stdout: str, *, single_file: str | None = None
    ) -> set[tuple[str, int]] | None:
        """Parse `path:line:...` from the agent's own output, or None if not that shape.

        Returns the set of (path, line) the command itself reported. Used to
        verify the rg anchors match what the agent saw.

        ``single_file`` is the one explicit file the search targeted, if any.
        grep omits the filename when searching a single file, so its output is
        ``line:...`` (no path); when we know that file a priori we accept the
        line-only form and attribute every line to ``single_file``. Without it,
        line-only output (e.g. grep without -n, or an unknowable path) returns
        None -> unverifiable.
        """
        found: set[tuple[str, int]] = set()
        any_line = False
        for ln in stdout.splitlines():
            any_line = True
            m = re.match(r"^(?:\./)?([^:]+):(\d+):", ln)
            if m:
                found.add((m.group(1), int(m.group(2))))
            elif single_file is not None:
                # Single-file grep: "line:content" with no path. Attribute it
                # to the known file so it reconciles with the rg anchors.
                lm = re.match(r"^(\d+):", ln)
                if lm:
                    found.add((single_file, int(lm.group(1))))
        if not any_line:
            return set()  # no output -> no matches, verifiable as empty
        if not found:
            return None  # output present but not file:line: shape -> can't verify
        return found

    @staticmethod
    def _parse_search(command: str) -> tuple[str | None, list[str], bool, bool]:
        """Extract (pattern, paths, ignore_case, fixed) from a grep/rg command.

        Best-effort, conservative: returns (None, ...) if the structure is
        anything we don't confidently understand, so harvesting is skipped.
        """
        toks = shlex.split(command)
        if not toks or toks[0] not in ("grep", "egrep", "rg"):
            return None, [], False, False
        ignore_case = False
        fixed = toks[0] == "egrep" and False
        pattern: str | None = None
        paths: list[str] = []
        i = 1
        positional: list[str] = []
        while i < len(toks):
            t = toks[i]
            if t == "--":
                positional.extend(toks[i + 1 :])
                break
            if t.startswith("-") and len(t) > 1:
                # long flags we understand
                if t in ("--ignore-case",):
                    ignore_case = True
                elif t in ("--fixed-strings",):
                    fixed = True
                elif t.startswith("--"):
                    # unknown long flag with potential value — bail to be safe
                    return None, [], False, False
                else:
                    letters = t[1:]
                    for ch in letters:
                        if ch == "i":
                            ignore_case = True
                        elif ch == "F":
                            fixed = True
                        elif ch in ("r", "R", "n", "H"):
                            pass  # recursive / line-number / with-filename: harmless
                        else:
                            return None, [], False, False
                i += 1
                continue
            positional.append(t)
            i += 1
        if not positional:
            return None, [], False, False
        pattern = positional[0]
        paths = positional[1:]
        return pattern, paths, ignore_case, fixed

    async def read(
        self,
        path: Annotated[str, spec(description="File path (relative to cwd or absolute)")],
        lines: Annotated[
            tuple[int, int] | None,
            spec(description="(start, end) 1-indexed inclusive, or None for whole file"),
        ] = None,
    ) -> Match:
        """
        Read a file (or line range) -> Match object.

        Print the Match to see numbered lines. Pass any Match to replace() for editing.
        Slice with match[start:end] to narrow the region.

        Args:
            path: File path (relative to cwd or absolute).
            lines: Optional (start, end) range, 1-indexed inclusive.

        Returns:
            Match with .text, .numbered, .path, .start, .end.
        """
        resolved = self._resolve_path(path)
        content = resolved.read_text()
        all_lines = content.splitlines(keepends=True)
        total = len(all_lines)

        if lines is not None:
            start, end = lines
            start = max(1, start)
            end = min(total, end)
            text = "".join(all_lines[start - 1 : end])
            return Match(str(path), start, end, text, resolved_path=resolved)

        return Match(str(path), 1, total, content, resolved_path=resolved)

    async def replace(
        self,
        target: Annotated[
            Any, spec(description="A Match (from read() or run().matches) or a file path string")
        ],
        old_or_new: Annotated[
            str,
            spec(
                description="For Match: replacement text. For path: text to find (must be unique)"
            ),
        ] = "",
        new: Annotated[
            str | None, spec(description="For path: replacement text. Leave None for Match.")
        ] = None,
    ) -> FileWrite:
        """
        Edit a file — two forms:

        1. replace(match, new_text) — replace the Match's line region.
        2. replace(path, old, new)  — old must match exactly once. new="" deletes.

        Args:
            target: A Match or file path string.
            old_or_new: For Match: the new text. For path: old text to find.
            new: Only for path form: the replacement text.
        """
        if isinstance(target, Match):
            new_text = old_or_new
            resolved = Path(target.resolved_path)
            content = resolved.read_text()
            all_lines = content.splitlines(keepends=True)

            before = all_lines[: target.start - 1]
            after = all_lines[target.end :]
            if new_text and not new_text.endswith("\n") and after:
                new_text += "\n"
            new_content = "".join(before) + new_text + "".join(after)
            resolved.write_text(new_content)

            diff = f"--- a/{target.path}\n+++ b/{target.path}\n"
            diff += f"@@ -{target.start},{target.end - target.start + 1} @@\n"
            return FileWrite(
                path=target.path,
                message=f"Edited {target.path} (replaced lines {target.start}-{target.end})",
                diff=diff,
                new_text=new_text,
            )

        elif isinstance(target, str):
            if new is None:
                raise ValueError(
                    "replace(path, old, new) requires 3 arguments. "
                    "Did you mean replace(match, new_text)?"
                )
            old_text = old_or_new
            resolved = self._resolve_path(target)
            content = resolved.read_text()

            count = content.count(old_text)
            if count == 0:
                raise ValueError(
                    f"old text not found in {target}. "
                    "It must match exactly once — check whitespace and indentation."
                )
            if count > 1:
                raise ValueError(
                    f"old text matched {count} times in {target}. "
                    "It must match exactly once — add surrounding context to make it unique."
                )

            new_content = content.replace(old_text, new, 1)
            resolved.write_text(new_content)

            return FileWrite(
                path=target,
                message=f"Edited {target}",
                diff=f"--- a/{target}\n+++ b/{target}",
                new_text=new,
            )
        else:
            raise TypeError(f"target must be a Match or file path str, got {type(target).__name__}")

    async def write_file(
        self,
        path: Annotated[str, spec(description="File path (relative to cwd or absolute)")],
        content: Annotated[str, spec(description="Full file content")],
    ) -> FileWrite:
        """
        Create or overwrite a file with content (no shell quoting needed).

        Args:
            path: File path (relative to cwd or absolute).
            content: Full file content.
        """
        resolved = self._resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return FileWrite(
            path=path,
            message=f"Created {path} ({line_count} lines)",
            new_text=content,
        )
