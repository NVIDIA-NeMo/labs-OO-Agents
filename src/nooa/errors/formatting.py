# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build concise, source-aware Python diagnostics for language-model feedback.

This module is the presentation boundary between Python exceptions and the text
shown to an agent.  It is intentionally responsible for one complete pipeline:

1. Classify syntax, static-validation, and runtime failures.
2. Render runtime failures with the standard-library ``traceback`` machinery,
   recursively filtering framework frames from causes, contexts, and exception
   groups while preserving generated-cell and downstream user-helper frames.
3. Normalize the result to an IPython-like form (``Cell In[N]`` locations,
   wrapper-line offsets, and hidden internal wrapper names).
4. Add narrowly targeted, best-effort guidance for mistakes that models often
   repeat, such as malformed heredocs and calls with the wrong signature.
5. Bound the final diagnostic before it enters model context.

The module does *not* execute code, register generated source with ``linecache``,
or serialize exceptions across process boundaries.  Those jobs belong to the
runtime and sandbox layers.  A trusted backend may pass its already-rendered
text to :func:`format_error_for_llm`; that preserves worker-side source context
but still applies the final size bound.

Formatting must never replace the original failure with a formatter failure.
Resolution of optional hints is therefore defensive, and the public entry point
falls back to a safe ``TypeName: message`` rendering when necessary.

Typical callers should use :func:`format_error_for_llm`.  The
:class:`IPythonErrorFormatter` class remains available for strategy-specific
formatter injection.
"""

import re
import traceback
from pathlib import Path, PurePath
from typing import Protocol

from nooa.agentdoc import TruncatingStringIO
from nooa.config.truncation_config import DEFAULT_TRUNCATION_CONFIG

# ---------------------------------------------------------------------------
# Traceback selection and IPython normalization policy
# ---------------------------------------------------------------------------

# The checked-out/installed NOOA package root. Match it as a path ancestor,
# never as the bare substring ``"nooa/"``: user repositories may themselves
# live under a directory named ``nooa``.
_NOOA_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_FRAMEWORK_PATH_PARTS = {"site-packages", "dist-packages"}

# User code filename pattern - matches "Cell In[N]" format
_CELL_PATTERN = re.compile(r"^Cell In\[\d+\]$")

# Internal wrapper function names to replace with <module>
_WRAPPER_NAMES = ("__repl_wrapper__", "__wrapper__")


# ---------------------------------------------------------------------------
# Targeted model-recovery hints
# ---------------------------------------------------------------------------

# SyntaxError messages that LLMs frequently hit when they embed a bash heredoc
# inside a single/double-quoted Python string. See issue 199.
_HEREDOC_TRIGGER_MSGS = (
    "unterminated string literal",
    "unexpected character after line continuation character",
    "invalid syntax. Perhaps you forgot a comma?",
)

# Matches a bash heredoc opener: `<<EOF`, `<< EOF`, `<<-EOF`, `<<'EOF'`, `<<"EOF"`.
# Note: `\w+` also matches numeric shifts like `<< 2`, so the false-positive
# guard for plain bit-shifts lives in `_looks_like_embedded_heredoc` (the
# quote-before-`<<` check), not in this regex.
_HEREDOC_RE = re.compile(r"<<-?\s*['\"]?\w+['\"]?")

# Appended to SyntaxErrors that look like heredoc-in-quoted-string failures.
_HEREDOC_HINT = '''\
Hint: this looks like a shell heredoc (`<<...`) embedded in a single- or
multiple-line Python string. Put the complete command in a triple-quoted
string, then pass that value using the callable's documented API:

    command = """cat <<'EOF'
content
EOF"""

Use `doc(...)` to inspect the available command runner and how it accepts
commands or standard input.'''


# Call-shape TypeError messages — a method/function called with the wrong
# signature. These name the bad arg but never the correct signature, so the
# model loops guessing synonyms (issue #245). We append the callable's concise
# agentdoc so the correction is actionable.
_BAD_CALL_RE = re.compile(
    r"^(?P<qual>[\w.]+)\(\) (?:"
    r"got an unexpected keyword argument"
    r"|missing \d+ required positional argument"
    r"|takes \d+ positional argument"
    r"|takes from \d+ to \d+ positional arguments"
    r"|takes no arguments"
    r")"
)


def _callable_qualname(value: object) -> str | None:
    """Best-effort ``__qualname__`` of a callable, with ``<locals>`` stripped.

    Used to confirm a resolved callable is *the* one named by the error, not a
    same-named decoy. Returns None if it has no usable qualname.
    """
    qn = getattr(value, "__qualname__", None)
    if not isinstance(qn, str):
        return None
    return ".".join(p for p in qn.split(".") if p != "<locals>")


def _resolve_called_callable(qual: str, error: Exception) -> object | None:
    """Find the callable named ``qual`` in the traceback's frames.

    A call-shape ``TypeError`` is raised at the *call site* (the callee is never
    entered), so the target object lives in some frame's locals/globals. ``qual``
    is the message's qualname, e.g. ``"ShellTools5.replace"`` or ``"my_func"``.

    The error gives only the qualname, not the identity of the object that
    raised — so we collect every candidate in the frames whose own qualname
    matches ``qual`` and return it ONLY if it is unique. Two distinct same-named
    callables in scope → ambiguous → ``None`` (don't show a confident-but-wrong
    signature). Attribute lookup uses ``inspect.getattr_static`` so a ``@property``
    (or any descriptor) on an unrelated in-scope object is never *invoked* during
    the scan. Returns the callable to ``doc()`` or ``None``.
    """
    import inspect

    # Drop "<locals>" segments — nested defs render as
    # "outer.<locals>.Cls.meth"; the owning class is the last non-<locals> part.
    parts = [p for p in qual.split(".") if p != "<locals>"]
    attr = parts[-1]
    owner = parts[-2] if len(parts) >= 2 else None

    tb = getattr(error, "__traceback__", None)
    frames = []
    while tb is not None:
        frames.append(tb.tb_frame)
        tb = tb.tb_next

    def _matches(candidate: object) -> bool:
        return callable(candidate) and _callable_qualname(candidate) == qual

    candidates: list[object] = []
    seen: set[int] = set()

    def _consider(candidate: object) -> None:
        if candidate is None or not _matches(candidate):
            return
        if id(candidate) in seen:
            return
        seen.add(id(candidate))
        candidates.append(candidate)

    for frame in frames:
        ns = {**frame.f_globals, **frame.f_locals}
        for value in ns.values():
            # Method call: an instance of the owning class. Resolve the attribute
            # statically (no descriptor invocation) and confirm its qualname.
            if owner is not None and type(value).__name__ == owner:
                try:
                    _consider(inspect.getattr_static(value, attr, None))
                except Exception:
                    pass
            # Bare function (no owner) — the value itself is the callable.
            if owner is None:
                _consider(value)
        # The owning class itself living in the namespace (unbound method).
        if owner is not None:
            cls = ns.get(owner)
            if isinstance(cls, type):
                try:
                    _consider(inspect.getattr_static(cls, attr, None))
                except Exception:
                    pass

    # Unique match only — a single resolvable callable whose qualname is exactly
    # what the error named. Zero or multiple (ambiguous) → best-effort None.
    return candidates[0] if len(candidates) == 1 else None


def _bad_call_agentdoc(error: BaseException, target: object | None = None) -> str | None:
    """Return concise agentdoc for a recognized bad call, when safely resolvable.

    ``target`` lets a broker that already resolved the called object preserve the
    hint across a process boundary. Without it, the target is resolved from the
    exception traceback. This is best-effort and never lets hostile ``__str__``
    or documentation hooks replace the original error.
    """
    if not isinstance(error, TypeError):
        return None
    try:
        match = _BAD_CALL_RE.match(str(error))
        if not match:
            return None
        resolved = (
            target if target is not None else _resolve_called_callable(match.group("qual"), error)
        )
        if resolved is None:
            return None
        resolved_qualname = _callable_qualname(resolved)
        if match.group("qual") not in {
            resolved_qualname,
            f"{resolved_qualname}.__init__",
        }:
            return None
        from nooa.agentdoc import doc

        rendered = doc(resolved, concise=True, inline_depth=0)
        return str(rendered).strip() or None
    except BaseException:
        return None


# ---------------------------------------------------------------------------
# Traceback rendering and normalization helpers
# ---------------------------------------------------------------------------


def _is_user_code_frame(filename: str) -> bool:
    """Check if a traceback frame is from user code (not framework internals).

    Returns True for:
    - Cell In[N] format (IPython-style REPL code)
    - <execute_code> format (legacy)
    - Any file NOT in framework paths (user's own agent files, etc.)

    Returns False for:
    - Framework paths (nooa/, site-packages/, lib/python, etc.)
    """
    if filename.startswith("<frozen"):
        return False

    path = PurePath(filename)
    parts = path.parts
    if _FRAMEWORK_PATH_PARTS.intersection(parts):
        return False
    if any(
        part.startswith("python")
        for index, part in enumerate(parts)
        if index and parts[index - 1] == "lib"
    ):
        return False

    # Relative module paths emitted by some traceback producers are safe to
    # classify only when ``nooa`` is the first complete component.
    if not path.is_absolute() and parts and parts[0] == "nooa":
        return False

    try:
        Path(filename).resolve().relative_to(_NOOA_PACKAGE_ROOT)
    except (OSError, ValueError):
        return True
    return False


def _is_validation_error(error: Exception) -> bool:
    """Check if error is a static validation error (no traceback needed)."""
    try:
        from nooa.errors import RestrictedCodeError, ValidationError

        return isinstance(error, (ValidationError, RestrictedCodeError))
    except ImportError:
        return type(error).__name__ in ("ValidationError", "RestrictedCodeError")


def _strip_file_prefix(text: str) -> str:
    """Strip 'File "..."' wrapper to match IPython output.

    Transforms:
        File "Cell In[1]", line 1 → Cell In[1], line 1
    """
    return re.sub(r'File "([^"]+)"', r"\1", text)


def _replace_wrapper_names(text: str) -> str:
    """Replace internal wrapper function names with <module>.

    Transforms:
        in __repl_wrapper__ → in <module>
        in __wrapper__ → in <module>
    """
    for name in _WRAPPER_NAMES:
        text = text.replace(f"in {name}", "in <module>")
    return text


def _looks_like_embedded_heredoc(line: str) -> bool:
    """True if `line` looks like a heredoc embedded in a quoted string.

    Requires both a heredoc opener (`<<EOF` etc) AND a quote character
    appearing somewhere before the `<<` on the same line. The quote-before-`<<`
    check is what distinguishes `shell.run("cat <<EOF` (a real failure) from
    a legitimate bit-shift like `func(a << foo b)`.
    """
    match = _HEREDOC_RE.search(line)
    if not match:
        return False
    before = line[: match.start()]
    return '"' in before or "'" in before


def _maybe_heredoc_hint(error: SyntaxError) -> str | None:
    """Return the heredoc hint to append, or None if the error doesn't match.

    Heuristic: the error's message must be one of the known LLM-confusing
    SyntaxError messages, AND the offending source line must contain a
    heredoc opener embedded in what looks like a quoted string.

    The quote-before-`<<` requirement co-locates the two tokens on the same
    source line, which `error.text` always carries — no broader source scan
    is needed. See issue 199.
    """
    msg = error.msg or ""
    if not any(trigger in msg for trigger in _HEREDOC_TRIGGER_MSGS):
        return None
    if error.text and _looks_like_embedded_heredoc(error.text):
        return _HEREDOC_HINT
    return None


def _filter_traceback_tree(
    diagnostic: traceback.TracebackException,
    seen: set[int] | None = None,
) -> None:
    """Filter framework frames from an entire chained-exception tree in place.

    ``TracebackException`` stores a tree rather than one linear traceback: a
    diagnostic can include explicit causes, implicit contexts, and nested
    exception-group children.  Filtering only the root would therefore leak
    internal framework frames from one of those branches.  Walk every branch
    and apply the same user-frame policy, starting at the first generated cell
    while retaining user helper frames reached from that cell.  ``seen`` makes
    the recursive walk safe when traceback nodes are shared or cyclic.
    """
    if seen is None:
        seen = set()
    if id(diagnostic) in seen:
        return
    seen.add(id(diagnostic))

    eligible = [frame for frame in diagnostic.stack if _is_user_code_frame(frame.filename)]
    # Once generated-cell execution begins, omit outer callers but retain later
    # user helpers: those downstream frames often contain the actual failing line.
    first_cell = next(
        (index for index, frame in enumerate(eligible) if _CELL_PATTERN.match(frame.filename)),
        None,
    )
    if first_cell is not None:
        eligible = eligible[first_cell:]
    diagnostic.stack = traceback.StackSummary.from_list(eligible)

    if diagnostic.__cause__ is not None:
        _filter_traceback_tree(diagnostic.__cause__, seen)
    if diagnostic.__context__ is not None:
        _filter_traceback_tree(diagnostic.__context__, seen)
    if diagnostic.exceptions is not None:
        for child in diagnostic.exceptions:
            _filter_traceback_tree(child, seen)


def _concise_exception(error: BaseException) -> str:
    """Return a safe type-and-message fallback for traceback-less errors."""
    name = type(error).__name__
    try:
        message = str(error)
    except BaseException:
        message = name
    return f"{name}: {message}" if message else name


def _wrapper_cell_filename(
    error: BaseException,
    seen: set[int] | None = None,
) -> str | None:
    """Find the generated-cell wrapper frame anywhere in an exception tree."""
    if seen is None:
        seen = set()
    if id(error) in seen:
        return None
    seen.add(id(error))

    traceback_cursor = error.__traceback__
    while traceback_cursor is not None:
        frame_code = traceback_cursor.tb_frame.f_code
        if frame_code.co_name in _WRAPPER_NAMES and _CELL_PATTERN.match(frame_code.co_filename):
            return frame_code.co_filename
        traceback_cursor = traceback_cursor.tb_next

    for related in (error.__cause__, error.__context__):
        if related is not None:
            filename = _wrapper_cell_filename(related, seen)
            if filename is not None:
                return filename
    if isinstance(error, BaseExceptionGroup):
        for child in error.exceptions:
            filename = _wrapper_cell_filename(child, seen)
            if filename is not None:
                return filename
    return None


def _format_runtime_error(error: BaseException) -> str:
    """Render a filtered runtime diagnostic with Python's traceback formatter."""
    if (
        error.__traceback__ is None
        and error.__cause__ is None
        and error.__context__ is None
        and not isinstance(error, BaseExceptionGroup)
    ):
        return _concise_exception(error)
    diagnostic = traceback.TracebackException.from_exception(
        error,
        compact=True,
        capture_locals=False,
    )
    _filter_traceback_tree(diagnostic)
    return "".join(diagnostic.format(chain=True))


def _adjust_line_numbers(
    text: str,
    offset: int,
    *,
    cell_only: bool = False,
    cell_filename: str | None = None,
) -> str:
    """Adjust matching traceback headers without rewriting source/messages.

    ``cell_filename`` limits adjustment to the currently wrapped cell. Persisted
    helpers are compiled from their original source, so frames from older cells
    already have source-relative line numbers and must not inherit the current
    cell's wrapper offset.
    """
    if offset <= 0:
        return text

    # Headers are the only lines that should change. In particular, source such
    # as ``# line 99`` and messages such as ``failed at line 12`` are verbatim.
    if cell_filename is not None:
        prefix = rf" *{re.escape(cell_filename)}, "
    elif cell_only:
        prefix = r" *Cell In\[\d+\], "
    else:
        prefix = r'(?: *File "[^"]+", | *Cell In\[\d+\], |)'
    header = re.compile(
        rf"(?m)^(?P<prefix>{prefix}line )"
        r"(?P<number>\d+)(?P<suffix>(?:, in .*)?)$"
    )

    def adjust_match(match: re.Match[str]) -> str:
        adjusted = max(1, int(match.group("number")) - offset)
        return f"{match.group('prefix')}{adjusted}{match.group('suffix')}"

    return header.sub(adjust_match, text)


# ---------------------------------------------------------------------------
# Formatter and public entry point
# ---------------------------------------------------------------------------


def _diagnostic_budget(
    max_error: int | None,
    tail_chars: int | None,
) -> tuple[int, int | None]:
    """Normalize an optional diagnostic budget defensively."""
    if isinstance(max_error, int) and not isinstance(max_error, bool) and max_error > 0:
        limit = max_error
    else:
        limit = DEFAULT_TRUNCATION_CONFIG.capture.max_error
    if tail_chars is None:
        tail: int | None = None
    elif (
        isinstance(tail_chars, int) and not isinstance(tail_chars, bool) and 0 <= tail_chars < limit
    ):
        tail = tail_chars
    else:
        default_tail = DEFAULT_TRUNCATION_CONFIG.capture.tail
        tail = None if default_tail is None else min(default_tail, limit - 1)
    return limit, tail


def _bound_diagnostic(
    text: str,
    max_error: int | None,
    tail_chars: int | None,
) -> str:
    """Apply the caller's error budget, falling back to framework defaults."""
    limit, tail = _diagnostic_budget(max_error, tail_chars)
    stream = TruncatingStringIO(limit=limit, tail_chars=tail)
    stream.write(text)
    return stream.getvalue()


def _hard_bound_text(text: str, limit: int, *, closing: str = "") -> str:
    """Cap text at ``limit``, optionally preserving a structural closing marker."""
    if len(text) <= limit:
        return text
    marker = "...<truncated>"
    suffix = closing if closing and text.endswith(closing) else ""
    keep = limit - len(marker) - len(suffix)
    if keep > 0:
        return text[:keep] + marker + suffix
    return marker[:limit]


class ErrorFormatter(Protocol):
    """Preferred strategy formatter contract.

    Implementations receive the exception, source context, and resolved per-call
    error budget. Custom strategy formatters must implement this complete contract.
    """

    def format(
        self,
        error: Exception,
        code: str | None = None,
        *,
        line_offset: int = 0,
        max_error: int | None = None,
        tail_chars: int | None = None,
    ) -> str: ...


class IPythonErrorFormatter:
    """IPython/Jupyter-style error formatter.

    Formats errors to match IPython output:
    - Uses "Cell In[N], line X" format (stripping File "..." wrapper)
    - Adjusts line numbers to account for wrapper code offset
    - Replaces internal wrapper names with <module>
    - Filters out framework tracebacks, showing only user code
    - Preserves Python's native syntax/runtime source markers
    - Source lines shown automatically (via linecache registration in actor.py)
    """

    def format(
        self,
        error: Exception,
        code: str | None = None,
        *,
        line_offset: int = 0,
        max_error: int | None = None,
        tail_chars: int | None = None,
    ) -> str:
        """Format and bound an error for LLM feedback.

        Args:
            error: The exception to format.
            code: Optional source code (used for syntax errors if text is missing).
            line_offset: Number of wrapper lines to subtract from line numbers.
            max_error: Maximum retained diagnostic characters. ``None`` uses the
                framework default.
            tail_chars: Characters reserved for the retained tail. ``None`` uses
                the standard half-head/half-tail split.

        Returns:
            Formatted error string with adjusted line numbers.
        """
        from nooa.runtime.sandbox.errors import SandboxExecutionError

        if isinstance(error, SandboxExecutionError):
            limit, _ = _diagnostic_budget(max_error, tail_chars)
            return _hard_bound_text(
                error.diagnostic.rstrip() or str(error),
                limit + 1_024,
                closing="\n</truncated-output>",
            )

        if isinstance(error, SyntaxError):
            formatted = self._format_syntax_error(error, code, line_offset)
        elif _is_validation_error(error):
            formatted = f"{type(error).__name__}: {error}"
        else:
            formatted = self._format_runtime_error(error, line_offset)
            # Issue #245: append the called callable's concise signature on a
            # call-shape TypeError (see _bad_call_agentdoc). Best-effort — unchanged
            # output on any miss.
            transported_hint = getattr(error, "_nooa_call_hint", None)
            agentdoc = (
                transported_hint if isinstance(transported_hint, str) else _bad_call_agentdoc(error)
            )
            if agentdoc:
                formatted = (
                    f"{formatted}\n\nThe callable you called has this signature:\n{agentdoc}"
                )

        return _bound_diagnostic(formatted, max_error, tail_chars)

    def _format_syntax_error(self, error: SyntaxError, code: str | None, line_offset: int) -> str:
        """Format SyntaxError using Python's traceback module, IPython-style.

        Output format:
            Cell In[1], line 1
              <invalid_code>
              ^
            SyntaxError: invalid syntax
        """
        # If error.text is missing but we have code, extract the line
        if not error.text and code and error.lineno:
            lines = code.split("\n")
            if 1 <= error.lineno <= len(lines):
                error.text = lines[error.lineno - 1]

        # Use Python's standard traceback formatting
        formatted = "".join(traceback.format_exception_only(type(error), error)).rstrip()

        # Strip the File "..." prefix to match IPython
        formatted = _strip_file_prefix(formatted)

        # Adjust line numbers for wrapper offset
        formatted = _adjust_line_numbers(formatted, line_offset)

        hint = _maybe_heredoc_hint(error)
        if hint:
            formatted = f"{formatted}\n\n{hint}"

        return formatted

    def _format_runtime_error(self, error: BaseException, line_offset: int) -> str:
        """Format runtime errors using the stdlib traceback implementation."""
        formatted = _strip_file_prefix(_format_runtime_error(error))
        current_cell = _wrapper_cell_filename(error)
        formatted = _adjust_line_numbers(
            formatted,
            line_offset,
            cell_only=current_cell is None,
            cell_filename=current_cell,
        )
        return _replace_wrapper_names(formatted).rstrip()


# Default formatter instance
_default_formatter = IPythonErrorFormatter()


def format_error_for_llm(
    error: Exception,
    code: str | None = None,
    *,
    line_offset: int = 0,
    max_error: int | None = None,
    tail_chars: int | None = None,
) -> str:
    """Format an error for LLM feedback.

    Uses IPython-style formatting:
    - Syntax errors retain Python's source marker when available
    - Validation errors show clean messages without tracebacks
    - Runtime errors filter to user code frames only
    - Line numbers are adjusted to account for wrapper code
    - Internal wrapper function names are replaced with <module>
    - Source code lines are shown (via linecache registration in actor.py)

    Args:
        error: The exception to format.
        code: Optional source code (used for syntax errors if text is missing).
        line_offset: Number of wrapper lines to subtract from line numbers.
            This compensates for lines added by the async wrapper (e.g.,
            "async def __repl_wrapper__():", "try:", etc.).
        max_error: Maximum retained diagnostic characters. ``None`` uses the
            framework default.
        tail_chars: Characters reserved for the retained tail. ``None`` uses
            the standard half-head/half-tail split.

    Returns:
        Formatted error string suitable for LLM consumption.
    """
    try:
        return _default_formatter.format(
            error,
            code,
            line_offset=line_offset,
            max_error=max_error,
            tail_chars=tail_chars,
        )
    except BaseException:
        return _bound_diagnostic(_concise_exception(error), max_error, tail_chars)
