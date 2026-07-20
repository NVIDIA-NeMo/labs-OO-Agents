# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error formatting for LLM feedback.

Formats errors to match IPython/Jupyter-style output:
- Uses "Cell In[N], line X" format (stripping File "..." wrapper)
- Adjusts line numbers to account for wrapper code offset
- Filters out framework tracebacks, showing only user code
- Source code lines are shown automatically via linecache registration

Usage:
    from nooa.errors.formatting import format_error_for_llm
    result = format_error_for_llm(error, code, line_offset=2)

    # Or use the formatter class directly
    from nooa.errors.formatting import IPythonErrorFormatter
    formatter = IPythonErrorFormatter()
    result = formatter.format(error, code, line_offset=2)
"""

import re
import traceback

# Framework path markers - frames containing these are filtered out
_FRAMEWORK_MARKERS = ("nooa/", "site-packages/", "lib/python", "<frozen")

# User code filename pattern - matches "Cell In[N]" format
_CELL_PATTERN = re.compile(r"^Cell In\[\d+\]$")

# Internal wrapper function names to replace with <module>
_WRAPPER_NAMES = ("__repl_wrapper__", "__wrapper__")

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
_HEREDOC_HINT = """\
Hint: this looks like a bash heredoc (`<<...`) embedded in a single- or
double-quoted Python string. Python rejects multi-line single/double-quoted
strings, which is why the parser errored before reaching the heredoc body.

Fix 1 (preferred) — use a triple-quoted Python string so newlines are legal:
    await shell.run(\"\"\"cat <<EOF
    content
    EOF\"\"\")

Fix 2 — write the script to a file first, then run it:
    await shell.write("/tmp/script.sh", script_text)
    await shell.run("bash /tmp/script.sh")"""


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


def _bad_call_agentdoc(error: Exception) -> str | None:
    """If ``error`` is a call-shape TypeError to a resolvable callable, return
    its concise agentdoc to append to the feedback. Best-effort; ``None`` on any
    miss so the caller falls back to the raw error (no regression)."""
    if not isinstance(error, TypeError):
        return None
    try:
        m = _BAD_CALL_RE.match(str(error))
        if not m:
            return None
        target = _resolve_called_callable(m.group("qual"), error)
        if target is None:
            return None
        from nooa.agentdoc import doc

        rendered = doc(target, concise=True)
        return str(rendered).strip() or None
    except Exception:
        return None


def _is_user_code_frame(filename: str) -> bool:
    """Check if a traceback frame is from user code (not framework internals).

    Returns True for:
    - Cell In[N] format (IPython-style REPL code)
    - <execute_code> format (legacy)
    - Any file NOT in framework paths (user's own agent files, etc.)

    Returns False for:
    - Framework paths (nooa/, site-packages/, lib/python, etc.)
    """
    # Filter out framework paths first
    if any(marker in filename for marker in _FRAMEWORK_MARKERS):
        return False

    # Everything else is user code (REPL cells, user's agent files, etc.)
    return True


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


def _adjust_line_numbers(text: str, offset: int) -> str:
    """Adjust line numbers in formatted output by subtracting offset.

    Transforms:
        Cell In[1], line 5 → Cell In[1], line 3 (with offset=2)
        line 5 → line 3 (with offset=2)
    """
    if offset <= 0:
        return text

    def adjust_match(match: re.Match[str]) -> str:
        prefix = match.group(1)
        line_num = int(match.group(2))
        adjusted = max(1, line_num - offset)  # Never go below line 1
        return f"{prefix}{adjusted}"

    # Match patterns like "Cell In[1], line 5" or "line 5"
    return re.sub(r"((?:Cell In\[\d+\], )?line )(\d+)", adjust_match, text)


class IPythonErrorFormatter:
    """IPython/Jupyter-style error formatter.

    Formats errors to match IPython output:
    - Uses "Cell In[N], line X" format (stripping File "..." wrapper)
    - Adjusts line numbers to account for wrapper code offset
    - Replaces internal wrapper names with <module>
    - Filters out framework tracebacks, showing only user code
    - Shows syntax errors with caret pointing to error location
    - Source lines shown automatically (via linecache registration in actor.py)
    """

    def format(self, error: Exception, code: str | None = None, *, line_offset: int = 0) -> str:
        """Format an error for LLM feedback.

        Args:
            error: The exception to format.
            code: Optional source code (used for syntax errors if text is missing).
            line_offset: Number of wrapper lines to subtract from line numbers.

        Returns:
            Formatted error string with adjusted line numbers.
        """
        if isinstance(error, SyntaxError):
            return self._format_syntax_error(error, code, line_offset)

        if _is_validation_error(error):
            return f"{type(error).__name__}: {error}"

        formatted = self._format_runtime_error(error, line_offset)
        # Issue #245: append the called callable's concise signature on a
        # call-shape TypeError (see _bad_call_agentdoc). Best-effort — unchanged
        # output on any miss.
        agentdoc = _bad_call_agentdoc(error)
        if agentdoc:
            formatted = f"{formatted}\n\nThe callable you called has this signature:\n{agentdoc}"
        return formatted

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

    def _format_runtime_error(self, error: Exception, line_offset: int) -> str:
        """Format runtime error using Python's traceback module, IPython-style.

        Filters to show only user code frames (excludes framework internals).
        Source lines are shown automatically via linecache registration.

        Output format:
            Cell In[1], line 3, in <module>
                x = 1 / 0
                    ~~^~~
            ZeroDivisionError: division by zero
        """
        if not error.__traceback__:
            return f"{type(error).__name__}: {error}"

        # Extract all frames as FrameSummary objects (filterable)
        extracted = traceback.extract_tb(error.__traceback__)

        # Filter to user code frames only
        user_frames = [frame for frame in extracted if _is_user_code_frame(frame.filename)]

        if not user_frames:
            # No user frames found, just show the error type and message
            return f"{type(error).__name__}: {error}"

        # Format the filtered frames + exception
        formatted_frames = traceback.format_list(user_frames)
        formatted_exception = traceback.format_exception_only(type(error), error)
        formatted = "".join(formatted_frames + formatted_exception).rstrip()

        # Strip the File "..." prefix to match IPython
        formatted = _strip_file_prefix(formatted)

        # Replace internal wrapper names with <module>
        formatted = _replace_wrapper_names(formatted)

        # Adjust line numbers for wrapper offset
        return _adjust_line_numbers(formatted, line_offset)


# Default formatter instance
_default_formatter = IPythonErrorFormatter()


def format_error_for_llm(error: Exception, code: str | None = None, *, line_offset: int = 0) -> str:
    """Format an error for LLM feedback.

    Uses IPython-style formatting:
    - Syntax errors show caret pointing to error location
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

    Returns:
        Formatted error string suitable for LLM consumption.
    """
    return _default_formatter.format(error, code, line_offset=line_offset)
