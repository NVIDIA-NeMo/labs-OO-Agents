# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core Stream type with method chaining — async-native."""

from __future__ import annotations

import re as _re
from collections import deque
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, TypeVar

from nooa_cli.tools.pyp.errors import Result

T = TypeVar("T")


def _to_async_iter(iterable) -> AsyncIterator:
    """Convert a sync iterable to an async iterator."""

    async def _gen():
        for item in iterable:
            yield item

    return _gen()


class Stream:
    """
    Lazy async stream of items (typically strings) with method-chaining transforms.

    Wraps any iterable or async iterable. Transforms return new Streams
    (lazy async generators). Consumption happens at terminal operations
    (await stream.collect(), etc).

    Usage:
        await cat("file.txt").grep("ERROR").head(10).collect()
        await run("ls -la").grep(r"\\.py$").sort().collect()
        await find("src", name="*.py").sort().text()
    """

    def __init__(self, iterable=None, *, _steps: list | None = None, _meta: dict | None = None):
        if iterable is None:
            self._aiterable = _to_async_iter([])
        elif hasattr(iterable, "__aiter__"):
            self._aiterable = iterable
        else:
            self._aiterable = _to_async_iter(iterable)
        self._steps: list[str] = _steps or []
        self._meta: dict = _meta or {}

    def __aiter__(self) -> AsyncIterator:
        return self._aiterable.__aiter__()

    def __or__(self, other):
        """Pipe operator: stream | callable for custom transforms."""
        if callable(other):
            return Stream(
                other(self._aiterable), _steps=self._steps + [repr(other)], _meta=self._meta
            )
        return NotImplemented

    def __repr__(self) -> str:
        steps = ".".join(self._steps) if self._steps else "Stream(empty)"
        return f"Stream({steps})"

    # ─── Transforms (return new Stream) ────────────────────────────────

    def grep(
        self, pattern: str, *, invert: bool = False, ignore_case: bool = False, fixed: bool = False
    ) -> Stream:
        """Filter lines matching a pattern (like grep)."""
        if fixed:

            def _match(line):
                if ignore_case:
                    return pattern.lower() in line.lower()
                return pattern in line
        else:
            flags = _re.IGNORECASE if ignore_case else 0
            regex = _re.compile(pattern, flags)

            def _match(line):
                return bool(regex.search(line))

        parent = self

        async def _gen():
            async for line in parent:
                matched = _match(line)
                if matched != invert:
                    yield line

        step = f"grep({pattern!r})"
        return Stream(_gen(), _steps=self._steps + [step], _meta=self._meta)

    def head(self, n: int = 10) -> Stream:
        """Take the first N items (like head)."""
        parent = self

        async def _gen():
            count = 0
            async for item in parent:
                if count >= n:
                    break
                yield item
                count += 1

        return Stream(_gen(), _steps=self._steps + [f"head({n})"], _meta=self._meta)

    def tail(self, n: int = 10) -> Stream:
        """Take the last N items (like tail)."""
        parent = self

        async def _gen():
            buf = deque(maxlen=n)
            async for item in parent:
                buf.append(item)
            for item in buf:
                yield item

        return Stream(_gen(), _steps=self._steps + [f"tail({n})"], _meta=self._meta)

    def sort(
        self, *, key: Callable | None = None, reverse: bool = False, numeric: bool = False
    ) -> Stream:
        """Sort items (like sort). Buffering."""
        parent = self

        async def _gen():
            items = []
            async for item in parent:
                items.append(item)
            sort_key = key
            if numeric and sort_key is None:

                def _num_key(x):
                    try:
                        return float(x)
                    except (ValueError, TypeError):
                        return float("inf")

                sort_key = _num_key
            items.sort(key=sort_key, reverse=reverse)
            for item in items:
                yield item

        return Stream(_gen(), _steps=self._steps + ["sort()"], _meta=self._meta)

    def uniq(self, *, all_unique: bool = False, count: bool = False) -> Stream:
        """Remove duplicates (consecutive or all)."""
        parent = self

        async def _gen_consecutive():
            _sentinel = object()
            prev = _sentinel
            cnt = 0
            async for item in parent:
                if item == prev:
                    cnt += 1
                else:
                    if prev is not _sentinel and cnt > 0:
                        if count:
                            yield f"{cnt} {prev}"
                        else:
                            yield prev
                    prev = item
                    cnt = 1
            if cnt > 0:
                if count:
                    yield f"{cnt} {prev}"
                else:
                    yield prev

        async def _gen_all():
            seen: set = set()
            async for item in parent:
                if item not in seen:
                    seen.add(item)
                    yield item

        gen = _gen_all if all_unique else _gen_consecutive
        return Stream(gen(), _steps=self._steps + ["uniq()"], _meta=self._meta)

    def cut(
        self, fields: list[int], *, sep: str | None = None, out_sep: str | None = None
    ) -> Stream:
        """Extract fields from each line (like cut)."""
        actual_out_sep = out_sep or sep or "\t"
        parent = self

        async def _gen():
            async for line in parent:
                parts = line.split(sep) if sep else line.split()
                selected = []
                for f in fields:
                    if f < len(parts):
                        selected.append(parts[f])
                    else:
                        selected.append("")
                yield actual_out_sep.join(selected)

        return Stream(_gen(), _steps=self._steps + [f"cut({fields})"], _meta=self._meta)

    def sed(self, pattern: str, repl: str, *, count: int = 0) -> Stream:
        """Regex substitution on each line (like sed s/pat/repl/)."""
        regex = _re.compile(pattern)
        parent = self

        async def _gen():
            async for line in parent:
                yield regex.sub(repl, line, count=count)

        return Stream(_gen(), _steps=self._steps + [f"sed({pattern!r})"], _meta=self._meta)

    def map(self, fn: Callable[[Any], Any]) -> Stream:
        """Apply a function to each item."""
        parent = self

        async def _gen():
            async for item in parent:
                yield fn(item)

        return Stream(_gen(), _steps=self._steps + ["map(...)"], _meta=self._meta)

    def filter(self, fn: Callable[[Any], bool]) -> Stream:
        """Keep items where predicate is true."""
        parent = self

        async def _gen():
            async for item in parent:
                if fn(item):
                    yield item

        return Stream(_gen(), _steps=self._steps + ["filter(...)"], _meta=self._meta)

    def wc(self, *, lines_only: bool = True) -> Stream:
        """Count lines/words/chars (like wc). Yields a single summary line."""
        parent = self

        async def _gen():
            line_count = 0
            word_count = 0
            char_count = 0
            async for line in parent:
                line_count += 1
                if not lines_only:
                    word_count += len(str(line).split())
                    char_count += len(str(line)) + 1
            if lines_only:
                yield str(line_count)
            else:
                yield f"{line_count} {word_count} {char_count}"

        return Stream(_gen(), _steps=self._steps + ["wc()"], _meta=self._meta)

    def skip(self, n: int = 1) -> Stream:
        """Skip the first N items."""
        parent = self

        async def _gen():
            count = 0
            async for item in parent:
                if count >= n:
                    yield item
                count += 1

        return Stream(_gen(), _steps=self._steps + [f"skip({n})"], _meta=self._meta)

    def tee(self, path: str, *, append: bool = False) -> Stream:
        """Pass items through while writing a copy to a file (like tee)."""
        import asyncio as _asyncio

        parent = self

        async def _gen():
            mode = "a" if append else "w"
            fh = await _asyncio.to_thread(open, path, mode)
            try:
                async for item in parent:
                    item_str = str(item)
                    line = item_str if item_str.endswith("\n") else item_str + "\n"
                    await _asyncio.to_thread(fh.write, line)
                    yield item
            finally:
                await _asyncio.to_thread(fh.close)

        return Stream(_gen(), _steps=self._steps + [f"tee({path!r})"], _meta=self._meta)

    def flatten(self, *, sep: str = "\n") -> Stream:
        """Split each line on a separator and yield individual items."""
        parent = self

        async def _gen():
            async for item in parent:
                parts = str(item).split(sep)
                for part in parts:
                    if part:
                        yield part

        return Stream(_gen(), _steps=self._steps + ["flatten()"], _meta=self._meta)

    def strip(self, chars: str | None = None) -> Stream:
        """Strip whitespace (or given chars) from each line."""
        parent = self

        async def _gen():
            async for item in parent:
                yield str(item).strip(chars)

        return Stream(_gen(), _steps=self._steps + ["strip()"], _meta=self._meta)

    def take_while(self, fn: Callable[[Any], bool]) -> Stream:
        """Yield items while predicate is true, then stop."""
        parent = self

        async def _gen():
            async for item in parent:
                if not fn(item):
                    break
                yield item

        return Stream(_gen(), _steps=self._steps + ["take_while(...)"], _meta=self._meta)

    def drop_while(self, fn: Callable[[Any], bool]) -> Stream:
        """Skip items while predicate is true, then yield the rest."""
        parent = self

        async def _gen():
            dropping = True
            async for item in parent:
                if dropping and fn(item):
                    continue
                dropping = False
                yield item

        return Stream(_gen(), _steps=self._steps + ["drop_while(...)"], _meta=self._meta)

    def pipe(self, *transforms) -> Stream:
        """Apply a sequence of transforms (for reusable pipelines)."""
        s = self
        for t in transforms:
            if callable(t):
                s = s | t
            else:
                raise TypeError(f"Transform must be callable, got {type(t)}")
        return s

    def xargs(self, fn: Callable) -> Stream:
        """Apply a function to each item, yielding results (like xargs).

        Supports both sync and async callables. Items where fn returns None
        are skipped.

        Args:
            fn: Function taking a single string argument. Can be sync or async.

        Usage:
            run("find . -name '*.py'").xargs(process_file).head(10)
        """
        import inspect as _inspect

        parent = self
        is_async = _inspect.iscoroutinefunction(fn)

        async def _gen():
            async for item in parent:
                if is_async:
                    result = await fn(item)
                else:
                    result = fn(item)
                if result is not None:
                    yield str(result)

        return Stream(_gen(), _steps=self._steps + ["xargs(...)"], _meta=self._meta)

    # ─── Terminal operations (sinks) — all async ───────────────────────

    def __await__(self):
        """Awaiting a Stream collects it — ``await find(...)`` ≡ ``await find(...).collect()``.

        Lets a Stream be the awaitable terminal directly when you just want the
        list, without remembering which sink verb to call.
        """
        return self.collect().__await__()

    async def paths(self) -> list[Path]:
        """Terminal: collect the stream as ``Path`` objects.

        Symmetric with ``rg(...).matches()`` — the natural sink for ``find(...)``
        when you want real paths rather than strings::

            for p in await shell.find("src", name="*.py").paths():
                ...
        """
        return [Path(line) for line in await self.collect()]

    async def collect(self) -> list[str]:
        """Consume the stream and return all items as a list."""
        result = []
        async for item in self._aiterable:
            result.append(item)
        return result

    async def result(self) -> Result:
        """Consume the stream and return a Result with metadata."""
        items = []
        async for item in self._aiterable:
            items.append(item)
        rc = self._meta.get("returncode", 0)
        stderr = self._meta.get("stderr", "")
        return Result(lines=items, returncode=rc, stderr=stderr)

    async def first(self) -> str | None:
        """Return the first item or None."""
        async for item in self._aiterable:
            return item
        return None

    async def last(self) -> str | None:
        """Return the last item or None."""
        last_item = None
        async for x in self._aiterable:
            last_item = x
        return last_item

    async def count(self) -> int:
        """Count items in the stream (consumes it)."""
        n = 0
        async for _ in self._aiterable:
            n += 1
        return n

    async def write(self, path: str | Path, *, mode: str = "w") -> int:
        """Write stream lines to a file (non-blocking). Returns number of lines written."""
        import asyncio as _asyncio

        p = Path(path)
        fh = await _asyncio.to_thread(open, p, mode)
        n = 0
        try:
            async for line in self._aiterable:
                line_str = str(line)
                await _asyncio.to_thread(
                    fh.write, line_str if line_str.endswith("\n") else line_str + "\n"
                )
                n += 1
        finally:
            await _asyncio.to_thread(fh.close)
        return n

    async def print(self, *, end: str = "\n") -> None:
        """Print each item to stdout."""
        import builtins

        async for item in self._aiterable:
            builtins.print(item, end=end)

    async def json(self) -> Any:
        """Parse the collected text as JSON."""
        import json as _json

        items = await self.collect()
        return _json.loads("\n".join(str(line) for line in items))

    async def to_set(self) -> set[str]:
        """Collect into a set."""
        result = set()
        async for item in self._aiterable:
            result.add(item)
        return result

    async def to_dict(self, sep: str = "=") -> dict[str, str]:
        """Split each line on sep and collect into a dict."""
        d = {}
        async for line in self._aiterable:
            line_str = str(line)
            if sep in line_str:
                k, v = line_str.split(sep, 1)
                d[k.strip()] = v.strip()
        return d

    async def text(self) -> str:
        """Collect and join lines with newlines into a single string."""
        items = await self.collect()
        return "\n".join(str(item) for item in items)

    async def table(self, *, sep: str | None = None, headers: list[str] | None = None) -> str:
        """
        Format stream as an aligned text table.

        Each line is split into columns. Columns are padded to align.
        """
        rows = []
        async for line in self._aiterable:
            line_str = str(line)
            cols = line_str.split(sep) if sep else line_str.split()
            rows.append([c.strip() for c in cols])

        if not rows:
            return ""

        all_rows = rows
        if headers:
            all_rows = [headers] + rows

        max_cols = max(len(r) for r in all_rows)
        widths = [0] * max_cols
        for row in all_rows:
            for i, col in enumerate(row):
                if i < max_cols:
                    widths[i] = max(widths[i], len(col))

        out_lines = []
        if headers:
            header_line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
            out_lines.append(header_line)
            out_lines.append("  ".join("-" * widths[i] for i in range(max_cols)))

        for row in rows:
            padded = []
            for i in range(max_cols):
                val = row[i] if i < len(row) else ""
                padded.append(val.ljust(widths[i]))
            out_lines.append("  ".join(padded))

        return "\n".join(out_lines)
