# Output files: `nooa.overflow`

`nooa.overflow` owns the files that hold output too large to keep in memory (a command's whole stdout, a cell's output). It is a small standalone package: it imports nothing from the rest of `nooa`. `nooa.snapshot`'s `TextStream` writes through it (`TextStream(overflow=folder.file())`), and a cut `Text.full` (an `Overflow`: path, kept, total) points to the file. The module docstring (`help(nooa.overflow)`) is the full contract; this page is the summary.

| Name | What it is |
|---|---|
| `OverflowDir(path=None, *, max_files=20, file_bytes=64 MiB)` | a bounded folder of files: a private `tempfile.mkdtemp` folder (0700), or a given existing one |
| `OverflowDir.file()` | a new `OverflowFile` in the folder (`tempfile.mkstemp`, 0600), or `None` when the budget is in use |
| `OverflowFile(path=None, max_bytes=64 MiB)` | one file; standalone or handed out by a folder; `open()` returns its writer |

## Guarantees

- **Bounded.** At most `max_files` files of at most `file_bytes` bytes. To make room, the oldest *closed* file is deleted first; a file being written (or handed out and not yet written) is never evicted.
- **Safe.** Files are created exclusively, mode 0600; an existing `OverflowFile(path=...)` is refused with `FileExistsError`. Removal never follows a symlink; in a given folder only the names the `OverflowDir` created are removed.
- **Explicit lifetime.** Files are deleted only when their writer finishes with complete output, by cap eviction, when their `OverflowDir` is closed (`close()` or a `with` block; idempotent), or at interpreter exit (`atexit`) for every `OverflowDir` never closed. **Never by garbage collection:** a dropped `OverflowDir` keeps its files until exit, so a path someone holds does not vanish at an arbitrary moment. The cost: an owner nobody closes keeps its capped files until exit.
- **A gone path says so.** Opening an evicted or closed file raises `FileNotFoundError`.
- **Threads.** `file()`, `close()` and writers finishing may run in any threads.

## Known limit

A process killed without running `atexit` (`SIGKILL`, an abort, `os._exit`) leaves its files behind, under the system temporary directory (`nooa-overflow-*`), until the operating system's temporary-file cleanup removes them.

Tests: `tests/overflow` (filesystem behaviour only).
