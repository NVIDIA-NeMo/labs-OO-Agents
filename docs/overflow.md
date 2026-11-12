# Output files: `nooa.overflow`

`nooa.overflow` owns the files that hold output too large to keep in memory (a command's whole stdout, a cell's output). It is a small standalone package: it imports nothing from the rest of `nooa`. `nooa.snapshot`'s `TextStream` writes through it (`TextStream(overflow=folder.file())`), and a cut `Text.full` (an `Overflow`: path, kept, total) points to the file. The module docstring (`help(nooa.overflow)`) is the full contract; this page is the summary.

| Name | What it is |
|---|---|
| `OverflowDir(path=None, *, max_files=20, file_bytes=64 MiB)` | a bounded folder of files: a private `tempfile.mkdtemp` folder (0700), or a given existing one |
| `OverflowDir.file()` | a new `OverflowFile` in the folder (exclusive, `O_NOFOLLOW`, 0600), or `None` when the budget is in use |
| `OverflowFile(path=None, max_bytes=64 MiB)` | one file; standalone or handed out by a folder; `open()` returns its `OverflowWriter` |

## Guarantees

- **Bounded.** At most `max_files` files of at most `file_bytes` bytes. To make room, the oldest *closed* file is deleted first; a file being written (or handed out and not yet written) is never evicted.
- **Safe.** Files are created exclusively, mode 0600, never through a symlink, relative to the folder's held descriptor, and written through the descriptor they were created with; an existing `OverflowFile(path=...)` is refused with `FileExistsError`.
- **Only what it created.** A file is deleted only while its name still names the file created (same device and inode), never through a symlink; a private folder is removed only while its path names the folder made. A renamed folder, or something put in place of a file or folder, is left alone (logged). In a given folder only the names the `OverflowDir` created are removed.
- **Failures are retried.** A file that could not be deleted stays counted against the caps and is retried by the next `file()`, `close()` and the exit; a `close()` that could not remove everything stays registered for the exit.
- **One process.** Only the process that created a folder deletes its files: in a forked child, an inherited folder is inert (`file()` returns `None`) and the child's exit leaves the parent's files alone.
- **Explicit lifetime.** Files are deleted only when their writer finishes with complete output, by cap eviction, when their `OverflowDir` is closed (`close()` or a `with` block; idempotent), or at interpreter exit (`atexit`) for every `OverflowDir` never closed. **Never by garbage collection:** a dropped `OverflowDir` keeps its files until exit, so a path someone holds does not vanish at an arbitrary moment. The cost: an owner nobody closes keeps its capped files until exit.
- **A gone path says so.** Opening an evicted or closed file raises `FileNotFoundError`.
- **Threads.** `file()`, `close()` and writers finishing may run in any threads.

## Known limits

The identity check and the delete are two system calls: a same-user process racing between them is not defended against. A device and inode cannot tell a file of ours someone else deleted from a new file that got the same inode number. Without `dir_fd` support (e.g. Windows) the paths are used, with the same identity check.

A process killed without running `atexit` (`SIGKILL`, an abort, `os._exit`) leaves its files behind, under the system temporary directory (`nooa-overflow-*`), until the operating system's temporary-file cleanup removes them.

Tests: `tests/overflow` (filesystem behaviour only).
