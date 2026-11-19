# Output files: `nooa.overflow`

`nooa.overflow` owns the files that hold output too large to keep in memory (a command's whole stdout, a cell's output). It is a small standalone package: it imports nothing from the rest of `nooa`. `nooa.snapshot`'s `TextStream` writes through it (`TextStream(overflow=folder, overflow_bytes=...)` opens a file with `folder.open(...)` only once its output is cut), and a cut `Text.full` (an `Overflow`: path, kept, total) points to the file. The module docstring (`help(nooa.overflow)`) is the full contract; this page is the summary.

| Name | What it is |
|---|---|
| `OverflowDir(*, max_files=20, file_bytes=64 MiB)` | a bounded private folder of files, always its own `tempfile.mkdtemp` folder (0700, unpredictable name) in the system temporary directory |
| `OverflowDir.open(max_bytes=None)` | the `OverflowWriter` of a new file in the folder (exclusive, `O_NOFOLLOW`, 0600), capped at `min(max_bytes, file_bytes)`, or `None` when the budget is in use, the folder is closed or the exit began |
| `OverflowWriter` | `write(data)` (unbuffered, straight to the descriptor), `finish(delete=...)` (returns whether the file is kept), read-only `path`, `max_bytes`, `written`, `closed` |

## Threat model

CPython's (`tempfile`, `shutil.rmtree`): an `OverflowDir` works only in its own private folder, which no other user can reach. The same user tampering inside their own private folder (renaming, replacing or linking its files) is out of scope: no file API can win there. Each file is created exclusively (`O_CREAT | O_EXCL | O_NOFOLLOW`) and written through the descriptor it was created with, so a name replaced after creation is never written; deletes are by name inside the folder, and `close()` removes the folder with `shutil.rmtree`.

## Guarantees

- **Bounded.** At most `max_files` files of at most `file_bytes` bytes. `open()` creates the new file first, then deletes closed files to make room, the oldest *finished* first ("keep the newest"); while it runs the new (empty) file can be one more. A file being written is never evicted. When the file cannot be created, nothing is evicted.
- **Explicit lifetime.** Files are deleted only when their writer finishes with `delete=True`, by cap eviction, when their `OverflowDir` is closed (`close()` or a `with` block; idempotent), or at interpreter exit (`atexit`) for every `OverflowDir` never closed. **Never by garbage collection:** a dropped `OverflowDir` keeps its files until exit, so a path someone holds does not vanish at an arbitrary moment. The cost: an owner nobody closes keeps its capped files until exit.
- **A gone path says so.** Reading an evicted or closed file raises `FileNotFoundError`.
- **Honest writers.** No Python buffer: `written` counts only bytes the file took; a write error raises `OSError`. `finish` never raises: a failed close is logged, the file deleted, and `finish` returns `False`. A later `finish(delete=True)` deletes a kept file.
- **Failures are logged.** A file that could not be deleted stays a closed file, counted against the caps, until a later eviction or the folder's removal deletes it; a folder that could not be removed stays registered for the next `close()` and the exit.
- **Fork.** An `os.register_at_fork` hook makes every inherited folder and writer inert in the child (no writes, deletes or exit cleanup) and gives fresh locks; a child's own `OverflowDir` works normally.
- **Threads.** `open()`, `close()` and writers finishing may run in any threads; one writer is used by one thread at a time.
- **The exit.** The exit cleanup never runs without a folder's lock and waits at most a second for it; a thread still holding it removes the folder when it lets go. During the exit `open()` returns `None`, and an `OverflowDir` made then (by an `atexit` handler that runs after the cleanup) gets no folder.

## Platforms

POSIX only. Where `os.O_NOFOLLOW` is missing (Windows), overflow files are off: `open()` returns `None` (logged once) and every stream keeps memory only, saying the full output was not kept.

## Known limits

A process that ends without running `atexit` (`SIGKILL`, an abort, `os._exit`, an `exec`) leaves its files behind, in the private folder under the system temporary directory, where the operating system's temporary-file cleanup may eventually remove them. A thread holding a folder's lock through the whole exit (a hung filesystem) leaves that folder (logged). An interrupt at the instant a file is created removes the file but can leak its descriptor.

Tests: `tests/overflow` (filesystem behaviour only).
