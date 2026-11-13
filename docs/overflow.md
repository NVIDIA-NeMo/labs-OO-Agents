# Output files: `nooa.overflow`

`nooa.overflow` owns the files that hold output too large to keep in memory (a command's whole stdout, a cell's output). It is a small standalone package: it imports nothing from the rest of `nooa`. `nooa.snapshot`'s `TextStream` writes through it (`TextStream(overflow=folder.file())`), and a cut `Text.full` (an `Overflow`: path, kept, total) points to the file. The module docstring (`help(nooa.overflow)`) is the full contract; this page is the summary.

| Name | What it is |
|---|---|
| `OverflowDir(parent=None, *, max_files=20, file_bytes=64 MiB)` | a bounded private folder of files, always its own `tempfile.mkdtemp` folder (0700, unpredictable name); `parent` only says where to create it |
| `OverflowDir.file()` | a new `OverflowFile` in the folder (exclusive, `O_NOFOLLOW`, 0600), or `None` when the budget is in use |
| `OverflowFile(path=None, max_bytes=64 MiB)` | one file; standalone or handed out by a folder; `open()` returns its `OverflowWriter`, `release()` gives a folder's unopened file back |
| `OverflowWriter` | `write(data)` (unbuffered, straight to the descriptor), `finish(delete=...)` (returns whether the file is kept), read-only `path`, `max_bytes`, `written`, `closed` |

## Threat model

CPython's (`tempfile`, `shutil.rmtree`): an `OverflowDir` works only in its own private folder, which no other user can reach. The same user tampering inside their own private folder (renaming, replacing or linking its files) is out of scope: no file API can win there. Files are created exclusively relative to the folder's held descriptor and written through the descriptor they were created with, so a name replaced after creation is never written; deletes are by name.

## Guarantees

- **Bounded.** At most `max_files` files of at most `file_bytes` bytes. `file()` creates the new file first, then deletes closed files to make room, the oldest *finished* first ("keep the newest"); while it runs the new (empty) file can be one more. A file being written, or handed out and neither opened nor released, is never evicted. When the file cannot be created, nothing is evicted.
- **Explicit lifetime.** Files are deleted only when their writer finishes with `delete=True`, by cap eviction, when their `OverflowDir` is closed (`close()` or a `with` block; idempotent), or at interpreter exit (`atexit`) for every `OverflowDir` never closed. **Never by garbage collection:** a dropped `OverflowDir` keeps its files until exit, so a path someone holds does not vanish at an arbitrary moment. The cost: an owner nobody closes keeps its capped files until exit.
- **A handed-out file holds its place.** A file from `file()` that is never opened keeps one of the `max_files` places until `OverflowFile.release()` or `close()`: release it when you will not open it (for example when building the stream fails).
- **A gone path says so.** Opening an evicted or closed file raises `FileNotFoundError`.
- **Honest writers.** No Python buffer: `written` counts only bytes the file took; a write error raises `OSError`. `finish` never raises: a failed close is logged, the file deleted, and `finish` returns `False`. A later `finish(delete=True)` deletes a kept file.
- **Failures are logged and retried.** A file that could not be deleted stays counted against the caps and is retried by the next `file()`, `close()` and the exit; a folder that could not be removed stays registered for the next `close()` and the exit.
- **Fork.** An `os.register_at_fork` hook makes every inherited folder and writer inert in the child (no writes, deletes or exit cleanup) and gives fresh locks; a child's own `OverflowDir` works normally.
- **Threads.** `file()`, `close()`, `open()`/`release()` and writers finishing may run in any threads; one writer is used by one thread at a time.
- **The exit.** The exit cleanup never runs without a folder's lock and waits at most a second for it; a thread still holding it removes the folder when it lets go. During the exit `file()` returns `None`, and an `OverflowDir` made then (by an `atexit` handler that runs after the cleanup) gets no folder.

## Platforms

Without `dir_fd` support (`os.supports_dir_fd`, e.g. Windows) full paths are used instead of the folder's descriptor. `O_NOFOLLOW` is used where it exists; Windows has none, so a symlink put inside the private folder would be followed there (same-user tampering, out of scope). `O_BINARY` is added where it exists, so Windows never translates newlines. Windows cannot delete an open file: a folder closed while a writer is open keeps that file until a later `close()` or the exit retries.

## Known limits

A process that ends without running `atexit` (`SIGKILL`, an abort, `os._exit`, an `exec`) leaves its files behind, in the private folder under the system temporary directory (or `parent`), where the operating system's temporary-file cleanup may eventually remove them. A thread holding a folder's lock through the whole exit (a hung filesystem) leaves that folder (logged). An interrupt at the instant a file is created removes the file but can leak its descriptor.

Tests: `tests/overflow` (filesystem behaviour only).
