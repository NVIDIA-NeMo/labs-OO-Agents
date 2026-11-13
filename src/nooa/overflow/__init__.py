# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output files: ``OverflowDir`` (a bounded private folder of them), ``OverflowFile`` (one of them) and its ``OverflowWriter``.

When a stream of output is too large to keep in memory, its whole content
can also go to a file, an *overflow file*. This package owns those files.
It is standalone: it imports nothing from the rest of ``nooa`` (the
snapshot ``TextStream`` writes through it, never the other way round).

Contract:

- **Private folders only.** An ``OverflowDir`` always creates its own
  folder (``tempfile.mkdtemp``: mode 0700, an unpredictable
  ``nooa-overflow-*`` name); its ``parent`` argument only says where. Each
  file is created exclusively with ``O_NOFOLLOW`` (mode 0600) relative to
  the folder's held descriptor and written through the descriptor it was
  created with. A standalone ``OverflowFile(path=...)`` that already
  exists is refused (``FileExistsError``), never truncated.
- **Threat model, CPython's** (``tempfile``, ``shutil.rmtree``): no other
  user can reach a private folder. The same user tampering inside their
  own private folder (renaming, replacing or linking its files) is out of
  scope: no file API can win there. Platform fallbacks: ``_overflow``.
- **Bounded.** An ``OverflowDir`` holds at most ``max_files`` files of at
  most ``file_bytes`` bytes each. ``file()`` creates the new file first
  and only then deletes closed files to make room, the oldest *finished*
  first ("keep the newest"); while it runs, the new file (still empty) can
  be one more. It never deletes a file still being written, or handed out
  and neither opened nor released (``OverflowFile.release()``). When
  nothing can go, or the file cannot be created, ``file()`` returns
  ``None`` (keep memory only) and nothing was deleted.
- **Explicit lifetime.** Files are deleted only (a) when their writer
  finishes with ``finish(delete=True)`` (``TextStream`` asks for that when
  its output was complete), (b) by cap eviction, (c) when their
  ``OverflowDir`` is closed (``close()``, or the end of a ``with`` block),
  or (d) at interpreter exit, for every ``OverflowDir`` never closed
  (``atexit``). **Never by garbage collection:** dropping an
  ``OverflowDir`` keeps its files until exit, so a path someone kept (in a
  ``Text.full``, a ``CommandResult``) does not vanish at an arbitrary
  moment. The cost: an owner nobody closes keeps its (capped) files until
  the process exits, and a file handed out and never opened keeps its
  place until it is released or the folder closes.
- **A path may be gone.** A path handed out earlier can name a file
  deleted since (evicted or closed). Opening it raises
  ``FileNotFoundError``: that is how a reader tells the output is gone.
- **Honest writers.** A writer has no buffer: ``write`` goes straight to
  the descriptor and ``written`` counts only bytes the file took. A write
  error raises ``OSError``; ``finish`` never raises, and when the close
  fails it logs it, deletes the file and returns ``False`` (not kept).
- **Errors.** File errors in eviction, ``close()`` and the exit cleanup
  are logged, never raised, never swallowed. A file that could not be
  deleted stays counted against the caps and is retried (next ``file()``,
  ``close()``, exit); a folder that could not be removed stays registered
  for the next ``close()`` and the exit.
- **Processes.** An ``os.register_at_fork`` hook makes every inherited
  folder and writer inert in a forked child (no writes, deletes or exit
  cleanup; fresh locks, so the child never waits on a lock held at the
  fork). A child's own ``OverflowDir`` works normally.
- **Threads.** ``file()``, ``close()``, ``OverflowFile.open()`` /
  ``release()`` and writers finishing may run in any threads; the
  bookkeeping is locked. One writer is used by one thread at a time.
- **The exit.** The exit cleanup never runs without a folder's lock and
  never waits more than a second for it: a thread still holding it then
  removes the folder when it lets go. Once the exit has begun, ``file()``
  returns ``None``, and an ``OverflowDir`` made then (say by an ``atexit``
  handler registered before this package was imported, which runs after
  its cleanup) gets no folder at all.

Known limits: a process that ends without running ``atexit`` (``SIGKILL``,
an abort, ``os._exit``, or an ``exec`` that replaces it) leaves its files
behind; a private folder is under the system temporary directory (unless
``parent`` says otherwise), where the operating system's temporary file
cleanup may eventually remove it. A thread that holds a folder's lock
through the whole exit (a hung filesystem) leaves that folder behind
(logged). An interrupt (``KeyboardInterrupt``) at the instant a file is
created removes the file but can leak its descriptor.

>>> from nooa.overflow import OverflowDir
>>> with OverflowDir(max_files=2, file_bytes=4) as folder:
...     writer = folder.file().open()
...     writer.write(b"0123456789")
...     writer.finish(delete=False)
4
True
"""

from nooa.overflow._overflow import OverflowDir, OverflowFile, OverflowWriter

__all__ = ["OverflowDir", "OverflowFile", "OverflowWriter"]
