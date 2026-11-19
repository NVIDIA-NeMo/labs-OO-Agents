# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output files: ``OverflowDir`` (a bounded private folder of them) and the ``OverflowWriter`` of one.

When a stream of output is too large to keep in memory, its whole content
can also go to a file, an *overflow file*. This package owns those files.
It is standalone: it imports nothing from the rest of ``nooa`` (the
snapshot ``TextStream`` writes through it, never the other way round).

Contract:

- **Private folders only.** An ``OverflowDir`` always creates its own
  folder in the system temporary directory (``tempfile.mkdtemp``: mode
  0700, an unpredictable ``nooa-overflow-*`` name). ``open()`` creates
  each file exclusively with ``O_NOFOLLOW`` (mode 0600) and its writer
  writes through the descriptor it was created with; deletes are by name
  inside the folder.
- **Threat model, CPython's** (``tempfile``, ``shutil.rmtree``): no other
  user can reach a private folder. The same user tampering inside their
  own private folder (renaming, replacing or linking its files) is out of
  scope: no file API can win there.
- **POSIX only.** Without ``os.O_NOFOLLOW`` (Windows) overflow files are
  off: ``open()`` returns ``None`` (logged once) and output stays in memory.
- **Bounded.** An ``OverflowDir`` holds at most ``max_files`` files of at
  most ``file_bytes`` bytes each. ``open()`` creates the new file first
  and only then deletes closed files to make room, the oldest *finished*
  first ("keep the newest"); while it runs, the new file (still empty) can
  be one more. It never deletes a file still being written. When nothing
  can go, or the file cannot be created, ``open()`` returns ``None`` (keep
  memory only) and nothing was deleted.
- **Explicit lifetime.** Files are deleted only (a) when their writer
  finishes with ``finish(delete=True)``, (b) by cap eviction, (c) when
  their ``OverflowDir`` is closed (``close()``, or the end of a ``with``
  block), or (d) at interpreter exit, for every ``OverflowDir`` never
  closed (``atexit``). **Never by garbage collection:** dropping an
  ``OverflowDir`` keeps its files until exit, so a path someone kept (in a
  ``Text.full``, a ``CommandResult``) does not vanish at an arbitrary
  moment. The cost: an owner nobody closes keeps its (capped) files until
  the process exits.
- **A path may be gone.** A path handed out earlier can name a file
  deleted since (evicted or closed). Opening it raises
  ``FileNotFoundError``: that is how a reader tells the output is gone.
- **Honest writers.** A writer has no buffer: ``write`` goes straight to
  the descriptor and ``written`` counts only bytes the file took. A write
  error raises ``OSError``; ``finish`` never raises, and when the close
  fails it logs it, deletes the file and returns ``False`` (not kept).
- **Errors.** File errors in eviction, ``close()`` and the exit cleanup
  are logged, never raised, never swallowed. A file that could not be
  deleted stays a closed file, counted against the caps, until a later
  eviction or the folder's removal deletes it; a folder that could not be
  removed stays registered for the next ``close()`` and the exit.
- **Processes.** An ``os.register_at_fork`` hook makes every inherited
  folder and writer inert in a forked child (no writes, deletes or exit
  cleanup; fresh locks, so the child never waits on a lock held at the
  fork). A child's own ``OverflowDir`` works normally.
- **Threads.** ``open()``, ``close()`` and writers finishing may run in any
  threads; the bookkeeping is locked. One writer is used by one thread at
  a time.
- **The exit.** The exit cleanup never runs without a folder's lock and
  never waits more than a second for it: a thread still holding it then
  removes the folder when it lets go. Once the exit has begun, ``open()``
  returns ``None``, and an ``OverflowDir`` made then (say by an ``atexit``
  handler registered before this package was imported, which runs after
  its cleanup) gets no folder at all.

Known limits: a process that ends without running ``atexit`` (``SIGKILL``,
an abort, ``os._exit``, or an ``exec`` that replaces it) leaves its files
behind in its private folder under the system temporary directory, where
the operating system's temporary file cleanup may eventually remove it. A
thread that holds a folder's lock through the whole exit (a hung
filesystem) leaves that folder behind (logged). An interrupt
(``KeyboardInterrupt``) at the instant a file is created removes the file
but can leak its descriptor.

>>> from nooa.overflow import OverflowDir
>>> with OverflowDir(max_files=2, file_bytes=4) as folder:
...     writer = folder.open()
...     writer.write(b"0123456789")
...     writer.finish(delete=False)
4
True
"""

from nooa.overflow._overflow import OverflowDir, OverflowWriter

__all__ = ["OverflowDir", "OverflowWriter"]
