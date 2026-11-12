# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Output files: ``OverflowDir`` (a bounded folder of them) and ``OverflowFile`` (one of them).

When a stream of output is too large to keep in memory, its whole content
can also go to a file, an *overflow file*. This package owns those files.
It is standalone: it imports nothing from the rest of ``nooa`` (the
snapshot ``TextStream`` writes through it, never the other way round).

Contract:

- **Bounded.** An ``OverflowDir`` holds at most ``max_files`` files of at
  most ``file_bytes`` bytes each. To make room for a new file it deletes
  its oldest *closed* file first; it never deletes a file still being
  written (or handed out and not yet written). When nothing can go,
  ``OverflowDir.file()`` returns ``None``: keep memory only.
- **Safe files.** A private folder is made with ``tempfile.mkdtemp`` (mode
  0700), each file with ``tempfile.mkstemp`` (exclusive, mode 0600). A
  standalone ``OverflowFile(path=...)`` that already exists is refused
  (``FileExistsError``), never truncated. Removal never follows a
  symlink.
- **Explicit lifetime.** Files are deleted only (a) when their writer
  finishes with complete output (``finish(delete=True)``), (b) by cap
  eviction, (c) when their ``OverflowDir`` is closed (``close()``, or the
  end of a ``with`` block), or (d) at interpreter exit, for every
  ``OverflowDir`` never closed (``atexit``). **Never by garbage
  collection:** dropping an ``OverflowDir`` keeps its files until exit, so
  a path someone kept (in a ``Text.full``, a ``CommandResult``) does not
  vanish at an arbitrary moment. The cost: an owner nobody closes keeps
  its (capped) files until the process exits.
- **A path may be gone.** A path handed out earlier can name a file
  deleted since (evicted or closed). Opening it raises
  ``FileNotFoundError``: that is how a reader tells the output is gone.
- **Errors.** File errors in eviction, ``close()`` and exit cleanup are
  logged, never raised.
- **Threads.** ``file()``, ``close()`` and writers finishing may run in any
  threads; the bookkeeping is locked.

Known limit: a process killed without running ``atexit`` (``SIGKILL``, an
abort, ``os._exit``) leaves its files behind; a private folder is under
the system temporary directory (``nooa-overflow-*``), where the operating
system's temporary file cleanup eventually removes it.

>>> from nooa.overflow import OverflowDir
>>> with OverflowDir(max_files=2, file_bytes=4) as folder:
...     writer = folder.file().open()
...     writer.write(b"0123456789")
...     writer.finish(delete=False)
4
"""

from nooa.overflow._overflow import OverflowDir, OverflowFile

__all__ = ["OverflowDir", "OverflowFile"]
