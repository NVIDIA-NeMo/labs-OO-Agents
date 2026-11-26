# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A cross-process write lock for small user-level files."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def file_write_lock(path: Path) -> Iterator[Path]:
    """Hold the exclusive write lock of ``path``; yield the resolved path.

    The lock is a separate ``.<name>.lock`` file next to ``path``, so a
    writer can replace ``path`` with ``os.replace`` while holding it. On
    native Windows it uses ``filelock`` (LockFileEx); elsewhere ``fcntl.flock``.
    """
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    from nooa._platform import is_native_windows

    if is_native_windows():
        # Windows has no fcntl; filelock uses LockFileEx there and manages the
        # lock file itself.
        from filelock import FileLock

        with FileLock(lock_path):
            yield path
        return
    import fcntl

    # Never delete the lock file: a new file would let a writer skip processes
    # still waiting on the old one.
    with lock_path.open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield path


__all__ = ["file_write_lock"]
