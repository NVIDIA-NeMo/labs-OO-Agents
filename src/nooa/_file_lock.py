# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exclusive locks held by open file descriptors on Unix and Windows.

The caller owns the descriptor and must keep the lock file at a stable path.
Closing the descriptor, including on process exit, releases its lock. Take a
lock only once per descriptor; separate opens contend even within one process.
"""

import os

if os.name == "nt":
    import ctypes
    import errno
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _lock_file_ex = _kernel32.LockFileEx
    _lock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _lock_file_ex.restype = wintypes.BOOL
    _unlock_file_ex = _kernel32.UnlockFileEx
    _unlock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _unlock_file_ex.restype = wintypes.BOOL

    # Windows byte-range locks also block reads through other handles. Lock
    # beyond the short owner record so read_lock_owner() can still read it.
    # LockFileEx permits a range beyond EOF without extending the file.
    _LOCK_OFFSET = 0x7FFF_FFFF
    _LOCKFILE_FAIL_IMMEDIATELY = 0x1
    _LOCKFILE_EXCLUSIVE_LOCK = 0x2
    _ERROR_LOCK_VIOLATION = 33

    def lock_file(fd: int, *, blocking: bool = True) -> None:
        """Acquire an exclusive lock, raising BlockingIOError if nonblocking and busy."""
        handle = msvcrt.get_osfhandle(fd)
        flags = _LOCKFILE_EXCLUSIVE_LOCK
        if not blocking:
            flags |= _LOCKFILE_FAIL_IMMEDIATELY
        overlapped = _Overlapped(Offset=_LOCK_OFFSET)
        if not _lock_file_ex(handle, flags, 0, 1, 0, ctypes.byref(overlapped)):
            error = ctypes.get_last_error()
            if error == _ERROR_LOCK_VIOLATION:
                raise BlockingIOError(errno.EAGAIN, "File is already locked")
            raise ctypes.WinError(error)

    def unlock_file(fd: int) -> None:
        """Release a previously acquired lock without closing its descriptor."""
        overlapped = _Overlapped(Offset=_LOCK_OFFSET)
        if not _unlock_file_ex(msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(overlapped)):
            raise ctypes.WinError(ctypes.get_last_error())

else:
    import fcntl

    def lock_file(fd: int, *, blocking: bool = True) -> None:
        """Acquire an exclusive lock, raising BlockingIOError if nonblocking and busy."""
        flags = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
        fcntl.flock(fd, flags)

    def unlock_file(fd: int) -> None:
        """Release a previously acquired lock without closing its descriptor."""
        fcntl.flock(fd, fcntl.LOCK_UN)
