# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Windows Job Objects for owning a shell's descendants, including orphaned ones.

Imported only on Windows. Nested jobs mark completed background processes;
their future descendants inherit the same protection. All operations on a PID
check membership using an open process handle, so PID reuse cannot kill an
unrelated process. This is process lifecycle management, not a sandbox.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes

_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class _BasicLimit(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint64)
        for name in (
            "ReadOperationCount",
            "WriteOperationCount",
            "OtherOperationCount",
            "ReadTransferCount",
            "WriteTransferCount",
            "OtherTransferCount",
        )
    ]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


_kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_kernel32.CreateJobObjectW.restype = wintypes.HANDLE
_kernel32.SetInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_kernel32.SetInformationJobObject.restype = wintypes.BOOL
_kernel32.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_kernel32.QueryInformationJobObject.restype = wintypes.BOOL
_kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
_kernel32.IsProcessInJob.argtypes = [
    wintypes.HANDLE,
    wintypes.HANDLE,
    ctypes.POINTER(wintypes.BOOL),
]
_kernel32.IsProcessInJob.restype = wintypes.BOOL
_kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_kernel32.OpenProcess.restype = wintypes.HANDLE
_kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_kernel32.GetExitCodeProcess.restype = wintypes.BOOL
_kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_kernel32.TerminateJobObject.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL

_PROCESS_RIGHTS = 0x0100 | 0x0001 | 0x1000  # SET_QUOTA | TERMINATE | QUERY_LIMITED_INFORMATION


class WindowsJob:
    """A non-inheritable job handle; closing normally terminates its process tree."""

    def __init__(self, *, kill_on_close: bool = True):
        self._handle = _kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            self._set_kill_on_close(kill_on_close)
        except BaseException:
            self.close()
            raise

    def _set_kill_on_close(self, enabled: bool) -> None:
        info = _ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = 0x2000 if enabled else 0
        if not _kernel32.SetInformationJobObject(
            self._handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise ctypes.WinError(ctypes.get_last_error())

    def _contains_handle(self, process: int) -> bool:
        inside = wintypes.BOOL()
        if not _kernel32.IsProcessInJob(process, self._handle, ctypes.byref(inside)):
            raise ctypes.WinError(ctypes.get_last_error())
        return bool(inside.value)

    def assign(
        self,
        pid: int,
        *,
        source: WindowsJob | None = None,
        exclude: tuple[WindowsJob, ...] = (),
    ) -> bool:
        """Assign a live member of ``source`` unless a protected job already owns it."""
        process = _kernel32.OpenProcess(_PROCESS_RIGHTS, False, pid)
        if not process:
            error = ctypes.get_last_error()
            if error == 87:  # exited between enumeration and OpenProcess
                return False
            raise ctypes.WinError(error)
        try:
            if source is not None and not source._contains_handle(process):
                return False
            if self._contains_handle(process) or any(j._contains_handle(process) for j in exclude):
                return False
            if not _kernel32.AssignProcessToJobObject(self._handle, process):
                error = ctypes.get_last_error()
                status = wintypes.DWORD()
                if (
                    _kernel32.GetExitCodeProcess(process, ctypes.byref(status))
                    and status.value != 259
                ):
                    return False
                raise ctypes.WinError(error)
            return True
        finally:
            _kernel32.CloseHandle(process)

    def pids(self) -> list[int]:
        """Query job membership; descendants remain listed even after their parent exits."""
        capacity = 64
        while True:
            # JOBOBJECT_BASIC_PROCESS_ID_LIST: two DWORDs, followed by ULONG_PTRs.
            buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            if _kernel32.QueryInformationJobObject(
                self._handle, 3, buffer, ctypes.sizeof(buffer), None
            ):
                count = wintypes.DWORD.from_buffer(buffer, 4).value
                return list((ctypes.c_size_t * count).from_buffer(buffer, 8))
            error = ctypes.get_last_error()
            if error != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, wintypes.DWORD.from_buffer(buffer).value)

    def collect_background(
        self, shell_pid: int, protected: tuple[WindowsJob, ...]
    ) -> list[WindowsJob]:
        """Mark surviving children and their future descendants as background work."""
        groups: list[WindowsJob] = []
        try:
            while True:
                assigned = False
                for pid in self.pids():
                    if pid != shell_pid:
                        # A venv launcher can put its child in a nested job of
                        # its own. Separate empty jobs can join each process's
                        # hierarchy; one shared job cannot span sibling jobs.
                        group = WindowsJob(kill_on_close=False)
                        try:
                            if group.assign(pid, source=self, exclude=(*protected, *groups)):
                                groups.append(group)
                                assigned = True
                            else:
                                group.close()
                        except BaseException:
                            group.close()
                            raise
                if not assigned:
                    return groups
        except BaseException:
            for group in groups:
                group.close()
            raise

    def terminate_foreground(self, protected: tuple[WindowsJob, ...]) -> None:
        """Kill unfinished work while sparing all completed background process trees.

        First put the foreground into another nested job. Any children it starts
        during cleanup then inherit that job and are killed atomically with it.
        """
        if not protected:
            if not _kernel32.TerminateJobObject(self._handle, 1):
                raise ctypes.WinError(ctypes.get_last_error())
            return
        foreground: list[WindowsJob] = []
        try:
            while True:
                assigned = False
                for pid in self.pids():
                    group = WindowsJob()
                    try:
                        if group.assign(pid, source=self, exclude=(*protected, *foreground)):
                            foreground.append(group)
                            assigned = True
                        else:
                            group.close()
                    except BaseException:
                        group.close()
                        raise
                if not assigned:
                    break
        finally:
            for group in foreground:
                group.close()

    def close(self, *, keep_processes: bool = False) -> None:
        if self._handle:
            if keep_processes:
                self._set_kill_on_close(False)
            _kernel32.CloseHandle(self._handle)
            self._handle = None

    def __del__(self) -> None:
        if getattr(self, "_handle", None):
            _kernel32.CloseHandle(self._handle)
