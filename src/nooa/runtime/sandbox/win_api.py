# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ctypes bindings for the Windows AppContainer sandbox backend.

Dependency-free on purpose: the worker runs from a private interpreter copy
whose site-packages are not guaranteed to contain ``pywin32``.

The design decisions encoded here come from ``experiments/windows-sandbox-probe``:

* an AppContainer is entered through ``PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES``,
  which exists only at process creation — it cannot be applied afterwards;
* the process is created **suspended**, assigned to its Job Object, and only
  then resumed, so there is no window in which container code runs outside the
  job;
* ``PROC_THREAD_ATTRIBUTE_HANDLE_LIST`` accepts inheritable handles only;
* teardown uses ``REVOKE_ACCESS``, not SDDL replay: ``GRANT_ACCESS`` merges
  into an existing ACE for the same SID, so replaying additions drifts.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import wintypes

__all__ = [
    "MODIFY",
    "Child",
    "Job",
    "container_sid",
    "create_profile",
    "delete_profile",
    "grant",
    "revoke",
    "sid_from_string",
    "sid_string",
    "spawn_in_container",
]

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_adv = ctypes.WinDLL("advapi32", use_last_error=True)
_user = ctypes.WinDLL("userenv", use_last_error=True)

INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value
WAIT_TIMEOUT = 0x00000102
WAIT_OBJECT_0 = 0x00000000
STILL_ACTIVE = 259

# --- process creation -------------------------------------------------------
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
HANDLE_FLAG_INHERIT = 0x00000001

# --- tokens -----------------------------------------------------------------
TOKEN_QUERY = 0x0008
TokenIsAppContainer = 29
TokenAppContainerSid = 31

# --- security ---------------------------------------------------------------
SE_FILE_OBJECT = 1
DACL_SECURITY_INFORMATION = 0x00000004
SE_DACL_PRESENT = 0x00000004
PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000

GRANT_ACCESS = 1
REVOKE_ACCESS = 4
NO_INHERITANCE = 0
SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
TRUSTEE_IS_SID = 0
TRUSTEE_IS_UNKNOWN = 0

# Access masks. Generic bits are what callers ask for; the effective mask is
# what the kernel stores.
FILE_GENERIC_READ = 0x00120089
FILE_GENERIC_EXECUTE = 0x001200A0
FILE_GENERIC_WRITE = 0x00120116
FILE_ALL_ACCESS = 0x001F01FF
READ_EXECUTE = FILE_GENERIC_READ | FILE_GENERIC_EXECUTE
# What a cell gets on its workspace: modify, without the two rights that change
# who may access the object. FILE_ALL_ACCESS includes WRITE_DAC and
# WRITE_OWNER, so a cell holding it can add an ACE for any SID — including
# ALL APPLICATION PACKAGES — and that ACE outlives the session, because revoke
# only removes the ACE for the container's own SID. Measured: after one
# session's cell ran icacls, a second session could write into its workspace.
MODIFY = FILE_GENERIC_READ | FILE_GENERIC_WRITE | FILE_GENERIC_EXECUTE | 0x00000001  # 0x001201BF

# --- job objects ------------------------------------------------------------
JobObjectBasicLimitInformation = 2
JobObjectExtendedLimitInformation = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004

# Capability SIDs are fixed well-known values. Derivation from a name is a
# package-model registry lookup, not a hash, so the documented constants are
# used instead of guessing one.
WELL_KNOWN_CAPABILITIES: dict[str, str] = {
    "internetClient": "S-1-15-3-1",
    "internetClientServer": "S-1-15-3-2",
    "privateNetworkClientServer": "S-1-15-3-3",
    "documentsLibrary": "S-1-15-3-7",
    "picturesLibrary": "S-1-15-3-8",
}


def _err() -> OSError:
    return ctypes.WinError(ctypes.get_last_error())


def _hr(hr: int, what: str) -> None:
    if hr < 0:
        raise OSError(hr & 0xFFFFFFFF, f"{what} failed")


# --- structures -------------------------------------------------------------
class _SidAndAttributes(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _SecurityCapabilities(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("Capabilities", ctypes.POINTER(_SidAndAttributes)),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


class _StartupInfoW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _StartupInfoExW(ctypes.Structure):
    _fields_ = [
        ("StartupInfo", _StartupInfoW),
        ("lpAttributeList", ctypes.c_void_p),
    ]


class _ProcessInformation(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class _TrusteeW(ctypes.Structure):
    _fields_ = [
        ("pMultipleTrustee", ctypes.c_void_p),
        ("MultipleTrusteeOperation", ctypes.c_int),
        ("TrusteeForm", ctypes.c_int),
        ("TrusteeType", ctypes.c_int),
        ("ptstrName", ctypes.c_void_p),
    ]


class _ExplicitAccessW(ctypes.Structure):
    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", _TrusteeW),
    ]


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


# --- prototypes -------------------------------------------------------------
_k32.LocalFree.argtypes = [ctypes.c_void_p]
_k32.LocalFree.restype = ctypes.c_void_p
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.CloseHandle.restype = wintypes.BOOL
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.GetCurrentProcessId.restype = wintypes.DWORD
_k32.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
_k32.SetHandleInformation.restype = wintypes.BOOL
_k32.InitializeProcThreadAttributeList.argtypes = [
    ctypes.c_void_p,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_size_t),
]
_k32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
_k32.UpdateProcThreadAttribute.argtypes = [
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_k32.UpdateProcThreadAttribute.restype = wintypes.BOOL
_k32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
_k32.CreateProcessW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPWSTR,
    ctypes.c_void_p,
    ctypes.c_void_p,
    wintypes.BOOL,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.LPCWSTR,
    ctypes.POINTER(_StartupInfoExW),
    ctypes.POINTER(_ProcessInformation),
]
_k32.CreateProcessW.restype = wintypes.BOOL
_k32.ResumeThread.argtypes = [wintypes.HANDLE]
_k32.ResumeThread.restype = wintypes.DWORD
_k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_k32.GetExitCodeProcess.restype = wintypes.BOOL
_k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
_k32.TerminateProcess.restype = wintypes.BOOL
_k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_k32.CreateJobObjectW.restype = wintypes.HANDLE
_k32.SetInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_k32.SetInformationJobObject.restype = wintypes.BOOL
_k32.QueryInformationJobObject.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.c_void_p,
]
_k32.QueryInformationJobObject.restype = wintypes.BOOL
_k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_k32.AssignProcessToJobObject.restype = wintypes.BOOL
_k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_k32.TerminateJobObject.restype = wintypes.BOOL

_adv.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
_adv.ConvertStringSidToSidW.restype = wintypes.BOOL
_adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
_adv.ConvertSidToStringSidW.restype = wintypes.BOOL
_adv.SetEntriesInAclW.argtypes = [
    wintypes.ULONG,
    ctypes.POINTER(_ExplicitAccessW),
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_void_p),
]
_adv.SetEntriesInAclW.restype = wintypes.DWORD
_adv.SetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]
_adv.SetNamedSecurityInfoW.restype = wintypes.DWORD
_adv.GetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,
    ctypes.c_int,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
    ctypes.POINTER(ctypes.c_void_p),
]
_adv.GetNamedSecurityInfoW.restype = wintypes.DWORD
_adv.OpenProcessToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]
_adv.OpenProcessToken.restype = wintypes.BOOL
_adv.GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_adv.GetTokenInformation.restype = wintypes.BOOL

_user.CreateAppContainerProfile.argtypes = [
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    wintypes.LPCWSTR,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(ctypes.c_void_p),
]
_user.CreateAppContainerProfile.restype = ctypes.c_long
_user.DeriveAppContainerSidFromAppContainerName.argtypes = [
    wintypes.LPCWSTR,
    ctypes.POINTER(ctypes.c_void_p),
]
_user.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
_user.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
_user.DeleteAppContainerProfile.restype = ctypes.c_long

_ERROR_ALREADY_EXISTS = 0x800700B7
_ERROR_FILE_NOT_FOUND = 0x80070002


# --- AppContainer profile ---------------------------------------------------
def create_profile(name: str) -> None:
    """Create the container profile, or reuse the existing one.

    Reuse is the normal case: the profile outlives individual sandbox sessions,
    and its SID must stay stable so ACEs from an earlier session can be revoked
    by SID rather than by name.
    """
    sid = ctypes.c_void_p()
    hr = _user.CreateAppContainerProfile(name, name, name, None, 0, ctypes.byref(sid))
    if sid:
        _k32.LocalFree(sid)
    if (hr & 0xFFFFFFFF) != _ERROR_ALREADY_EXISTS:
        _hr(hr, "CreateAppContainerProfile")


def delete_profile(name: str) -> None:
    """Remove the profile. Only for tests: it invalidates the SID."""
    hr = _user.DeleteAppContainerProfile(name)
    if (hr & 0xFFFFFFFF) not in (_ERROR_FILE_NOT_FOUND,):
        _hr(hr, "DeleteAppContainerProfile")


def container_sid(name: str) -> ctypes.c_void_p:
    sid = ctypes.c_void_p()
    _hr(
        _user.DeriveAppContainerSidFromAppContainerName(name, ctypes.byref(sid)),
        "DeriveAppContainerSidFromAppContainerName",
    )
    return sid


def sid_string(sid: ctypes.c_void_p) -> str:
    out = wintypes.LPWSTR()
    if not _adv.ConvertSidToStringSidW(sid, ctypes.byref(out)):
        raise _err()
    try:
        return out.value
    finally:
        _k32.LocalFree(out)


def sid_from_string(text: str) -> ctypes.c_void_p | None:
    """The SID a string names, or None if it is not a valid SID.

    Needed wherever a grant must be revoked after its profile may be gone:
    ``DeriveAppContainerSidFromAppContainerName`` keeps working for a deleted
    name, but the caller cannot tell the two cases apart, so a SID that was
    recorded when the ACE was written is the only dependable source.
    """
    sid = ctypes.c_void_p()
    if not _adv.ConvertStringSidToSidW(text, ctypes.byref(sid)):
        return None
    return sid


def capability_sid(name: str) -> ctypes.c_void_p:
    text = WELL_KNOWN_CAPABILITIES.get(name)
    if text is None:
        raise ValueError(f"unknown capability {name!r}; known: {sorted(WELL_KNOWN_CAPABILITIES)}")
    sid = ctypes.c_void_p()
    if not _adv.ConvertStringSidToSidW(text, ctypes.byref(sid)):
        raise _err()
    return sid


# --- DACL -------------------------------------------------------------------
# A DACL edit is read-modify-write: GetNamedSecurityInfoW, SetEntriesInAclW,
# SetNamedSecurityInfoW. Two edits that interleave lose one of them, and this
# process does edit from several threads — grants run on the caller's thread
# while teardown revokes on a worker (executor._aterminate_worker uses
# asyncio.to_thread). Measured before the lock: 15 rounds of "grant A, revoke A
# on one thread while granting B on another" lost B's ACE 10 times.
#
# One lock for every path, not one per path: the interleaving that matters is
# between different SIDs on the *same* object, and a per-path lock would still
# serialise the whole runtime root (which is what most edits touch). The cost
# is that concurrent edits to unrelated directories wait for each other; an
# edit takes milliseconds, so that is cheaper than the bookkeeping a finer lock
# would need.
_DACL_LOCK = threading.RLock()


def _set_entry(path: str, sid: ctypes.c_void_p, access: int, mode: int, inheritance: int) -> None:
    with _DACL_LOCK:
        _set_entry_locked(path, sid, access, mode, inheritance)


def _set_entry_locked(
    path: str, sid: ctypes.c_void_p, access: int, mode: int, inheritance: int
) -> None:
    sd = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    rc = _adv.GetNamedSecurityInfoW(
        path,
        SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(sd),
    )
    if rc != 0:
        raise OSError(rc, f"GetNamedSecurityInfoW failed for {path}")
    try:
        ea = _ExplicitAccessW()
        ea.grfAccessPermissions = access
        ea.grfAccessMode = mode
        ea.grfInheritance = inheritance
        ea.Trustee.TrusteeForm = TRUSTEE_IS_SID
        ea.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN
        ea.Trustee.ptstrName = sid
        new_dacl = ctypes.c_void_p()
        rc = _adv.SetEntriesInAclW(1, ctypes.byref(ea), dacl, ctypes.byref(new_dacl))
        if rc != 0:
            raise OSError(rc, f"SetEntriesInAclW failed for {path}")
        try:
            rc = _adv.SetNamedSecurityInfoW(
                path,
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                None,
                None,
                new_dacl,
                None,
            )
            if rc != 0:
                raise OSError(rc, f"SetNamedSecurityInfoW failed for {path}")
        finally:
            _k32.LocalFree(new_dacl)
    finally:
        _k32.LocalFree(sd)


def grant(path: str, sid: ctypes.c_void_p, access: int) -> None:
    """Add an inheritable allow-ACE for ``sid`` on ``path``."""
    _set_entry(path, sid, access, GRANT_ACCESS, SUB_CONTAINERS_AND_OBJECTS_INHERIT)


def revoke(path: str, sid: ctypes.c_void_p) -> None:
    """Remove every explicit ACE for ``sid``; inheritance re-propagates.

    Reliable teardown, unlike replaying a saved SDDL: ``GRANT_ACCESS`` merges
    into an existing ACE for the same SID, and the ``AI`` bit of an inheriting
    directory is recomputed on every write.
    """
    _set_entry(path, sid, 0, REVOKE_ACCESS, NO_INHERITANCE)


def read_dacl(path: str) -> str:
    """Current DACL as SDDL text (audit only — never used as the undo path)."""
    sd = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    rc = _adv.GetNamedSecurityInfoW(
        path,
        SE_FILE_OBJECT,
        DACL_SECURITY_INFORMATION,
        None,
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(sd),
    )
    if rc != 0:
        raise OSError(rc, f"GetNamedSecurityInfoW failed for {path}")
    try:
        out = wintypes.LPWSTR()
        length = wintypes.DWORD()
        if not _adv.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            sd, 1, DACL_SECURITY_INFORMATION, ctypes.byref(out), ctypes.byref(length)
        ):
            raise _err()
        try:
            return out.value
        finally:
            _k32.LocalFree(out)
    finally:
        _k32.LocalFree(sd)


# --- Job Object -------------------------------------------------------------
class Job:
    """A Job Object that owns the container process and all its descendants."""

    def __init__(self, *, kill_on_close: bool = True, name: str | None = None):
        self._h = _k32.CreateJobObjectW(None, name)
        if not self._h:
            raise _err()
        self._limits = _ExtendedLimit()
        self._flags = 0
        # Keep the requested values as fields, not as whatever the previous
        # struct happened to hold. Rebuilding ``_ExtendedLimit`` from scratch on
        # each call would otherwise drop a limit set by an earlier call or leave
        # its flag set with a zero value.
        self._memory_mb = 0
        self._cpu_seconds = 0
        if kill_on_close:
            self._limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            self._flags |= JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            self._apply()

    def _apply(self) -> None:
        if not _k32.SetInformationJobObject(
            self._h,
            JobObjectExtendedLimitInformation,
            ctypes.byref(self._limits),
            ctypes.sizeof(self._limits),
        ):
            raise _err()

    def set_limits(self, *, memory_mb: int = 0, cpu_seconds: int = 0) -> None:
        """Apply absolute process-memory and total-job CPU limits.

        ``memory_mb`` here is an *absolute* process-memory ceiling. The
        executor converts the configured headroom into one by measuring the
        worker's baseline first (see ``win_launcher``), because the Linux
        backend's ``RLIMIT_AS`` value is headroom above the worker's baseline.

        Limits accumulate: omitting an argument keeps the previously applied
        value, so a later call cannot silently unset an earlier limit.
        """
        if memory_mb:
            self._memory_mb = int(memory_mb)
        if cpu_seconds:
            self._cpu_seconds = int(cpu_seconds)

        info = _ExtendedLimit()
        info.BasicLimitInformation.LimitFlags = self._flags
        if self._cpu_seconds:
            info.BasicLimitInformation.PerJobUserTimeLimit = self._cpu_seconds * 10_000_000
            info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_TIME
        if self._memory_mb:
            info.ProcessMemoryLimit = self._memory_mb * 1024 * 1024
            info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_PROCESS_MEMORY
        self._limits = info
        self._flags = info.BasicLimitInformation.LimitFlags
        self._apply()

    def peak_memory_bytes(self) -> int:
        info = _ExtendedLimit()
        if not _k32.QueryInformationJobObject(
            self._h,
            JobObjectExtendedLimitInformation,
            ctypes.byref(info),
            ctypes.sizeof(info),
            None,
        ):
            raise _err()
        return int(info.PeakProcessMemoryUsed)

    def assign(self, process_handle: int) -> None:
        if not _k32.AssignProcessToJobObject(self._h, process_handle):
            raise _err()

    def terminate(self, code: int = 1) -> None:
        _k32.TerminateJobObject(self._h, code)

    def close(self) -> None:
        if self._h:
            _k32.CloseHandle(self._h)
            self._h = None

    def __enter__(self) -> Job:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --- process creation -------------------------------------------------------
class Child:
    """A container process, suspended until :meth:`resume`."""

    def __init__(self, pi: _ProcessInformation, cmdline: str):
        self.pi = pi
        self.pid = pi.dwProcessId
        self.cmdline = cmdline

    def resume(self) -> None:
        if _k32.ResumeThread(self.pi.hThread) == 0xFFFFFFFF:
            raise _err()

    def wait(self, timeout_ms: int = 60000) -> int:
        _k32.WaitForSingleObject(self.pi.hProcess, timeout_ms)
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(self.pi.hProcess, ctypes.byref(code)):
            raise _err()
        return code.value

    def close(self) -> None:
        for handle in (self.pi.hProcess, self.pi.hThread):
            if handle:
                _k32.CloseHandle(handle)
        self.pi.hProcess = self.pi.hThread = None


def spawn_in_container(
    argv: list[str],
    *,
    container_name: str,
    capabilities: tuple[str, ...] = (),
    inherit_handles: tuple[int, ...] = (),
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> Child:
    """Create a suspended process inside the AppContainer.

    ``SECURITY_CAPABILITIES`` is the only way in, and it is a creation-time
    attribute — which is why this cannot be expressed as "start first, confine
    later". The caller must assign the process to its Job Object and only then
    resume it.
    """
    app_sid = container_sid(container_name)
    cap_sids = [capability_sid(name) for name in capabilities]
    if cap_sids:
        cap_array = (_SidAndAttributes * len(cap_sids))()
        for index, cap_sid in enumerate(cap_sids):
            cap_array[index].Sid = cap_sid
            cap_array[index].Attributes = 0x00000004  # SE_GROUP_ENABLED
    else:
        cap_array = ctypes.POINTER(_SidAndAttributes)()

    caps = _SecurityCapabilities()
    caps.AppContainerSid = app_sid
    caps.Capabilities = cap_array
    caps.CapabilityCount = len(cap_sids)
    caps.Reserved = 0

    attribute_count = 1 + (1 if inherit_handles else 0)
    size = ctypes.c_size_t()
    _k32.InitializeProcThreadAttributeList(None, attribute_count, 0, ctypes.byref(size))
    if size.value == 0:
        raise _err()
    buffer = ctypes.create_string_buffer(size.value)
    if not _k32.InitializeProcThreadAttributeList(buffer, attribute_count, 0, ctypes.byref(size)):
        raise _err()

    try:
        if not _k32.UpdateProcThreadAttribute(
            buffer,
            0,
            PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
            ctypes.byref(caps),
            ctypes.sizeof(caps),
            None,
            None,
        ):
            raise _err()

        handle_array = None
        if inherit_handles:
            handle_array = (wintypes.HANDLE * len(inherit_handles))(*inherit_handles)
            if not _k32.UpdateProcThreadAttribute(
                buffer,
                0,
                PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                handle_array,
                ctypes.sizeof(handle_array),
                None,
                None,
            ):
                raise _err()

        startup = _StartupInfoExW()
        startup.StartupInfo.cb = ctypes.sizeof(_StartupInfoExW)
        startup.lpAttributeList = ctypes.cast(buffer, ctypes.c_void_p)

        info = _ProcessInformation()
        command = ctypes.create_unicode_buffer(_command_line(argv))
        flags = (
            EXTENDED_STARTUPINFO_PRESENT
            | CREATE_SUSPENDED
            | CREATE_UNICODE_ENVIRONMENT
            | CREATE_NO_WINDOW
        )
        env_block = _environment_block(env) if env is not None else None
        ok = _k32.CreateProcessW(
            None,
            command,
            None,
            None,
            bool(inherit_handles),
            flags,
            ctypes.cast(env_block, ctypes.c_void_p) if env_block is not None else None,
            cwd,
            ctypes.byref(startup),
            ctypes.byref(info),
        )
        if not ok:
            raise _err()
        # Keep the SID arrays alive until CreateProcessW has consumed them.
        _ = handle_array, cap_sids, app_sid
        return Child(info, command.value)
    finally:
        _k32.DeleteProcThreadAttributeList(buffer)


def _command_line(argv: list[str]) -> str:
    """Quote ``argv`` the way ``CommandLineToArgvW`` will split it back.

    Three things the obvious ``' '.join`` gets wrong:

    * an **empty** argument has no space in it, so it was emitted as nothing at
      all and the neighbouring arguments shifted into its slot — measured with
      the launcher's own argv, where a missing workspace made the tokenizer
      cache path land in the crash-log position;
    * an argument containing a **quote** was passed through unescaped, letting
      it split into several arguments or end the quoting early;
    * a trailing backslash before a quote needs doubling, or it escapes the
      quote that was meant to close the argument.

    The child runs with ``-I`` and a fixed argv shape, so this only has to be
    correct for the strings the launcher passes — but it is written to the
    documented rules rather than to those strings.

    The caller must pass a program name first. ``CommandLineToArgvW`` parses
    the first token with different rules (it may treat it as a path rather than
    a quoted argument), so quoting a lone argument yields a different split
    than the same argument in second position — measured: ``'"say \\"hi\\""'``
    splits into two tokens, while ``'prog "say \\"hi\\""'`` splits into
    ``['prog', 'say "hi"']``. The launcher always starts with the interpreter
    path, so this holds by construction.
    """
    parts: list[str] = []
    for arg in argv:
        if arg and not any(c in arg for c in ' \t"'):
            parts.append(arg)
            continue
        # Backslashes are literal except immediately before a quote, where they
        # are doubled and the quote escaped. Doubling every backslash corrupts
        # ordinary paths — verified by round-tripping through
        # CommandLineToArgvW, which rejected the first version of this.
        quoted = ['"']
        backslashes = 0
        for char in arg:
            if char == "\\":
                backslashes += 1
                continue
            if char == '"':
                quoted.append("\\" * (backslashes * 2 + 1))
                quoted.append('"')
            else:
                quoted.append("\\" * backslashes)
                quoted.append(char)
            backslashes = 0
        # Backslashes at the end would escape the closing quote.
        quoted.append("\\" * (backslashes * 2))
        quoted.append('"')
        parts.append("".join(quoted))
    return " ".join(parts)


def _environment_block(env: dict[str, str]) -> ctypes.Array:
    """Sorted ``K=V\\0`` block with a trailing NUL, as CreateProcessW expects."""
    items = sorted(f"{key}={value}" for key, value in env.items())
    return ctypes.create_unicode_buffer("\0".join(items) + "\0\0")


# --- token queries (used by the in-container self-check) --------------------
def is_app_container(handle: int | None = None) -> bool:
    """True if the current (or given) process token belongs to a container."""
    token = wintypes.HANDLE()
    process = wintypes.HANDLE(handle) if handle else _k32.GetCurrentProcess()
    if not _adv.OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
        raise _err()
    try:
        value = wintypes.DWORD()
        needed = wintypes.DWORD()
        if not _adv.GetTokenInformation(
            token,
            TokenIsAppContainer,
            ctypes.byref(value),
            ctypes.sizeof(value),
            ctypes.byref(needed),
        ):
            raise _err()
        return bool(value.value)
    finally:
        _k32.CloseHandle(token)


def app_container_sid(handle: int | None = None) -> str | None:
    """SID string of the current process's container, or None outside one."""
    token = wintypes.HANDLE()
    process = wintypes.HANDLE(handle) if handle else _k32.GetCurrentProcess()
    if not _adv.OpenProcessToken(process, TOKEN_QUERY, ctypes.byref(token)):
        raise _err()
    try:
        # TokenAppContainerSid copies the SID itself into the buffer, so it must
        # be sized for the structure, not for a pointer.
        buffer = ctypes.create_string_buffer(256)
        needed = wintypes.DWORD()
        if not _adv.GetTokenInformation(
            token,
            TokenAppContainerSid,
            buffer,
            ctypes.sizeof(buffer),
            ctypes.byref(needed),
        ):
            return None
        sid = ctypes.c_void_p.from_buffer(buffer).value
        if not sid:
            return None
        return sid_string(sid)
    finally:
        _k32.CloseHandle(token)
