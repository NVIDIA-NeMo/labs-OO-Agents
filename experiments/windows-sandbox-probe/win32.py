# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal Win32 bindings for the AppContainer + Job Object feasibility probe.

Deliberately dependency-free (ctypes only) so the probe can run under a bare
interpreter copy, which is the situation the real sandbox worker will be in.

Nothing here is production code. The real backend will need the same calls but
with error handling, handle lifetime discipline, and a rollback journal.
"""

from __future__ import annotations

import ctypes
import msvcrt
import os
from ctypes import wintypes

# --- libraries --------------------------------------------------------------
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_adv = ctypes.WinDLL("advapi32", use_last_error=True)
_user = ctypes.WinDLL("userenv", use_last_error=True)

INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

# --- flags ------------------------------------------------------------------
EXTENDED_STARTUPINFO_PRESENT = 0x00080000
CREATE_SUSPENDED = 0x00000004
CREATE_NO_WINDOW = 0x08000000
CREATE_UNICODE_ENVIRONMENT = 0x00000400
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009
PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002

STARTF_USESTDHANDLES = 0x00000100

TOKEN_DUPLICATE = 0x0002
TOKEN_QUERY = 0x0008
TOKEN_ASSIGN_PRIMARY = 0x0001
TOKEN_ADJUST_DEFAULT = 0x0080

_DACL_SECURITY_INFORMATION = 0x00000004
_SE_FILE_OBJECT = 1

_JOB_OBJECT_BASIC_LIMIT_INFORMATION = 2
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_BASIC_PROCESS_ID_LIST = 3
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
_JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
_JOB_OBJECT_LIMIT_JOB_TIME = 0x00000004

_JOB_OBJECT_MSG_END_OF_JOB_TIME = 1
_JOB_OBJECT_MSG_END_OF_PROCESS_TIME = 2
_JOB_OBJECT_MSG_ACTIVE_PROCESS_LIMIT = 3
_JOB_OBJECT_MSG_ABNORMAL_EXIT_PROCESS = 8
_JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT = 9
_JOB_OBJECT_MSG_JOB_MEMORY_LIMIT = 10

# ACE
_ACCESS_ALLOWED_ACE_TYPE = 0
_ACL_REVISION = 2
_GRANT_ACCESS = 1
_SUB_CONTAINERS_AND_OBJECTS_INHERIT = 0x3
_TRUSTEE_IS_SID = 0
_TRUSTEE_IS_UNKNOWN = 0

_SE_DACL_SECURITY_INFORMATION = 0x00000004
_OBJECT_INHERIT_ACE = 0x01
_CONTAINER_INHERIT_ACE = 0x02
_DACL_SECURITY_INFORMATION = _SE_DACL_SECURITY_INFORMATION
_DACL_PRESENT = 0x00000004

# Access rights used by the probe grants.
FILE_GENERIC_READ = 0x00120089
FILE_GENERIC_EXECUTE = 0x001200A0
FILE_GENERIC_WRITE = 0x00120116
FILE_ALL_ACCESS = 0x001F01FF


def _err() -> OSError:
    return ctypes.WinError(ctypes.get_last_error())


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


class _SecurityAttributes(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
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
    _fields_ = [(name, ctypes.c_uint64) for name in (
        "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
        "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
    )]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimit),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
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


# --- prototypes -------------------------------------------------------------
_k32.LocalFree.argtypes = [ctypes.c_void_p]
_k32.LocalFree.restype = ctypes.c_void_p
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.CloseHandle.restype = wintypes.BOOL
_k32.GetCurrentProcess.restype = wintypes.HANDLE
_k32.GetCurrentProcessId.restype = wintypes.DWORD
_k32.SetHandleInformation.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD]
_k32.SetHandleInformation.restype = wintypes.BOOL
_k32.CreatePipe.argtypes = [ctypes.POINTER(wintypes.HANDLE), ctypes.POINTER(wintypes.HANDLE),
                            ctypes.POINTER(_SecurityAttributes), wintypes.DWORD]
_k32.CreatePipe.restype = wintypes.BOOL
_k32.InitializeProcThreadAttributeList.argtypes = [ctypes.c_void_p, wintypes.DWORD,
                                                   wintypes.DWORD, ctypes.POINTER(ctypes.c_size_t)]
_k32.InitializeProcThreadAttributeList.restype = wintypes.BOOL
_k32.UpdateProcThreadAttribute.argtypes = [ctypes.c_void_p, wintypes.DWORD, ctypes.c_size_t,
                                           ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p,
                                           ctypes.c_void_p]
_k32.UpdateProcThreadAttribute.restype = wintypes.BOOL
_k32.DeleteProcThreadAttributeList.argtypes = [ctypes.c_void_p]
_k32.CreateProcessW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p,
                                ctypes.c_void_p, wintypes.BOOL, wintypes.DWORD,
                                ctypes.c_void_p, wintypes.LPCWSTR,
                                ctypes.POINTER(_StartupInfoExW),
                                ctypes.POINTER(_ProcessInformation)]
_k32.CreateProcessW.restype = wintypes.BOOL
_k32.ResumeThread.argtypes = [wintypes.HANDLE]
_k32.ResumeThread.restype = wintypes.DWORD
_k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_k32.GetExitCodeProcess.restype = wintypes.BOOL
_k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
_k32.TerminateProcess.restype = wintypes.BOOL
_k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
_k32.CreateJobObjectW.restype = wintypes.HANDLE
_k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                         wintypes.DWORD]
_k32.SetInformationJobObject.restype = wintypes.BOOL
_k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
_k32.AssignProcessToJobObject.restype = wintypes.BOOL
_k32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
_k32.TerminateJobObject.restype = wintypes.BOOL

_adv.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)]
_adv.ConvertStringSidToSidW.restype = wintypes.BOOL
_adv.SetEntriesInAclW.argtypes = [wintypes.ULONG, ctypes.POINTER(_ExplicitAccessW),
                                  ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
_adv.SetEntriesInAclW.restype = wintypes.DWORD
_adv.SetNamedSecurityInfoW.argtypes = [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                       ctypes.c_void_p]
_adv.SetNamedSecurityInfoW.restype = wintypes.DWORD
_adv.GetNamedSecurityInfoW.argtypes = [wintypes.LPWSTR, ctypes.c_int, wintypes.DWORD,
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_void_p),
                                       ctypes.POINTER(ctypes.c_void_p)]
_adv.GetNamedSecurityInfoW.restype = wintypes.DWORD
_adv.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
    ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, ctypes.POINTER(wintypes.LPWSTR),
    ctypes.POINTER(wintypes.DWORD)]
_adv.ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = wintypes.BOOL

_user.CreateAppContainerProfile.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.LPCWSTR,
                                            ctypes.c_void_p, wintypes.DWORD,
                                            ctypes.POINTER(ctypes.c_void_p)]
_user.CreateAppContainerProfile.restype = ctypes.c_long
_user.DeriveAppContainerSidFromAppContainerName.argtypes = [wintypes.LPCWSTR,
                                                            ctypes.POINTER(ctypes.c_void_p)]
_user.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long
_user.DeleteAppContainerProfile.argtypes = [wintypes.LPCWSTR]
_user.DeleteAppContainerProfile.restype = ctypes.c_long


# --- AppContainer -----------------------------------------------------------
def create_profile(name: str) -> None:
    """Create (or reuse) an AppContainer profile by name."""
    sid = ctypes.c_void_p()
    hr = _user.CreateAppContainerProfile(name, name, name, None, 0, ctypes.byref(sid))
    # 0x800700B7 == ERROR_ALREADY_EXISTS; reuse is fine.
    if hr < 0 and (hr & 0xFFFFFFFF) != 0x800700B7:
        raise OSError(f"CreateAppContainerProfile failed: 0x{hr & 0xFFFFFFFF:08X}")
    if sid:
        _k32.LocalFree(sid)


def delete_profile(name: str) -> None:
    hr = _user.DeleteAppContainerProfile(name)
    if hr < 0 and (hr & 0xFFFFFFFF) != 0x80070002:  # not found
        raise OSError(f"DeleteAppContainerProfile failed: 0x{hr & 0xFFFFFFFF:08X}")


def container_sid(name: str) -> ctypes.c_void_p:
    sid = ctypes.c_void_p()
    hr = _user.DeriveAppContainerSidFromAppContainerName(name, ctypes.byref(sid))
    if hr < 0:
        raise OSError(f"DeriveAppContainerSidFromAppContainerName failed: 0x{hr & 0xFFFFFFFF:08X}")
    return sid


# Well-known capability SIDs. These are fixed values, not derived from the
# capability name — the derivation is a package-model registry lookup, so the
# probe hardcodes the documented ones instead of guessing a hash.
WELL_KNOWN_CAPABILITIES: dict[str, str] = {
    "internetClient": "S-1-15-3-1",
    "internetClientServer": "S-1-15-3-2",
    "privateNetworkClientServer": "S-1-15-3-3",
    "documentsLibrary": "S-1-15-3-7",
    "picturesLibrary": "S-1-15-3-8",
}


def capability_sid(name: str) -> ctypes.c_void_p:
    """Capability SIDs come from a fixed well-known set."""
    text = WELL_KNOWN_CAPABILITIES.get(name)
    if text is None:
        raise ValueError(
            f"unknown capability {name!r}; known: {sorted(WELL_KNOWN_CAPABILITIES)}"
        )
    sid = ctypes.c_void_p()
    if not _adv.ConvertStringSidToSidW(text, ctypes.byref(sid)):
        raise _err()
    return sid


# --- ACL grant (with backup for the reversibility test) ---------------------
def _sd_sddl(sd: ctypes.c_void_p) -> str:
    out = wintypes.LPWSTR()
    length = wintypes.DWORD()
    if not _adv.ConvertSecurityDescriptorToStringSecurityDescriptorW(
        sd, 1, _DACL_SECURITY_INFORMATION, ctypes.byref(out), ctypes.byref(length)
    ):
        raise _err()
    try:
        return out.value
    finally:
        _k32.LocalFree(out)


def read_dacl_sddl(path: str) -> str:
    """Current DACL as SDDL text, for the before/after reversibility comparison."""
    sd = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    rc = _adv.GetNamedSecurityInfoW(
        path, _SE_FILE_OBJECT, _DACL_SECURITY_INFORMATION,
        None, None, ctypes.byref(dacl), None, ctypes.byref(sd),
    )
    if rc != 0:
        raise OSError(rc, f"GetNamedSecurityInfoW failed for {path}")
    try:
        return _sd_sddl(sd)
    finally:
        _k32.LocalFree(sd)


_adv.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
_adv.ConvertSidToStringSidW.restype = wintypes.BOOL
_REVOKE_ACCESS = 4
_NO_INHERITANCE = 0


def sid_string(sid: ctypes.c_void_p) -> str:
    out = wintypes.LPWSTR()
    if not _adv.ConvertSidToStringSidW(sid, ctypes.byref(out)):
        raise _err()
    try:
        return out.value
    finally:
        _k32.LocalFree(out)


def grant(path: str, sid: ctypes.c_void_p, access: int) -> None:
    """Add an inheritable allow-ACE for ``sid`` on ``path`` (no explicit deny)."""
    _set_entry(path, sid, access, _GRANT_ACCESS, _SUB_CONTAINERS_AND_OBJECTS_INHERIT)


def revoke(path: str, sid: ctypes.c_void_p) -> None:
    """Remove every explicit ACE for ``sid`` on ``path``; inheritance re-propagates."""
    _set_entry(path, sid, 0, _REVOKE_ACCESS, _NO_INHERITANCE)


def _set_entry(path: str, sid: ctypes.c_void_p, access: int, mode: int, inheritance: int) -> None:
    sd = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    rc = _adv.GetNamedSecurityInfoW(
        path, _SE_FILE_OBJECT, _DACL_SECURITY_INFORMATION,
        None, None, ctypes.byref(dacl), None, ctypes.byref(sd),
    )
    if rc != 0:
        raise OSError(rc, f"GetNamedSecurityInfoW failed for {path}")
    try:
        ea = _ExplicitAccessW()
        ea.grfAccessPermissions = access
        ea.grfAccessMode = mode
        ea.grfInheritance = inheritance
        ea.Trustee.TrusteeForm = _TRUSTEE_IS_SID
        ea.Trustee.TrusteeType = _TRUSTEE_IS_UNKNOWN
        ea.Trustee.ptstrName = sid
        new_dacl = ctypes.c_void_p()
        rc = _adv.SetEntriesInAclW(1, ctypes.byref(ea), dacl, ctypes.byref(new_dacl))
        if rc != 0:
            raise OSError(rc, f"SetEntriesInAclW failed for {path}")
        try:
            rc = _adv.SetNamedSecurityInfoW(
                path, _SE_FILE_OBJECT, _DACL_SECURITY_INFORMATION,
                None, None, new_dacl, None,
            )
            if rc != 0:
                raise OSError(rc, f"SetNamedSecurityInfoW failed for {path}")
        finally:
            _k32.LocalFree(new_dacl)
    finally:
        _k32.LocalFree(sd)


def set_dacl_sddl(path: str, sddl: str) -> None:
    """Restore a DACL from SDDL text (the rollback path)."""
    sd = ctypes.c_void_p()
    if not _adv.ConvertStringSecurityDescriptorToSecurityDescriptorW(sddl, 1,
                                                                   ctypes.byref(sd), None):
        raise _err()
    try:
        dacl = ctypes.c_void_p()
        present = wintypes.BOOL()
        defaulted = wintypes.BOOL()
        if not _adv.GetSecurityDescriptorDacl(sd, ctypes.byref(present),
                                              ctypes.byref(dacl), ctypes.byref(defaulted)):
            raise _err()
        rc = _adv.SetNamedSecurityInfoW(
            path, _SE_FILE_OBJECT, _DACL_SECURITY_INFORMATION,
            None, None, dacl if present.value else None, None,
        )
        if rc != 0:
            raise OSError(rc, f"SetNamedSecurityInfoW(restore) failed for {path}")
    finally:
        _k32.LocalFree(sd)


_adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
    wintypes.LPCWSTR, wintypes.DWORD, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.ULONG)]
_adv.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
_adv.GetSecurityDescriptorDacl.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.BOOL),
                                           ctypes.POINTER(ctypes.c_void_p),
                                           ctypes.POINTER(wintypes.BOOL)]
_adv.GetSecurityDescriptorDacl.restype = wintypes.BOOL


# --- pipes ------------------------------------------------------------------
def make_pipe(read_inheritable: bool) -> tuple[int, int]:
    """Return ``(parent_handle, child_handle)`` for an anonymous pipe.

    ``read_inheritable`` picks which end the child gets; the other end is marked
    non-inheritable so it cannot leak into the container process.

    ``wintypes.HANDLE`` is already an integer on 64-bit Windows — do not
    ``ctypes.cast`` it, which would reinterpret the numeric value as an address.
    """
    sa = _SecurityAttributes()
    sa.nLength = ctypes.sizeof(sa)
    sa.lpSecurityDescriptor = None
    sa.bInheritHandle = True
    r = wintypes.HANDLE()
    w = wintypes.HANDLE()
    if not _k32.CreatePipe(ctypes.byref(r), ctypes.byref(w), ctypes.byref(sa), 0):
        raise _err()
    # Clear the inherit flag on the end the child must NOT receive.
    keep = r if read_inheritable else w
    drop = w if read_inheritable else r
    _k32.SetHandleInformation(drop, 0x00000001, 0)  # HANDLE_FLAG_INHERIT -> 0
    if read_inheritable:
        return (w.value, r.value)
    return (r.value, w.value)


def to_socket(handle: int) -> "object":
    """Wrap a pipe handle as a socket-like object usable by mp Connection."""
    import socket as _socket
    return _socket.socket(fileno=msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY))


# --- process creation -------------------------------------------------------
class Child:
    """A started container process (suspended until ``resume``)."""

    def __init__(self, pi: _ProcessInformation, cmdline: str):
        self.pi = pi
        self.pid = pi.dwProcessId
        self.cmdline = cmdline

    def resume(self) -> None:
        _k32.ResumeThread(self.pi.hThread)

    def wait(self, timeout_ms: int = 60000) -> int:
        _k32.WaitForSingleObject(self.pi.hProcess, timeout_ms)
        code = wintypes.DWORD()
        if not _k32.GetExitCodeProcess(self.pi.hProcess, ctypes.byref(code)):
            raise _err()
        return code.value

    def kill(self) -> None:
        _k32.TerminateProcess(self.pi.hProcess, 1)

    def close(self) -> None:
        for h in (self.pi.hProcess, self.pi.hThread):
            if h:
                _k32.CloseHandle(h)


def spawn_in_container(
    argv: list[str],
    *,
    container_name: str,
    capabilities: tuple[str, ...] = (),
    inherit_handles: tuple[int, ...] = (),
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> Child:
    """Create a process inside the AppContainer, suspended, with job-ready handle.

    SECURITY_CAPABILITIES is the only way to enter an AppContainer: it is a
    creation-time attribute, so it cannot be applied after the fact.
    """
    app_sid = container_sid(container_name)

    cap_sids = [capability_sid(c) for c in capabilities]
    # The SECURITY_CAPABILITIES struct and its SID array must stay alive until
    # the attribute list has been populated.
    if cap_sids:
        n = len(cap_sids)
        arr = (_SidAndAttributes * n)()
        for i, s in enumerate(cap_sids):
            arr[i].Sid = s
            arr[i].Attributes = 0x00000004  # SE_GROUP_ENABLED
        cap_sid_arr = arr
    else:
        cap_sid_arr = ctypes.POINTER(_SidAndAttributes)()
    _ = cap_sid_arr  # keep alive

    caps = _SecurityCapabilities()
    caps.AppContainerSid = app_sid
    caps.Capabilities = cap_sid_arr
    caps.CapabilityCount = len(cap_sids)
    caps.Reserved = 0

    attr_count = 1 + (1 if inherit_handles else 0)
    size = ctypes.c_size_t()
    _k32.InitializeProcThreadAttributeList(None, attr_count, 0, ctypes.byref(size))
    buffer = ctypes.create_string_buffer(size.value)
    if not _k32.InitializeProcThreadAttributeList(buffer, attr_count, 0, ctypes.byref(size)):
        raise _err()

    try:
        if not _k32.UpdateProcThreadAttribute(
            buffer, 0, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES,
            ctypes.byref(caps), ctypes.sizeof(caps), None, None,
        ):
            raise _err()

        handle_array = None
        if inherit_handles:
            n = len(inherit_handles)
            handle_array = (wintypes.HANDLE * n)(*inherit_handles)
            if not _k32.UpdateProcThreadAttribute(
                buffer, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
                handle_array, ctypes.sizeof(handle_array), None, None,
            ):
                raise _err()

        si = _StartupInfoExW()
        si.StartupInfo.cb = ctypes.sizeof(_StartupInfoExW)
        si.lpAttributeList = ctypes.cast(buffer, ctypes.c_void_p)

        pi = _ProcessInformation()
        cmd = ctypes.create_unicode_buffer(" ".join(f'"{a}"' if " " in a else a for a in argv))
        flags = EXTENDED_STARTUPINFO_PRESENT | CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT
        if env is not None:
            block = _environment_block(env)
            ok = _k32.CreateProcessW(
                None, cmd, None, None, bool(inherit_handles), flags,
                ctypes.cast(block, ctypes.c_void_p), cwd, ctypes.byref(si), ctypes.byref(pi),
            )
        else:
            ok = _k32.CreateProcessW(
                None, cmd, None, None, bool(inherit_handles), flags,
                None, cwd, ctypes.byref(si), ctypes.byref(pi),
            )
        if not ok:
            raise _err()
        _ = handle_array, cap_sids, app_sid
        return Child(pi, cmd.value)
    finally:
        _k32.DeleteProcThreadAttributeList(buffer)


def _environment_block(env: dict[str, str]) -> ctypes.Array:
    """Sorted ``K=V\\0`` block with a trailing NUL, as CreateProcessW expects."""
    items = sorted(f"{k}={v}" for k, v in env.items())
    return ctypes.create_unicode_buffer("\0".join(items) + "\0\0")


# --- Job Object -------------------------------------------------------------
class Job:
    def __init__(self, *, kill_on_close: bool = True):
        self._h = _k32.CreateJobObjectW(None, None)
        if not self._h:
            raise _err()
        if kill_on_close:
            info = _ExtendedLimit()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not _k32.SetInformationJobObject(self._h, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                                ctypes.byref(info), ctypes.sizeof(info)):
                raise _err()

    def set_limits(self, *, memory_mb: int = 0, cpu_seconds: int = 0) -> None:
        info = _ExtendedLimit()
        if cpu_seconds:
            info.BasicLimitInformation.PerJobUserTimeLimit = cpu_seconds * 10_000_000
            info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_JOB_TIME
        if memory_mb:
            info.ProcessMemoryLimit = memory_mb * 1024 * 1024
            info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_PROCESS_MEMORY
            info.BasicLimitInformation.LimitFlags |= _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not _k32.SetInformationJobObject(self._h, _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
                                            ctypes.byref(info), ctypes.sizeof(info)):
            raise _err()

    def assign(self, process_handle: int) -> None:
        if not _k32.AssignProcessToJobObject(self._h, process_handle):
            raise _err()

    def terminate(self, code: int = 1) -> None:
        _k32.TerminateJobObject(self._h, code)

    def close(self) -> None:
        if self._h:
            _k32.CloseHandle(self._h)
            self._h = None


PROCESS_RIGHTS = 0x0100 | 0x0001 | 0x1000 | 0x0400  # SET_QUOTA|TERMINATE|QUERY_LTD|QUERY_INFO
