# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owner-only permissions for newly created credential files."""

import os


def make_owner_only(path: str | os.PathLike[str]) -> None:
    """Restrict a file to its owner, using a protected DACL on Windows.

    Apply this to the staging file before writing a credential. Failures are
    propagated so an unprotected file is never published in its place.
    """
    if os.name != "nt":
        os.chmod(path, 0o600)
        return

    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    convert = advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        ctypes.POINTER(wintypes.DWORD),
    ]
    convert.restype = wintypes.BOOL
    set_security = advapi32.SetFileSecurityW
    set_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID]
    set_security.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.LPVOID]
    kernel32.LocalFree.restype = wintypes.LPVOID

    descriptor = wintypes.LPVOID()
    # Protected DACL, one allow ACE: full file access for OWNER RIGHTS.
    # OW identifies the file's owner without depending on account-name locale.
    if not convert("D:P(A;;FA;;;OW)", 1, ctypes.byref(descriptor), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        # DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION
        if not set_security(os.fspath(path), 0x00000004 | 0x80000000, descriptor):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel32.LocalFree(descriptor)
