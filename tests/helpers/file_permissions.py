# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check actual access permissions, including Windows DACLs."""

import os
import re


def assert_owner_only(path):
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
        return

    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_security = advapi32.GetFileSecurityW
    get_security.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    get_security.restype = wintypes.BOOL
    needed = wintypes.DWORD()
    get_security(str(path), 4, None, 0, ctypes.byref(needed))
    assert needed.value, ctypes.WinError(ctypes.get_last_error())
    descriptor = ctypes.create_string_buffer(needed.value)
    if not get_security(str(path), 4, descriptor, needed.value, ctypes.byref(needed)):
        raise ctypes.WinError(ctypes.get_last_error())

    convert = advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW
    convert.argtypes = [
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(wintypes.DWORD),
    ]
    convert.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [wintypes.LPVOID]
    kernel32.LocalFree.restype = wintypes.LPVOID
    text = wintypes.LPWSTR()
    if not convert(descriptor, 1, 4, ctypes.byref(text), None):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        sddl = text.value
        assert sddl is not None
        assert "P" in sddl.split("(", 1)[0], sddl
        assert re.findall(r"\([^)]*\)", sddl) == ["(A;;FA;;;OW)"], sddl
    finally:
        kernel32.LocalFree(text)
