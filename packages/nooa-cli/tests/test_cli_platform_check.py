# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `nooa` command refuses native Windows like `import nooa`, without importing nooa."""

import sys

import nooa_cli
import pytest

from nooa import _platform


def test_cli_policy_matches_library():
    assert nooa_cli._SKIP_PLATFORM_CHECK_ENV == _platform.SKIP_PLATFORM_CHECK_ENV
    assert nooa_cli._UNSUPPORTED_PLATFORM_MESSAGE == _platform.UNSUPPORTED_PLATFORM_MESSAGE


def test_cli_refuses_native_windows_with_one_line(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.delenv("NOOA_SKIP_PLATFORM_CHECK", raising=False)
    with pytest.raises(SystemExit) as info:
        nooa_cli._refuse_unsupported_platform()
    assert info.value.code == f"Error: {_platform.UNSUPPORTED_PLATFORM_MESSAGE}"


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_cli_override_matches_library(monkeypatch, value):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("NOOA_SKIP_PLATFORM_CHECK", value)
    nooa_cli._refuse_unsupported_platform()
    assert _platform.platform_check_skipped()


def test_cli_check_is_noop_on_posix(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("NOOA_SKIP_PLATFORM_CHECK", raising=False)
    nooa_cli._refuse_unsupported_platform()
