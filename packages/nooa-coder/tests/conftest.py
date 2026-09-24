# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared fixtures for the nooa-coder tests."""

import pytest


@pytest.fixture(autouse=True)
def _user_dir(tmp_path, monkeypatch):
    """Point the user-level NOOA directory (and so the default sessions dir) at tmp_path."""
    user_dir = tmp_path / "user"
    monkeypatch.setenv("NEMO_OO_USER_DIR", str(user_dir))
    return user_dir


@pytest.fixture
def sessions_dir(tmp_path):
    return tmp_path / "sessions"
