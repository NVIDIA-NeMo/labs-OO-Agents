# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nooa.paths import get_user_dir


def test_xdg_user_dir_uses_nooa_name(tmp_path, monkeypatch):
    monkeypatch.delenv("NEMO_OO_USER_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    assert get_user_dir() == tmp_path / "nooa"
