# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Only actual file mentions become workspace links."""

import pytest
from nooa_coder.coding.mentions import expand_mentions


@pytest.mark.parametrize("token", ["@", "@?", "@.", "@..", "@/", "@./", "@!"])
def test_punctuation_does_not_link_workspace_root(token, tmp_path):
    text = f"look {token} here"
    assert expand_mentions(text, base_dir=tmp_path) == text


def test_mentions_preserve_real_punctuation_filenames_and_sentence_suffix(tmp_path):
    (tmp_path / "report.md").write_text("report")
    (tmp_path / "why?").write_text("answer")
    assert expand_mentions("@report.md.", base_dir=tmp_path) == (
        f"[report.md](<{tmp_path / 'report.md'}>)."
    )
    assert expand_mentions("@why?", base_dir=tmp_path) == f"[why?](<{tmp_path / 'why?'}>)"
