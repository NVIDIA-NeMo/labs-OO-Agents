# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pandas doc() adapter gives a concise, construction-focused DataFrame/Series view."""

from __future__ import annotations

import pytest

pd = pytest.importorskip("pandas")

from nooa.agentdoc import doc  # noqa: E402
from nooa.agentdoc.adapters import register_all  # noqa: E402


@pytest.fixture(autouse=True)
def _register():
    # Reload to re-run the @spec.define_doc decorators even if a sibling agentdoc test
    # cleared the registry (register_all() alone is a no-op once the module is imported).
    import importlib

    import nooa.agentdoc.adapters.pandas as _pandas_adapter

    importlib.reload(_pandas_adapter)


class TestPandasAdapter:
    def test_register_all_includes_pandas(self):
        assert "pandas" in register_all()

    def test_dataframe_doc_is_concise_and_shows_construction(self):
        out = doc(pd.DataFrame)
        # construction guidance the CodeAct fallback relies on
        assert "pd.DataFrame({" in out
        # concise: the curated view is far shorter than pandas' ~50-line constructor docstring
        assert len(out.splitlines()) < 25

    def test_series_doc_shows_construction(self):
        out = doc(pd.Series)
        assert "pd.Series(" in out
