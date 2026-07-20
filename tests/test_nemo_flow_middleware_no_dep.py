# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for NeMo Flow middleware when nemo_flow is NOT installed.

These tests verify that install_nemo_flow() and nemo_flow_scope() raise ImportError
with helpful install instructions when nemo_flow is not available. They run
regardless of whether nemo_flow is installed by monkeypatching the flag.
"""

from unittest.mock import MagicMock

import pytest

import nooa.nemo_flow_middleware as nm


@pytest.fixture()
def _no_nemo_flow(monkeypatch):
    """Simulate nemo_flow not being installed."""
    monkeypatch.setattr(nm, "_HAS_NEMO_FLOW", False)


class TestImportErrorWhenMissing:
    """Verify ImportError is raised when nemo_flow is not available."""

    @pytest.mark.usefixtures("_no_nemo_flow")
    def test_install_nemo_flow_raises_import_error(self):
        """install_nemo_flow() raises ImportError when nemo_flow is not installed."""
        from nooa.runtime.event_manager import EventManager

        em = EventManager()
        with pytest.raises(ImportError, match="nemo_flow is required"):
            nm.install_nemo_flow(em)

    @pytest.mark.usefixtures("_no_nemo_flow")
    @pytest.mark.asyncio
    async def test_nemo_flow_scope_raises_import_error(self):
        """nemo_flow_scope() raises ImportError when nemo_flow is not installed."""
        agent = MagicMock()
        agent.event_manager = MagicMock()
        with pytest.raises(ImportError, match="nemo_flow is required"):
            async with nm.nemo_flow_scope(agent, "test-scope"):
                pass
