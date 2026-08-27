# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``_enable_tracing`` must ask core tracing to stay quiet.

``bootstrap()`` runs *after* TUIApplication installs the stray-stream
forwarder, so anything ``enable_tracing`` writes to stdout/stderr is captured
into the transcript. The library-side ``quiet`` switch only helps if the TUI
actually passes it — these tests pin the call site, on both the auto-probe and
explicit-``trace_dir`` branches.
"""

from __future__ import annotations

import pytest
from nooa_cli.tui.bootstrap import _enable_tracing
from nooa_cli.tui.config import Config


@pytest.fixture
def recorded_calls(monkeypatch):
    """Replace core ``enable_tracing`` with a recorder."""
    import nooa.tracing as tracing_module

    calls: list[dict] = []

    def fake_enable_tracing(exporters=None, **kwargs):
        calls.append({"exporters": exporters, **kwargs})

    monkeypatch.setattr(tracing_module, "enable_tracing", fake_enable_tracing)
    return calls


def test_auto_probe_branch_passes_quiet(recorded_calls):
    config = Config()
    config.tui.trace_dir = None

    enabled, _set_session = _enable_tracing(config, [])

    assert enabled is True
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["quiet"] is True


def test_explicit_trace_dir_branch_passes_quiet(recorded_calls, tmp_path):
    config = Config()
    config.tui.trace_dir = tmp_path / "traces"

    enabled, _set_session = _enable_tracing(config, [])

    assert enabled is True
    assert len(recorded_calls) == 1
    assert recorded_calls[0]["quiet"] is True
    # The explicit branch still wires up its exporters.
    assert recorded_calls[0]["exporters"]


def test_no_trace_short_circuits(recorded_calls):
    config = Config()
    config.no_trace = True

    enabled, set_session = _enable_tracing(config, [])

    assert enabled is False
    assert set_session is None
    assert recorded_calls == []
