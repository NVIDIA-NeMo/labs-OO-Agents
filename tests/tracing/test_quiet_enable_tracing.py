# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for enable_tracing(quiet=True).

Hosts that own the terminal (the TUI paints before it bootstraps, and
captures stray stdout/stderr into its scrollback) must be able to configure
tracing without a raw print landing in their transcript.
"""

import tempfile

import pytest


class TestQuietEnableTracing:
    """enable_tracing(quiet=True) writes nothing to stdout/stderr."""

    def test_explicit_exporters_print_target_by_default(self, capsys):
        from nooa.tracing import enable_tracing, exporters

        with tempfile.TemporaryDirectory() as tmpdir:
            enable_tracing(exporters=[exporters.jsonl(tmpdir)])

        assert "OTel tracing enabled" in capsys.readouterr().out

    def test_explicit_exporters_are_silent_when_quiet(self, capsys):
        from nooa.tracing import enable_tracing, exporters

        with tempfile.TemporaryDirectory() as tmpdir:
            enable_tracing(exporters=[exporters.jsonl(tmpdir)], quiet=True)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_exporter_replacement_is_silent_when_quiet(self, capsys):
        """The already-enabled path prints its own target report too."""
        from nooa.tracing import enable_tracing, exporters

        with (
            tempfile.TemporaryDirectory() as dir1,
            tempfile.TemporaryDirectory() as dir2,
        ):
            enable_tracing(exporters=[exporters.jsonl(dir1)], quiet=True)
            enable_tracing(exporters=[exporters.jsonl(dir2)], quiet=True)

        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    def test_quiet_still_enables_tracing(self):
        """Silencing the report must not silence the provider."""
        import nooa.tracing as module
        from nooa.tracing import enable_tracing, exporters

        with tempfile.TemporaryDirectory() as tmpdir:
            enable_tracing(exporters=[exporters.jsonl(tmpdir)], quiet=True)
            assert module._enabled is True
            assert module._provider is not None

    @pytest.mark.parametrize("quiet, expected", [(False, True), (True, False)])
    def test_unreachable_explicit_endpoint_warning_honours_quiet(
        self, monkeypatch, capsys, quiet, expected
    ):
        """An opted-in OTLP_ENDPOINT that fails to probe warns unless quiet."""
        import nooa.tracing as module

        monkeypatch.setenv("OTLP_ENDPOINT", "http://127.0.0.1:1/v1/traces")
        monkeypatch.setattr(module, "probe_otlp_endpoint", lambda endpoint: False)

        assert module._default_exporters(quiet=quiet) is None

        assert ("is not reachable" in capsys.readouterr().err) is expected
