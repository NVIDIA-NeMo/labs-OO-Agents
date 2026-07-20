# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for probe_otlp_endpoint."""

import urllib.error
from unittest.mock import MagicMock, patch

import pytest

from nooa.tracing import probe_otlp_endpoint


class TestProbeOtlpEndpoint:
    def test_returns_true_when_server_responds_200(self):
        mock_response = MagicMock()
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock_response):
            assert probe_otlp_endpoint("http://localhost:5001/v1/traces") is True

    def test_returns_true_on_http_error(self):
        # Server is up but returns 4xx/5xx — still reachable
        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.HTTPError(None, 404, "Not Found", {}, None),
        ):
            assert probe_otlp_endpoint("http://localhost:5001/v1/traces") is True

    def test_returns_false_on_connection_refused(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=ConnectionRefusedError(),
        ):
            assert probe_otlp_endpoint("http://localhost:5001/v1/traces") is False

    def test_returns_false_on_timeout(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=TimeoutError(),
        ):
            assert probe_otlp_endpoint("http://localhost:5001/v1/traces") is False

    def test_returns_false_on_socket_error(self):
        with patch(
            "urllib.request.urlopen",
            side_effect=OSError("Network unreachable"),
        ):
            assert probe_otlp_endpoint("http://localhost:5001/v1/traces") is False

    def test_probes_health_endpoint_not_traces(self):
        """Must GET /api/eval/health, not POST to /v1/traces (avoids phantom sessions)."""
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.get_full_url()
            captured["method"] = req.method
            mock = MagicMock()
            mock.__enter__ = lambda s: s
            mock.__exit__ = MagicMock(return_value=False)
            return mock

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            probe_otlp_endpoint("http://localhost:5001/v1/traces")

        assert captured["url"] == "http://localhost:5001/api/eval/health"
        assert captured["method"] == "GET"

    @pytest.mark.parametrize(
        "endpoint, expected_health_url",
        [
            ("http://localhost:5001/v1/traces", "http://localhost:5001/api/eval/health"),
            ("http://localhost:5001/v1", "http://localhost:5001/api/eval/health"),
            ("http://localhost:5001", "http://localhost:5001/api/eval/health"),
            ("http://myhost:9000/v1/traces", "http://myhost:9000/api/eval/health"),
        ],
    )
    def test_strips_otlp_path_suffix(self, endpoint, expected_health_url):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["url"] = req.get_full_url()
            mock = MagicMock()
            mock.__enter__ = lambda s: s
            mock.__exit__ = MagicMock(return_value=False)
            return mock

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            probe_otlp_endpoint(endpoint)

        assert captured["url"] == expected_health_url

    def test_custom_timeout_is_passed(self):
        captured = {}

        def fake_urlopen(req, timeout):
            captured["timeout"] = timeout
            mock = MagicMock()
            mock.__enter__ = lambda s: s
            mock.__exit__ = MagicMock(return_value=False)
            return mock

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            probe_otlp_endpoint("http://localhost:5001/v1/traces", timeout=2.0)

        assert captured["timeout"] == 2.0
