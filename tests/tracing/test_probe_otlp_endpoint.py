# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for probe_otlp_endpoint."""

import io
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


def _https_only_400():
    return urllib.error.HTTPError(
        None,
        400,
        "Bad Request",
        {},
        io.BytesIO(b"Client sent an HTTP request to an HTTPS server.\n"),
    )


class TestHttpsOnlyServers:
    def test_a_plain_http_request_to_an_https_server_is_not_reachable(self):
        """Every export would fail the same way, so the http URL is not usable."""
        with patch("urllib.request.urlopen", side_effect=_https_only_400()):
            assert probe_otlp_endpoint("http://viewer:5443/v1/traces") is False

    def test_other_400s_still_count_as_reachable(self):
        error = urllib.error.HTTPError(None, 400, "Bad Request", {}, io.BytesIO(b"bad query"))
        with patch("urllib.request.urlopen", side_effect=error):
            assert probe_otlp_endpoint("http://viewer:5443/v1/traces") is True

    def test_resolve_upgrades_to_https_when_only_that_answers(self, caplog):
        from nooa.tracing import resolve_otlp_endpoint

        def fake_urlopen(req, timeout):
            if req.get_full_url().startswith("http://"):
                raise _https_only_400()
            mock = MagicMock()
            mock.__enter__ = lambda s: s
            mock.__exit__ = MagicMock(return_value=False)
            return mock

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            with caplog.at_level("WARNING", logger="nooa.tracing"):
                assert (
                    resolve_otlp_endpoint("http://viewer:5443/v1/traces")
                    == "https://viewer:5443/v1/traces"
                )
        # The upgrade is not silent: one line names both endpoints.
        [record] = [r for r in caplog.records if r.name == "nooa.tracing"]
        assert "http://viewer:5443/v1/traces" in record.getMessage()
        assert "https://viewer:5443/v1/traces" in record.getMessage()

    def test_resolve_keeps_a_working_http_endpoint(self):
        from nooa.tracing import resolve_otlp_endpoint

        mock = MagicMock()
        mock.__enter__ = lambda s: s
        mock.__exit__ = MagicMock(return_value=False)
        with patch("urllib.request.urlopen", return_value=mock):
            assert resolve_otlp_endpoint("http://viewer:5001/v1/traces") == (
                "http://viewer:5001/v1/traces"
            )

    def test_resolve_is_none_when_nothing_answers(self):
        from nooa.tracing import resolve_otlp_endpoint

        with patch("urllib.request.urlopen", side_effect=ConnectionRefusedError()):
            assert resolve_otlp_endpoint("http://viewer:5001/v1/traces") is None


class TestHttpsOnlyDetectionIsNarrow:
    def test_a_400_that_merely_mentions_https_is_still_reachable(self):
        body = b'{"error": "invalid field endpoint; see https://docs.example/otlp"}'
        error = urllib.error.HTTPError(None, 400, "Bad Request", {}, io.BytesIO(body))
        with patch("urllib.request.urlopen", side_effect=error):
            assert probe_otlp_endpoint("http://collector:4318/v1/traces") is True

    def test_nginx_plain_http_to_https_port_is_recognised(self):
        body = b"<html><center>The plain HTTP request was sent to HTTPS port</center></html>"
        error = urllib.error.HTTPError(None, 400, "Bad Request", {}, io.BytesIO(body))
        with patch("urllib.request.urlopen", side_effect=error):
            assert probe_otlp_endpoint("http://viewer:443/v1/traces") is False

    def test_resolve_probes_once_when_nothing_is_listening(self):
        from nooa.tracing import resolve_otlp_endpoint

        calls: list[str] = []

        def refused(req, timeout):
            calls.append(req.get_full_url())
            raise urllib.error.URLError(ConnectionRefusedError())

        with patch("urllib.request.urlopen", side_effect=refused):
            assert resolve_otlp_endpoint("http://localhost:5001/v1/traces") is None
        assert len(calls) == 1
