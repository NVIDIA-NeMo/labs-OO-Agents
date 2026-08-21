# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the WebPublisher tool."""

from unittest.mock import MagicMock, patch

import pytest

from nooa.tools.web_publisher import RichOutput, WebPublisher


class TestWebPublisher:
    @pytest.fixture
    def publisher(self):
        return WebPublisher()

    @pytest.fixture
    def mock_event_manager(self):
        return MagicMock()

    def test_attach_sets_event_manager(self, publisher, mock_event_manager):
        agent = MagicMock()
        agent.event_manager = mock_event_manager

        publisher.attach(agent)

        assert publisher._event_manager is mock_event_manager

    def test_post_adds_rich_output_to_event_manager(self, publisher, mock_event_manager):
        publisher._event_manager = mock_event_manager

        payload = {"kind": "markdown", "text": "hello"}
        publisher._post(payload)

        mock_event_manager.add.assert_called_once()
        event = mock_event_manager.add.call_args[0][0]
        assert isinstance(event, RichOutput)
        assert event.payload == payload

    def test_post_skips_clear_in_event_manager(self, publisher, mock_event_manager):
        publisher._event_manager = mock_event_manager

        payload = {"kind": "clear"}
        publisher._post(payload)

        mock_event_manager.add.assert_not_called()

    @patch("nooa.tools.web_publisher.os.environ.get", return_value="http://localhost:8080")
    def test_post_sends_httpx_request_if_url_set(self, mock_getenv, publisher):
        """When NEMO_OO_RICH_URL is set, _post should call httpx.post."""
        import httpx

        payload = {"kind": "markdown", "text": "hello"}
        with patch.object(httpx, "post") as mock_httpx_post:
            publisher._post(payload)
            mock_httpx_post.assert_called_once_with("http://localhost:8080", json=payload, timeout=5.0)

    @patch("nooa.tools.web_publisher.os.environ.get", return_value=None)
    def test_post_skips_httpx_if_no_url(self, mock_getenv, publisher):
        """When NEMO_OO_RICH_URL is not set, _post should not call httpx.post."""
        import httpx

        payload = {"kind": "markdown", "text": "hello"}
        with patch.object(httpx, "post") as mock_httpx_post:
            publisher._post(payload)
            mock_httpx_post.assert_not_called()

    def test_html(self, publisher):
        with patch.object(publisher, "_post") as mock_post:
            publisher.html("<p>Hi</p>", title="Greeting")
            mock_post.assert_called_once_with({"kind": "html", "html": "<p>Hi</p>", "title": "Greeting"})

    def test_markdown(self, publisher):
        with patch.object(publisher, "_post") as mock_post:
            publisher.markdown("**Bold**", title="Info")
            mock_post.assert_called_once_with({"kind": "markdown", "text": "**Bold**", "title": "Info"})

    def test_image(self, publisher):
        with patch.object(publisher, "_post") as mock_post:
            publisher.image("data:image/png;base64,123", alt="alt-text", title="Image")
            mock_post.assert_called_once_with({
                "kind": "image",
                "src": "data:image/png;base64,123",
                "alt": "alt-text",
                "title": "Image"
            })

    def test_json(self, publisher):
        with patch.object(publisher, "_post") as mock_post:
            publisher.json({"key": "value"}, title="Data")
            mock_post.assert_called_once_with({"kind": "json", "data": {"key": "value"}, "title": "Data"})

    def test_clear(self, publisher):
        with patch.object(publisher, "_post") as mock_post:
            publisher.clear()
            mock_post.assert_called_once_with({"kind": "clear"})

    def test_plot(self, publisher):
        with patch.object(publisher, "_post") as mock_post:
            mock_fig = MagicMock()
            mock_fig.to_json.return_value = '{"data": []}'

            publisher.plot(mock_fig, title="Plot")
            mock_post.assert_called_once_with({
                "kind": "plotly",
                "figure_json": '{"data": []}',
                "title": "Plot"
            })

    def test_plot_handles_exception(self, publisher):
        with patch.object(publisher, "_post") as mock_post:
            mock_fig = MagicMock()
            mock_fig.to_json.side_effect = Exception("serialization error")

            publisher.plot(mock_fig)
            mock_post.assert_not_called()
