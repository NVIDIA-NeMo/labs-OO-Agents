# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WebPublisher — push rich content to the web terminal side-channel."""

import logging
import os
from typing import TYPE_CHECKING, Any

from pydantic import Field

from nooa.context_blocks import Metadata
from nooa.skill import Skill

_log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from nooa.runtime.event_manager import EventManager


class RichOutput(Metadata):  # type: ignore[misc]
    """Rich content published by the agent — stored in session history, never shown to LLM.

    Each payload is a dict with at least a ``kind`` field (``"plotly"``,
    ``"html"``, ``"image"``, ``"markdown"``, ``"json"``).  On session resume
    the PTY server scans for ``RichOutput`` events and replays them to the
    browser, reconstructing the rich content panel.
    """

    payload: dict[str, Any] = Field(default_factory=dict)


class WebPublisher(Skill):
    """Inline rich output — interactive charts, images, HTML, and formatted data rendered in the web panel.

    Available when the agent runs inside ``nooa term``.  Call methods on
    ``self.web`` to render content inline at the current position in the
    terminal scroll buffer.  All methods are fire-and-forget; if the browser
    is not connected the call is silently skipped.  Content is persisted in
    the session and replayed automatically when resuming with ``--continue``.

    All modules are pre-loaded — do NOT write import statements.
    Use np, pd, px, go, make_subplots, base64, io directly.

    Plotly figures (most common — prefer this for any chart)::

        # Scatter
        fig = px.scatter(df, x="year", y="gdp", color="country",
                         size="population", hover_name="country",
                         title="GDP over time")
        self.web.plot(fig)

        # Line with multiple traces
        fig = px.line(df, x="date", y="value", color="metric", title="Metrics")
        self.web.plot(fig)

        # Bar (grouped or stacked)
        fig = px.bar(df, x="category", y="revenue", color="region",
                     barmode="group", title="Revenue by region")
        self.web.plot(fig)

        # Histogram
        fig = px.histogram(df, x="age", nbins=30, color="group",
                           title="Age distribution")
        self.web.plot(fig)

        # Box plot
        fig = px.box(df, x="department", y="salary", color="level",
                     title="Salary distribution")
        self.web.plot(fig)

        # Heatmap / correlation matrix
        corr = df.corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns,
                                   y=corr.index, colorscale="RdBu",
                                   zmid=0))
        self.web.plot(fig, title="Correlation matrix")

        # Choropleth map
        fig = px.choropleth(df, locations="iso_alpha", color="value",
                            hover_name="country", title="World map")
        self.web.plot(fig)

        # Sunburst / treemap
        fig = px.sunburst(df, path=["continent", "country"], values="gdp")
        self.web.plot(fig, title="GDP breakdown")

        # Subplots (make_subplots is pre-loaded from plotly.subplots)
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Before", "After"))
        fig.add_trace(go.Histogram(x=before), row=1, col=1)
        fig.add_trace(go.Histogram(x=after),  row=1, col=2)
        self.web.plot(fig, title="Before vs After")

        # 3-D scatter
        fig = px.scatter_3d(df, x="x", y="y", z="z", color="label")
        self.web.plot(fig)

        # Animated chart
        fig = px.scatter(df, x="gdp", y="life_exp", color="continent",
                         size="population", animation_frame="year",
                         title="Gapminder")
        self.web.plot(fig)

    Markdown (formatted summaries, tables, code blocks)::

        self.web.markdown("## Results\\n- accuracy: **94.2 %**\\n- f1: **0.91**")

        self.web.markdown(f"```python\\n{code}\\n```")

        # Markdown table
        rows = "\\n".join(f"| {k} | {v} |" for k, v in metrics.items())
        self.web.markdown(f"| Metric | Value |\\n|--------|-------|\\n{rows}")

    HTML fragments (custom tables, styled output — no <html>/<head>/<script> tags)::

        self.web.html("<table><tr><th>Name</th><th>Score</th></tr>...</table>")

        self.web.html('<p style="color:salmon">Warning: missing values in column age</p>')

        # DataFrame as HTML table
        self.web.html(df.head(20).to_html(index=False))

    Images (data URIs or URLs — base64, io are pre-loaded)::

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight")
        src = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        self.web.image(src, title="matplotlib figure")

        # PIL image
        img_bytes = io.BytesIO()
        pil_img.save(img_bytes, format="PNG")
        src = "data:image/png;base64," + base64.b64encode(img_bytes.getvalue()).decode()
        self.web.image(src)

    JSON data (pretty-printed, useful for dicts and lists)::

        self.web.json({"accuracy": 0.94, "loss": 0.12, "epochs": 50})
        self.web.json(results_list, title="Top matches")
        self.web.json(df.describe().to_dict(), title="Summary statistics")

    Clear all inline items::

        self.web.clear()
    """

    def attach(self, agent: Any) -> None:
        """Wire event_manager from the agent so RichOutput events persist for session replay."""
        super().attach(agent)
        if self._event_manager is None and hasattr(agent, "event_manager"):
            self._event_manager = agent.event_manager

    def __init__(self, event_manager: "EventManager | None" = None) -> None:
        super().__init__()
        self._event_manager = event_manager

    def _post(self, payload: dict[str, Any]) -> None:
        # Persist to session storage (skip "clear" — nothing to replay)
        if self._event_manager is not None and payload.get("kind") != "clear":
            try:
                self._event_manager.add(RichOutput(payload=payload))
            except Exception as exc:
                _log.debug("WebPublisher: could not store RichOutput event: %s", exc)

        url = os.environ.get("NEMO_OO_RICH_URL")
        if not url:
            return
        try:
            import httpx

            httpx.post(url, json=payload, timeout=5.0)
        except ImportError:
            _log.warning("WebPublisher requires httpx. Install with: uv add httpx")
        except Exception as exc:
            _log.debug("WebPublisher POST failed: %s", exc)

    def plot(self, fig: Any, title: str = "") -> None:
        """Push a Plotly figure inline into the terminal.

        Args:
            fig: A ``plotly.graph_objects.Figure`` instance.
            title: Optional panel header text.
        """
        try:
            figure_json = fig.to_json()
        except Exception as exc:
            _log.debug("WebPublisher.plot: could not serialise figure: %s", exc)
            return
        self._post({"kind": "plotly", "figure_json": figure_json, "title": title})

    def html(self, html: str, title: str = "") -> None:
        """Push an HTML fragment inline into the terminal.

        Fragment HTML is rendered in a Shadow DOM.  Full pages (with ``<html>``
        or ``<!doctype>``) are rendered in a sandboxed iframe.  Do not include
        ``<script>`` tags in fragments — use ``plot()`` for interactive content.

        Args:
            html: HTML string (fragment or full page).
            title: Optional panel header text.
        """
        self._post({"kind": "html", "html": html, "title": title})

    def image(self, src: str, alt: str = "", title: str = "") -> None:
        """Push an image inline into the terminal.

        Args:
            src: A ``data:`` URI or any URL accessible from the browser.
            alt: Alt text for accessibility.
            title: Optional panel header text.
        """
        self._post({"kind": "image", "src": src, "alt": alt, "title": title})

    def markdown(self, text: str, title: str = "") -> None:
        """Push rendered Markdown inline into the terminal.

        Args:
            text: Markdown-formatted string.
            title: Optional panel header text.
        """
        self._post({"kind": "markdown", "text": text, "title": title})

    def json(self, data: Any, title: str = "") -> None:
        """Push a JSON-serialisable object inline as pretty-printed JSON.

        Args:
            data: Any JSON-serialisable value.
            title: Optional panel header text.
        """
        self._post({"kind": "json", "data": data, "title": title})

    def clear(self) -> None:
        """Remove all inline items from the terminal output."""
        self._post({"kind": "clear"})
