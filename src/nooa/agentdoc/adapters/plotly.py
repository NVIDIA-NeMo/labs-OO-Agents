# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""agentdoc adapter for Plotly — curated doc() views for plotly, plotly.express, and plotly.graph_objects.

Import to register:

    import nooa.agentdoc.adapters.plotly

Without this adapter, doc(px) is ~2000 lines of generated parameter docs.
With it, doc(px) is ~60 lines covering every chart type.
"""

import plotly
import plotly.express as px
import plotly.graph_objects as go

from nooa.agentdoc import spec
from nooa.agentdoc.ext import CallableInfo, ModuleInfo

# ---------------------------------------------------------------------------
# plotly root
# ---------------------------------------------------------------------------


@spec.define_doc(plotly)
def _plotly_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="plotly",
        docstring="Plotly — interactive visualisation library.\n\n"
        "Typical workflow:\n\n"
        "    import plotly.express as px          # high-level, one-liner charts\n"
        "    import plotly.graph_objects as go    # low-level, full control\n\n"
        "    fig = px.scatter(df, x='col_a', y='col_b', color='group')\n"
        "    fig.show()   # opens in browser\n"
        "    fig.write_html('chart.html')\n"
        "    fig.write_image('chart.png')  # requires kaleido",
        functions=[],
        classes=[],
        values=[("__version__", repr(plotly.__version__))],
        submodules=[
            ("plotly.express", "High-level API — one function per chart type, returns Figure."),
            (
                "plotly.graph_objects",
                "Low-level API — Figure, Layout, and individual trace classes.",
            ),
            ("plotly.io", "Figure I/O: show(), write_html(), write_image(), read_json()."),
            (
                "plotly.colors",
                "Named colour sequences and scales (e.g. px.colors.qualitative.Plotly).",
            ),
            ("plotly.data", "Built-in sample datasets (gapminder, tips, iris, ...)."),
        ],
    )


# ---------------------------------------------------------------------------
# plotly.express
# ---------------------------------------------------------------------------


@spec.define_doc(px)
def _px_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="plotly.express",
        docstring=(
            "High-level plotting API. Every function accepts a DataFrame (or dict / array)\n"
            "and returns a ``plotly.graph_objects.Figure``.\n\n"
            "Common parameters shared by almost all functions:\n\n"
            "    data_frame  — DataFrame (or dict/array); column names used for x/y/color/...\n"
            "    x, y        — column name, Series, or array\n"
            "    color       — column whose values control marker/line colour\n"
            "    facet_row / facet_col — split into sub-plots by column value\n"
            "    hover_name  — column shown in bold in tooltip\n"
            "    hover_data  — extra columns shown in tooltip\n"
            "    labels      — dict to rename axis / legend labels, e.g. {'x': 'Date'}\n"
            "    category_orders — dict to fix the order of categorical values\n"
            "    color_discrete_sequence / color_continuous_scale — colour palette\n"
            "    template    — Plotly theme: 'plotly', 'plotly_dark', 'ggplot2', 'seaborn', ...\n"
            "    title       — figure title string\n"
            "    width, height — figure size in pixels"
        ),
        functions=[
            # --- Scatter / point clouds ---
            CallableInfo(
                "scatter",
                "(data_frame=None, x=None, y=None, color=None, size=None, symbol=None,"
                " text=None, trendline=None, marginal_x=None, marginal_y=None,"
                " log_x=False, log_y=False, ...)",
                "Figure",
                "Scatter plot — one mark per row.",
            ),
            CallableInfo(
                "scatter_3d",
                "(data_frame=None, x=None, y=None, z=None, color=None, size=None, symbol=None, ...)",
                "Figure",
                "3-D scatter plot.",
            ),
            CallableInfo(
                "scatter_matrix",
                "(data_frame=None, dimensions=None, color=None, symbol=None, ...)",
                "Figure",
                "Scatter-plot matrix (SPLOM) — all pairs of dimensions.",
            ),
            CallableInfo(
                "scatter_polar",
                "(data_frame=None, r=None, theta=None, color=None, symbol=None, ...)",
                "Figure",
                "Polar scatter plot.",
            ),
            CallableInfo(
                "scatter_geo",
                "(data_frame=None, lat=None, lon=None, color=None, size=None, locations=None, ...)",
                "Figure",
                "Geographic scatter plot on a map projection.",
            ),
            CallableInfo(
                "scatter_map",
                "(data_frame=None, lat=None, lon=None, color=None, size=None, zoom=8, ...)",
                "Figure",
                "Scatter plot on a tile map (replaces scatter_mapbox).",
            ),
            # --- Line ---
            CallableInfo(
                "line",
                "(data_frame=None, x=None, y=None, color=None, line_group=None,"
                " line_dash=None, markers=False, log_x=False, log_y=False, ...)",
                "Figure",
                "Line plot — rows connected in order.",
            ),
            CallableInfo(
                "line_3d",
                "(data_frame=None, x=None, y=None, z=None, color=None, line_group=None, ...)",
                "Figure",
                "3-D line plot.",
            ),
            CallableInfo(
                "line_polar",
                "(data_frame=None, r=None, theta=None, color=None, line_group=None, ...)",
                "Figure",
                "Polar line plot.",
            ),
            CallableInfo(
                "line_geo",
                "(data_frame=None, lat=None, lon=None, color=None, locations=None, ...)",
                "Figure",
                "Lines on a geographic map projection.",
            ),
            CallableInfo(
                "line_map",
                "(data_frame=None, lat=None, lon=None, color=None, line_group=None, zoom=8, ...)",
                "Figure",
                "Lines on a tile map.",
            ),
            # --- Distribution ---
            CallableInfo(
                "histogram",
                "(data_frame=None, x=None, y=None, color=None, nbins=None,"
                " barnorm=None, histnorm=None, cumulative=False, marginal=None, ...)",
                "Figure",
                "Histogram — bin and count values.",
            ),
            CallableInfo(
                "box",
                "(data_frame=None, x=None, y=None, color=None, points='outliers', notched=False, ...)",
                "Figure",
                "Box plot — median, quartiles, outliers.",
            ),
            CallableInfo(
                "violin",
                "(data_frame=None, x=None, y=None, color=None, points='outliers', box=False, ...)",
                "Figure",
                "Violin plot — KDE-shaped distribution.",
            ),
            CallableInfo(
                "strip",
                "(data_frame=None, x=None, y=None, color=None, stripmode='overlay', ...)",
                "Figure",
                "Strip plot — individual points jittered by category.",
            ),
            CallableInfo(
                "ecdf",
                "(data_frame=None, x=None, y=None, color=None, markers=False, lines=True, ...)",
                "Figure",
                "Empirical cumulative distribution function.",
            ),
            # --- Bar / categorical ---
            CallableInfo(
                "bar",
                "(data_frame=None, x=None, y=None, color=None, barmode='relative', text_auto=False, ...)",
                "Figure",
                "Bar chart.",
            ),
            CallableInfo(
                "bar_polar",
                "(data_frame=None, r=None, theta=None, color=None, barmode='relative', ...)",
                "Figure",
                "Polar bar chart (wind rose).",
            ),
            CallableInfo(
                "funnel",
                "(data_frame=None, x=None, y=None, color=None, ...)",
                "Figure",
                "Funnel chart — stages with decreasing values.",
            ),
            # --- Part-of-whole ---
            CallableInfo(
                "pie",
                "(data_frame=None, names=None, values=None, color=None, hole=0, ...)",
                "Figure",
                "Pie chart.",
            ),
            CallableInfo(
                "sunburst",
                "(data_frame=None, names=None, values=None, parents=None, path=None, color=None, ...)",
                "Figure",
                "Sunburst — hierarchical pie chart.",
            ),
            CallableInfo(
                "treemap",
                "(data_frame=None, names=None, values=None, parents=None, path=None, color=None, ...)",
                "Figure",
                "Treemap — hierarchical rectangles sized by value.",
            ),
            CallableInfo(
                "icicle",
                "(data_frame=None, names=None, values=None, parents=None, path=None, color=None, ...)",
                "Figure",
                "Icicle chart — hierarchical icicle chart.",
            ),
            # --- Heatmap / 2-D density ---
            CallableInfo(
                "density_heatmap",
                "(data_frame=None, x=None, y=None, z=None, nbinsx=None, nbinsy=None, histfunc='count', ...)",
                "Figure",
                "2-D histogram as a colour heatmap.",
            ),
            CallableInfo(
                "density_contour",
                "(data_frame=None, x=None, y=None, z=None, nbinsx=None, nbinsy=None, ...)",
                "Figure",
                "2-D histogram as filled contours.",
            ),
            CallableInfo(
                "density_map",
                "(data_frame=None, lat=None, lon=None, z=None, zoom=8, radius=None, ...)",
                "Figure",
                "Density heatmap on a tile map.",
            ),
            CallableInfo(
                "imshow",
                "(img, zmin=None, zmax=None, origin=None, labels=None,"
                " x=None, y=None, aspect=None, color_continuous_scale=None, ...)",
                "Figure",
                "Display an image or 2-D array as a heatmap.",
            ),
            # --- Financial / time-series ---
            CallableInfo(
                "area",
                "(data_frame=None, x=None, y=None, color=None, line_group=None, ...)",
                "Figure",
                "Area chart — filled line plot.",
            ),
            CallableInfo(
                "timeline",
                "(data_frame=None, x_start=None, x_end=None, y=None, color=None, ...)",
                "Figure",
                "Gantt / timeline chart.",
            ),
            # --- Multivariate ---
            CallableInfo(
                "parallel_coordinates",
                "(data_frame=None, dimensions=None, color=None, ...)",
                "Figure",
                "Parallel coordinates — one axis per numeric dimension.",
            ),
            CallableInfo(
                "parallel_categories",
                "(data_frame=None, dimensions=None, color=None, ...)",
                "Figure",
                "Parallel categories — like parallel coordinates for categoricals.",
            ),
            # --- Geographic choropleth ---
            CallableInfo(
                "choropleth",
                "(data_frame=None, locations=None, color=None, locationmode='ISO-3', geojson=None, ...)",
                "Figure",
                "Choropleth map — regions coloured by value.",
            ),
            CallableInfo(
                "choropleth_map",
                "(data_frame=None, locations=None, color=None, geojson=None, zoom=8, ...)",
                "Figure",
                "Choropleth on a tile map (replaces choropleth_mapbox).",
            ),
        ],
        classes=[],
        values=[],
        submodules=[],
    )


# ---------------------------------------------------------------------------
# plotly.graph_objects
# ---------------------------------------------------------------------------


@spec.define_doc(go)
def _go_doc(mod) -> ModuleInfo:
    return ModuleInfo(
        name="plotly.graph_objects",
        docstring=(
            "Low-level figure construction. Build figures by composing trace objects into a Figure.\n\n"
            "    fig = go.Figure()\n"
            "    fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines+markers', name='series'))\n"
            "    fig.update_layout(title='My Chart', xaxis_title='X', yaxis_title='Y')\n"
            "    fig.show()\n\n"
            "Prefer ``plotly.express`` for one-liners; use ``graph_objects`` when you need\n"
            "precise control over traces, axes, annotations, or mixed chart types."
        ),
        functions=[],
        classes=[
            # Core
            ("Figure", "Top-level figure container. Holds data (traces), layout, and frames."),
            ("FigureWidget", "Figure with ipywidgets integration for Jupyter interactivity."),
            ("Layout", "Controls axes, titles, legend, margins, background, annotations."),
            # Common 2-D traces
            ("Scatter", "Scatter / line trace. mode='markers'|'lines'|'lines+markers'|'text'."),
            ("Bar", "Bar trace. orientation='v'|'h'."),
            ("Histogram", "Histogram trace — bins continuous values automatically."),
            ("Box", "Box plot trace — shows median, quartiles, whiskers, outliers."),
            ("Violin", "Violin trace — KDE-shaped distribution."),
            ("Heatmap", "Heatmap trace — 2-D colour grid from a z matrix."),
            ("Contour", "Contour trace — iso-lines or filled contours from a z matrix."),
            ("Pie", "Pie / donut trace. Use hole=0.4 for a donut."),
            ("Funnel", "Funnel trace for stage-by-stage value drop-off."),
            ("Waterfall", "Waterfall trace — running total with up/down segments."),
            ("Candlestick", "OHLC candlestick trace for financial time series."),
            # 3-D traces
            ("Scatter3d", "3-D scatter / line trace."),
            ("Surface", "3-D surface plot from a z matrix."),
            ("Mesh3d", "3-D mesh / triangulation from x, y, z + optional i, j, k indices."),
            ("Cone", "3-D vector field arrows (cone glyphs)."),
            # Hierarchical / part-of-whole
            ("Sunburst", "Sunburst trace — hierarchical pie chart. Needs ids, labels, parents."),
            ("Treemap", "Treemap trace — hierarchical rectangles. Needs ids, labels, parents."),
            ("Icicle", "Icicle trace — top-down hierarchical chart."),
            # Geographic
            ("Choropleth", "Choropleth map trace — regions filled by a z value."),
            ("Scattergeo", "Geographic scatter / line trace on a map projection."),
            ("Scattermap", "Scatter / line trace on a tile-map (replaces Scattermapbox)."),
            ("Densitymap", "Density heatmap on a tile-map (replaces Densitymapbox)."),
            # Other
            ("Indicator", "KPI / gauge / bullet trace — single numeric display."),
            ("Image", "Raster image trace from a 3-D (H×W×RGBA) array."),
            ("Sankey", "Sankey diagram — flow between nodes via weighted links."),
            ("Splom", "Scatter-plot matrix trace."),
            ("Parcoords", "Parallel coordinates trace for multivariate numeric data."),
            ("Parcats", "Parallel categories trace for multivariate categorical data."),
            ("Ohlc", "OHLC bar trace for financial time series (alternative to Candlestick)."),
        ],
        values=[],
        submodules=[],
    )
