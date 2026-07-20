# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T3: every route the journal exporter posts to is registered on every receiver.

We were burned by exactly this asymmetry: the exporter posted to
``/v1/journal/blocks`` while the viewer and the eval-pipeline headless
backend only registered ``/v1/journal/messages`` + ``/v1/journal/calls``.
Block POSTs returned 405 Method Not Allowed and were silently dropped on
both backends.

This test is a static contract check: the set of ``/v1/journal/*`` POST
routes the exporter uses (read off each ``_Destination``'s ``blocks_url``
/ ``calls_url``) must be a subset of the POST routes registered on both
the viewer's FastAPI app and the headless OTLP backend's FastAPI app.
"""

from __future__ import annotations

from urllib.parse import urlparse


def _exporter_routes() -> set[str]:
    """Inspect ``MessageJournalCallback`` to discover which ``/v1/journal/*``
    paths it actually POSTs to, so adding a new wire route automatically
    pulls it into this contract test."""
    from nooa.tracing._litellm_journal import MessageJournalCallback

    cb = MessageJournalCallback("http://example.invalid")
    paths: set[str] = set()
    for dest in cb._destinations:
        for attr in ("blocks_url", "calls_url"):
            v = getattr(dest, attr, None)
            if isinstance(v, str) and "/v1/journal/" in v:
                paths.add(urlparse(v).path)
    return paths


def _post_routes_on_app(app) -> set[str]:
    """Return the set of paths registered as POST on a FastAPI/Starlette app."""
    paths: set[str] = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", set()) or set()
        if path and "POST" in methods:
            paths.add(path)
    return paths


def test_exporter_journal_paths_are_registered_on_viewer_app():
    from nooa.viewer.main import app as viewer_app

    exporter_paths = _exporter_routes()
    assert exporter_paths, (
        "MessageJournalCallback exposes no /v1/journal/* URLs; either the "
        "implementation moved or this test is stale"
    )

    registered = _post_routes_on_app(viewer_app)
    missing = exporter_paths - registered
    assert not missing, (
        f"viewer is missing POST handlers for journal routes the exporter "
        f"uses: {sorted(missing)}.\n"
        f"  exporter posts to: {sorted(exporter_paths)}\n"
        f"  viewer registers:  "
        f"{sorted(p for p in registered if p.startswith('/v1/journal/'))}"
    )


def test_exporter_journal_paths_are_registered_on_headless_backend_app():
    """The eval-pipeline's in-process backend must register the same routes;
    otherwise eval runs silently lose journal data on the way to its temp DB,
    even before the data is forwarded to a viewer."""
    from eval_pipeline.headless_backend import _make_headless_app

    app = _make_headless_app()
    exporter_paths = _exporter_routes()
    registered = _post_routes_on_app(app)
    missing = exporter_paths - registered
    assert not missing, (
        f"headless OTLP backend is missing POST handlers for journal routes "
        f"the exporter uses: {sorted(missing)}.\n"
        f"  exporter posts to: {sorted(exporter_paths)}\n"
        f"  headless registers: "
        f"{sorted(p for p in registered if p.startswith('/v1/journal/'))}"
    )
