# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unified trace and evaluation viewer for nooa."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent


def _resolve_frontend_dir() -> Path:
    """Locate the React frontend build output directory.

    Search order:
    1. PACKAGE_DIR/frontend-react/dist/ — consolidated layout
    2. workspace_root/frontend-react/dist/ — old workspace/editable install
    3. Bundled frontend at package level — wheel install
    """
    # Consolidated layout: frontend-react is next to the viewer package
    local_dist = PACKAGE_DIR / "frontend-react" / "dist"
    if local_dist.is_dir() and (local_dist / "index.html").exists():
        return local_dist

    # Old layout: frontend-react at workspace root (two dirs up from package)
    workspace_root = PACKAGE_DIR.parent.parent
    react_dist = workspace_root / "frontend-react" / "dist"
    if react_dist.is_dir() and (react_dist / "index.html").exists():
        return react_dist

    bundled = PACKAGE_DIR / "frontend"
    if bundled.is_dir():
        return bundled

    raise RuntimeError(
        "Frontend not found. Run 'npm run build' in frontend-react/, "
        "or install from a wheel that bundles the frontend."
    )


FRONTEND_DIR = _resolve_frontend_dir()
