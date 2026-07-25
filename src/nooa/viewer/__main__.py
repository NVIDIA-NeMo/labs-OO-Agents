# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allow running the viewer with `python -m nooa.viewer`."""

import logging
import os

import uvicorn

from .main import app

log = logging.getLogger(__name__)

port = int(
    os.environ.get("NOOA_TRACE_VIEWER_PORT") or os.environ.get("NEMO_OO_TRACE_VIEWER_PORT", "5001")
)
log.info("Starting viewer on port %d", port)
uvicorn.run(app, host="0.0.0.0", port=port)
