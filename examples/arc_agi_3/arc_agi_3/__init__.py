# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Vendored ARC-AGI-3 environment wrapper.

A copy of the environment + scoring layer this example needs, on top of the
public ARC-AGI-3 SDK (``arc-agi`` / ``arcengine``, installed via the optional
``arc`` extra). Kept as a local package so the example is self-contained — it
does not depend on any external submodule.
"""
