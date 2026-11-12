# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Adapter-side numeric projection and cap application to a prepared attempt."""

from collections.abc import Mapping

from nooa.unifiedllm.limits import REPLY_CAP_KEYS


def _apply_decision(params, estimated, quality, decision, *, responses=False):
    layers = [params, params.get("extra_body")]
    selected = [
        (layer, key)
        for layer in layers
        if isinstance(layer, Mapping)
        for key in REPLY_CAP_KEYS
        if key in layer
    ]
    caps = [layer[key] for layer, key in selected if type(layer[key]) is int and layer[key] > 0]
    cap = decision(estimated, quality, min(caps) if caps else None)
    if cap is None:
        return
    # Keep the prepared family's spelling/location; remove every conflicting alias.
    layer, key = (
        selected[0] if selected else (params, "max_output_tokens" if responses else "max_tokens")
    )
    for target in layers:
        if isinstance(target, dict):
            for alias in REPLY_CAP_KEYS:
                target.pop(alias, None)
    layer[key] = cap
