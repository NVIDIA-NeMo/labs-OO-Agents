# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit wizard review of advertisements, not automatic media permissions."""


def offer_suggestions(state, previous):
    """Render selected-model hints; return whether the user chose permissions."""
    from nooa.unifiedllm import connect

    from . import _connect_prompts as prompts
    from . import _connect_view as view

    hints = state.media_hints
    report = connect.advertised_media_suggestions(
        hints,
        source=(hints or {}).get(
            "media_source", "published model information" if hints else "unknown"
        ),
        api_style=state.api_style,
        decision_image_transport=state.decision_image_transport or "json",
        typesafe_danger_zone_images=state.typesafe_danger_zone_images,
    )
    if any(report["evidence"].values()):
        if report["advertised"]:
            view.line("Listed media inputs: " + ", ".join(report["advertised"]))
        source = report["source"]
        if source == connect.CATALOGUE_URL:
            view.line("Model details from OpenRouter")
        elif source not in {"", "unknown", "published model information", "endpoint listing"}:
            view.line("Model details from " + source)
        elif source == "endpoint listing" and state.endpoint_models_url:
            view.line("Model details from " + state.endpoint_models_url)
        for note in report["notes"]:
            view.line(note)
        for kind, reason in report["unavailable_checks"].items():
            view.line(("PDF" if kind == "pdf" else kind.capitalize()) + ": " + reason)
    else:
        view.line(
            "Image support isn't listed. You can test it or leave images off."
            if state.api_style in connect.DECISION_API_STYLES
            else "Media support isn't listed. You can test inputs or choose them yourself."
        )
    view.line("Current media inputs: " + (", ".join(previous) or "off"))
    if (
        state.media_inputs is not None
        or state.test_media_inputs is not None
        or state.yes
        or not report["suggested"]
    ):
        return False
    options = []
    can_test = not state.no_probe and state.approval == "all" and bool(report["checkable"])
    if can_test:
        options.append(("test", "Enable listed inputs with test"))
    options.extend(
        (
            ("suggested", "Enable listed inputs without testing"),
            ("manual", "Select media inputs manually"),
            ("none", "Turn media inputs off"),
        )
    )
    action = prompts.labelled_choice(
        "Choose how to use the listed media inputs",
        tuple(options),
        default="test" if can_test else "manual",
    )
    if action == "test":
        state.test_media_inputs = list(report["checkable"])
        state.media_inputs = list(previous)
        state.inherit_media_inputs = True
    else:
        state.media_inputs = (
            list(report["suggested"])
            if action == "suggested"
            else prompts.choose_media_inputs(previous)
            if action == "manual"
            else []
        )
        state.inherit_media_inputs = False
    return True
