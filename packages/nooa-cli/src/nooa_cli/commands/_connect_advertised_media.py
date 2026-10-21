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
    view.line("Advertised media inputs: " + (", ".join(report["advertised"]) or "none known"))
    view.line(
        "Media information source: "
        + report["source"]
        + (" (" + report["model_id"] + ")" if report["model_id"] else "")
    )
    for note in report["notes"]:
        view.line(note)
    for kind, reason in report["unavailable_checks"].items():
        view.line(kind + ": " + reason)
    view.line(
        "Advertisements are unverified suggestions, not permission. Current inputs: "
        + (", ".join(previous) or "none (text-only)")
    )
    if (
        state.media_inputs is not None
        or state.test_media_inputs is not None
        or state.yes
        or not report["suggested"]
    ):
        return False
    options = [
        ("keep", "Keep current inputs: " + (", ".join(previous) or "text only")),
        (
            "suggested",
            "Enable suggested inputs manually (unverified): " + ", ".join(report["suggested"]),
        ),
    ]
    if not state.no_probe and state.approval == "all" and report["checkable"]:
        options.append(
            (
                "test",
                "Test suggested inputs with generated samples, using the approved budget: "
                + ", ".join(report["checkable"]),
            )
        )
    options.extend((("manual", "Change inputs manually"), ("none", "Use no media (text only)")))
    action = prompts.labelled_choice(
        "Review advertised media suggestions", tuple(options), default="keep"
    )
    if action == "test":
        state.test_media_inputs = list(report["checkable"])
        state.media_inputs = list(previous)
        state.inherit_media_inputs = True
    else:
        state.media_inputs = (
            list(previous)
            if action == "keep"
            else list(report["suggested"])
            if action == "suggested"
            else prompts.choose_media_inputs(previous)
            if action == "manual"
            else []
        )
    return True
