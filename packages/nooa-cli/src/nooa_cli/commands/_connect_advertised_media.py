# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wizard step that shows which media inputs the model is said to accept.

Published model information (the model listing, server details or the
OpenRouter catalogue) may list input types such as images or audio. This step
shows those as suggestions and lets the user decide: test them, enable them
without testing, pick them by hand, or turn media off. Listed media is never
enabled automatically.
"""


def offer_suggestions(state, previous):
    """Show the media inputs listed for the selected model and ask how to use them.

    ``state`` is the wizard state; ``previous`` is the list of media inputs the
    entry had before (shown as "Current inputs"). Depending on the answer this
    sets ``state.test_media_inputs`` (inputs to test) or ``state.media_inputs``
    (inputs to enable without testing).

    Returns True if the user made a choice here, and False if no question was
    asked: the choice already came from command-line options, ``--yes`` was
    given, or nothing was suggested.
    """
    from nooa.unifiedllm import connect

    from . import _connect_prompts as prompts
    from . import _connect_view as view

    if not state.metadata_displayed:
        from ._connect_wizard import show_model_metadata

        show_model_metadata(state)
    from ._connect_input_policy import media_route
    from ._connect_wizard import model_media_hints

    hints = model_media_hints(state)
    report = connect.advertised_media_suggestions(
        hints,
        source=(hints or {}).get(
            "media_source", "published model information" if hints else "unknown"
        ),
        api_style=state.api_style,
        entry=media_route(state),
        decision_image_transport=state.decision_image_transport or "json",
        typesafe_danger_zone_images=state.typesafe_danger_zone_images,
    )
    from ._connect_wizard import scrub_setup

    view.section("Media setup")
    for note in report["notes"]:
        if note not in {
            "Media support isn't listed. You can test inputs or choose them yourself.",
            "No media is listed.",
        }:
            view.note(scrub_setup(state, note))
    from ._connect_input_policy import available_media

    supported = available_media(state)
    # Formats this interface can't send are already excluded from the choices.
    explained = [
        (view.media_label(kind) + " test", reason)
        for kind, reason in report["unavailable_checks"].items()
        if kind in supported
    ]
    if explained:
        view.fields(scrub_setup(state, explained))

    def input_label(kind):
        return {"audio": "Audio clips", "video": "Videos"}.get(kind, view.media_label(kind))

    suggested_label = ", ".join(input_label(kind) for kind in report["suggested"])
    checkable_label = ", ".join(input_label(kind) for kind in report["checkable"])
    view.fields(
        [
            ("Suggested inputs", suggested_label),
            ("Current inputs", ", ".join(view.media_label(kind) for kind in previous) or "Off"),
        ]
        if suggested_label
        else [("Current inputs", ", ".join(view.media_label(kind) for kind in previous) or "Off")]
    )
    if (
        state.media_inputs is not None
        or state.test_media_inputs is not None
        or state.yes
        or not report["suggested"]
    ):
        return False
    options = []
    can_test = not state.no_probe and state.approval == "all" and bool(report["checkable"])
    all_inputs = (
        ["image"]
        if state.api_style in connect.DECISION_API_STYLES
        else ["image", "audio", "video", "pdf"]
    )
    all_report = connect.advertised_media_suggestions(
        {"id": state.model, "input_modalities": all_inputs},
        api_style=state.api_style,
        entry=media_route(state),
        decision_image_transport=state.decision_image_transport or "json",
        typesafe_danger_zone_images=state.typesafe_danger_zone_images,
    )
    can_test_all = not state.no_probe and state.approval == "all" and bool(all_report["checkable"])
    if can_test:
        options.append(("test", f"Test and enable {checkable_label}"))
    if can_test_all:
        options.append(("test-all", "Test all media inputs"))
    options.extend(
        (
            ("suggested", f"Enable {suggested_label} without testing"),
            ("manual", "Select media inputs manually"),
            ("none", "Turn media inputs off"),
        )
    )
    action = prompts.labelled_choice(
        "Choose how to use the listed media inputs",
        tuple(options),
        default="test" if can_test else "test-all" if can_test_all else "manual",
    )
    if action in {"test", "test-all"}:
        state.test_media_inputs = (
            list(report["checkable"]) if action == "test" else list(all_inputs)
        )
        state.media_inputs = list(previous)
        state.inherit_media_inputs = True
        if action == "test-all":
            supported = available_media(state)
            for kind, reason in all_report["unavailable_checks"].items():
                if kind not in report["unavailable_checks"] and kind in supported:
                    view.fields(scrub_setup(state, [(view.media_label(kind) + " test", reason)]))
    elif action == "manual":
        from ._connect_wizard import select_manual_media

        select_manual_media(
            state,
            previous,
            **({"available": ("image",)} if state.api_style in connect.DECISION_API_STYLES else {}),
        )
    else:
        state.media_inputs = list(report["suggested"]) if action == "suggested" else []
        state.inherit_media_inputs = False
    return True
