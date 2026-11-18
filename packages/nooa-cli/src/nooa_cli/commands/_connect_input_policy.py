# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which media inputs the wizard may offer for the chosen API style.

Each API style can send only some media kinds (for example, the Anthropic
Messages and OpenAI Responses clients send images and PDFs, Chat Completions
also sends audio and video, decision clients send images). The rules live in
``nooa.media_policy``; this module applies them to the wizard state. It limits
the media picker to the kinds the API style can send, and asks the user to fix
an entry whose saved media inputs include kinds the API style cannot send.
"""

import click


def media_route(state):
    """Return a model entry describing the endpoint and API style chosen so far.

    The entry is built from the wizard state without any network request. It
    is only used to look up which media kinds that API style can send; it says
    nothing about what the model itself accepts.
    """
    if state.editing is not None:
        # When editing, the existing entry's client and provider settings are
        # kept unchanged, and they decide how requests are sent at runtime.
        # The wizard's api_style label may not match them, so use the entry.
        return state.editing
    style = state.api_style or "chat"
    prefix = "anthropic/" if style == "anthropic" else "openai/"
    return {
        "model_name": prefix + (state.model or ""),
        "api_base": state.endpoint,
        "client_type": "decision"
        if style in {"systemone", "openai"}
        else "responses"
        if style == "responses"
        else "completion",
    }


def available_media(state, *, entry=None):
    """Return the media kinds the API style of ``entry`` (or the wizard state) can send."""
    from nooa.media_policy import entry_input_interface, implemented_input_kinds

    return implemented_input_kinds(
        entry_input_interface(entry if entry is not None else media_route(state))
    )


def picker_options(state, *, entry=None):
    """Return keyword arguments that limit the media picker to the kinds that can be sent."""
    available = available_media(state, entry=entry)
    # When all four kinds can be sent (Chat Completions), return no arguments so
    # the picker is called exactly as before, with its default list of kinds.
    return {"available": available} if len(available) != 4 else {}


def repair_native_media(state, current, *, entry=None, explicit=False):
    """Make sure the enabled media inputs ``current`` can be sent with this API style.

    Only the Anthropic Messages and OpenAI Responses styles are checked here.
    If every enabled kind can be sent, the list is returned unchanged. If not,
    the user chooses: drop the kinds that cannot be sent, turn media off, or
    cancel. The chosen list is stored on ``state`` and returned; cancel returns
    None. Nothing is removed without the user's choice, and the saved entry
    being edited is not modified here (only the new entry being built is).

    With ``explicit`` (the list came from ``--media-inputs``) or ``--yes``,
    there is no one to ask, so a ``click.UsageError`` is raised instead.
    This only concerns enabled inputs; media tests chosen with "Test all" are
    handled separately and may report kinds that were skipped.
    """
    from nooa.media_inputs import validate_media_inputs
    from nooa.media_policy import (
        entry_input_interface,
        implemented_input_kinds,
        unsupported_input_reason,
    )

    from . import _connect_prompts as prompts
    from . import _connect_view as view

    enabled = list(validate_media_inputs(current))
    interface = entry_input_interface(entry if entry is not None else media_route(state))
    if interface not in {"anthropic", "responses"}:
        return enabled
    available = implemented_input_kinds(interface)
    unsupported = [kind for kind in enabled if kind not in available]
    if not unsupported:
        return enabled
    reason = unsupported_input_reason(interface, unsupported)
    if explicit or state.yes:
        raise click.UsageError(
            reason + " Use --media-inputs image,pdf or --media-inputs - to repair, or choose Chat."
        )
    view.warning(reason)
    action = prompts.labelled_choice(
        "Repair unsupported media permissions for this interface",
        (
            ("supported", "Remove unsupported inputs and keep image/PDF settings"),
            ("off", "Turn all media inputs off"),
            ("cancel", "Cancel without changing or saving; choose another interface"),
        ),
        default="cancel",
    )
    if action == "cancel":
        return None
    selected = [kind for kind in enabled if kind in available] if action == "supported" else []
    state.media_inputs = selected
    state.inherit_media_inputs = False
    state.guided_media_revised = True
    return selected


def validate_native_declaration(state, *, entry=None):
    """Raise ``click.UsageError`` if ``--media-inputs`` names a kind this API style cannot send.

    Runs before any request is sent, so an invalid option fails early.
    """
    if state.media_inputs is None:
        return
    from ._connect_prompts import parse_media_inputs

    repair_native_media(state, parse_media_inputs(state.media_inputs), entry=entry, explicit=True)
