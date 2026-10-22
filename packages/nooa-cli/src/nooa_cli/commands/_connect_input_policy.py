# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit repair and picker ceilings using the core request-input contract."""

import click


def media_route(state):
    """A no-I/O preview of the selected Connect route, not model capability."""
    if state.editing is not None:
        # The editor preserves this route's client/provider controls verbatim;
        # its api_style label need not describe the actual runtime dispatch.
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
    from nooa.media_policy import entry_input_interface, implemented_input_kinds

    return implemented_input_kinds(
        entry_input_interface(entry if entry is not None else media_route(state))
    )


def picker_options(state, *, entry=None):
    available = available_media(state, entry=entry)
    # Preserve existing generic Chat picker invocation and its API contract.
    return {"available": available} if len(available) != 4 else {}


def repair_native_media(state, current, *, entry=None, explicit=False):
    """Return an explicit valid selection, or None for cancellation; never erase.

    Old raw profile permissions/evidence stay on the source entry. Only a user
    choice repairs a detached candidate. --yes and invalid explicit flags fail.
    Requested tests are separate: Test All may report unsupported skipped rows.
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
    """Reject explicit unsupported declarations before provider execution."""
    if state.media_inputs is None:
        return
    from ._connect_prompts import parse_media_inputs

    repair_native_media(state, parse_media_inputs(state.media_inputs), entry=entry, explicit=True)
