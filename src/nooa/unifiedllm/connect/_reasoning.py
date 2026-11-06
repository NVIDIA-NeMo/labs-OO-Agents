# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reasoning ("thinking") options read from the Anthropic Models API.

The Anthropic Models API lists, per model, whether extended thinking and effort
levels are supported (``capabilities.thinking``/``capabilities.effort``). This
module copies those fields and turns them into reasoning levels the user can
pick. Nothing is guessed from the model name.
"""

from copy import deepcopy

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def reasoning_metadata(model):
    """Copy the reasoning-related ``capabilities`` fields of one Anthropic model record.

    Only the documented ``thinking``, ``effort`` and ``structured_outputs``
    fields are kept, each reduced to its boolean ``supported`` flags. A missing
    field, an empty field and ``supported: False`` stay distinguishable. Other
    content of the record is never copied. These are claims from the listing,
    not proof that a request using them will work.
    """
    caps = model.get("capabilities") if isinstance(model, dict) else None
    if not isinstance(caps, dict):
        return {}
    result = {}
    for key, children in (("thinking", ("enabled", "adaptive", "disabled")), ("effort", EFFORTS)):
        raw = caps.get(key)
        if not isinstance(raw, dict):
            continue
        clean = {}
        if isinstance(raw.get("supported"), bool):
            clean["supported"] = raw["supported"]
        container = raw.get("types") if key == "thinking" else raw
        modes = {}
        if isinstance(container, dict):
            for name in children:
                support = container.get(name)
                if isinstance(support, dict) and isinstance(support.get("supported"), bool):
                    modes[name] = {"supported": support["supported"]}
        if key == "thinking":
            if isinstance(raw.get("types"), dict):
                clean["types"] = modes
            between = raw.get("between_tools")
            if isinstance(between, dict) and isinstance(between.get("supported"), bool):
                clean["between_tools"] = {"supported": between["supported"]}
        else:
            clean.update(modes)
        result[key] = clean
    structured = caps.get("structured_outputs")
    if isinstance(structured, dict) and isinstance(structured.get("supported"), bool):
        result["structured_outputs"] = {"supported": structured["supported"]}
    return {"capabilities": result} if result else {}


def metadata_fields(model):
    """Map each field kept by ``reasoning_metadata`` to its path in the listing response.

    For example ``capabilities.thinking.supported`` maps to
    ``data[].capabilities.thinking.supported``, so the UI can say where each
    value came from.
    """
    fields = {}

    def walk(value, path):
        for key, child in value.items():
            name = path + "." + key if path else key
            if isinstance(child, dict):
                walk(child, name)
            else:
                fields[name] = "data[]." + name

    walk(reasoning_metadata(model), "")
    return fields


def native_metadata_matches(model, *, model_id, api_style, source, endpoint):
    """Return True if ``model`` describes exactly this model on this endpoint.

    Requires the Anthropic API style, ``model["id"] == model_id``, and that
    ``source`` (the listing URL) is the ``/models`` URL of ``endpoint``, with or
    without the ``/v1`` prefix.
    """
    return (
        api_style == "anthropic"
        and isinstance(model, dict)
        and model.get("id") == model_id
        and isinstance(endpoint, str)
        and source
        in {
            endpoint.rstrip("/") + "/models",
            endpoint.rstrip("/").removesuffix("/v1") + "/v1/models",
        }
    )


def native_levels(model, *, model_id, api_style, source, endpoint):
    """Build the reasoning levels offered for ``model`` from its Anthropic listing.

    Returns a dict of level name -> request settings to merge into requests,
    ``{}`` if the listing says thinking is not supported, or ``None`` if there
    is no usable information (wrong source or model, or no levels found).
    Only the listing of the selected model from the same endpoint is used,
    never a third-party catalogue.

    Adaptive thinking with effort support yields one level per supported
    effort. A model with only manual ("enabled") thinking gets a single "on"
    level with a small fixed token budget; effort levels are not invented. An
    "off" level is added when disabling thinking is supported.
    """
    if not native_metadata_matches(
        model, model_id=model_id, api_style=api_style, source=source, endpoint=endpoint
    ):
        return None
    caps = reasoning_metadata(model).get("capabilities", {})
    thinking = caps.get("thinking", {})
    modes = thinking.get("types", {})
    effort = caps.get("effort", {})
    levels = {}
    if (
        thinking.get("supported") is True
        and modes.get("adaptive", {}).get("supported") is True
        and effort.get("supported") is True
    ):
        for label in EFFORTS:
            if effort.get(label, {}).get("supported") is True:
                levels[label] = {
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": label},
                }
    if (
        thinking.get("supported") is True
        and modes.get("enabled", {}).get("supported") is True
        and modes.get("adaptive", {}).get("supported") is False
    ):
        levels["on"] = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    if thinking.get("supported") is True and modes.get("disabled", {}).get("supported") is True:
        levels["off"] = {"thinking": {"type": "disabled"}}
    if thinking.get("supported") is False:
        return {}
    return deepcopy(levels) if levels else None
