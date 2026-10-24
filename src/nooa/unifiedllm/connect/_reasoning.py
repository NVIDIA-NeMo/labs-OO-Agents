# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded reasoning facts and known controls, never model-name inference."""

from copy import deepcopy

EFFORTS = ("low", "medium", "high", "xhigh", "max")


def reasoning_metadata(model):
    """Detach published Anthropic ModelCapabilities paths only.

    Missing, empty and false remain distinct. No arbitrary request blocks are
    accepted. Advertised facts are not evidence of successful inference.
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
    """Field-level provenance for precisely the normalized capability paths."""
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
    """Exact native ID and same Messages base, allowing its equivalent /v1 spelling."""
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
    """Compile exact selected, same-source native facts, not foreign catalogues.

    Manual-only thinking offers a minimal On patch, never invented effort levels.
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
