# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, bounded mappings of published metadata, never runtime permissions.

Only documented field paths are read. Names/descriptions, output modalities,
model-family guesses and arbitrary nested flags are not input evidence.
"""

FORMATS = ("image", "audio", "video", "pdf")


def _strings(value):
    return (
        list(value)
        if isinstance(value, list)
        and len(value) <= 100
        and all(
            isinstance(v, str) and len(v) <= 200 and not any(ord(c) < 32 for c in v) for v in value
        )
        else None
    )


def media_metadata(metadata):
    """Detach only known media fields from one selected model (no recursion)."""
    if not isinstance(metadata, dict):
        return {}
    result = {}
    architecture = metadata.get("architecture")
    if isinstance(architecture, dict):
        modalities = _strings(architecture.get("input_modalities"))
        if modalities is not None:
            result["architecture"] = {"input_modalities": modalities}
    modalities = _strings(metadata.get("input_modalities"))
    if modalities is not None:
        result["input_modalities"] = modalities
    # Anthropic Models API / official SDK ModelCapabilities (0.109.2):
    # image_input/pdf_input are CapabilitySupport objects, not bare booleans.
    capabilities = metadata.get("capabilities")
    if isinstance(capabilities, dict):
        flags = {}
        for key in ("image_input", "pdf_input"):
            flag = capabilities.get(key)
            if isinstance(flag, dict) and isinstance(flag.get("supported"), bool):
                flags[key] = {"supported": flag["supported"]}
        if flags:
            result["capabilities"] = flags
    return result


def advertised_media_suggestions(
    metadata=None,
    *,
    source="unknown",
    api_style="chat",
    decision_image_transport="json",
    typesafe_danger_zone_images=False,
):
    """Report advertised inputs and check limits; do not accept or enable them.

    ``metadata`` is one selected model, ``source`` its listing URL/label. Both
    inputs are bounded. ``suggested`` is constrained to implemented permissions;
    ``checkable`` is narrower and is not authorization to run generated checks.
    Generic ``file`` does not establish PDF support.
    """
    facts = media_metadata(metadata)
    evidence = {}
    unknown = []
    for path, values in (
        ("architecture.input_modalities", facts.get("architecture", {}).get("input_modalities")),
        ("input_modalities", facts.get("input_modalities")),
    ):
        if values is not None:
            evidence[path] = values
            unknown.extend(v for v in values if v not in (*FORMATS, "text", "file"))
    advertised = [kind for kind in FORMATS if any(kind in v for v in evidence.values())]
    for kind, key in (("image", "image_input"), ("pdf", "pdf_input")):
        flag = facts.get("capabilities", {}).get(key)
        if flag is not None:
            evidence["capabilities." + key + ".supported"] = flag["supported"]
            if flag["supported"] and kind not in advertised:
                advertised.append(kind)
            elif not flag["supported"] and kind in advertised:
                advertised.remove(kind)
    advertised = [kind for kind in FORMATS if kind in advertised]
    notes = []
    if not evidence:
        notes.append("Media information unknown; choose inputs manually if needed.")
    if any(isinstance(v, list) and "file" in v for v in evidence.values()):
        pdf_known = "capabilities.pdf_input.supported" in evidence or any(
            isinstance(v, list) and "pdf" in v for v in evidence.values()
        )
        notes.append(
            "Generic file input alone does not establish PDF support."
            if pdf_known
            else "Generic file input is advertised; PDF support is unknown without specific PDF evidence."
        )
    if unknown:
        notes.append("Unmapped input categories: " + ", ".join(dict.fromkeys(unknown)) + ".")
    decision = api_style in {"systemone", "openai"}
    suggested = [kind for kind in advertised if not decision or kind == "image"]
    unavailable = {}
    for kind in advertised:
        if decision and kind != "image":
            unavailable[kind] = (
                "Native decisions implement image inputs only; this format is not testable."
            )
        elif api_style in {"anthropic", "responses"} and kind in {"audio", "video"}:
            unavailable[kind] = (
                "Generated audio/video checks are unavailable on this interface; advertised support is not verified."
            )
        elif (
            api_style == "systemone"
            and kind == "image"
            and not (
                decision_image_transport == "typesafe_multipart"
                and typesafe_danger_zone_images is True
            )
        ):
            unavailable[kind] = (
                "System One JSON image mapping is unverified and not testable; no transport is selected from metadata."
            )
    source = (
        source
        if isinstance(source, str) and len(source) <= 2048 and not any(ord(c) < 32 for c in source)
        else "unknown"
    )
    if source == "https://openrouter.ai/api/v1/models":
        notes.append("OpenRouter facts describe its route, not verified native API behavior.")
    model_id = (
        metadata.get("media_model_id", metadata.get("id")) if isinstance(metadata, dict) else None
    )
    if not isinstance(model_id, str) or len(model_id) > 500 or any(ord(c) < 32 for c in model_id):
        model_id = None
    return {
        "source": source,
        "model_id": model_id,
        "advertised": advertised,
        "suggested": suggested,
        "checkable": [kind for kind in suggested if kind not in unavailable],
        "unavailable_checks": unavailable,
        "evidence": evidence,
        "notes": notes,
        "permissions_changed": False,
    }
