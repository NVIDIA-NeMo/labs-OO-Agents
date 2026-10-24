# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Small, bounded mappings of published metadata, never runtime permissions.

Only documented field paths are read. Names/descriptions, output modalities,
model-family guesses and arbitrary nested flags are not input evidence.
"""

from nooa.media_policy import ALL_INPUT_KINDS as FORMATS


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
    entry=None,
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
        notes.append("No media is listed.")
    if any(isinstance(v, list) and "file" in v for v in evidence.values()):
        pdf_known = "capabilities.pdf_input.supported" in evidence or any(
            isinstance(v, list) and "pdf" in v for v in evidence.values()
        )
        notes.append(
            "Files are listed separately from PDFs; use the PDF setting for PDF inputs."
            if pdf_known
            else "PDF support isn't listed. Test a generated PDF or leave PDFs off."
        )
    if unknown:
        notes.append("Inputs unavailable in Connect: " + ", ".join(dict.fromkeys(unknown)) + ".")
    decision = api_style in {"systemone", "openai"}
    from nooa.media_policy import (
        entry_input_interface,
        implemented_input_kinds,
        unsupported_input_reason,
    )

    interface = (
        entry_input_interface(entry)
        if entry is not None
        else ("decision" if decision else api_style)
    )
    implemented = implemented_input_kinds(interface)
    suggested = [kind for kind in advertised if kind in implemented]
    unavailable = {}
    for kind in advertised:
        if interface == "decision" and kind != "image":
            unavailable[kind] = (
                "The NOOA decision client currently supports image attachments only. Use a chat model for this input format."
            )
        elif kind not in implemented:
            unavailable[kind] = unsupported_input_reason(interface, [kind])
        elif (
            api_style == "systemone"
            and kind == "image"
            and not (
                decision_image_transport == "typesafe_multipart"
                and typesafe_danger_zone_images is True
            )
        ):
            unavailable[kind] = (
                "Image testing isn't available for this connection. Use OpenAI Decisions or direct TypeSafe multipart with experimental images enabled, or leave images off."
            )
    source = (
        source
        if isinstance(source, str) and len(source) <= 2048 and not any(ord(c) < 32 for c in source)
        else "unknown"
    )
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
