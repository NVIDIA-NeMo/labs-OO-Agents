# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Implemented input contracts, independent of model capability and Connect.

Permissions and historical proof are not capability discovery. Chat extensions
are implemented, but backend acceptance remains unverified without a test.
Resolve the actual request adapter, never a Claude name or an api_style label.
"""

from collections.abc import Mapping
from typing import Any

from nooa.media_inputs import MediaInput, attachment_kind, validate_media_inputs

ALL_INPUT_KINDS: tuple[MediaInput, ...] = ("image", "audio", "video", "pdf")


def implemented_input_kinds(
    interface: str = "chat", provider: str | None = None
) -> tuple[MediaInput, ...]:
    """The implementation ceiling, not a claim about the selected model."""
    if interface == "decision":
        return ("image",)
    if interface == "responses" or interface == "anthropic" or provider == "anthropic":
        return ("image", "pdf")
    return ALL_INPUT_KINDS


def resolved_input_interface(
    model: str, config: Mapping[str, Any], *, client_type: str = "completion"
) -> str:
    """Match registry/client dispatch, with no key lookup, network or metadata inference.

    client_type selects Responses/Decisions. Completion dispatch is resolved by
    LiteLLM with the actual model/base/provider override; api_style is not used.
    Unknown Chat routes still permit truthful untested declarations.
    """
    if client_type in {"responses", "decision"}:
        return client_type
    if not model:
        return "chat"  # No dispatch to resolve in a partial/unknown profile.
    import litellm

    try:
        _, provider, _, _ = litellm.get_llm_provider(
            model=model,
            api_base=config.get("api_base") or config.get("base_url"),
            custom_llm_provider=config.get("custom_llm_provider"),
        )
    except Exception:
        return "chat"
    return "anthropic" if provider == "anthropic" else "chat"


def entry_input_interface(entry: Mapping[str, Any]) -> str:
    return resolved_input_interface(
        entry.get("model_name") or "",
        entry,
        client_type=entry.get("client_type") or "completion",
    )


def unsupported_input_reason(interface: str, kinds) -> str:
    name = {
        "anthropic": "native Anthropic Messages",
        "responses": "Responses",
        "decision": "Decisions",
    }.get(interface, interface)
    available = ", ".join(implemented_input_kinds(interface))
    return (
        f"Unsupported media input for {name}: {', '.join(kinds)}. "
        f"This request interface implements only {available}. "
        "Explicitly repair media_inputs by choosing supported formats or text-only; "
        "enabling without testing cannot support these inputs."
    )


def validate_implemented_media_inputs(value: object, interface: str) -> tuple[MediaInput, ...]:
    """Prospective declarations must fit the implementation before calls/writes."""
    enabled = validate_media_inputs(value)
    unsupported = [kind for kind in enabled if kind not in implemented_input_kinds(interface)]
    if unsupported:
        raise ValueError(unsupported_input_reason(interface, unsupported))
    return enabled


def reject_unsupported_media(messages, interface: str) -> None:
    """Validate explicit public/projected slots AFTER permissions were filtered.

    Never visit tool arguments, reasoning or arbitrary nested JSON. Native turn
    projection owns opaque state and its existing disabled-media guard.
    """
    allowed = implemented_input_kinds(interface)
    if allowed == ALL_INPUT_KINDS:
        return
    kinds = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if message.get("audio") is not None:
            kinds.append("audio")
        kind = attachment_kind(message)
        if kind in {"audio", "video"}:
            kinds.append(kind)
        for slot in ("content", "output"):
            blocks = message.get(slot)
            if isinstance(blocks, (list, tuple)):
                kinds.extend(
                    attachment_kind(block) for block in blocks if isinstance(block, Mapping)
                )
    unsupported = [kind for kind in ALL_INPUT_KINDS if kind in kinds and kind not in allowed]
    if unsupported:
        raise ValueError(unsupported_input_reason(interface, unsupported))
