# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which media kinds each request interface can carry.

NOOA sends requests through one of four interfaces:

- ``chat``: OpenAI-compatible Chat Completions. Every ``CompletionClient`` route
  that is not native Anthropic uses this format; LiteLLM translates it for other
  providers. It can carry images, audio, video and PDFs.
- ``anthropic``: native Anthropic Messages, used when LiteLLM dispatches a
  ``CompletionClient`` request to the ``anthropic`` provider. Images and PDFs only.
- ``responses``: the OpenAI Responses API, used by ``ResponsesClient``. Images and
  PDFs only.
- ``decision``: native decision endpoints. Images only.

These limits describe what NOOA's request code can send. They are not a claim
about what a particular model accepts: a Chat backend may still reject audio or
video. The interface is resolved from the actual LiteLLM dispatch (model, endpoint
and ``custom_llm_provider``), never from the model name or ``api_style``.
"""

from collections.abc import Mapping
from typing import Any

from nooa.media_inputs import MediaInput, attachment_kind, validate_media_inputs

ALL_INPUT_KINDS: tuple[MediaInput, ...] = ("image", "audio", "video", "pdf")


def interface_for_provider(provider: str | None) -> str:
    """Return the wire format a ``CompletionClient`` request uses for ``provider``.

    ``provider`` is the LiteLLM provider name for the request. LiteLLM sends
    ``anthropic`` requests as native Anthropic Messages; every other provider is
    given an OpenAI-compatible Chat request. ``ResponsesClient`` does not use this
    function: its requests always use the ``responses`` interface.
    """
    return "anthropic" if provider == "anthropic" else "chat"


def implemented_input_kinds(
    interface: str = "chat", provider: str | None = None
) -> tuple[MediaInput, ...]:
    """Return the media kinds NOOA can send on ``interface``.

    ``provider`` lets a caller pass the LiteLLM provider instead of an interface
    name; ``anthropic`` gives the native Anthropic limits. The result is an upper
    bound for the request code, not proof that a model accepts these kinds.
    """
    if interface == "decision":
        return ("image",)
    if interface == "responses" or interface == "anthropic" or provider == "anthropic":
        return ("image", "pdf")
    return ALL_INPUT_KINDS


def resolved_input_interface(
    model: str, config: Mapping[str, Any], *, client_type: str = "completion"
) -> str:
    """Return the interface a client built from ``model`` and ``config`` would use.

    ``client_type`` ``responses`` and ``decision`` fix the interface directly. For
    completion clients, LiteLLM resolves the provider from the model, the endpoint
    (``api_base`` or ``base_url``) and ``custom_llm_provider``, exactly as it does
    when sending. No network call, key lookup or metadata is used. If the provider
    cannot be resolved (for example an incomplete profile), the result is ``chat``,
    which does not restrict any media kind.
    """
    if client_type in {"responses", "decision"}:
        return client_type
    if not model:
        return "chat"
    import litellm

    try:
        _, provider, _, _ = litellm.get_llm_provider(
            model=model,
            api_base=config.get("api_base") or config.get("base_url"),
            custom_llm_provider=config.get("custom_llm_provider"),
        )
    except Exception:
        # LiteLLM raises for model strings it cannot route; treat them as Chat,
        # which is also how CompletionClient would try to send them.
        return "chat"
    return interface_for_provider(provider)


def entry_input_interface(entry: Mapping[str, Any]) -> str:
    """Return the interface for a registry entry (``model_name`` plus settings)."""
    return resolved_input_interface(
        entry.get("model_name") or "",
        entry,
        client_type=entry.get("client_type") or "completion",
    )


def unsupported_input_reason(interface: str, kinds) -> str:
    """Return the error message for media ``kinds`` that ``interface`` cannot send."""
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
    """Validate a ``media_inputs`` setting and check every kind fits ``interface``.

    Returns the normalized tuple. Raises ``ValueError`` if ``value`` is not a valid
    ``media_inputs`` setting or lists a kind ``interface`` cannot send.
    """
    enabled = validate_media_inputs(value)
    unsupported = [kind for kind in enabled if kind not in implemented_input_kinds(interface)]
    if unsupported:
        raise ValueError(unsupported_input_reason(interface, unsupported))
    return enabled


def reject_unsupported_media(messages, interface: str) -> None:
    """Raise ``ValueError`` if ``messages`` attach a kind ``interface`` cannot send.

    Call this after disabled media has been filtered out, so only media the
    caller allowed is checked. The check looks at the places a request carries
    attachments: a message that is itself an attachment item, an assistant
    ``audio`` field, and the blocks of ``content`` or ``output`` lists. It does not
    inspect tool arguments, reasoning or other nested JSON.
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
        # Responses input can contain attachment items at the top level, beside
        # role/content messages.
        if (kind := attachment_kind(message)) is not None:
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
