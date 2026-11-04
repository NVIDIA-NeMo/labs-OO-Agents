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
video. Which interface a client uses is decided in ``nooa.request_interface``.
"""

from collections.abc import Mapping

from nooa.media_inputs import MediaInput, attachment_kind, validate_media_inputs

# Interface resolution lives in nooa.request_interface; these names stay
# importable from here for existing callers.
from nooa.request_interface import entry_input_interface as entry_input_interface
from nooa.request_interface import interface_for_provider as interface_for_provider
from nooa.request_interface import resolved_input_interface as resolved_input_interface

ALL_INPUT_KINDS: tuple[MediaInput, ...] = ("image", "audio", "video", "pdf")


def implemented_input_kinds(interface: str = "chat") -> tuple[MediaInput, ...]:
    """Return the media kinds NOOA can send on ``interface``.

    The result is an upper bound for the request code, not proof that a model
    accepts these kinds. Use interface_for_provider() to map a LiteLLM provider
    to its interface first.
    """
    if interface == "decision":
        return ("image",)
    if interface in ("responses", "anthropic"):
        return ("image", "pdf")
    return ALL_INPUT_KINDS


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
