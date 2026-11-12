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

# Compatibility exports: pure interface policy has no provider resolver dependency.
from nooa.media_inputs import MediaInput as MediaInput
from nooa.media_inputs import attachment_kind as attachment_kind
from nooa.media_inputs import validate_media_inputs as validate_media_inputs
from nooa.media_interfaces import ALL_INPUT_KINDS as ALL_INPUT_KINDS
from nooa.media_interfaces import implemented_input_kinds as implemented_input_kinds
from nooa.media_interfaces import reject_unsupported_media as reject_unsupported_media
from nooa.media_interfaces import unsupported_input_reason as unsupported_input_reason
from nooa.media_interfaces import (
    validate_implemented_media_inputs as validate_implemented_media_inputs,
)
from nooa.request_interface import entry_input_interface as entry_input_interface
from nooa.request_interface import interface_for_provider as interface_for_provider
from nooa.request_interface import resolved_input_interface as resolved_input_interface
from nooa.request_interface import resolved_provider as resolved_provider
