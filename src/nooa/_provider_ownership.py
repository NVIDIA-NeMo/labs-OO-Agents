# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Private call-bound LiteLLM journal ownership, never caller metadata.

The marker lives solely on the SDK Logging object's details. Unlike an ambient
scope or transient call-ID registry, it survives copied/redacted callback details,
fresh worker contexts, and callbacks delivered after the provider scope exits.
Only built-in transports holding an explicit one-send owner may prepare it.
"""

from contextlib import contextmanager

_OWNED_LOGGING_KEY = "_nooa_logical_call_owned"


def _is_owned_provider_details(details: dict) -> bool:
    return details.get(_OWNED_LOGGING_KEY) is True


@contextmanager
def _owned_provider_region(kwargs, *, owner, call_type, prepare):
    """Scoped OTel suppression plus durable SDK journal ownership at entry.

    Async setup precedes SDK cache-handler construction (unlike the ordinary
    async wrapper); the constructor itself remains exclusively SDK-owned.
    """
    if owner is None or owner.writer.record.logical_id is None:
        yield kwargs
        return
    from nooa.unifiedllm._accounting import _current_account

    account = _current_account.get()
    if account is not None:
        account.describe_request(model=kwargs.get("model"))
    prepared = prepare(kwargs, owner=owner, call_type=call_type)
    from opentelemetry.context import (
        _SUPPRESS_INSTRUMENTATION_KEY,
        attach,
        detach,
        set_value,
    )

    token = attach(set_value(_SUPPRESS_INSTRUMENTATION_KEY, True))
    try:
        yield prepared
    finally:
        detach(token)
