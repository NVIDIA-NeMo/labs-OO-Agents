# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load the legacy dependency only when a legacy client needs it."""

import importlib

_initialized = False


def _module():
    global _initialized
    module = importlib.import_module("litellm")
    if not _initialized:
        module.modify_params = True
        module.disable_aiohttp_transport = True
        _initialized = True
    return module


class _LazyLiteLLM:
    def __getattr__(self, name):
        return getattr(_module(), name)

    def __setattr__(self, name, value):
        setattr(_module(), name, value)

    def __delattr__(self, name):
        delattr(_module(), name)


litellm = _LazyLiteLLM()
