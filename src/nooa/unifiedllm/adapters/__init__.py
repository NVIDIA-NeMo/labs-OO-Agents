# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One adapter per wire format; the only place for provider-specific code."""

from nooa.unifiedllm.adapters.base import Adapter, Capabilities

__all__ = ["Adapter", "Capabilities"]
