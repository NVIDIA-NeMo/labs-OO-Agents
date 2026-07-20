# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Core runtime tests - Layer 2: Runtime (pure infrastructure).

Tests focus on:
- Task/signal queuing (serialized generation, concurrent execution)
- Code caching (ONCE vs AGENT lifetime)
- Code execution via sandbox
- Task introspection (TaskWrapper API)
- Child runtime creation
- Event logging
"""
