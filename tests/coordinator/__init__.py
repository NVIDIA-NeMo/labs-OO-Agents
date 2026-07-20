# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coordinator tests - Layer 3: Generation Coordinator.

Tests focus on:
- Code generation via LLM
- REPL exploration loops
- Code validation (AST, forbidden features)
- Retry on validation failures
"""
