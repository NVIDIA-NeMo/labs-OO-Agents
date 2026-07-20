# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Smoke tests to ensure agents can import successfully.

These tests verify that the agents and their dependencies are correctly installed
and can be imported without errors. They don't test functionality, just basic
import/initialization capability.
"""

import os
import sys
from pathlib import Path

import pytest


def test_nooa_runtime_imports():
    """Test that nooa runtime with agentdoc dependency works."""
    from nooa.runtime.actor import ActorRuntime
    from nooa.tracing import enable_tracing

    assert enable_tracing is not None
    assert ActorRuntime is not None


def test_agentdoc_imports():
    """Test that agentdoc package is installed and importable."""
    from nooa.agentdoc import doc
    from nooa.agentdoc.introspect import methods, variables

    assert doc is not None
    assert methods is not None
    assert variables is not None


@pytest.mark.skipif(
    not os.getenv("SLACK_BOT_TOKEN"),
    reason="SLACK_BOT_TOKEN not set - required for LibrarianAgent initialization",
)
def test_librarian_agent_imports():
    """Test that LibrarianAgent can be imported."""
    # Add agents directory to path
    agents_path = Path(__file__).parent.parent.parent / "agents" / "librarian-agent"
    sys.path.insert(0, str(agents_path))

    from librarian_agent import LibrarianAgent, generate_trace_id, get_trace_link

    assert LibrarianAgent is not None
    assert generate_trace_id is not None
    assert get_trace_link is not None


@pytest.mark.skipif(
    not os.getenv("SLACK_BOT_TOKEN"),
    reason="SLACK_BOT_TOKEN not set - required for TPMAgent initialization",
)
def test_tpm_agent_imports():
    """Test that TPM agent can be imported."""
    # Add agents directory to path
    agents_path = Path(__file__).parent.parent.parent / "agents" / "tpm-agent"
    sys.path.insert(0, str(agents_path))

    from runner import TPMAgentRunner

    assert TPMAgentRunner is not None
