# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Internal names stay out of the model's python_cell context and <self> block."""

import re

from atom_test_agents import CellLLM
from nooa_atom.agent.agent import AtomAgent, AtomCodeAct

from nooa.agentdoc import doc
from nooa.runtime.channels import ChannelReader

# Module-level helpers of agent.py and plumbing attributes; none is model API.
LEAKS = ("get_harness_metrics", "import re", "ProducersSkill", "timeout_hint")


async def test_rendered_cell_context_and_self_have_no_internal_names(tmp_path):
    agent = AtomAgent(cwd=tmp_path, llm=CellLLM([]))
    try:
        context = await AtomCodeAct().python_cell_context(agent.runtime)
        rendered = doc(agent)
    finally:
        await agent.aclose()
    for name in LEAKS:
        assert name not in context, name
        assert name not in rendered, name
    assert not re.search(r"^\s*producers:", rendered, re.MULTILINE)


def test_channel_reader_doc_hides_the_owner_timeout_hint():
    assert "timeout_hint" not in doc(ChannelReader)
