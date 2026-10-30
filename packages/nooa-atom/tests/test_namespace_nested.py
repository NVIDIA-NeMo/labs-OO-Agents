# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.agent import AtomAgent

from nooa import strategy
from nooa.config import CodeActConfig
from nooa.interactive import Done
from nooa.strategies import CodeActV2


class NestedAtom(AtomAgent):
    @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
    async def other(self) -> int:
        """Use an independent Python namespace; never delete handle's variables."""
        ...


async def test_nested_generated_method_has_independent_owner_and_del(tmp_path):
    llm = CellLLM(
        [
            cell(
                "scratch=7\nanswer=await self.other()\nassert answer == 9 and scratch == 7\nreturn_result(Done(explanation='isolated'))"
            ),
            cell(
                "try:\n scratch\nexcept NameError:\n pass\nelse:\n raise AssertionError('leaked')\nscratch=9\nanswer=scratch\ndel scratch\nreturn_result(answer)"
            ),
            cell("assert scratch == 7\nreturn_result(Done(explanation='retained'))"),
        ]
    )
    agent = NestedAtom(cwd=tmp_path, llm=llm)
    try:
        assert await agent.handle({"user_messages": ["one"]}) == Done(explanation="isolated")
        assert await agent.handle({"user_messages": ["two"]}) == Done(explanation="retained")
        assert len(llm.calls) == 3
    finally:
        await agent.aclose()
