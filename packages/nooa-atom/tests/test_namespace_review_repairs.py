# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict fake-model regressions for independently reproduced core defects."""

import pytest
from atom_test_agents import CellLLM, cell
from nooa_atom.agent.agent import AtomAgent, AtomCodeAct

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.interactive import Done
from nooa.strategies import CodeActV2


class LegacyChild(Agent):
    @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
    async def answer(self, saved) -> int:
        """Complete this generation only with its own capability."""
        ...


class OwnedChild(AtomAgent):
    @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
    async def answer(self, saved) -> int:
        """Complete this generation only with its own capability."""
        ...


@pytest.mark.parametrize(
    "outer_timeout,child_timeout", [(None, None), (5, None), (None, 5), (5, 5)]
)
@pytest.mark.parametrize("child_kind", ["non_atom", "atom"])
async def test_completion_checks_actual_generation_for_every_engine_and_timeout(
    tmp_path,
    outer_timeout,
    child_timeout,
    child_kind,
):
    child_llm = CellLLM(
        [
            cell(
                "print('innermost call:', self.runtime.current_call.method_name)\n"
                "try:\n saved(73)\nexcept RuntimeError:\n return_result(-1)"
            )
        ]
    )
    child = (
        LegacyChild(llm=child_llm)
        if child_kind == "non_atom"
        else OwnedChild(cwd=tmp_path, llm=child_llm)
    )
    outer_llm = CellLLM(
        [
            cell(
                "saved=return_result\nvalue=await self.child.answer(saved, _strategy=self.child_strategy)\nassert value == -1, value"
            ),
            # An Atom cell ends with ``return``: its return_result alias completes
            # nothing, in a nested generation or later (NameError with the hint).
            cell(
                "try:\n saved(Done(explanation='alias'))\nexcept NameError:\n"
                " return Done(explanation=str(value))"
            ),
        ]
    )
    outer = AtomAgent(cwd=tmp_path, llm=outer_llm)
    outer.child = child
    outer.child_strategy = CodeActV2(config=CodeActConfig(prefill=None, cell_timeout=child_timeout))
    try:
        result = await outer.handle(
            {"user_messages": ["probe"]},
            _strategy=AtomCodeAct(config=CodeActConfig(prefill=None, cell_timeout=outer_timeout)),
        )
        assert result == Done(explanation="-1")
        assert len(outer_llm.calls) == 2 and len(child_llm.calls) == 1
        outputs = list(child.events.query(type="PythonOutput"))
        assert len(outputs) == 1 and not outputs[0].error
        assert outputs[0].stdout == "innermost call: answer\n"
    finally:
        await outer.aclose()
        await child.aclose()


def fresh_class():
    # Deliberately no real skill install and no module-global class export.
    return type("FreshSkill", (), {"__module__": "dynamic_skill_example"})


async def test_newly_advertised_dynamic_class_is_available_in_default_session(tmp_path):
    llm = CellLLM(
        [
            cell("return Done(explanation='init')"),
            cell("assert isinstance(self.fresh, FreshSkill)\nreturn Done(explanation='available')"),
        ]
    )
    agent = AtomAgent(cwd=tmp_path, llm=llm)
    try:
        await agent.handle({"user_messages": ["one"]})
        owner = agent._handle_namespace
        agent.fresh = fresh_class()()
        assert "FreshSkill" in await AtomCodeAct().python_cell_context(agent.runtime)
        assert await agent.handle({"user_messages": ["two"]}) == Done(explanation="available")
        assert owner is agent._handle_namespace
        assert "FreshSkill" not in owner.globals  # framework fallback, not user state
    finally:
        await agent.aclose()


async def test_provider_replacement_and_deactivation_preserve_alias_and_globals_identity(tmp_path):
    llm = CellLLM(
        [
            cell("def lookup():\n return FreshSkill\nreturn Done(explanation='defined')"),
            cell(
                "alias=FreshSkill\nassert lookup() is type(self.fresh)\nreturn Done(explanation='A')"
            ),
            cell(
                "assert lookup() is type(self.fresh) and lookup() is not alias\nreturn Done(explanation='B')"
            ),
            cell("try:\n lookup()\nexcept NameError:\n return Done(explanation='gone')"),
        ]
    )
    agent = AtomAgent(cwd=tmp_path, llm=llm)
    try:
        await agent.handle({"user_messages": ["one"]})
        ns = agent._handle_namespace
        globals_id = id(ns.globals)
        first = fresh_class()
        agent.fresh = first()
        await agent.handle({"user_messages": ["two"]})
        agent.fresh = fresh_class()()
        await agent.handle({"user_messages": ["three"]})
        del agent.fresh
        assert await agent.handle({"user_messages": ["four"]}) == Done(explanation="gone")
        assert ns.globals["alias"] is first
        assert id(ns.globals) == globals_id and ns.globals["lookup"].__globals__ is ns.globals
    finally:
        await agent.aclose()


@pytest.mark.parametrize(
    "assignment", ["FreshSkill=FreshSkill", "FreshSkill='user'", "import math as FreshSkill"]
)
async def test_user_store_same_object_or_import_wins_on_replacement_and_removal(
    tmp_path, assignment
):
    llm = CellLLM(
        [
            cell(
                assignment + "\ndef lookup():\n return FreshSkill\nreturn Done(explanation='owned')"
            ),
            cell("assert lookup() is FreshSkill\nreturn Done(explanation='replaced provider')"),
            cell("assert lookup() is FreshSkill\nreturn Done(explanation='removed provider')"),
        ]
    )
    agent = AtomAgent(cwd=tmp_path, llm=llm)
    agent.fresh = fresh_class()()
    try:
        await agent.handle({"user_messages": ["one"]})
        ns = agent._handle_namespace
        user_value = ns.globals["FreshSkill"]
        agent.fresh = fresh_class()()
        await agent.handle({"user_messages": ["two"]})
        assert ns.globals["FreshSkill"] is user_value
        del agent.fresh
        await agent.handle({"user_messages": ["three"]})
        assert ns.globals["FreshSkill"] is user_value
        assert ns.globals["lookup"]() is user_value
    finally:
        await agent.aclose()


@pytest.mark.parametrize("provider_present", [False, True])
async def test_explicit_deletion_tombstone_prevents_reactivation_but_not_explicit_store(
    tmp_path, provider_present
):
    initial = (
        "alias=FreshSkill\ndel FreshSkill" if provider_present else "FreshSkill=42\ndel FreshSkill"
    )
    llm = CellLLM(
        [
            cell(initial + "\nreturn Done(explanation='deleted')"),
            cell("try:\n FreshSkill\nexcept NameError:\n return Done(explanation='still deleted')"),
            cell("FreshSkill=type(self.fresh)\nreturn Done(explanation='explicitly rebound')"),
            cell("assert FreshSkill is restored\nreturn Done(explanation='kept')"),
        ]
    )
    agent = AtomAgent(cwd=tmp_path, llm=llm)
    original = fresh_class()
    if provider_present:
        agent.fresh = original()
    try:
        await agent.handle({"user_messages": ["one"]})
        ns = agent._handle_namespace
        agent.fresh = fresh_class()()
        assert "FreshSkill" not in await AtomCodeAct().python_cell_context(agent.runtime)
        assert await agent.handle({"user_messages": ["two"]}) == Done(explanation="still deleted")
        await agent.handle({"user_messages": ["three"]})
        ns.globals["restored"] = ns.globals["FreshSkill"]
        del agent.fresh
        await agent.handle({"user_messages": ["four"]})
        if provider_present:
            assert ns.globals["alias"] is original
    finally:
        await agent.aclose()


class ChangingTools(AtomAgent):
    def activate_fresh(self):
        self.fresh = fresh_class()()

    def deactivate_fresh(self):
        del self.fresh


async def test_dynamic_availability_updates_between_cells_of_one_handle(tmp_path):
    llm = CellLLM(
        [
            cell("self.activate_fresh()"),
            cell(
                "alias=FreshSkill\nassert isinstance(self.fresh, FreshSkill)\nself.deactivate_fresh()"
            ),
            cell(
                "try:\n FreshSkill\nexcept NameError:\n return Done(explanation='gone in same call')"
            ),
        ]
    )
    agent = ChangingTools(cwd=tmp_path, llm=llm)
    try:
        assert await agent.handle({"user_messages": ["go"]}) == Done(
            explanation="gone in same call"
        )
        assert len(llm.calls) == 3 and "alias" in agent._handle_namespace.globals
    finally:
        await agent.aclose()


@pytest.mark.parametrize("raw_present", [False, True])
async def test_private_global_deletion_in_actual_atom_handle(tmp_path, raw_present):
    seed = "__x='unrelated raw'\n" if raw_present else ""
    first = (
        seed
        + """
class C:
    def drop():
        def inner():
            global __x
            __x = 'private'
            del __x
        inner()
C.drop()
return Done(explanation='private deleted')
"""
    )
    second = (
        "assert __x == 'unrelated raw'\n"
        if raw_present
        else "try:\n __x\nexcept NameError:\n pass\nelse:\n raise AssertionError('unexpected raw binding')\n"
    )
    second += "try:\n _C__x\nexcept NameError:\n return Done(explanation='correct bindings')"
    llm = CellLLM([cell(first), cell(second)])
    agent = AtomAgent(cwd=tmp_path, llm=llm)
    try:
        assert await agent.handle({"user_messages": ["one"]}) == Done(explanation="private deleted")
        assert await agent.handle({"user_messages": ["two"]}) == Done(
            explanation="correct bindings"
        )
        ns = agent._handle_namespace
        assert ns.deleted_names == {"_C__x"}
        assert "del __x" in ns.globals["C"].drop._generated_source
        assert "_nooa_ns_delete" not in ns.globals["C"].drop._generated_source
        assert len(llm.calls) == 2
    finally:
        await agent.aclose()


_COLLISIONS = [
    "self",
    "pprint",
    "show",
    "notification",
    "return_result",
    "python_cell_state",
    "print",
    "Out",
    "__builtins__",
    "_nooa_ns_source",
    "_nooa_ns_probe",
    "OrdinaryDynamic",
]


@pytest.mark.parametrize("name", _COLLISIONS)
async def test_runtime_binding_and_advertisement_resist_dynamic_provider_collisions(tmp_path, name):
    # Host receives the actual captured objects, not only a type-name assertion.
    captures = []
    checks = """
class FunctionCreationCheck:
    def method(self):
        return 'metadata still works'
assert FunctionCreationCheck().method() == 'metadata still works'
self.capture(notification, self, print, pprint, show, return_result, python_cell_state, Out)
"""
    if name == "OrdinaryDynamic":
        checks += "assert isinstance(self.tool, OrdinaryDynamic)\n"
    checks += "return Done(explanation='runtime bindings win')"
    llm = CellLLM([cell("return Done(explanation='init')"), cell(checks)])
    agent = CapturingAtom(cwd=tmp_path, llm=llm, captures=captures)
    try:
        await agent.handle({"user_messages": ["one"]})
        provider = type(name, (), {"__module__": "dynamic_probe"})
        agent.tool = provider()
        context = await AtomCodeAct().python_cell_context(agent.runtime)
        assert ("dynamic_probe" in context) is (name == "OrdinaryDynamic")
        inputs = {"user_messages": ["two"]}
        assert await agent.handle(inputs) == Done(explanation="runtime bindings win")
        [captured] = captures
        assert captured[0] is inputs and captured[1] is agent
        assert all(value is not provider for value in captured)
        import builtins

        from nooa.runtime.media_capture import show
        from nooa.runtime.out_accessor import OutAccessor
        from nooa.runtime.pprint import pprint

        assert captured[2] is builtins.print and captured[3] is pprint and captured[4] is show
        assert (
            callable(captured[5]) and callable(captured[6]) and isinstance(captured[7], OutAccessor)
        )
        assert isinstance(captured[6](), dict)
        assert len(llm.calls) == 2
    finally:
        await agent.aclose()


class CapturingAtom(AtomAgent):
    def __init__(self, *, captures, **kwargs):
        super().__init__(**kwargs)
        self.captures = captures

    def capture(self, *values):
        self.captures.append(values)


class ParameterCollisionAgent(Agent):
    @strategy(CodeActV2(config=CodeActConfig(prefill=None)))
    async def answer(self, payload: dict) -> str:
        """Use the genuine payload, not a discovered provider named payload."""
        ...


async def test_current_call_parameters_keep_identity_and_are_not_provider_advertised():
    llm = CellLLM([cell("assert payload is self.input_payload\nreturn payload['token']")])
    agent = ParameterCollisionAgent(llm=llm)
    agent.tool = type("payload", (), {"__module__": "dynamic_probe"})()
    agent.input_payload = {"token": "genuine input"}
    try:
        assert await agent.answer(agent.input_payload) == "genuine input"
        assert "dynamic_probe" not in str(llm.calls[0].messages)
    finally:
        await agent.aclose()


async def test_shared_autoimport_fix_preserves_explicit_non_atom_user_binding():
    llm = CellLLM(
        [
            cell("print = 'explicit user binding'"),
            cell("assert print == 'explicit user binding'\nreturn 7"),
        ]
    )
    agent = LegacyChild(llm=llm)
    agent.tool = type("print", (), {"__module__": "dynamic_probe"})()
    try:
        assert await agent.answer(None) == 7
        assert len(llm.calls) == 2
    finally:
        await agent.aclose()
