# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for visibility consistency between doc(self) and exec_globals.

Verifies that module-level types and constants are visible in exec_globals
(simplified model: everything visible by default). Also verifies @hidden
methods/fields and that types used only in hidden API are not required to
be in the module.
"""

import inspect

import pytest
from pydantic import BaseModel

from nooa import Agent, hidden
from nooa.agentdoc import doc
from nooa.agentdoc.visibility import filter_module_globals
from nooa.runtime.actor import ActorRuntime
from nooa.unifiedllm import FakeLLMClient

_TEST_LLM = FakeLLMClient()


# ── Module-level types and constants (visible by default in simplified model) ──


class VisibleResult(BaseModel):
    item: str
    quantity: int


SOME_CONSTANT = 42


# ═════════════════════════════════════════════════════════════════════════════
# Module-level types are visible in exec_globals
# ═════════════════════════════════════════════════════════════════════════════


class TestVisibleTypesAccepted:
    """Agent classes using only visible types should be created without error."""

    def test_visible_return_type_accepted(self):
        class GoodAgent(Agent, llm=_TEST_LLM):
            async def process(self, request: str) -> VisibleResult:
                """Process {request}."""
                ...

    def test_visible_field_type_accepted(self):
        class GoodAgent(Agent, llm=_TEST_LLM):
            data: VisibleResult = VisibleResult(item="x", quantity=0)

    def test_visible_parameter_type_accepted(self):
        class GoodAgent(Agent, llm=_TEST_LLM):
            async def process(self, data: VisibleResult) -> str:
                """Process {data}."""
                ...

    def test_builtin_types_always_accepted(self):
        class GoodAgent(Agent, llm=_TEST_LLM):
            items: list[str] = []
            count: int = 0

            async def process(self, text: str) -> dict[str, int]:
                """Process {text}."""
                ...

    def test_visible_type_usable_in_generated_code(self):
        class GoodAgent(Agent, llm=_TEST_LLM):
            data: VisibleResult = VisibleResult(item="default", quantity=0)

            async def process(self, request: str) -> VisibleResult:
                """Process {request}."""
                ...

        agent = GoodAgent()
        doc_output = doc(agent)
        assert "VisibleResult" in doc_output

        module = inspect.getmodule(GoodAgent)
        filtered = filter_module_globals(module)
        assert "VisibleResult" in filtered

    @pytest.mark.asyncio
    async def test_visible_type_constructable_in_runtime(self):
        class GoodAgent(Agent, llm=_TEST_LLM):
            data: VisibleResult = VisibleResult(item="default", quantity=0)

        agent = GoodAgent()
        runtime = ActorRuntime(agent)
        code = 'self.data = VisibleResult(item="widget", quantity=5)'
        result = await runtime.execute_code(code)

        assert result.error is None
        assert agent.data.item == "widget"


# ═════════════════════════════════════════════════════════════════════════════
# ═════════════════════════════════════════════════════════════════════════════
# Module-level types (e.g. HiddenResult) are visible in filter_module_globals
# ═════════════════════════════════════════════════════════════════════════════


class HiddenResult(BaseModel):
    """Module-level type; visible in exec_globals by default."""

    item: str
    quantity: int


class AnotherHiddenType(BaseModel):
    value: float


class TestModuleLevelTypesInFilteredGlobals:
    """Types defined at module level are visible in filter_module_globals (simplified model)."""

    def test_return_type_defined_in_module_in_filtered(self):
        class PromoAgent(Agent, llm=_TEST_LLM):
            async def process(self, request: str) -> HiddenResult:
                """Process {request}."""
                ...

        module = inspect.getmodule(PromoAgent)
        filtered = filter_module_globals(module)
        assert "HiddenResult" in filtered
        assert filtered["HiddenResult"] is HiddenResult

    def test_parameter_type_defined_in_module_in_filtered(self):
        class PromoAgent(Agent, llm=_TEST_LLM):
            async def process(self, data: HiddenResult) -> str:
                """Process {data}."""
                ...

        module = inspect.getmodule(PromoAgent)
        filtered = filter_module_globals(module)
        assert "HiddenResult" in filtered

    def test_field_type_defined_in_module_in_filtered(self):
        class PromoAgent(Agent, llm=_TEST_LLM):
            data: HiddenResult = HiddenResult(item="x", quantity=0)

        module = inspect.getmodule(PromoAgent)
        filtered = filter_module_globals(module)
        assert "HiddenResult" in filtered

    def test_nested_type_in_signature_in_filtered(self):
        class PromoAgent(Agent, llm=_TEST_LLM):
            async def process(self, request: str) -> list[HiddenResult]:
                """Process {request}."""
                ...

        module = inspect.getmodule(PromoAgent)
        filtered = filter_module_globals(module)
        assert "HiddenResult" in filtered

    def test_multiple_module_types_in_filtered(self):
        class PromoAgent(Agent, llm=_TEST_LLM):
            data: HiddenResult = HiddenResult(item="x", quantity=0)

            async def process(self, request: str) -> HiddenResult:
                """Process {request}."""
                ...

        module = inspect.getmodule(PromoAgent)
        filtered = filter_module_globals(module)
        assert "HiddenResult" in filtered

    @pytest.mark.asyncio
    async def test_module_type_usable_at_runtime(self):
        class PromoAgent(Agent, llm=_TEST_LLM):
            data: HiddenResult = HiddenResult(item="x", quantity=0)

            async def process(self, request: str) -> HiddenResult:
                """Process {request}."""
                ...

        agent = PromoAgent()
        runtime = ActorRuntime(agent)
        code = 'self.data = HiddenResult(item="widget", quantity=5)'
        result = await runtime.execute_code(code)
        assert result.error is None
        assert agent.data.item == "widget"
        assert agent.data.quantity == 5


class NestedAddress(BaseModel):
    """Nested type: referenced by OrderWithAddress."""

    street: str
    city: str


class OrderWithAddress(BaseModel):
    """Uses NestedAddress in a field; both are module-level so in filtered."""

    item: str
    quantity: int
    address: NestedAddress


class TestNestedModuleTypesInFilteredGlobals:
    """Nested types defined at module level are visible in filter_module_globals."""

    def test_nested_type_in_filtered(self):
        class PromoAgent(Agent, llm=_TEST_LLM):
            async def process(self, request: str) -> OrderWithAddress:
                """Process {request}."""
                ...

        module = inspect.getmodule(PromoAgent)
        filtered = filter_module_globals(module)
        assert "OrderWithAddress" in filtered
        assert "NestedAddress" in filtered
        assert filtered["OrderWithAddress"] is OrderWithAddress
        assert filtered["NestedAddress"] is NestedAddress


# ═════════════════════════════════════════════════════════════════════════════
# @hidden methods and Annotated[T, hidden] fields bypass the check
# ═════════════════════════════════════════════════════════════════════════════


class TestHiddenBypassesCheck:
    """@hidden methods and Annotated[T, hidden] fields are excluded from doc/exec_globals."""

    def test_hidden_method_with_nonvisible_type_accepted(self):
        class OkAgent(Agent, llm=_TEST_LLM):
            @hidden
            async def internal(self, request: str) -> HiddenResult:
                """Internal processing of {request}."""
                ...

    def test_hidden_field_with_nonvisible_type_accepted(self):
        from typing import Annotated

        class OkAgent(Agent, llm=_TEST_LLM):
            data: Annotated[HiddenResult, hidden] = HiddenResult(item="x", quantity=0)

    def test_hidden_method_with_module_type_accepted(self):
        """Types used only in @hidden methods: agent still works; type is module-level so in filtered."""

        class OkAgent(Agent, llm=_TEST_LLM):
            @hidden
            async def internal(self, request: str) -> AnotherHiddenType:
                """Internal processing of {request}."""
                ...

        module = inspect.getmodule(OkAgent)
        filtered = filter_module_globals(module)
        assert "AnotherHiddenType" in filtered

    def test_mix_of_hidden_and_visible(self):
        """Hidden method with non-visible type + visible method with visible type."""
        from typing import Annotated

        class OkAgent(Agent, llm=_TEST_LLM):
            data: Annotated[HiddenResult, hidden] = HiddenResult(item="x", quantity=0)
            visible_data: VisibleResult = VisibleResult(item="y", quantity=1)

            @hidden
            async def internal(self, request: str) -> HiddenResult:
                """Internal processing of {request}."""
                ...

            async def public_method(self, request: str) -> VisibleResult:
                """Public processing of {request}."""
                ...


# ═════════════════════════════════════════════════════════════════════════════
# Module-level filter_module_globals (simplified: all visible by default)
# ═════════════════════════════════════════════════════════════════════════════


class TestModuleGlobalsFiltering:
    """filter_module_globals includes all module names by default."""

    def setup_method(self):
        self.module = inspect.getmodule(TestModuleGlobalsFiltering)
        self.filtered = filter_module_globals(self.module)

    def test_visible_type_in_filtered_globals(self):
        assert "VisibleResult" in self.filtered
        assert self.filtered["VisibleResult"] is VisibleResult

    def test_visible_constant_in_filtered_globals(self):
        assert "SOME_CONSTANT" in self.filtered
