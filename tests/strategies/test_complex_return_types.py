# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit guards for complex / nested / non-serializable return types (no network).

Covers gaps found while hardening structured output for nested types (KDD-cup
feedback): nested Pydantic models, subtype reuse, ``dict[str, Model]`` value typing,
and non-JSON-serializable types (``pandas.DataFrame`` / ``numpy.ndarray``) — including
Pydantic models that *contain* such a field.

Strategy split:
- **PredictStrategy** needs a JSON schema, so non-serializable types fail at model
  build with a clear, actionable error.
- **CodeActStrategy** routes non-serializable types to an ``Any`` tool schema and lets
  the model construct the value in ``execute_python`` — so it must NOT crash building
  the tool schema for them.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ConfigDict

from nooa import Agent
from nooa.decorators import strategy
from nooa.errors import GenerationError
from nooa.strategies import PredictStrategy
from nooa.strategies.codeact import CodeActStrategy
from nooa.unifiedllm import FakeLLMClient
from nooa.unifiedllm.unifiedllm import _resolve_schema_refs, _schema_strict_compatible

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


# ── Fixtures: nested + subtype reuse + non-serializable field ───────────────
class Address(BaseModel):
    street: str
    city: str


class Person(BaseModel):
    name: str
    address: Address  # nested model


class Route(BaseModel):
    origin: Address  # subtype reused twice
    destination: Address


class Report(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    title: str
    table: pd.DataFrame  # non-JSON-serializable nested field


# ── PredictStrategy ─────────────────────────────────────────────────────────
class TestPredictDictValueTyping:
    """`dict[str, Person]` must validate/construct values as Person, not leave dicts."""

    def test_dict_of_models_constructs_values(self):
        s = PredictStrategy()
        model = s._create_response_model(dict[str, Person], "m")
        out = s._validate_response(
            {"london": {"name": "Ada", "address": {"street": "1 King", "city": "London"}}},
            model,
            dict[str, Person],
        )
        assert isinstance(out, dict)
        assert all(isinstance(v, Person) for v in out.values())
        assert out["london"].address.city == "London"

    def test_bare_dict_still_accepts_any(self):
        s = PredictStrategy()
        model = s._create_response_model(dict, "m")
        out = s._validate_response({"a": 1, "b": "x"}, model, dict)
        assert out == {"a": 1, "b": "x"}


class TestPredictNonSerializable:
    """PredictStrategy can't schema DataFrame/ndarray (or models containing them)."""

    def test_dataframe_raises_actionable_error(self):
        s = PredictStrategy()
        with pytest.raises(GenerationError) as ei:
            s._create_response_model(pd.DataFrame, "blah")
        msg = str(ei.value)
        assert "CodeActStrategy" in msg  # points to the supported path
        assert "columns" in msg  # suggests a serializable proxy

    @pytest.mark.asyncio
    async def test_model_with_dataframe_field_raises(self):
        # A Pydantic model with a non-serializable field passes _create_response_model
        # (returned as-is) but the runtime schema probe in _execute_inner must reject it
        # with the clear error — before any LLM call (FakeLLMClient has no scripts).
        class DFAgent(Agent, llm=FakeLLMClient()):
            @strategy(PredictStrategy())
            async def build(self) -> Report:
                """Return a report."""
                ...

        with pytest.raises(GenerationError) as ei:
            await DFAgent().build()
        assert "CodeActStrategy" in str(ei.value)


class TestNestedSchemaRouting:
    """Nested/reused models stay strict-compatible; free-form dicts do not."""

    def _schema(self, return_type):
        m = PredictStrategy()._create_response_model(return_type, "m")
        return _resolve_schema_refs(m.model_json_schema())

    def test_nested_model_strict_compatible(self):
        assert _schema_strict_compatible(self._schema(Person))

    def test_reused_subtype_strict_compatible(self):
        assert _schema_strict_compatible(self._schema(Route))

    def test_dict_of_models_not_strict_compatible(self):
        # free-form object (additionalProperties) → routed to non-strict
        assert not _schema_strict_compatible(self._schema(dict[str, Person]))


# ── CodeActStrategy ───────────────────────────────────────────────────────
class TestCodeActPydanticCompat:
    """The return_result tool must fall back to Any for non-serializable types."""

    def test_serializable_types_compatible(self):
        s = CodeActStrategy()
        for t in (int, str, list[str], dict[str, int], Person, Route):
            assert s._is_pydantic_compatible(t), t

    def test_dataframe_and_ndarray_not_compatible(self):
        s = CodeActStrategy()
        assert not s._is_pydantic_compatible(pd.DataFrame)
        assert not s._is_pydantic_compatible(np.ndarray)

    def test_model_with_nonserializable_field_not_compatible(self):
        # Regression: previously True (only create_model was probed), then the tool
        # schema build crashed at model_json_schema() time. Must be False → Any fallback.
        assert not CodeActStrategy()._is_pydantic_compatible(Report)

    def test_build_return_result_tool_for_nested_df_does_not_crash(self):
        s = CodeActStrategy()
        tool = s._build_return_result_tool(Report, "m")
        schema = tool.get_parameter_schema()  # must not raise
        # falls back to Any → the result property carries no nested DataFrame schema
        assert "result" in schema["properties"]
        assert f"{Report.__name__}" in tool.description  # type hint still surfaced

    def test_build_return_result_tool_for_nested_model_keeps_schema(self):
        import json

        s = CodeActStrategy()
        tool = s._build_return_result_tool(Person, "m")
        schema = tool.get_parameter_schema()
        # serializable nested model keeps a typed result schema
        assert schema["properties"]["result"].get("type") == "object"
        # and the tool schema fully expands the nested type (Person -> address -> street/city),
        # i.e. nested structure was never hidden in this channel (only the prefill was).
        flat = json.dumps(schema)
        assert "address" in flat and "street" in flat and "city" in flat


class TestCodeActReturnTypeDocFallback:
    """Opaque (Any-fallback) return types get construction guidance via doc(); the
    JSON-schemable ones don't need it (their structure is in the tool schema)."""

    @pytest.fixture(autouse=True)
    def _ensure_pandas_adapter(self):
        # A sibling agentdoc test may have cleared the registry; reload the adapter so its
        # @spec.define_doc decorators re-run (register_all alone is a no-op once imported).
        import importlib

        import nooa.agentdoc.adapters.pandas as _pandas_adapter

        importlib.reload(_pandas_adapter)

    def test_opaque_type_description_includes_doc_reference(self):
        s = CodeActStrategy()
        tool = s._build_return_result_tool(pd.DataFrame, "m")
        # the pandas adapter (auto-registered by the fallback) makes this concise
        assert "Return type reference:" in tool.description
        assert "pd.DataFrame({" in tool.description

    def test_pydantic_type_description_omits_doc_reference(self):
        s = CodeActStrategy()
        tool = s._build_return_result_tool(Person, "m")
        # Person's structure already lives in the JSON schema, so no doc dump is added
        assert "Return type reference:" not in tool.description

    def test_render_return_type_doc_is_best_effort(self):
        # Never raises, even for odd inputs.
        assert CodeActStrategy()._render_return_type_doc(pd.DataFrame)  # non-empty
        # truncation guard
        s = CodeActStrategy()
        out = s._render_return_type_doc(pd.DataFrame, max_chars=40)
        assert out is not None and len(out) <= 120  # truncated + suffix


class TestPrefillNestedReturnType:
    """The CodeAct prefill must expand nested return types (KDD-cup feedback).

    For `Type2 {t1: Type1}`, concise mode alone hides Type1's fields; the prefill
    must request `inline_depth=1` so the model sees how to construct the nested type.
    """

    def test_prefill_renders_nested_referenced_types(self):
        from nooa.agentdoc import doc
        from nooa.strategies.current_call import CurrentCall
        from nooa.strategies.prefill import InspectInputsPrefill

        call = CurrentCall(
            id="c",
            method_name="make",
            decorator="agent",
            signature=None,
            args=[],
            kwargs={"seed": 1},
            return_type=Person,  # Person {address: Address} — nested
        )
        code = InspectInputsPrefill().get_code(call)
        # the prefill expands one level of referenced types for the return type
        assert "inline_depth=1" in code
        # and that rendering actually surfaces the nested type's fields
        rendered = doc(Person, concise=True, inline_depth=1)
        assert "Referenced Types" in rendered
        assert "street" in rendered and "city" in rendered  # Address fields
