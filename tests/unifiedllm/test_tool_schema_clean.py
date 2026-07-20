# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for Tool.get_parameter_schema() — $ref resolution and noise stripping."""

import json
from typing import Literal

from pydantic import BaseModel, create_model

from nooa.unifiedllm import Tool


class TestToolSchemaSimple:
    """Simple tool schemas from callable signatures."""

    def test_simple_string_param(self):
        def execute_python(code: str) -> str:
            """Execute code."""

        tool = Tool(name="execute_python", description="Execute code.", callable=execute_python)
        schema = tool.get_parameter_schema()

        assert schema == {
            "type": "object",
            "properties": {"code": {"type": "string"}},
            "required": ["code"],
        }

    def test_multiple_params(self):
        def add(a: int, b: int) -> int:
            """Add numbers."""
            return a + b

        tool = Tool(name="add", description="Add.", callable=add)
        schema = tool.get_parameter_schema()

        assert schema["properties"]["a"] == {"type": "integer"}
        assert schema["properties"]["b"] == {"type": "integer"}
        assert set(schema["required"]) == {"a", "b"}
        assert "title" not in json.dumps(schema)

    def test_no_title_anywhere(self):
        def fn(name: str, count: int, active: bool) -> str:
            """Test."""

        tool = Tool(name="fn", description="test", callable=fn)
        schema = tool.get_parameter_schema()
        serialized = json.dumps(schema)

        assert "title" not in serialized


class TestToolSchemaNestedModels:
    """Tool schemas with nested Pydantic models (produces $ref/$defs)."""

    def test_ref_resolved_inline(self):
        class Inner(BaseModel):
            name: str
            value: int

        ReturnModel = create_model("Ret", result=(Inner, ...))

        def rr(result=None):
            pass

        tool = Tool(
            name="return_result", description="Return.", callable=rr, parameters_model=ReturnModel
        )
        schema = tool.get_parameter_schema()

        # $ref and $defs should be gone
        serialized = json.dumps(schema)
        assert "$ref" not in serialized
        assert "$defs" not in serialized

        # The nested model should be inlined with proper type
        result_schema = schema["properties"]["result"]
        assert result_schema["type"] == "object"
        assert "name" in result_schema["properties"]
        assert result_schema["properties"]["name"]["type"] == "string"
        assert result_schema["properties"]["value"]["type"] == "integer"

    def test_no_title_in_nested(self):
        class Address(BaseModel):
            street: str
            city: str

        class Person(BaseModel):
            name: str
            address: Address

        ReturnModel = create_model("Ret", result=(Person, ...))

        def rr(result=None):
            pass

        tool = Tool(name="rr", description="d", callable=rr, parameters_model=ReturnModel)
        schema = tool.get_parameter_schema()

        assert "title" not in json.dumps(schema)

    def test_dict_with_model_values(self):
        """dict[str, SomeModel] produces additionalProperties with $ref."""

        class Item(BaseModel):
            name: str
            score: float

        class Results(BaseModel):
            items: dict[str, Item]

        ReturnModel = create_model("Ret", result=(Results, ...))

        def rr(result=None):
            pass

        tool = Tool(name="rr", description="d", callable=rr, parameters_model=ReturnModel)
        schema = tool.get_parameter_schema()

        serialized = json.dumps(schema)
        assert "$ref" not in serialized
        assert "$defs" not in serialized
        # The additionalProperties should have the Item schema inlined
        items_schema = schema["properties"]["result"]["properties"]["items"]
        assert "additionalProperties" in items_schema
        add_props = items_schema["additionalProperties"]
        assert add_props["type"] == "object"
        assert "name" in add_props["properties"]

    def test_union_type_anyof(self):
        """Union types produce anyOf with $ref."""

        class TypeA(BaseModel):
            kind: Literal["a"]
            data: str

        class TypeB(BaseModel):
            kind: Literal["b"]
            count: int

        ReturnModel = create_model("Ret", result=(TypeA | TypeB, ...))

        def rr(result=None):
            pass

        tool = Tool(name="rr", description="d", callable=rr, parameters_model=ReturnModel)
        schema = tool.get_parameter_schema()

        serialized = json.dumps(schema)
        assert "$ref" not in serialized
        assert "$defs" not in serialized
        # anyOf items should be inlined
        result_schema = schema["properties"]["result"]
        assert "anyOf" in result_schema
        assert len(result_schema["anyOf"]) == 2
        assert all("type" in item for item in result_schema["anyOf"])

    def test_list_of_models(self):
        """list[SomeModel] produces items with $ref."""

        class Entry(BaseModel):
            text: str
            score: float

        ReturnModel = create_model("Ret", result=(list[Entry], ...))

        def rr(result=None):
            pass

        tool = Tool(name="rr", description="d", callable=rr, parameters_model=ReturnModel)
        schema = tool.get_parameter_schema()

        serialized = json.dumps(schema)
        assert "$ref" not in serialized
        result_schema = schema["properties"]["result"]
        assert result_schema["type"] == "array"
        assert result_schema["items"]["type"] == "object"
        assert "text" in result_schema["items"]["properties"]

    def test_allof_with_description(self):
        """allOf wrapping (Pydantic v2 with Field(description=...)) is unwrapped."""
        from pydantic import Field

        class Config(BaseModel):
            host: str
            port: int

        class ServerResult(BaseModel):
            config: Config = Field(description="Server configuration")

        ReturnModel = create_model("Ret", result=(ServerResult, ...))

        def rr(result=None):
            pass

        tool = Tool(name="rr", description="d", callable=rr, parameters_model=ReturnModel)
        schema = tool.get_parameter_schema()

        serialized = json.dumps(schema)
        assert "$ref" not in serialized
        assert "$defs" not in serialized
        assert "allOf" not in serialized
        result_props = schema["properties"]["result"]["properties"]
        assert "config" in result_props
        config_schema = result_props["config"]
        assert "properties" in config_schema
