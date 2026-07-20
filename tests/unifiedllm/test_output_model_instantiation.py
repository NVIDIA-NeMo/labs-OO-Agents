# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for output model instantiation with RootModel support.

These tests verify that the _instantiate_output_model helper correctly handles:
- Regular Pydantic BaseModel (instantiated with **kwargs)
- RootModel for dict types (instantiated with positional arg)
- RootModel for list types (instantiated with positional arg)
"""

from typing import Any

import pytest
from pydantic import BaseModel, RootModel

from nooa.unifiedllm.unifiedllm import _instantiate_output_model


class TestInstantiateOutputModelWithRegularModels:
    """Tests for regular Pydantic BaseModel instantiation."""

    def test_simple_model(self):
        """Regular BaseModel with simple fields."""

        class Person(BaseModel):
            name: str
            age: int

        json_data = {"name": "Alice", "age": 30}
        result = _instantiate_output_model(Person, json_data)

        assert isinstance(result, Person)
        assert result.name == "Alice"
        assert result.age == 30

    def test_nested_model(self):
        """Regular BaseModel with nested structure."""

        class Address(BaseModel):
            city: str
            country: str

        class Person(BaseModel):
            name: str
            address: Address

        json_data = {"name": "Bob", "address": {"city": "NYC", "country": "USA"}}
        result = _instantiate_output_model(Person, json_data)

        assert isinstance(result, Person)
        assert result.name == "Bob"
        assert isinstance(result.address, Address)
        assert result.address.city == "NYC"

    def test_model_with_optional_fields(self):
        """Regular BaseModel with optional fields."""

        class Config(BaseModel):
            enabled: bool
            timeout: int | None = None

        json_data = {"enabled": True}
        result = _instantiate_output_model(Config, json_data)

        assert isinstance(result, Config)
        assert result.enabled is True
        assert result.timeout is None

    def test_model_with_value_field(self):
        """Wrapped basic type model (like what we use for str/int return types)."""
        from pydantic import create_model

        WrappedStr = create_model("WrappedStr", value=(str, ...))
        json_data = {"value": "hello world"}
        result = _instantiate_output_model(WrappedStr, json_data)

        assert hasattr(result, "value")
        assert result.value == "hello world"  # type: ignore[attr-defined]


class TestInstantiateOutputModelWithRootModels:
    """Tests for RootModel instantiation (dict/list return types)."""

    def test_dict_root_model(self):
        """RootModel for dict type should accept dict directly."""

        class DictResponse(RootModel[dict[str, Any]]):
            pass

        json_data = {"message": "Hello!", "action": "greet"}
        result = _instantiate_output_model(DictResponse, json_data)

        assert isinstance(result, DictResponse)
        assert result.root == {"message": "Hello!", "action": "greet"}

    def test_dict_root_model_with_nested_structure(self):
        """RootModel for dict with nested values."""

        class DictResponse(RootModel[dict[str, Any]]):
            pass

        json_data = {
            "user": {"name": "Alice", "age": 30},
            "metadata": {"created": "2025-01-01"},
        }
        result = _instantiate_output_model(DictResponse, json_data)

        assert isinstance(result, DictResponse)
        assert result.root["user"]["name"] == "Alice"
        assert result.root["metadata"]["created"] == "2025-01-01"

    def test_list_root_model_strings(self):
        """RootModel for list[str] should accept list directly."""

        class ListResponse(RootModel[list[str]]):
            pass

        json_data = ["positive", "negative", "neutral"]
        result = _instantiate_output_model(ListResponse, json_data)

        assert isinstance(result, ListResponse)
        assert result.root == ["positive", "negative", "neutral"]

    def test_list_root_model_integers(self):
        """RootModel for list[int] should accept list directly."""

        class ListResponse(RootModel[list[int]]):
            pass

        json_data = [1, 2, 3, 4, 5]
        result = _instantiate_output_model(ListResponse, json_data)

        assert isinstance(result, ListResponse)
        assert result.root == [1, 2, 3, 4, 5]

    def test_list_root_model_dicts(self):
        """RootModel for list of dicts."""

        class ListResponse(RootModel[list[dict[str, Any]]]):
            pass

        json_data = [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
        result = _instantiate_output_model(ListResponse, json_data)

        assert isinstance(result, ListResponse)
        assert len(result.root) == 2
        assert result.root[0]["name"] == "Item 1"

    def test_empty_list_root_model(self):
        """RootModel for empty list."""

        class ListResponse(RootModel[list[str]]):
            pass

        json_data = []
        result = _instantiate_output_model(ListResponse, json_data)

        assert isinstance(result, ListResponse)
        assert result.root == []

    def test_empty_dict_root_model(self):
        """RootModel for empty dict."""

        class DictResponse(RootModel[dict[str, Any]]):
            pass

        json_data = {}
        result = _instantiate_output_model(DictResponse, json_data)

        assert isinstance(result, DictResponse)
        assert result.root == {}


class TestInstantiateOutputModelEdgeCases:
    """Edge cases and error scenarios."""

    def test_regular_model_with_dict_fails_with_list(self):
        """Regular BaseModel should fail when given a list (not a dict)."""

        class Person(BaseModel):
            name: str

        with pytest.raises(TypeError):
            _instantiate_output_model(Person, ["Alice"])

    def test_issubclass_check_works_for_dynamic_root_models(self):
        """Dynamically created RootModel subclasses should be detected."""

        # This simulates what our strategy code does
        class DynamicDictModel(RootModel[dict[str, Any]]):
            pass

        DynamicDictModel.__name__ = "CustomResponse"
        DynamicDictModel.__qualname__ = "CustomResponse"

        json_data = {"key": "value"}
        result = _instantiate_output_model(DynamicDictModel, json_data)

        assert result.root == {"key": "value"}  # type: ignore[attr-defined]


class TestInstantiateOutputModelIntegration:
    """Integration-style tests mimicking real usage patterns."""

    def test_sentiment_classification_list_response(self):
        """Test pattern: sentiment classification returning list of labels."""

        class ClassifyResponse(RootModel[list[str]]):
            pass

        # Simulates LLM returning sentiment labels for multiple texts
        json_data = [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
        ]
        result = _instantiate_output_model(ClassifyResponse, json_data)

        assert result.root == ["positive", "negative", "neutral", "positive", "negative"]  # type: ignore[attr-defined]

    def test_order_processing_dict_response(self):
        """Test pattern: order processing returning dict with message and action."""

        class ProcessMessageResponse(RootModel[dict[str, Any]]):
            pass

        # Simulates LLM returning order processing result
        json_data = {
            "message": "I've added a large pizza to your order.",
            "action_taken": "added pizza",
        }
        result = _instantiate_output_model(ProcessMessageResponse, json_data)

        assert result.root["message"] == "I've added a large pizza to your order."  # type: ignore[attr-defined]
        assert result.root["action_taken"] == "added pizza"  # type: ignore[attr-defined]

    def test_analysis_pydantic_model_response(self):
        """Test pattern: analysis returning structured Pydantic model."""

        class AnalysisResult(BaseModel):
            score: float
            confidence: float
            label: str
            reasoning: str

        json_data = {
            "score": 0.85,
            "confidence": 0.92,
            "label": "positive",
            "reasoning": "The text contains positive language.",
        }
        result = _instantiate_output_model(AnalysisResult, json_data)

        assert isinstance(result, AnalysisResult)
        assert result.score == 0.85
        assert result.label == "positive"
