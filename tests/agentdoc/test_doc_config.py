# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from pydantic import BaseModel, ValidationError

from nooa.agentdoc.doc_config import DocConfig


def test_doc_config_is_pydantic_model():
    assert issubclass(DocConfig, BaseModel)


def test_doc_config_defaults():
    c = DocConfig()
    assert c.max_value_chars == 50
    assert c.max_list_items == 10
    assert c.hidden_prefixes == ("_",)
    assert c.hidden_names == frozenset()
    assert c.include_types is True
    assert c.include_docstrings is True
    assert c.include_hints is True


def test_doc_config_frozen():
    c = DocConfig()
    with pytest.raises(ValidationError):
        c.max_value_chars = 100


def test_hidden_prefixes_is_immutable_tuple():
    c = DocConfig()
    assert isinstance(c.hidden_prefixes, tuple)
    with pytest.raises(AttributeError):
        c.hidden_prefixes.append("test")  # type: ignore[attr-defined]  # tuples have no append


def test_hidden_names_is_frozenset():
    c = DocConfig()
    assert isinstance(c.hidden_names, frozenset)


def test_should_hide_by_prefix():
    c = DocConfig()
    assert c.should_hide("_private") is True
    assert c.should_hide("public") is False


def test_should_hide_by_name():
    c = DocConfig(hidden_names=frozenset({"secret"}))
    assert c.should_hide("secret") is True
    assert c.should_hide("public") is False
