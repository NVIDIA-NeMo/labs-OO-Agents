# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test that @property methods on @dataclass are rendered by doc()."""

from dataclasses import dataclass

from nooa.agentdoc import doc, hidden
from nooa.agentdoc._structured import extract_type_info


class TestDataclassPropertyRendering:
    """Properties on dataclasses should be visible to agents via doc()."""

    def test_property_on_dataclass_appears_in_extract_type_info(self):
        """extract_type_info should include @property fields from a dataclass."""

        @dataclass
        class Result:
            """A result with computed properties."""

            items: list[str]
            total: int

            @property
            def summary(self) -> str:
                """Human-readable summary."""
                return f"{self.total} items"

        info = extract_type_info(Result)
        field_names = [f.name for f in info.fields]

        assert "items" in field_names
        assert "total" in field_names
        assert "summary" in field_names, f"@property 'summary' not in fields: {field_names}"

    def test_property_on_dataclass_appears_in_doc_output(self):
        """doc() output should show @property fields from a dataclass."""

        @dataclass
        class ViewResult:
            """View result with computed content."""

            lines: list[str]
            start_line: int

            @property
            def content(self) -> str:
                """Numbered display for LLM consumption."""
                return "\n".join(f"{self.start_line + i}|{ln}" for i, ln in enumerate(self.lines))

            @property
            def raw(self) -> str:
                """Raw file content without line numbers."""
                return "\n".join(self.lines)

        output = doc(ViewResult)
        assert "content" in output, f"@property 'content' not in doc() output:\n{output}"
        assert "raw" in output, f"@property 'raw' not in doc() output:\n{output}"

    def test_property_description_from_docstring(self):
        """Property description should come from its getter docstring."""

        @dataclass
        class Data:
            value: int

            @property
            def doubled(self) -> int:
                """The value times two."""
                return self.value * 2

        info = extract_type_info(Data)
        prop_field = next((f for f in info.fields if f.name == "doubled"), None)
        assert prop_field is not None, "Property 'doubled' not found in fields"
        assert prop_field.description == "The value times two."
        assert prop_field.type == "int"

    def test_hidden_property_on_dataclass_is_excluded(self):
        """@hidden @property on a @dataclass should NOT appear in doc() or extract_type_info."""

        @dataclass
        class Data:
            value: int

            @property
            def visible(self) -> int:
                """A visible computed field."""
                return self.value + 1

            @hidden
            @property
            def secret(self) -> int:
                """A hidden computed field."""
                return self.value * 2

        info = extract_type_info(Data)
        field_names = [f.name for f in info.fields]
        assert "visible" in field_names, f"visible not in {field_names}"
        assert "secret" not in field_names, f"hidden 'secret' should be excluded: {field_names}"

        output = doc(Data)
        assert "visible" in output
        assert "secret" not in output
