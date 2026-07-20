# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, ConfigDict

from nooa.agentdoc._docs import spec


@spec(expand=False)  # type: ignore[misc]
class DocConfig(BaseModel):
    """Configuration for documentation generation.

    Controls formatting, filtering, and level of detail in generated documentation.

    Attributes:
        max_value_chars: Maximum length for string values before truncation
        max_list_items: Maximum number of list/tuple items to display
        hidden_prefixes: Tuple of prefixes to hide (e.g., ("_",) hides private attrs)
        hidden_names: Frozenset of specific attribute names to hide
        include_types: Whether to show type annotations
        include_docstrings: Whether to include docstring summaries
        include_hints: Whether to show drill-down hints like "# doc(self.items)"
    """

    model_config = ConfigDict(frozen=True)

    max_value_chars: int = 50
    max_list_items: int = 10
    hidden_prefixes: tuple[str, ...] = ("_",)
    hidden_names: frozenset[str] = frozenset()
    include_types: bool = True
    include_docstrings: bool = True
    include_hints: bool = True

    def should_hide(self, name: str) -> bool:
        """Check if an attribute name should be hidden."""
        if name in self.hidden_names:
            return True
        return any(name.startswith(prefix) for prefix in self.hidden_prefixes)
