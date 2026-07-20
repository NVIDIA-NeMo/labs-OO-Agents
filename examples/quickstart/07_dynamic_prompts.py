# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F403,F405
"""Quickstart 07: Dynamic prompts — {self.attr} template expansion in docstrings.

uv run python examples/quickstart/07_dynamic_prompts.py
"""

from nooa.util.quickstart import *


class TranslatorAgent(Agent, llm=llm):
    """Agent that translates text with configurable behavior."""

    def __init__(self, target_language: str = "Spanish", **kwargs):
        super().__init__(**kwargs)
        self.target_language = target_language
        self.translation_count = 0

    async def translate(self, text: str) -> str:
        """Translate the text to {self.target_language}.

        Keep the translation natural and idiomatic.
        """
        ...

    async def translate_formal(self, text: str) -> str:
        """Translate the text to {self.target_language} using formal register.

        Use polite/formal forms (e.g., 'usted' in Spanish, 'Sie' in German).
        """
        ...


@autorun
async def main():
    # The {self.target_language} in docstrings is expanded at runtime
    spanish = TranslatorAgent(target_language="Spanish")
    german = TranslatorAgent(target_language="German")

    text = "Hello, how are you today?"

    print(f"Original: {text}\n")

    result_es = await spanish.translate(text)
    print(f"Spanish: {result_es}")

    result_de = await german.translate(text)
    print(f"German: {result_de}")

    result_formal = await spanish.translate_formal(text)
    print(f"Spanish (formal): {result_formal}")
