# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarization agents."""

from nooa import Agent


class SummarizeAgent(Agent):
    """Agent that summarizes text."""

    async def summarize_single(self, document: str) -> str:
        """Summarize a single document.

        Args:
            document: The document to summarize

        Returns:
            A concise summary of the document
        """
        ...


class SummarizeBatchAgent(Agent):
    """Agent that summarizes multiple documents."""

    async def summarize_batch(self, documents: list[str]) -> list[str]:
        """Summarize multiple documents.

        Args:
            documents: List of documents to summarize

        Returns:
            List of summaries for each document
        """
        ...
