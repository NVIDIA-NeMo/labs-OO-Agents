# SPDX-FileCopyrightText: Copyright (c) 2026 Sampoorn Nagpal
# SPDX-License-Identifier: Apache-2.0
"""Run a credential-free demonstration, or opt into a configured model alias."""

import argparse
import asyncio
import json

from nooa.unifiedllm import FakeLLMClient, LLMResponse, get_llm_client

from .agent import DocumentAgent, retrieve_and_answer
from .index import DocumentIndex, Section


def fixture() -> DocumentIndex:
    return DocumentIndex(
        "deployment-handbook",
        [
            Section("production", "Production", "Production changes require approval."),
            Section("prod-rollback", "Rollback", "Run deploy rollback --production.", "production"),
            Section("staging", "Staging", "Staging changes do not require approval."),
            Section("stage-rollback", "Rollback", "Run deploy rollback --staging.", "staging"),
        ],
    )


async def main(model: str | None) -> None:
    index = fixture()
    if model:
        llm = get_llm_client(model)
    else:
        outline, _ = index.outline(6000)
        sid = json.loads(outline)[1]["section_id"]
        responses = [
            {"section_ids": [sid]},
            {
                "text": "Run deploy rollback --production.",
                "citations": [{"section_id": sid, "quote": "Run deploy rollback --production."}],
                "abstained": False,
            },
        ]
        llm = FakeLLMClient(
            scripted_responses=[
                LLMResponse(
                    raw_response=None,
                    content=json.dumps(value),
                    tool_calls=[],
                    finish_reason="stop",
                )
                for value in responses
            ]
        )
    result = await retrieve_and_answer(
        DocumentAgent(llm=llm), index, "How do I roll back production?"
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="Existing NOOA model alias (makes real model calls)")
    asyncio.run(main(parser.parse_args().model))
