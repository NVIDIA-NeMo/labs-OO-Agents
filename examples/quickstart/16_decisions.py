# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Quickstart 16: Decision models — calibrated classification without a chat LLM.

OPENROUTER_API_KEY=... uv run python examples/quickstart/16_decisions.py

Set DECISION_ENDPOINT to use a different decisions API, such as a gateway that
adds credentials itself, and DECISION_MODEL to use a different model.
"""

import asyncio
import os
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel

from nooa import (
    Agent,
    ChoiceDecision,
    Criteria,
    DecideStrategy,
    DecisionClient,
    Instructions,
    Threshold,
    strategy,
)


class Department(StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"


# Criteria tell the decision model what each department means. Declare them
# once and reuse them for plain and detailed results.
DEPARTMENT_CRITERIA = Criteria(
    by_value={
        Department.BILLING: "Payments, invoices, and refunds.",
        Department.TECHNICAL: "Bugs, outages, and integrations.",
        Department.ACCOUNT: "Sign-in, profile, and access changes.",
    }
)
DepartmentChoice = Annotated[Department, DEPARTMENT_CRITERIA]
DepartmentDecision = Annotated[ChoiceDecision[Department], DEPARTMENT_CRITERIA]


class Triage(BaseModel):
    department: Annotated[DepartmentChoice, Instructions("Select the team that owns the request.")]
    urgent: Annotated[
        bool,
        Instructions("Decide whether the request needs immediate action."),
        Criteria(
            by_value={
                True: "Customers are blocked or harmed right now.",
                False: "The request can wait for normal business hours.",
            }
        ),
    ]


# A decision-only agent: every generated method uses DecideStrategy, so no chat
# LLM is configured. Pass llm=... as well to mix in chat generation methods.
class SupportRouter(Agent):
    @strategy(DecideStrategy())
    async def department(self, message: str) -> DepartmentChoice:
        """Choose the team that should handle the message."""
        ...

    # ChoiceDecision keeps the evidence. With a Threshold, .value is None when
    # the selected department's probability is below 0.8; .selected is kept.
    @strategy(DecideStrategy())
    async def confident_department(
        self, message: str
    ) -> Annotated[DepartmentDecision, Threshold(0.8)]:
        """Choose the team that should handle the message."""
        ...

    # One request answers both questions.
    @strategy(DecideStrategy())
    async def triage(self, message: str) -> Triage:
        """Use only the supplied message."""
        ...


MESSAGES = [
    "I was charged twice for my March invoice.",
    "The API has returned 500 errors for every request since 9am.",
    "Since the update I'm being billed for seats I removed and my teammates lost access.",
]


async def main() -> None:
    decision_model = DecisionClient(
        os.getenv("DECISION_MODEL", "typesafe/jev-1.13"),
        endpoint=os.getenv("DECISION_ENDPOINT", "https://openrouter.ai/api/alpha/decisions"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )
    try:
        router = SupportRouter(decision_model=decision_model)
        for message in MESSAGES:
            print(f"\n{message}")
            print(f"  department: {await router.department(message)}")

            decision = await router.confident_department(message)
            # The threshold compares the selected option's probability, not
            # .confidence, which is the service's separate overall estimate.
            selected_probability = decision.probabilities[decision.selected]
            outcome = decision.value or "manual review"
            print(
                f"  selected: {decision.selected} (p={selected_probability:.2f}, "
                f"confidence={decision.confidence:.2f}) -> {outcome}"
            )

            triage = await router.triage(message)
            print(f"  triage: {triage.department}, urgent={triage.urgent}")
    finally:
        # DecisionClient owns its HTTP connection pool unless you pass client=.
        await decision_model.aclose()


if __name__ == "__main__":
    asyncio.run(main())

# Example output from typesafe/jev-1.13 (probabilities vary by model version):
#
# I was charged twice for my March invoice.
#   department: billing
#   selected: billing (p=1.00, confidence=1.00) -> billing
#   triage: billing, urgent=False
#
# The API has returned 500 errors for every request since 9am.
#   department: technical
#   selected: technical (p=1.00, confidence=1.00) -> technical
#   triage: technical, urgent=True
#
# Since the update I'm being billed for seats I removed and my teammates lost access.
#   department: billing
#   selected: billing (p=0.45, confidence=0.18) -> manual review
#   triage: billing, urgent=True
