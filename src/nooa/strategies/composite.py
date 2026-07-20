# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CompositeStrategy - Base class for strategies that compose other strategies.

Composite strategies use @strategy methods to delegate to simpler strategies,
enabling natural composition and hierarchical strategy design.
"""

from nooa.strategies.base import GenerationStrategy


class CompositeStrategy(GenerationStrategy):
    """Base class for strategies that compose other strategies.

    Almost all strategies except TemplateStrategy are composite strategies.
    They use @strategy methods with simpler strategies to build complex behaviors.

    Strategy Hierarchy:
        TemplateStrategy (pure - no composition)
            ↓ used by
        PurePythonStrategy, PredictStrategy (compose TemplateStrategy)
            ↓ used by
        ReflexionStrategy, PlanExecuteStrategy, etc. (compose multiple strategies)

    Usage Pattern:
        class MyStrategy(CompositeStrategy):
            @strategy(TemplateStrategy())
            async def build_prompt(self, runtime: RuntimeServices, task: str) -> str:
                '''Build prompt for {task}'''
                ...

            @strategy(PredictStrategy())
            async def analyze(self, runtime: RuntimeServices, prompt: str) -> Result:
                '''{prompt}'''
                ...

            async def execute(self, runtime: RuntimeServices, call: CurrentCall) -> Any:
                prompt = await self.build_prompt(runtime, task=call.docstring)
                result = await self.analyze(runtime, prompt=prompt)
                return result

    Note:
        - Not all composite strategies have a "base" strategy
        - PurePythonStrategy and PredictStrategy only compose TemplateStrategy
        - Higher-level strategies like Reflexion have a base strategy for generation
        - Each strategy decides what it needs
    """

    pass  # Marker class - subclasses implement their own logic
