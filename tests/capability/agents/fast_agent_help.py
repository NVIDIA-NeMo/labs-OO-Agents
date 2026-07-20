# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fast agent help-calling test — tests whether the LLM can call self.call_for_help() via execute_python.

Weaker models often try to call agent methods directly as tool calls instead
of wrapping them in execute_python(). This test exercises that pattern: the
agent has a deterministic call_for_help() method that MUST be invoked via
execute_python("result = self.call_for_help(request)").
"""

from nooa import Agent, strategy
from nooa.config import CodeActConfig
from nooa.strategies.codeact import CodeActStrategy


class FastHelpAgent(Agent):
    """You are Amy, a voice assistant. You speak in short, natural sentences.

    You can handle on your own: greetings, small talk, simple math, and general knowledge you already know.

    For anything you are unsure about or cannot do yourself, call self.call_for_help(request).
    This includes: real-time data, weather, actions, orders, timers, reminders, or anything
    that requires external information.

    When in doubt whether you can answer correctly, always call for help rather than guess.
    """

    def call_for_help(self, request: str) -> str:
        """Ask for help with a task you cannot handle on your own.

        Call this whenever the task requires real-time data, external lookups,
        performing actions, or anything beyond your built-in knowledge.
        When in doubt, call for help rather than guessing.

        Args:
            request: Natural language description of what needs to be done.

        Returns:
            The result from the helper. Relay this to the user faithfully.
        """
        return f"[HELP_CALLED: {request}]"

    async def reply(self, user_message: str) -> str:
        """Reply to the user as Amy.
        If you can answer from your own knowledge (greetings, math, general facts), answer directly.
        Otherwise, call for help by running: result = self.call_for_help(user_message)"""
        ...


class FastHelpAgentTranslate(FastHelpAgent):
    """Same as FastHelpAgent but with translate_tool_calls enabled."""

    @strategy(CodeActStrategy(config=CodeActConfig(translate_tool_calls=True)))
    async def reply(self, user_message: str) -> str:
        """Reply to the user as Amy.
        If you can answer from your own knowledge (greetings, math, general facts), answer directly.
        Otherwise, call for help by running: result = self.call_for_help(user_message)"""
        ...


class FastHelpAgentWrapper(Agent):
    """Wrapper without translation (baseline).

    Note: subclasses Agent because the eval runner instantiates with llm=.
    """

    async def reply(self, user_message: str) -> str:
        """Reply to the user message."""
        agent = FastHelpAgent()
        return await agent.reply(user_message)


class FastHelpAgentTranslateWrapper(Agent):
    """Wrapper with translate_tool_calls=True.

    Note: subclasses Agent because the eval runner instantiates with llm=.
    """

    async def reply(self, user_message: str) -> str:
        """Reply to the user message."""
        agent = FastHelpAgentTranslate()
        return await agent.reply(user_message)
