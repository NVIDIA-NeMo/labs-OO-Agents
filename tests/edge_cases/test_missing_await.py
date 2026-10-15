# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test cases for detecting missing await on async method calls.

This test suite verifies that the system can detect when LLM-generated code
calls async methods without await, which would return coroutine objects
instead of actual results.
"""

import pytest

from nooa import Agent, strategy
from nooa.errors import GenerationError
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
    )


@pytest.mark.asyncio
async def test_missing_await_on_async_method():
    """Test that missing await on async method is detected and reported."""

    # Buggy code that will be generated (missing await)
    buggy_code = """
sentiments = []
for text in texts:
    sentiment = self.classify_single(text)  # BUG: Missing await!
    sentiments.append(sentiment)
return sentiments
"""

    # Create test LLM with scripted responses
    test_llm = FakeLLMClient(
        scripted_responses=[_resp(buggy_code)] * 4  # Queue enough to exceed max_retries
    )

    class SentimentAgent(Agent, llm=test_llm):
        """Agent for sentiment classification tasks."""

        async def classify_single(self, text: str) -> str:
            """Classify the sentiment of this single text."""
            # Implemented method (not generated)
            if "good" in text.lower() or "great" in text.lower():
                return "positive"
            elif "bad" in text.lower() or "terrible" in text.lower():
                return "negative"
            return "neutral"

        async def classify_batch(self, texts: list[str]) -> list[str]:
            """Classify the sentiment of each text in the batch."""
            ...  # LLM generates code

    agent_instance = SentimentAgent(llm=test_llm)

    # Should raise GenerationError due to validation failure
    with pytest.raises(GenerationError) as exc_info:
        await agent_instance.classify_batch(
            ["This is great!", "This is terrible!", "This is okay."]
        )

    # The error should signal that the LLM never produced a valid tool call.
    # CodeAct routes stop responses through return_result() validation and
    # aborts after max_consecutive_text_only consecutive failures (issue 185).
    error_msg = str(exc_info.value).lower()
    assert (
        "max_retries" in error_msg
        or "generation failed" in error_msg
        or "validation failed" in error_msg
        or "max_consecutive_text_only" in error_msg
        or "codeact aborted" in error_msg
    )


@pytest.mark.asyncio
async def test_missing_await_multiple_methods():
    """Test detection when multiple async methods are called without await."""

    buggy_code = """
results = []
for key in keys:
    data = self.fetch_data(key)  # BUG: Missing await!
    transformed = self.transform_data(data)  # BUG: Missing await!
    results.append(transformed)
return results
"""

    test_llm = FakeLLMClient(scripted_responses=[_resp(buggy_code)] * 4)

    class DataProcessor(Agent, llm=test_llm):
        """Agent that processes data through multiple async steps."""

        async def fetch_data(self, key: str) -> dict:
            """Fetch data for the given key."""
            return {"key": key, "value": "data"}

        async def transform_data(self, data: dict) -> dict:
            """Transform the data."""
            return {**data, "transformed": True}

        async def process_pipeline(self, keys: list[str]) -> list[dict]:
            """Process multiple keys through the full pipeline."""
            ...

    agent_instance = DataProcessor(llm=test_llm)

    with pytest.raises(GenerationError):
        await agent_instance.process_pipeline(["a", "b", "c"])


@pytest.mark.asyncio
async def test_correct_await_usage_passes():
    """Test that correct await usage passes validation."""

    correct_code = """
results = []
for num in numbers:
    result = await self.helper_method(num)  # Correct: has await!
    results.append(result)
return results
"""

    test_llm = FakeLLMClient(scripted_responses=[_resp(correct_code)])

    class CorrectAgent(Agent, llm=test_llm):
        """Agent with correct async/await usage."""

        async def helper_method(self, x: int) -> int:
            """Helper that doubles the input."""
            return x * 2

        @strategy(PurePythonStrategy())
        async def process_list(self, numbers: list[int]) -> list[int]:
            """Process a list of numbers using helper_method."""
            ...

    agent_instance = CorrectAgent(llm=test_llm)

    # This should work without errors
    result = await agent_instance.process_list([1, 2, 3])
    assert result == [2, 4, 6]


@pytest.mark.asyncio
async def test_asyncio_gather_pattern_allowed():
    """Test that list comprehension pattern (for asyncio.gather) is allowed."""

    # Simpler test - just execute coroutines in list comp and await them manually
    gather_code = """
# List comprehension with coroutines (common pattern for gather)
tasks = [self.process_item(item) for item in items]

# Manually await each (simplified version)
results = []
for task in tasks:
    result = await task
    results.append(result)

return results
"""

    test_llm = FakeLLMClient(scripted_responses=[_resp(gather_code)])

    class ParallelAgent(Agent, llm=test_llm):
        """Agent that processes items in parallel."""

        async def process_item(self, item: str) -> str:
            """Process a single item."""
            return f"processed_{item}"

        @strategy(PurePythonStrategy())
        async def process_parallel(self, items: list[str]) -> list[str]:
            """Process items in parallel using coroutines."""
            ...

    agent_instance = ParallelAgent(llm=test_llm)

    # This pattern should pass validation (list comp with coroutines is allowed)
    result = await agent_instance.process_parallel(["a", "b", "c"])
    assert result == ["processed_a", "processed_b", "processed_c"]


@pytest.mark.asyncio
async def test_validation_unit():
    """Unit test the validation method directly."""
    from nooa.strategies.generated_code import GeneratedCodeValidator

    validator = GeneratedCodeValidator()

    # Create a mock agent with async methods
    test_llm = FakeLLMClient()

    class TestAgent(Agent, llm=test_llm):
        async def async_method(self, x: int) -> int:
            """An async method."""
            return x + 1

        async def sync_method(self, x: int) -> int:
            """A sync method."""
            return x + 2

    agent_instance = TestAgent(llm=test_llm)

    # Test 1: Missing await should be invalid
    code_with_error = """
result = self.async_method(5)
return result
"""
    errors = validator.validate(code_with_error, agent_instance)
    assert len(errors) > 0
    assert "async_method" in errors[0]
    assert "await" in errors[0].lower()

    # Test 2: Correct await should be valid
    code_correct = """
result = await self.async_method(5)
return result
"""
    errors = validator.validate(code_correct, agent_instance)
    assert len(errors) == 0

    # Test 3: List comprehension should be allowed (for gather)
    code_with_listcomp = """
tasks = [self.async_method(x) for x in range(10)]
results = await asyncio.gather(*tasks)
return results
"""
    errors = validator.validate(code_with_listcomp, agent_instance)
    assert len(errors) == 0


# --- #447: attribute chains on the agent (self.shell.read, self.repo.symbols) ---


class _ChainShell:
    async def read(self, path):
        return path

    async def run(self, command):
        return command


class _ChainSkills:
    async def activate(self, names):
        return names

    def search(self, query):
        return query

    def read(self, name):
        return name

    def doc(self, name):
        return name


class _ChainTodo:
    def add(self, title):
        return title

    def status(self):
        return ""


class _ChainEvents:
    def collapse(self, start, end, summary):
        return summary


class _ChainAgent:
    def __init__(self):
        self.shell = _ChainShell()
        self.skills = _ChainSkills()
        self.todo = _ChainTodo()
        self.events = _ChainEvents()

    @property
    def lazy(self):
        raise AssertionError("validation must not run properties")


def _chain_issues(code):
    import ast

    from nooa.runtime.code_validator import REPLPolicyValidator, ValidationContext
    from nooa.strategies.generated_code import GeneratedCodeValidator

    agent = _ChainAgent()
    generated = GeneratedCodeValidator().validate(code, agent)
    tree = ast.parse(
        "async def _cell():\n" + "".join(f"    {line}\n" for line in code.splitlines())
    )
    e301 = [
        issue.message
        for issue in REPLPolicyValidator().validate(tree, ValidationContext(code=code, agent=agent))
        if issue.code == "E301"
    ]
    return generated, e301


@pytest.mark.parametrize(
    ("code", "chain"),
    [
        # value discarded (bare expression statement)
        ('self.shell.read("f.py")', "self.shell.read"),
        ("self.skills.activate(['web'])", "self.skills.activate"),
        # result used as if awaited: attribute, subscript, iteration
        ('print(self.shell.run("ls").stdout)', "self.shell.run"),
        ('text = self.shell.read("f.py").numbered', "self.shell.read"),
        ('first = self.shell.run("ls")[0]', "self.shell.run"),
        ('for line in self.shell.run("ls"):\n    print(line)', "self.shell.run"),
        ('lines = [x for x in self.shell.run("ls")]', "self.shell.run"),
        ('texts = [self.shell.read(p).numbered for p in ["a"]]', "self.shell.read"),
        ('text = (await self.shell.read("f.py").numbered)', "self.shell.read"),
    ],
)
def test_unawaited_async_attribute_chain_is_flagged(code, chain):
    generated, e301 = _chain_issues(code)
    assert generated == [
        f"`{chain}` is async and must be called with `await`.\n"
        f"   Example: result = await {chain}(...)"
    ]
    assert e301 == [
        f"`{chain}` is async and must be called with `await`. Example: result = await {chain}(...)"
    ]


@pytest.mark.parametrize(
    "code",
    [
        'x = await self.shell.read("f.py")',
        "self.skills.search('web')",
        "self.skills.read('web')",
        "self.skills.doc('web')",
        "self.todo.add('t')",
        "self.todo.status()",
        "self.events.collapse('1', '9', 'summary')",
        "rs = await asyncio.gather(*[self.shell.read(p) for p in ['a', 'b']])",
        "tasks = [self.shell.read(p) for p in ['a']]",
        "rs = await asyncio.gather(self.shell.read('a'), self.shell.run('ls'))",
        "t = asyncio.create_task(self.shell.run('make'))",
        "t = asyncio.ensure_future(self.shell.run('make'))",
        "r = await asyncio.wait_for(self.shell.run('make'), 5)",
        "self.lazy.read('x')",
        "self.missing.read('x')",
        # #447 review: valid ways to hold a coroutine for later must pass
        'x = self.shell.read("f.py")',
        "coros = []\nfor p in ['a']:\n    coros.append(self.shell.read(p))\n"
        "rs = await asyncio.gather(*coros)",
        "rs = await asyncio.gather(*[self.shell.run('a'), self.shell.run('b')])",
        "cs = [self.shell.run('a'), self.shell.run('b')]",
        "cs = (self.shell.run('a'), self.shell.run('b'))",
        "cs = {self.shell.run('a')}",
        "d = {'a': self.shell.read('a')}",
        "d = {p: self.shell.read(p) for p in ['a']}",
        "s = {self.shell.read(p) for p in ['a']}",
        "coro = self.shell.run('x')\nr = await coro",
        "done, pending = await asyncio.wait([self.shell.run('a'), self.shell.run('b')])",
        "results = await asyncio.wait(self.shell.run('a') for _ in range(2))",
        "t = asyncio.create_task(self.shell.run('make'), name='m')",
        "q.put_nowait(self.shell.run('a'))",
    ],
)
def test_awaited_sync_and_gathered_chains_are_not_flagged(code):
    assert _chain_issues(code) == ([], [])


class _DirectAgent(_ChainAgent):
    async def fetch(self, x):
        return x


@pytest.mark.parametrize(
    ("code", "flagged"),
    [
        # self.<async method>() keeps the pre-#447 rule: awaited, list/generator
        # comprehension, or a gather/create_task argument; anything else is flagged.
        ("x = self.fetch(1)", True),
        ("self.fetch(1)", True),
        ("cs = [self.fetch(1)]", True),
        ("x = await self.fetch(1)", False),
        ("cs = [self.fetch(i) for i in range(3)]", False),
        ("rs = await asyncio.gather(self.fetch(1), self.fetch(2))", False),
    ],
)
def test_direct_agent_async_method_rule_is_unchanged(code, flagged):
    from nooa.strategies.generated_code import GeneratedCodeValidator

    errors = GeneratedCodeValidator().validate(code, _DirectAgent())
    assert bool(errors) is flagged, errors
    if flagged:
        assert errors[0].startswith("Method `fetch` is async and must be called with `await`")


def test_validators_and_formatter_share_one_missing_await_rule_module():
    """#447 review finding 8: one implementation of the chain resolution and exemptions."""
    import nooa.errors.formatting as formatting
    import nooa.runtime.async_calls as async_calls
    import nooa.runtime.code_validator as code_validator
    import nooa.strategies.generated_code as generated_code

    assert code_validator.async_self_chain is async_calls.async_self_chain
    assert code_validator.is_gather_argument is async_calls.is_gather_argument
    for module in (code_validator, generated_code, formatting):
        for private_copy in (
            "_resolve_static_chain",
            "_is_coroutine_function",
            "_unwrapped_is_coroutine_function",
            "_self_attribute_chain",
        ):
            assert not hasattr(module, private_copy), (module.__name__, private_copy)
    assert not hasattr(generated_code.GeneratedCodeValidator, "_is_in_await_or_comprehension")
