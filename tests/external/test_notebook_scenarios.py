# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Test all scenarios from agent_experiments.ipynb notebook.

This validates that the new self-contained agent API works for all notebook examples.
"""

import asyncio

import pytest

from nooa import Agent, strategy
from nooa.strategies.pure_python import PurePythonStrategy
from nooa.unifiedllm import FakeLLMClient, LLMResponse


def _resp(content: str) -> LLMResponse:
    """Create a test LLM response with the given content."""
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=[],
        finish_reason="stop",
        assistant_message={"role": "assistant", "content": content},
    )


# Module-level test LLM (can be overridden at instantiation)
_TEST_LLM = FakeLLMClient()

# Tool base class removed - plain classes work fine

# ============================================================================
# Part 1: Basic Agent - Testing Developer Experience
# ============================================================================


class SimpleAgent(Agent, llm=_TEST_LLM):
    """A simple agent for testing basic functionality."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.message = None
        self.count = 0
        self.items = []

    @strategy(
        PurePythonStrategy(),
    )
    async def perform_task(self) -> str:
        """
        Perform a simple task to test basic agent functionality.

        Steps:
        1. Set self.message to "Hello from agent!"
        2. Set self.count to 42
        3. Add three items to self.items: "apple", "banana", "cherry"
        4. Return a status message
        """
        ...


@pytest.mark.asyncio
async def test_simple_agent():
    """Test basic agent creation and execution with state modification."""

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                'self.message = "Hello from agent!"\n'
                "self.count = 42\n"
                'self.items = ["apple", "banana", "cherry"]\n'
                'return "Task completed successfully"'
            )
        ]
    )

    # NEW API: Create agent directly (no runtime.create())
    agent = SimpleAgent(llm=fake_llm)

    # Initial state
    assert agent.message is None
    assert agent.count == 0
    assert agent.items == []

    # Call agent method
    result = await agent.perform_task()

    # Verify state was modified
    assert agent.message == "Hello from agent!"
    assert agent.count == 42
    assert agent.items == ["apple", "banana", "cherry"]
    assert "Task completed successfully" in result

    # Verify runtime exists and code was cached (PERSISTENT lifetime)
    assert agent.runtime is not None
    code = agent.runtime.get_code("perform_task")
    assert code is not None
    assert "self.message" in code
    assert "self.count" in code


# ============================================================================
# Part 2: Async Behavior - Testing Concurrency
# ============================================================================


class AsyncAgent(Agent, llm=_TEST_LLM):
    """Agent for testing concurrent async operations."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.results = []
        self.completed_count = 0

    @strategy(PurePythonStrategy())
    async def process_concurrent(self, items: list[str]) -> dict:
        """
        Process multiple items concurrently using asyncio.gather().

        For each item:
        1. Simulate async processing with asyncio.sleep(0.01)
        2. Add result to self.results
        3. Increment self.completed_count

        Return a summary dict with total count and results.
        """
        ...


@pytest.mark.asyncio
async def test_async_agent():
    """Test agent handling concurrent async operations."""

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                "async def process_item(item):\n"
                "    await asyncio.sleep(0.01)\n"
                '    self.results.append(f"processed_{item}")\n'
                "    self.completed_count += 1\n"
                "\n"
                "await asyncio.gather(*[process_item(item) for item in items])\n"
                'return {"total": self.completed_count, "results": self.results}'
            )
        ]
    )

    # NEW API: Create agent directly
    agent = AsyncAgent(llm=fake_llm)

    # Initial state
    assert agent.results == []
    assert agent.completed_count == 0

    # Process items concurrently
    result = await agent.process_concurrent(["item1", "item2", "item3"])

    # Verify concurrent processing worked
    assert agent.completed_count == 3
    assert len(agent.results) == 3
    assert all("processed_" in r for r in agent.results)
    assert result["total"] == 3


# ============================================================================
# Part 3: Subagents - Testing Parent-Child Delegation
# ============================================================================


class WorkerAgent(Agent, llm=_TEST_LLM):
    """Worker agent that processes individual tasks."""

    def __init__(self, worker_id: int = 0, **kwargs):
        super().__init__(**kwargs)
        self.worker_id = worker_id
        self.tasks_completed = 0

    @strategy(PurePythonStrategy())
    async def process_task(self, task: str) -> str:
        """
        Process a single task.

        1. Increment self.tasks_completed
        2. Return formatted result with worker_id and task
        """
        ...


class CoordinatorAgent(Agent, llm=_TEST_LLM):
    """Coordinator that delegates to worker agents."""

    # Register child agent class for sandbox access
    WorkerAgent = WorkerAgent

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total_tasks = 0

    @strategy(PurePythonStrategy())
    async def coordinate_tasks(self, tasks: list[str]) -> dict:
        """
        Coordinate multiple tasks by delegating to worker agents.

        For each task:
        1. Create a WorkerAgent with worker_id (from enumerate)
        2. Call worker.process_task(task)
        3. Collect results

        Update self.total_tasks and return summary.
        """
        ...


@pytest.mark.asyncio
async def test_subagents():
    """Test parent agent creating and delegating to child agents."""

    # Worker response - REPL-style
    worker_response = _resp(
        'self.tasks_completed += 1\nreturn f"Worker {self.worker_id} processed {task}"'
    )

    # Coordinator response - REPL-style
    coordinator_response = _resp(
        "workers = [self.WorkerAgent(worker_id=i, llm=self._llm) for i in range(len(tasks))]\n"
        "results = await asyncio.gather(*[workers[i].process_task(tasks[i]) for i in range(len(tasks))])\n"
        "self.total_tasks = len(tasks)\n"
        'return {"total": self.total_tasks, "results": list(results)}'
    )

    # Build response list: coordinator first, then worker responses for each worker
    all_responses = [
        coordinator_response,  # Coordinator generates its method
        worker_response,  # Worker 0 generates process_task
        worker_response,  # Worker 1 generates process_task
        worker_response,  # Worker 2 generates process_task
        # Extras in case of retries
        coordinator_response,
        worker_response,
        worker_response,
        worker_response,
    ]

    shared_llm = FakeLLMClient(scripted_responses=all_responses)

    # NEW API: Create coordinator directly (children will inherit llm_client)
    coordinator = CoordinatorAgent(llm=shared_llm)

    # Coordinate tasks (will create child workers internally)
    result = await coordinator.coordinate_tasks(["task1", "task2", "task3"])

    # Verify coordination worked
    assert coordinator.total_tasks == 3
    assert len(result["results"]) == 3
    assert all("Worker" in r and "processed" in r for r in result["results"])


# ============================================================================
# Part 4: Tools - Testing Tool Registration and Usage
# ============================================================================


class Calculator:
    """Simple calculator tool - plain class, no Tool inheritance required."""

    async def add(self, a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    async def multiply(self, a: int, b: int) -> int:
        """Multiply two numbers."""
        return a * b


@pytest.mark.asyncio
async def test_tools():
    """Test agent using registered tools."""

    # Create calculator tool instance
    calc = Calculator()

    # Define agent class with tool assigned directly as class attribute

    class MathAgent(Agent, llm=_TEST_LLM):
        """Agent that uses calculator tool."""

        # Assign tool directly as class attribute
        calculator = calc

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.calculation_history = []

        @strategy(
            PurePythonStrategy(),
        )
        async def calculate(self, x: int, y: int) -> dict:
            """
            Use calculator to add and multiply.

            Steps:
            1. Use self.calculator.add(x, y)
            2. Use self.calculator.multiply(x, y)
            3. Add results to self.calculation_history
            4. Return dict with both results
            """
            ...

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                "sum_result = await self.calculator.add(x, y)\n"
                "product_result = await self.calculator.multiply(x, y)\n"
                'entry = {"add": sum_result, "multiply": product_result}\n'
                "self.calculation_history.append(entry)\n"
                'return {"sum": sum_result, "product": product_result}'
            )
        ]
    )

    # NEW API: Create agent directly
    math_agent = MathAgent(llm=fake_llm)

    # Use tool through agent
    result = await math_agent.calculate(5, 3)

    # Verify tool usage
    assert result["sum"] == 8
    assert result["product"] == 15
    assert len(math_agent.calculation_history) == 1
    assert math_agent.calculation_history[0]["add"] == 8
    assert math_agent.calculation_history[0]["multiply"] == 15


# ============================================================================
# Part 4: PLAN Strategy with Multi-Round - Removed (EXECUTE strategy removed)
# ============================================================================

# Note: EXECUTE strategy has been removed from the system.
# PLAN strategy now handles multi-round exploration with REPL.


class DataAnalyzer(Agent, llm=_TEST_LLM):
    """Agent that analyzes data using the PLAN strategy."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = []
        self.analysis_results = {}

    async def analyze_numbers(self, numbers: list[int]) -> dict:
        """
        Analyze a list of numbers using PLAN strategy with REPL exploration.

        Your task:
        1. Store the numbers in self.data
        2. Define a helper method `calculate_stats` that computes sum, mean, min, max
        3. Call the helper method to get the stats
        4. Return the stats as a dict
        """
        ...


# ============================================================================
# Part 8: Callbacks - Testing message and reasoning events
# ============================================================================


@pytest.mark.asyncio
async def test_message_callback():
    """Test message callback receives messages from generated code via message()."""

    messages_received = []

    def capture_message(event):
        """Capture Message."""
        messages_received.append(event.content)

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                'message("I\'m implementing the task now!")\n'
                'self.message = "Hello from agent!"\n'
                "self.count = 42\n"
                'self.items = ["apple", "banana", "cherry"]\n'
                'message("Task implementation complete!")\n'
                'return "Task completed successfully"'
            )
        ]
    )

    # Create agent and subscribe to message events
    agent = SimpleAgent(llm=fake_llm)
    agent.event_manager.on("Message", capture_message)

    # Call method
    await agent.perform_task()

    # Verify callbacks were called
    assert len(messages_received) == 2, f"Expected 2 messages, got {len(messages_received)}"
    assert "I'm implementing the task now!" in messages_received[0]
    assert "Task implementation complete!" in messages_received[1]


@pytest.mark.asyncio
async def test_no_reasoning_events_emitted():
    """The reasoning() builtin was removed — generated code emits no Reasoning events."""

    reasoning_received = []

    def capture_reasoning(event):
        """Capture Reasoning."""
        reasoning_received.append(event.content)

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                'self.message = "Hello from agent!"\n'
                "self.count = 42\n"
                'self.items = ["apple", "banana", "cherry"]\n'
                'return "Task completed successfully"'
            )
        ]
    )

    # Create agent and subscribe to reasoning events
    agent = SimpleAgent(llm=fake_llm)
    agent.event_manager.on("Reasoning", capture_reasoning)

    result = await agent.perform_task()

    assert result == "Task completed successfully"
    assert agent.message == "Hello from agent!"
    assert agent.count == 42
    assert agent.items == ["apple", "banana", "cherry"]
    # No Reasoning events are emitted — the builtin no longer exists
    assert reasoning_received == []


# ============================================================================
# Part 9: Variable Substitution - Testing {expression} expansion
# ============================================================================


class QueueStatusAgent(Agent, llm=_TEST_LLM):
    """Agent that reports status with variable substitution."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.queue = []
        self.processed = []
        self.is_processing = False

    def simple_status(self) -> str:
        """Get simple status without generation."""
        return f"Queue: {len(self.queue)}, Processed: {len(self.processed)}"

    @strategy(PurePythonStrategy())
    async def query_status(self, question: str) -> None:
        """
        Answer questions about queue status.

        The question is: {question}

        Explain the current status using variable substitution:
        - self.queue has the pending items
        - self.processed has the completed items
        - self.is_processing indicates if we're currently processing

        Use {len(self.queue)}, {self.queue}, {len(self.processed)},
        {self.processed}, and {self.is_processing} in your message.
        """
        ...


@pytest.mark.asyncio
async def test_variable_substitution():
    """Test that f-strings work in message() calls for variable expansion."""

    # Capture messages
    messages = []

    def capture_message(event):
        messages.append(event.content)

    fake_llm = FakeLLMClient(
        scripted_responses=[
            _resp(
                "message(\n"
                '    f"There is {len(self.queue)} item left in the queue: {self.queue}. "\n'
                '    f"{len(self.processed)} item has been processed: {self.processed}. "\n'
                "    f\"{'I am currently busy processing.' if self.is_processing else 'I am idle.'}\"\n"
                ")"
            )
        ]
    )

    # Create agent and subscribe to messages
    agent = QueueStatusAgent(llm=fake_llm)
    agent.event_manager.on("Message", capture_message)

    # Set up state
    agent.queue = ["task1"]
    agent.processed = ["task0"]
    agent.is_processing = True

    # Call method
    await agent.query_status("What's the status?")

    # Verify message was sent with f-string variable expansion
    assert len(messages) >= 1, f"Expected at least 1 message, got {len(messages)}"
    message = messages[0]

    # Should contain expanded values (f-strings work in generated code)
    assert "1 item left in the queue: ['task1']" in message, (
        f"Missing expanded queue info: {message}"
    )
    assert "1 item has been processed: ['task0']" in message, (
        f"Missing expanded processed info: {message}"
    )
    assert "busy processing" in message, f"Missing processing status: {message}"


if __name__ == "__main__":
    # Run tests manually
    print("Running notebook scenario tests...")
    print()

    print("1. Testing simple agent...")
    asyncio.run(test_simple_agent())
    print("   ✅ Simple agent works\n")

    print("2. Testing async agent...")
    asyncio.run(test_async_agent())
    print("   ✅ Async behavior works\n")

    print("3. Testing subagents...")
    asyncio.run(test_subagents())
    print("   ✅ Subagents work\n")

    print("4. Testing tools...")
    asyncio.run(test_tools())
    print("   ✅ Tools work\n")

    print("5. Testing message callback...")
    asyncio.run(test_message_callback())
    print("   ✅ Message callback works\n")

    print("6. Testing variable substitution...")
    asyncio.run(test_variable_substitution())
    print("   ✅ Variable substitution works\n")

    print("🎉 All notebook scenarios work with new API!")
