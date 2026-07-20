# Test Suite

## Structure

### Root-level tests
Broad framework tests — metaclass behavior, event system, sandbox, error formatting, actor, decorator/strategy wiring, method definition, and module imports.

### `runtime/` — Runtime internals
Context building, event manager, code execution, hooks, pure Python executor/REPL, structured output executor, async deadlock prevention, span relationships, and truncation behavior.

### `strategies/` — Generation strategies
`CodeActStrategy`, `PurePythonStrategy`, `ReflexionStrategy`, `TemplateStrategy`, argument validation, return type validation, helper method manager, and `RuntimeServices`.

### `core_runtime/` — Task lifecycle
Task queuing and serialization, code caching (ONCE vs AGENT lifetime), execution, LLM client reuse, and implemented plan behavior.

### `coordinator/` — Code validation
AST validation, forbidden features, and retry logic.

### `external/` — Public API surface
Decorator semantics, stub layer, agent method requirements, provider configuration, and end-to-end notebook scenarios (gold standard for user-facing behavior).

### `integration/` — Cross-cutting tests
Nested generation, concurrent traces, hook failure traces, nested agent history, and CodeAct structured output.

### `edge_cases/` — Boundary conditions
Generation lock contention, nested generation, child agent edge cases, sandbox edge cases, signal edge cases, missing await detection, builtin shadowing, and agent initialization.

### `agents/` — Agent-level tests
Agent imports and summarization agent behavior.

### `capability/` — Capability routing
Router repeated runs and class method replacement.

### `evaluation/` — Evaluation backend
Unified backend with LLM tests.

### `onboarding/` — Model onboarding
Tests for evaluating model performance on framework capabilities (code generation, REPL behavior, validation retry, working context). Used for model onboarding optimization. See `tests/onboarding/README.md`.

### `tools/` — Built-in tools
Bash tool and file tool tests.

### `utils/` — Utility modules
`doc`, `logger`, `message`, and `task` utility tests.

### `performance/` — Performance benchmarks
Client creation overhead.

### `unit/` — Isolated unit tests
Mechanical checks schema.

### `fixtures/` and `helpers/`
Shared test fixtures and helper utilities.

---

## Running Tests

```bash
uv run pytest                          # all tests
uv run pytest tests/runtime/ -v        # single directory
uv run pytest tests/test_metaclass.py  # single file
uv run pytest -k "test_codeact" -v     # by name pattern
```
