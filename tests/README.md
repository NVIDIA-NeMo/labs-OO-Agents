# Test Suite

## Structure

### Root-level tests
Broad framework tests: metaclass behavior, event system, sandbox, error formatting, actor, decorator/strategy wiring, method definition, media handling, storage compatibility, skill registry, visibility, and module imports.

### `agentdoc/` - Agent documentation and visibility
`doc(self)` rendering, visibility rules, referenced types, annotated parameters and fields, truncating `pformat`/`pprint`, metadata, protocols, and fixture modules under `agentdoc/fixtures/`.

### `agents/` - Agent-level behavior
Agent imports, render configuration, method summarization, and summarization agents.

### `atif/` - Agent Trace Interchange Format
ATIF schema validation, exporter state machines, event typing, nested traces, standalone entrypoints, and end-to-end CodeAct traces.

### `capability/` - Capability routing
Router repeated runs, class method replacement, scorer behavior, structured output scoring, and support agents/data under `capability/agents/` and `capability/data/`.

### `config/` - Configuration resolution
Execution, strategy, summarizer, blocked, and resolved configuration behavior.

### `context_blocks/` - Context block rendering
Context block models, renderers, scoped blocks, cached rendering, event formatting, stats, and truncation.

### `coordinator/` - Code validation
AST validation, forbidden features, and retry logic.

### `core_runtime/` - Task lifecycle
Task queuing and serialization, code caching (ONCE vs AGENT lifetime), execution, LLM client reuse, and implemented plan behavior.

### `edge_cases/` - Boundary conditions
Generation lock contention, nested generation, child agent edge cases, sandbox edge cases, signal edge cases, missing await detection, builtin shadowing, task wrappers, and agent initialization.

### `external/` - Public API surface
Decorator semantics, stub layer, agent method requirements, provider configuration, and end-to-end notebook scenarios (gold standard for user-facing behavior).

### `helpers/` - Shared support modules
Reusable helper modules for cross-module inheritance, OpenTelemetry assertions, and signature utilities. These are support files, not standalone test suites.

### `integration/` - Cross-cutting tests
Nested generation, concurrent traces, hook failure traces, nested agent history, CodeAct structured output, archival behavior, journal fanout, import round-trips, nested bug fixtures under `integration/nested_bug/`, and fixture data under `integration/fixtures/`.

### `onboarding/` - Model onboarding
Tests for evaluating model performance on framework capabilities (code generation, REPL behavior, validation retry, working context), with sample inputs under `onboarding/test_data/`. Used for model onboarding optimization. See `tests/onboarding/README.md`.

### `performance/` - Performance benchmarks
Client creation overhead.

### `provider_compat/` - Provider compatibility
Provider compatibility checks across model backends and API shapes.

### `runtime/` - Runtime internals
Context building, event manager, code execution, hooks, pure Python executor/REPL, structured output executor, async deadlock prevention, span relationships, truncation behavior, and runtime evaluation.

### `runtime/sandbox/` - Sandbox executor
Sandbox executor behavior, broker deadlines, CodeAct sandbox integration, guard enforcement, sandbox config, nested async proxies, read-only module state, teardown behavior, and implicit return handling.

### `storage/` - Storage backends
In-memory storage, snapshot markers, serialization, snapshot variables, and snapshot persistence.

### `strategies/` - Generation strategies
`CodeActStrategy`, `PurePythonStrategy`, `ReflexionStrategy`, `TemplateStrategy`, argument validation, return type validation, helper method manager, and `RuntimeServices`.

### `test_mcp/` - MCP client and tool integration
MCP client, tool calls, OAuth discovery, browser detection, exception descriptions, and timeout refresh behavior.

### `tools/` - Built-in tools
Bash tool and file tool tests.

### `trace_explorer/` - Trace explorer storage and queries
Trace explorer cache, client, database queries, filtered loading, OTLP span conversion, fast path loading, and OpenInference backcompat.

### `tracing/` - Tracing pipeline
Span context isolation, exporters, journal invariants, metadata, OpenInference conformance, OTLP probing, secret scrubbing, subprocess hooks, viewer attributes, and wire-format stripping.

### `unifiedllm/` - Unified LLM client
Provider detection, cache control, context windows, retries, HTTP config/logging, JSON parsing, schema sanitization, Responses client behavior, strict schema fallback, and tool schema cleanup.

### `unit/` - Isolated unit tests
Focused coverage for context vars, actor behavior, prompts, pragma replacements, skill loading, strategy behavior, utilities, and remaining coverage gaps.

### `utils/` - Utility modules
`doc`, `logger`, `message`, and `task` utility tests.

### `viewer/` - Trace viewer API and stores
Viewer main routes, journal and OTLP stores, OTLP ingest, stress coverage, image resolution, message resolution, and block export.

---

## Running Tests

### Prerequisites

- `ripgrep` (`rg`) and `grep` must be on `PATH`. The 145 tests in
  `tests/tools/test_shell_tools_modern.py` skipif-away without them, and a
  clean skip is indistinguishable from a pass in the summary line. Install
  with `apt install ripgrep` on Debian/Ubuntu or `brew install ripgrep` on
  macOS.

```bash
uv run pytest                          # all tests
uv run pytest tests/runtime/ -v        # single directory
uv run pytest tests/runtime/sandbox/   # nested runtime area
uv run pytest tests/test_metaclass.py  # single file
uv run pytest -k "test_codeact" -v     # by name pattern
```
