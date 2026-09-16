# Changelog

All notable changes to this project are documented here. The format is loosely
based on [Keep a Changelog](https://keepachangelog.com/), and the project aims
to follow semantic versioning.

## [Unreleased]

- Responses clients now honor the cached renderer's stable-prefix boundary by default,
  without a cache setting in the model registry. Requests without a usable boundary
  retain provider-default caching; `cache_breakpoint=None` opts out of NOOA markers.

- Security: the sandbox parent no longer unpickles worker bytes. Brokered `self.*`
  arguments, `self.x = value` assignments, cell return values and `return_result`
  payloads now cross as msgpack; rich values are rebuilt only from a fixed set of
  value types, numpy arrays, and the agent's declared pydantic models / dataclasses
  / enums (validated on the way in). Anything else is a `CellSerializationError`
  instead of code running in the parent. Adds the `msgpack` dependency.
- Breaking: custom CodeAct error formatters must implement
  `format(error, code=None, *, line_offset=0, max_error=None, tail_chars=None)`.
  Reduced legacy signatures are no longer supported.
- Breaking: sandboxed user-code failures are exposed as `SandboxExecutionError`;
  inspect `original_type`, `original_error`, and `diagnostic` for worker-side details.
- Add composable, context-scoped instrumentation hooks and trace-session scopes so hosts can observe NOOA execution without replacing native tracing.
- Initial public release of NVIDIA Object-Oriented Agents (NOOA).
- Security: MCP server configurations no longer expand host environment variables
  from `${VAR}` placeholders. Trusted caller code must resolve secrets and pass
  their values explicitly.
- Fixed: generator agent methods (`def`/`async def` containing `yield`) are now
  traced correctly. Their span previously covered only the *creation* of the
  generator, so LLM calls made by the body were recorded as children of whichever
  method drained it. Body calls now nest under the generator, and calls the
  consumer makes between yields do not.
- Breaking: a generator method with the `...` generation marker (including
  `yield ...`) now raises `TypeError` at class-creation time. Generation
  strategies commit one final result and do not define a stream protocol.
  Deterministic generators remain supported without `@strategy`.
