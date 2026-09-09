# Explicit context-cache boundary implementation

## Objective

Implement the approved `docs/context.md` contract without changing rendered prompt content: views explicitly place cache boundaries; rendering preserves them; UnifiedLLM alone maps them to supported provider annotations. View-assembled requests with no boundary receive no implicit cache marker.

## Plan

1. Add a frozen, fieldless `CacheBoundary` to the public context IR and include it in `ContextItem`, collection validation, exports, and budget handling.
2. Make `DefaultAgentView` add at most one boundary after prefix/custom-skill/history items and before trailing blocks. Keep custom-view output untouched.
3. Preserve boundary positions through every block formatter, including `CachedBlockFormatter` and `PlainCodeActBlockFormatter`:
   - split/flush adjacent message coalescing at each boundary;
   - mark the preceding neutral `RenderedMessage` as ending a cacheable prefix;
   - place a boundary after a tool event after its complete tool-call/result expansion;
   - treat leading/consecutive boundaries without preceding content as no-ops.
   - segmenting prevents `PlainCodeActBlockFormatter` from merging `PythonOutput` across a boundary.
4. Have stock provider formatters preserve the neutral marker on the exact final wire item produced for that message. `AnthropicProviderFormatter` rejects system-boundary layouts its flattened system string cannot preserve; this does not add a new native Anthropic client path. UnifiedLLM consumes and removes the internal marker without mutating caller messages, then:
   - maps explicit boundaries to existing provider-specific `cache_control` representation where supported;
   - discards them without content changes where unsupported;
   - maps Completion boundaries at the exact message; Responses maps supported Anthropic message/input boundaries, treats extracted-instructions and non-Anthropic boundaries as unsupported no-ops, and preserves native and legacy tool transformations;
   - ignores configured role-based injection for view-assembled agent requests because `ActorRuntime.generate()` passes request-scoped `cache_control_injection_points=[]` at middleware, fast-path, and recovery calls, including requests with no boundary;
   - retains legacy role-based injection for direct UnifiedLLM callers.
5. Remove formatter content recovery: an incomplete `ToolCallEvent` raises `UnsupportedContextLayout` in every formatter. Audit CodeAct failure and cancellation paths; producer-created tool events must carry a result before control can return or persist, while genuinely incomplete external/history events fail visibly.
6. Correct context-budget accounting so replacement notices contribute their own token cost.
7. Add focused tests for type validation, default/custom placement, empty/no-trailing/system-only context, adjacent-role flushing, complete tool-event expansion, all block/provider formatters, native/legacy tool and multimodal inputs, Completion/Responses sync and async calls, supported/no-op provider paths, no implicit custom-view marker, mixed direct/agent calls, middleware/recovery, incomplete events, budgeting, and existing override/static-key behavior.
8. Treat the existing FakeLLM experiment as assembly parity: normalize only the exact internal marker and request-scoped empty injection option, with negative comparator tests proving similarly named user/schema/tool data remains significant. Compare an identifiable candidate commit with upstream `main` and the pre-boundary branch commit. Add mocked SDK-boundary tests for the actual prepared Completion and Responses payloads; do not normalize budget-fix content differences.
9. Run `uv run` formatting/lint/type commands from project configuration, focused suites, then the repository test suite with any exclusions recorded. Run live Predict, CodeAct tool, and skill quickstarts against `openai/openai/openai/gpt-5.6-terra` at NVIDIA internal inference with durable JSONL/OTel traces; inspect exact system/event messages, tool pairing, boundary translation, results, and error spans. Update the parity README/report with quantitative evidence.
10. Obtain GPT-6 Astra implementation approval, address findings, rerun affected checks, commit, and update the existing internal MR using `oe internal-mr --dry-run` then `oe internal-mr`; record reproducible evidence without credentials.

## Compatibility and edge cases

- `CacheBoundary` changes transport metadata only; it never renders text, affects token counts, or becomes evictable.
- Multiple explicit boundaries retain order; the default adds only one. A provider may ignore unsupported caching but may not move a boundary.
- Middleware sees and must preserve the internal marker when retaining the associated message; exact position remains attached to the message rather than stored as a fragile numeric index.
- Request-scoped legacy suppression is passed explicitly and never mutates shared client configuration; direct and agent calls may safely share one client.
- Existing direct UnifiedLLM cache-injection configuration remains supported outside view-assembled agent calls.
- Existing context-view precedence, skill compatibility, block placement, events, tracing, and prompt text remain unchanged.
