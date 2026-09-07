# Context View Implementation Plan

## Objective

Make `DefaultAgentView` the visible owner of default context assembly while preserving all public context, event, strategy, scoped, and skill interfaces.

## Plan

1. **Complete `CurrentCall`.** Snapshot the resolved model, provider, context window, context budget, strategy, and event query. Do not expose the LLM client or credentials.
2. **Extract generic helpers.** Add bounded `context_text()`, optional `evaluate_context_expression()`, context collection, and explicit budget utilities.
3. **Extract source helpers.** Move `ContextManager` materialization and `EventManager` selection behind public source-specific functions. Preserve overrides, disabled/protected blocks, dynamic cache updates, and event-query behavior.
4. **Move policy into views.** Implement ordering, skill composition, event placement, optional adaptation, and budgeting in `DefaultAgentView`. Implement legacy `Skill.context_block` through `DefaultSkillView` with agent-scoped expression evaluation.
5. **Reduce `ActorRuntime`.** Leave only call/view resolution, tuple collection, rendering, and LLM invocation. Remove `_assemble_default_context()` and other runtime-owned assembly policy after migration.
6. **Verify compatibility.** Run focused context/view tests, the full unit suite, all quickstarts, the capability pipeline, and live model tests. Compare rendered prompts for representative Predict, CodeAct, skill, scoped-context, and event-history cases.

## Non-goals

- No `BlockSpec` core type.
- No `AssembledContext` object or alias; use explicit tuples/sequences.
- No framework-level `PromptAdapter` stage.
- No content policy in block or provider formatters.
- No public API break for existing context, event, strategy, scoped, or skill usage.

## Completion criteria

- `DefaultAgentView` can be read, copied, and modified without private runtime calls.
- Custom views can reuse generic helpers or ignore manager-based sources entirely.
- Existing behavior and prompt order remain unchanged under the default view.
- The full verification matrix passes without framework or API errors.

## Status

Steps 1–5 are implemented. Focused tests, repository lint, and type checks pass. Five representative quickstarts pass, the one-sample Claude Haiku capability matrix passes 37/40 with no framework errors, and both custom and default view paths pass live on Nemotron Super v3.
