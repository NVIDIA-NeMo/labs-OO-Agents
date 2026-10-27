# Follow-ups outside the context-view PR

## Decorator context leaks across agent boundaries

**Pre-existing on main; not introduced by this PR.**

`@strategy(context=...)` overrides can reach a different agent called by the
parent. Method wrappers reset scoped context at subagent boundaries, but not
`_decorator_context_var`; the runtime subsequently inherits and merges it.
The summarizer currently has a local workaround.

Fix the shared subagent boundary: clear decorator context when entering a
different agent and restore it on exit, including exceptions and cancellation.
Keep inheritance within the same agent. Cover async, sync, and generator paths,
then remove the summarizer workaround if no longer needed.

Relevant code: `src/nooa/runtime/method_wrapper.py`,
`src/nooa/runtime/actor.py`, and `src/nooa/agents/summarization.py`.
