# Shell services graded after agent completion

Shells normally own their process groups and remove background jobs on close.
A harness that grades services after agent completion can opt into an
invocation-scoped policy:

```python
from nooa.tools.shell_lifecycle import preserve_background_services

async with preserve_background_services() as shells:
    shells.adopt(agent.shell.session)  # If constructed before entering the scope.
    await agent.solve(task)
    await agent.aclose()
# Verify services here, then tear down their processes/sandbox.
```

Every BashSession created inside this async scope, including through a fresh
ShellTools instance, inherits preservation and is held until scope exit. This
also applies when a caller explicitly supplies `keep_background_on_close=False`:
the invocation owner chooses the lifecycle. Child asyncio tasks inherit the
scope; unrelated concurrent invocations do not. Threads/processes created
without context propagation do not inherit it.

Scope exit closes all owned Bash sessions through the existing preservation
path, including after an exception or cancellation. The shell exits and a
separate drainer keeps background output pipes readable. The services remain
owned by the external harness, which must clean them up after verification or
on an aborted episode. Scope exit does not itself kill services.

The destructor respects the preservation flag and terminates only Bash rather
than its process group. Outside an opted-in scope, default cleanup and timeout
recovery retain their existing behavior. Creating new shells from an inherited
scope after it has closed raises an error instead of silently orphaning them.

On Linux, timeout recovery discovers command descendants directly through
`/proc`, so task images do not need `ps`. Each PID is paired with its process
start time before signaling to protect against PID reuse. Earlier background
jobs and their descendants remain excluded from command-timeout cleanup.
Other platforms retain the `ps` fallback.

Integrations that track terminal model errors can use
`nooa.agents.summarization.summary_fork_active()` to identify best-effort
summary calls. A caught summary failure must not mark the parent rollout fatal,
and a successful summary must not clear an unresolved parent model failure.
