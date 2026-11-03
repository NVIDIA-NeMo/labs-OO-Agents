# Safety

CodeAct can execute Python chosen by an LLM. Treat that execution as untrusted
code, even when the prompt is trusted and the model is strong.

## The containment boundary is outside the Python process

NOOA validates generated syntax, blocks known-dangerous constructs, and can
restrict imports and calls. These controls reduce accidents and protect the
event loop. They are defense in depth, not a security sandbox.

Run code-executing agents in an operating-system isolation boundary appropriate
to the data and capabilities involved, such as a container, VM, or
[NVIDIA OpenShell](https://github.com/NVIDIA/OpenShell). Limit mounted files,
network access, credentials, CPU, memory, and wall-clock time.

Predict does not execute generated Python, but it can still produce unsafe
content or influence later application actions. Validate outputs and apply
authorization at the action boundary regardless of strategy.

## Expose capabilities deliberately

Generated code can call visible methods and tools. Give an agent only the
surface required for its task:

- prefer `read_customer(customer_id)` over a raw database connection;
- start filesystem tools in the current job directory;
- keep stateful tools per agent instance;
- hide administration, deletion, and credential-management operations;
- separate read and write roles when their authority differs.

A narrow wrapper is the model's intended capability interface and a useful
place to validate inputs and apply policy. It is not, by itself, an
authorization boundary.

For example, `ShellTools(cwd=job_dir)` sets the shell's starting directory and
bounds its direct path helpers, but `run()` launches an ordinary shell that can
use absolute paths or change directories. Restrict filesystem access with OS
mounts or sandboxing, not `cwd`.

## Keep secrets out of model-visible state

Hide secrets and internal clients:

```python
from typing import Annotated

from nooa import hidden

with hidden:
    from my_service import Client


class ServiceAgent(Agent, llm=llm):
    _client: Annotated[Client, hidden]
    api_key: Annotated[str, hidden] = ""

    def __init__(self, client: Client, api_key: str, **kwargs):
        super().__init__(**kwargs)
        # Both are already scoped to read-only access for this service.
        self._client = client
        self.api_key = api_key

    def lookup_customer(self, customer_id: str) -> dict:
        """Return the customer fields this task is allowed to use."""
        return self._authorized_request(f"/customers/{customer_id}")

    @hidden
    def _authorized_request(self, path: str) -> dict:
        return self._client.get(path, token=self.api_key)
```

In practice, expose a smaller public method that performs the one authorized
operation the model needs. Hiding a value removes it from the documented agent
surface, but generated Python can still try to access a guessed private name.
Enforce authority with least-privilege credentials and services, plus the
process isolation and secret-management controls described above.

## Keep untrusted data in the data channel

Method arguments are rendered as inputs. Do not interpolate raw user messages,
documents, repository contents, or tool output into docstring instructions.
This preserves truncation controls and reduces instruction/data confusion.

Treat retrieved text and MCP responses as untrusted data too. A remote tool's
output is not a trusted system instruction merely because it arrived through a
tool call.

## Verify effects in deterministic code

The model may propose an action or report that work succeeded. Python should
decide whether the external effect is acceptable:

```python
candidate = await agent.implement(request)
result = await shell.run("pytest -q")
if not result.success:
    raise RuntimeError("Verification failed")
return candidate
```

Use schemas and validators for properties of the returned value. Use
application code for filesystem state, test results, database writes, API
responses, permissions, and policy checks.

## Bound resources and failure modes

Production runners should set limits outside the model prompt:

- execution timeouts and process limits;
- maximum model turns and retry budgets;
- filesystem and network allowlists;
- output and trace retention limits;
- explicit cancellation and cleanup of child processes;
- human approval for consequential writes where appropriate.

Prompt instructions can explain policy, but they do not enforce these limits.

## Safety checklist

- Is generated code inside an OS-level sandbox?
- Does the agent have only the files, network, and credentials it needs?
- Are secrets and administrative APIs hidden or absent?
- Are stateful tools isolated per job?
- Are untrusted inputs passed as data rather than prompt instructions?
- Are external claims verified before results are accepted or persisted?
- Can the runner time out, cancel, and clean up the work?
- Are durable traces available for review?

## Continue

- [Tools and visibility](tools-and-visibility.md)
- [Orchestration](orchestration.md)
- [Tracing](tracing.md)
- Root [Quick Start safety note](../../README.md#quick-start)
