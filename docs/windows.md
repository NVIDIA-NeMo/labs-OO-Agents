# Native Windows support

The core SDK runs directly in Windows Python 3.12 and 3.13. The workflows below
use native Python and `uv`; they do not require WSL, Docker, or Rust.

## Supported scope

| Feature | Native Windows status |
| --- | --- |
| Agent classes, deterministic Python tools, and typed generation | Supported |
| CodeAct with the default `inprocess` execution backend | Supported |
| SQLite events, snapshots, session ownership, and recovery | Supported |
| Model registry and layered configuration reads/writes | Supported; files use UTF-8 |
| Credential file writes | Supported; protected Windows ACLs restrict access to the owner |
| Basic CLI commands and the trace viewer | Supported |
| `ShellTools` and persistent PowerShell sessions | Supported; PowerShell 7 preferred, Windows PowerShell 5.1 fallback |
| `RepoTools` definitions, references, and editable anchors | Supported; native Python search with optional Tree-sitter parsing |
| Shared `CodingAgent` search, edit, and test workflow | Supported with the `inprocess` backend and native PowerShell commands |
| `nooa-acp` / `nooa acp` protocol entry points | Supported; native commands, activity updates, cancellation, and durable session reload |
| Persistent Bash sessions | Require a Unix environment |
| Shell-based producer `monitor()` | Supported; each monitor owns its PowerShell session and child processes |
| The `sandbox` execution backend | Requires a supported Unix process model; enforced guards require Linux |
| Coding/benchmark workflows that depend on Bash or Linux isolation | Outside this Windows support scope |

In-process CodeAct executes Python with the application's permissions. It does
not provide OS isolation. Requesting the sandbox backend on native Windows
raises `SandboxUnavailable`, including when `sandbox.require=False`: disabling
guards cannot supply the missing `fork` process model. For sandbox execution,
run the agent and its dependencies in a suitable Linux environment.

## Work from this repository

In PowerShell, from the repository root:

```powershell
uv sync --frozen --python 3.12 --no-dev --group test --extra cli --extra viewer --extra tracing
uv run --no-sync python -c "from nooa import Agent; print(Agent.__name__)"
uv run --no-sync nooa --help
uv run --no-sync nooa config show
```

Use `--python 3.13` to select the other supported Python version. The `test`
dependency group installs the test runner independently of the benchmark and
evaluation development stack. Add only the extras required by the workflow;
`--all-extras` also selects optional integrations outside the native Windows scope.

Start the local trace viewer with:

```powershell
uv run --no-sync nooa start-dev --host 127.0.0.1
```

The default port is 5001. Visit [the local viewer](http://127.0.0.1:5001).

## Native shell commands

`ShellTools()` selects PowerShell on Windows and Bash on Unix. Commands and
`init_command` must use the selected shell's language; `shell.shell` exposes
`"powershell"` or `"bash"`. Explicit selection is available as
`ShellTools(shell="powershell")` or `ShellTools(shell="bash")`, but each backend
requires its native platform. Installing Git Bash does not provide the Unix
process APIs required by `BashSession`. `pwsh` or `powershell` must be on `PATH`.

```python
import asyncio
from nooa.tools import ShellTools

async def main():
    shell = ShellTools()
    try:
        result = await shell.run("$name = 'world'; Write-Output hello")
        print(result.stdout, result.returncode)
        print((await shell.run("Write-Output $name")).stdout)
        # The same working directory is used by read/replace/write_file.
        await shell.write_file("example.txt", "hello\n")
    finally:
        await shell.close()

asyncio.run(main())
```

Variables, functions, environment changes, and `Set-Location` persist across
commands. `cwd=` changes directory for one command; `stdin=` gives a native
child process UTF-8 bytes followed by EOF, without shell interpolation. Without
`stdin=`, native input receives EOF. PowerShell pipelines consume objects;
use `[Console]::In` when PowerShell itself needs to read the supplied input.
PowerShell formats command output as text, so native line endings can change.

`run()` returns `.stdout`, `.stderr`, `.returncode`, `.timed_out`, and `.shell`.
Large results use the framework's bounded head/tail output format.
`run_stream()` yields output as it arrives and finishes with a `StreamDone`.
When stopping iteration early, use `contextlib.aclosing()` to close the stream.
PowerShell searches do not produce `.matches` anchors; use `read()` to obtain
an editable `Match`, then `replace()`.

`ActivityShellTools` forwards the selected shell and per-command `cwd=` while
recording activity events. `CodingAgent` gives the model the selected shell's
language in its context, so commands use PowerShell on Windows. File reads,
writes, replacements, and activity diffs use UTF-8 regardless of the Windows
locale.

Timeouts return code 124 with `timed_out=True`; an ordinary native exit code
124 has `timed_out=False`. A timeout, cancellation, explicit `exit`, or early
stream closure retires the host and its unfinished child processes. The next
command restarts the host, restores the last confirmed directory, and reruns
`init_command`; shell variables and functions from the old host are lost.

Background processes started by completed commands survive command recovery
but belong to the session until `close()`. To transfer their lifetime to the
caller, explicitly use `keep_background_on_close=True` or
`preserve_background_services()`.
For a detached service use `Start-Process -WindowStyle Hidden`; PowerShell
`Start-Job` jobs depend on their host. Windows Job Objects manage descendants,
including orphaned processes; they do not provide sandbox isolation. `close()`
waits for the active command, so cancel that command first to interrupt it.

## Repository navigation and coding

`RepoTools` shares the coding agent's shell session and returns editable
`Match` anchors from `symbols()` and `refs()`. Windows searches source files
directly with Python, without requiring Git Bash or ripgrep. Bash sessions
retain their ripgrep fast path and fall back to Python if ripgrep is missing.

The Python traversal skips hidden directories and common generated directories
such as `node_modules`, `build`, and `dist` below the requested search root.
An explicitly requested root is still searched, including repositories inside
a `.codex` parent directory. This traversal does not interpret `.gitignore`.
Absolute Windows paths, spaces, quotes, brackets, and Unicode filenames work
with anchors; editing an anchor still targets its original file after the
shell changes directory.

Regex search is available with the basic CLI install. For AST-aware navigation,
including more precise reference matching, install the optional grammars:

```powershell
uv sync --frozen --package nooa --package nooa-cli --no-dev --group test --extra cli --extra viewer --extra tracing --extra ast
```

The shared coding workflow is verified through `CodingAgent.handle()`: search
for a definition or reference, edit the returned anchor, run a local test
command, and report its output and file diff through activity events. These
checks use scripted model responses and real tools; they do not require model
credentials.

## ACP clients and editors

For a source-free installation using built wheels, see
[Windows wheel bundles](windows-wheel-installation.md). That guide covers
clean installation, standalone acceptance checks, and pointing Zed at the
installed executable.

Install the ACP extra to expose both `nooa-acp` and `nooa acp`. From this
checkout, in PowerShell:

```powershell
uv sync --frozen --no-dev --group test --extra acp
$env:NOOA_MODEL = "my-model-alias"
uv run --no-sync nooa-acp
```

An ACP client launches this command and communicates over stdin/stdout;
starting it alone waits for protocol input. Use a configured model alias or
provider model name. See [ACP setup](../packages/nooa-acp/README.md) for editor
configuration.

The Windows acceptance tests launch both installed command entry points with
PowerShell 7 and 5.1. They use a local HTTP model responder while keeping the
model client, ACP transport, code execution, repository edits, and test commands
real. Tests cover Unicode and special-character paths, file diffs, stdout and
stderr, and successful or failed command status. Every stdout line must remain
a JSON-RPC message; ordinary diagnostics use stderr.

Cancellation checks wait until a native process tree has actually started,
then verify that it exits and the same session can run another prompt. Closing
a session preserves its history; loading it in the same or a restarted ACP
server replays Unicode messages and supplies that history to the next model
call. EOF shutdown releases live session resources.

For a separate check using a real configured model:

```powershell
uv run --no-sync python scripts/smoke_acp_coding.py --model my-model-alias --report .nooa/acp-live-smoke.json
```

This opt-in command makes real model requests. It creates a small temporary
repository, asks the agent to fix an arithmetic function through ACP, and
independently reruns the unchanged acceptance tests. Its JSON report includes
exit codes and protocol error counts, without credentials. Use
`--entrypoint nooa` to check the `nooa acp` entry point. Editor UI rendering and
editor-specific extensions are outside these protocol checks.

## Streaming command producers

`monitor()` uses native PowerShell commands on Windows and the system shell on
Unix. It yields complete stdout and stderr lines without a command deadline.
Concurrent writes to the two streams do not have a guaranteed combined order.
Use `aclosing()` when stopping iteration early:

```python
from contextlib import aclosing

from nooa.runtime.producers import monitor

async def watch_tests():
    async with aclosing(monitor("uv run --no-sync python -m pytest -q")) as lines:
        async for line in lines:
            print(line)
```

Each Windows monitor owns its session. By default, completion, cancellation,
and explicit stream closure release that session and its remaining child
processes without stopping other monitors. An explicit
`preserve_background_services()` scope transfers completed background services
to the caller, using the same ownership policy as `ShellTools`.

## Files and process ownership

SQLite sessions use OS locks on both platforms: `flock` on Unix and
`LockFileEx` on Windows. A second process cannot open or delete an active
session. The lock's owner record remains readable, and process exit releases
the lock. Retained `.lock` files are normal; their existence alone does not
mean the session is active.

Model registry updates use the same lock implementation, with waiting enabled
to serialize concurrent edits. Writes preserve the source file's line endings
and UTF-8 comments. Snapshot timestamps can tie on a fast sequence of writes;
insertion order breaks ties when selecting the latest snapshot.

Credential files use POSIX mode `0600` on Unix. On Windows a protected DACL
grants full file access to the owner without inheriting broader parent-directory
permissions. Permission setup happens before writing credentials to the staging
file, and a failure prevents publication. Windows `stat().st_mode` does not
describe these ACLs.

## Verification

The `windows-core` job in [CI](../.github/workflows/ci.yml) runs on native
Windows with Python 3.12 and 3.13 and UTF-8 mode disabled. It checks the SDK,
storage, model configuration, CLI, viewer requests, selected tracing contracts,
and real PowerShell 7/5.1 sessions without live model calls. Shell checks cover
state, UTF-8 input/output, exit codes, streaming, cancellation, and background
process ownership. Repository tests exercise Tree-sitter and regex fallbacks;
coding tests execute the full search, edit, test, and activity sequence. ACP
tests also cover installed entry points, protocol output, cancellation, and
session recovery through process restarts.

For a short local check after the installation above:

```powershell
uv run --no-sync python -m pytest -q --timeout=60 tests/test_platform_core.py tests/storage/test_session_locking.py tests/test_secrets.py
uv run --no-sync python -m pytest -q --timeout=60 tests/tools/test_powershell_session.py tests/tools/test_shell_session_platform.py packages/nooa-cli/tests/test_coding_activity.py
uv run --no-sync python -m pytest -q --timeout=60 packages/nooa-cli/tests/test_repo_tools.py packages/nooa-cli/tests/test_coding_workflow.py tests/runtime/test_producers.py src/nooa/runtime/tests/test_producers.py
uv run --no-sync python -m pytest -q --timeout=60 packages/nooa-acp/tests
```

The full selected suite is listed in the CI job. Symbolic-link tests skip when
Windows does not grant symlink creation; other filesystem errors remain failures.
Linux-only capability tests are separate from the Windows support contract.
