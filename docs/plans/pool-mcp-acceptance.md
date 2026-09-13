# Pool-to-NOOA MCP forwarding test

The prepared workspace is `/localhome/local-pfurgale/dev/nooa-pool-mcp-test`.
Its `.poolside/settings.local.yaml` registers only one probe, `pool_probe`.
It launches this checkout's dependency-free `scripts/pool_mcp_probe.py` through
`uv`. No probe definition has been added to NOOA's workspace settings.

Pool documents project-local MCP configuration in
[its settings reference](https://docs.poolside.ai/settings-file-reference#mcp-servers).
The test determines whether the installed Pool version forwards that configuration
to a third-party ACP agent. ACP itself accepts `mcpServers` on both
[session creation and load](https://agentclientprotocol.com/protocol/v1/session-setup).

## Run

Use the same Pool `nooa-acp` server entry and NOOA credentials/configuration as the
previous acceptance tests. Record `pool --version`. In a terminal:

```bash
cd /localhome/local-pfurgale/dev/nooa-pool-mcp-test
export NOOA_MODEL='nvidia/zai-org/glm-5.3'
export NEMO_OO_LLM_CONFIG='/localhome/local-pfurgale/dev/.nooa/llm_config.yaml'
export NOOA_ACP_MCP_TRACE='/localhome/local-pfurgale/dev/nooa-pool-mcp-test/acp-mcp-handoff.jsonl'
env -u NEMO_OO_PROJECT_DIR pool --agent-server nooa-acp
```

Start a **new** session. Before sending any prompt, inspect the protocol handoff
from another terminal:

```bash
cat /localhome/local-pfurgale/dev/nooa-pool-mcp-test/acp-mcp-handoff.jsonl
```

The opt-in NOOA trace appends a `trace_started` record per server process and
records the names and transports sent in raw `session/new` and `session/load`
requests. It excludes commands, arguments, URLs, environment variables, headers,
and conversation content. Look at the records after the latest `trace_started`:

```json
{"pid": 123, "event": "session/new", "mcpServersField": "list", "servers": [{"name": "pool_probe", "transport": "stdio"}]}
```

If `servers` is empty, Pool did not pass the probe in that request. If
`mcpServersField` is `missing`, the field itself was absent. A missing trace file
does **not** establish either result: check that the configured server runs this
checkout and receives `NOOA_ACP_MCP_TRACE`. The trace is disabled when that
variable is unset. A file error is reported on server stderr and does not stop ACP.

If the handoff contains `pool_probe`, ask:

> Call `await self.pool_probe.probe(nonce="pool-test-1")` using the MCP tool
> already attached to you and show its result. Do not launch the server yourself,
> register it manually, or read its journal. If the tool is absent, report that.

The result should be JSON containing `event: "probe"`, `nonce: "pool-test-1"`,
a PID, and a generated `server_token`. In another terminal, inspect:

```bash
cat /localhome/local-pfurgale/dev/nooa-pool-mcp-test/probe.jsonl
```

A matching probe entry proves the tool call reached the server. Together with
the handoff trace, this verifies client forwarding and tool invocation. Multiple start
entries are normal: NOOA opens MCP connections for discovery and tool calls.
The token in the returned result must match the token in its probe entry.

Exit Pool, resume the session through Pool's picker, and repeat with nonce
`pool-test-2`. Check the `session/load` handoff record before prompting again.
This checks forwarding on session/load as well as session/new.

As a negative control, start native NOOA in the same directory and ask whether
`mcp.pool_probe` is in `self.skills.activated()`. It should be absent: the probe
was configured only in Pool. Do not remember it through workspace_settings
until this forwarding test is finished.

## Interpret a failure

- An agent asks for `/mcp approve pool_probe ...`: it is using NOOA's separately
  registered MCP configuration. Do not approve it for this test. Reading Pool's
  settings and manually registering the probe does not test ACP forwarding.
  Inspect the handoff trace and start a fresh session; do not infer the handoff
  from the agent's explanation.
- Empty handoff: from the test directory, check `pool mcp list` for `pool_probe`
  and record `pool --version`. The probe was not forwarded in that request.
- Handoff contains the probe, but no probe journal: report NOOA's startup
  warnings. The failure occurred after the client sent the server definition.
- Start entries but no probe entry: something launched the server, but the
  requested tool call has not reached it. Report NOOA's startup warnings and
  whether `mcp.pool_probe` appears in `self.skills.activated()`.
- Handoff contains the probe, with matching nonce and token in the probe journal:
  forwarding, discovery, and invocation worked.

This checkout tests the probe against the actual NOOA ACP adapter for new and
loaded sessions. The manual Pool 1.0.16 test below received empty MCP lists on
session/new, including with a CLI-visible global registration. Pool forwarding
on session/load has not been manually verified.

## Observed result: Pool 1.0.16

The user reported this handoff from the prepared workspace on 2026-09-13:

```json
{"pid": 484799, "event": "trace_started"}
{"pid": 484799, "event": "session/new", "mcpServersField": "list", "servers": []}
```

`pool --version` returned `1.0.16`; `pool mcp list` returned
`No MCP servers configured`. No MCP definition reached NOOA in this request.
The agent's earlier manual registration of `pool_probe` and request for NOOA
approval did not test client forwarding.

Pool's [MCP documentation](https://docs.poolside.ai/mcp-servers) explicitly
supports `.poolside/settings.local.yaml`, but the documented `mcp list` command
does not specify whether it includes project settings. Those initial observations alone
do not distinguish configuration discovery from missing external-agent forwarding.
The prepared directory is also not a Git repository; whether that affects Pool's
project discovery has not been established.

### Global configuration control

Add a distinct temporary server using Pool's CLI. This writes to the user's
`~/.config/poolside/settings.yaml`, making the probe available across projects
until removed. It does not register or approve anything in NOOA.

```bash
cd /localhome/local-pfurgale/dev/nooa-pool-mcp-test
pool mcp add pool_probe_global -- /usr/local/bin/uv run --no-project --no-config python \
  /localhome/local-pfurgale/dev/labs-OO-Agents-shared-interactive/scripts/pool_mcp_probe.py \
  --journal /localhome/local-pfurgale/dev/nooa-pool-mcp-test/probe.jsonl
pool mcp list
```

Once `pool_probe_global` is listed, repeat the launch above and inspect the new
`session/new` trace before prompting. If this name is present, global forwarding
works and project configuration discovery remains the question. If the list is
still empty, even the CLI-visible registration was not forwarded in that run.
If forwarded, use `self.pool_probe_global.probe` for the invocation/resume steps.

The user completed this control. `pool mcp list` reported one configured server,
`pool_probe_global`, with the expected command and arguments. Subsequent server
processes still recorded empty handoffs:

```json
{"pid": 487555, "event": "trace_started"}
{"pid": 487555, "event": "session/new", "mcpServersField": "list", "servers": []}
{"pid": 487826, "event": "trace_started"}
{"pid": 487826, "event": "session/new", "mcpServersField": "list", "servers": []}
```

This establishes that Pool 1.0.16 did not forward the CLI-visible global stdio
registration to the external NOOA ACP agent in these runs. The empty list was
captured before NOOA's session setup or MCP connection code ran. NOOA approvals
and tool discovery cannot explain an absent definition at that boundary.

After completing this control, remove only its temporary global registration:

```bash
pool mcp remove pool_probe_global
```

### Report draft

**Title:** Pool 1.0.16 sends an empty session/new mcpServers list to an external
ACP agent despite a configured global MCP server

**Reproduction:** Add the stdio probe with `pool mcp add` as above, confirm it
appears in `pool mcp list`, then start `pool --agent-server nooa-acp` with the
handoff trace enabled. Create a fresh session and inspect the incoming request
before sending any model prompt. NOOA advertises HTTP/SSE MCP support as well as
implementing the baseline stdio transport.

**Expected for this integration:** The configured probe is included in the
session/new `mcpServers` list, or Pool documents the configuration needed to
forward it to an external ACP agent.

**Actual:** The global registration is listed by Pool but `mcpServers` is an
empty list. The sanitized records above include independent process starts.

**Scope:** Observed with Pool 1.0.16, a local external NOOA ACP agent, and a stdio
MCP server. This does not establish behavior for other Pool versions,
transports, or session/load. The protocol permits a client to send no MCP
servers; this report concerns interoperability, not a malformed ACP request.

This report is a local draft; it has not been submitted to Poolside.

### Path for shared NOOA sessions

Configure MCP definitions in NOOA's shared workspace settings for both native
and ACP agents: register with `self.mcp.register(...)`, then persist with
`self.workspace_settings.remember_mcp(name)`. Approval remains separate; the
current first-time approval UI is the native TUI's `/mcp approve` command.
After approving the exact saved definition there, fresh native and ACP sessions
can auto-connect it. This path is covered by the automated native/ACP parity
test; it does not depend on Pool forwarding or reading Pool's private settings.
