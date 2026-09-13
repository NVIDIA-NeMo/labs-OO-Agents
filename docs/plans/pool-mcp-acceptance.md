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
loaded sessions. The installed Pool application's forwarding remains the part
verified by this manual test.
