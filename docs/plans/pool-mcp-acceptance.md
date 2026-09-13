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
env -u NEMO_OO_PROJECT_DIR pool --agent-server nooa-acp
```

Ask:

> Call `await self.pool_probe.probe(nonce="pool-test-1")` using the MCP tool
> already attached to you and show its result. Do not launch the server yourself,
> register it manually, or read its journal. If the tool is absent, report that.

The result should be JSON containing `event: "probe"`, `nonce: "pool-test-1"`,
a PID, and a generated `server_token`. In another terminal, inspect:

```bash
cat /localhome/local-pfurgale/dev/nooa-pool-mcp-test/probe.jsonl
```

A matching probe entry proves the tool call reached the server. Multiple start
entries are normal: NOOA opens MCP connections for discovery and tool calls.
The token in the returned result must match the token in its probe entry.

Exit Pool, resume the session through Pool's picker, and repeat with nonce
`pool-test-2`. This checks forwarding on session/load as well as session/new.

As a negative control, start native NOOA in the same directory and ask whether
`mcp.pool_probe` is in `self.skills.activated()`. It should be absent: the probe
was configured only in Pool. Do not remember it through workspace_settings
until this forwarding test is finished.

## Interpret a failure

- No journal: the probe has not started. From the test directory, check
  `pool mcp list` for `pool_probe`, then inspect Pool's `/logs` for the
  `mcpServers` field in session/new or session/load. Report server names only;
  other server definitions may include credentials.
- Start entries but no probe entry: something launched the server, but the
  requested tool call has not reached it. Report NOOA's startup warnings and
  whether `mcp.pool_probe` appears in `self.skills.activated()`.
- Matching nonce and token: forwarding, discovery, and invocation worked.

This checkout tests the probe against the actual NOOA ACP adapter for new and
loaded sessions. The installed Pool application's forwarding remains the part
verified by this manual test.
