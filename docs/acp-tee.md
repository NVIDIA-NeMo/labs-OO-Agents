# Recording ACP traffic with the tee

`nooa coder` speaks the Agent Client Protocol (ACP) as JSON-RPC over its
standard input and output. To see exactly what a client such as Pool sends
and what the server answers, record every frame with the tee. There are two
ways to run it.

## The two tees

**Inside the server** (`nooa coder --tee PATH`). The server records the frames
its ACP connection reads and writes, as parsed JSON-RPC messages. Use this
with `nooa coder` itself.

**Around any server** (`python -m nooa_coder.acp.tee --log PATH -- COMMAND...`). A relay
starts `COMMAND` as a child process and copies its standard input and output
unchanged, line by line, logging each line. Use this to record a client
against a different server, for example the older `nooa-acp`:

```bash
python -m nooa_coder.acp.tee --log ~/acp-old.jsonl -- nooa-acp --model my-alias
```

The relay forwards SIGTERM to the child, closes the child's standard input
when its own input ends, passes the child's standard error through, and exits
with the child's exit status.

Both write one JSON object per line:

```json
{"ts": 1790000000.123, "dir": "in", "frame": {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"...": "..."}}}
{"ts": 1790000000.456, "dir": "out", "frame": {"jsonrpc": "2.0", "id": 1, "result": {"...": "..."}}}
```

`in` is client to server, `out` is server to client, `ts` is Unix time in
seconds. A line that is not JSON is recorded as a string. The log is appended
to, and created with mode 0600: `session/new` carries the client's MCP server
definitions, including their environment variables and headers.

## Pool settings

Pool reads named agent servers from its settings file. The entry below runs
`nooa coder` from a checkout at `/localhome/local-pfurgale/dev/wt-p3` whose
environment was created with `UV_PROJECT_ENVIRONMENT=.venv-host uv sync
--all-extras`, and records the traffic with the in-server tee. Replace
`MODEL_ALIAS` with a model alias from your NOOA model configuration:

```yaml
agent_servers:
  nooa-coder:
    command: /usr/bin/env
    args:
      - UV_PROJECT_ENVIRONMENT=.venv-host
      - uv
      - run
      - --project
      - /localhome/local-pfurgale/dev/wt-p3
      - nooa
      - coder
      - --model
      - MODEL_ALIAS
      - --tee
      - /tmp/nooa-coder-acp.jsonl
```

Then start Pool in the repository you want to work on:

```bash
pool --agent-server nooa-coder
```

Sessions are stored in the workspace Pool was started in, under
`.nooa/sessions`, next to the sessions of the older `nooa-acp` server. Add
`--sessions-dir DIR` (or set `NOOA_SESSIONS_DIR`) to keep the sessions of all
workspaces in one directory.

Without `--tee` the server records nothing. To record a different server, put
`python -m nooa_coder.acp.tee --log PATH --` in front of its command in the same way
(with `uv run --project ...` in front when the server runs from a checkout).

The `command`/`args` shape follows Pool's `mcp_servers` entries; if the
installed Pool version names the fields differently, keep the command line and
adapt the keys. `/usr/bin/env` sets `UV_PROJECT_ENVIRONMENT` without relying on
an `env` field.

## What the log answers

The design (`notes/nooa-session-tree-design.md` §6) has six open questions
about Pool. Each is answered from the `in` frames Pool sends:

1. **Does Pool open more than one session per agent process?** Count the
   `session/new` and `session/load` requests between one `initialize` and the
   end of the log. More than one, with different `sessionId`s in the results or
   params, means several sessions share the process.
2. **Does it keep a background session working when the user switches?**
   Switch sessions while a prompt runs. If the first session's
   `session/prompt` request gets its response (an `out` frame with that `id`)
   after requests for the second session appear, and no `session/cancel` for
   the first was sent, Pool left it running.
3. **Does it render session updates that arrive outside a prompt?** The log
   shows only what was sent, so pair it with the screen: find
   `session/update` notifications (`out`) that fall between a prompt's
   response and the next `session/prompt` request (for example the
   `available_commands_update` sent just after `session/new`), and check
   whether Pool displayed them.
4. **Can a user send a prompt while one is open?** Look for a second
   `session/prompt` request for the same `sessionId` before the first one's
   response. If it appears, Pool allows it; `nooa coder` treats it as a steer.
5. **Does Pool set `clientCapabilities.elicitation.form`?** Read the params of
   the first `initialize` request.
6. **Does it render enum and free-text forms?** Ask the agent something that
   ends its turn with a question (with options, and without). The log shows the
   `elicitation/create` request (`out`) with its `requestedSchema`, and Pool's
   response (`in`): `accept` with `content`, `decline` or `cancel`. An
   `accept` whose `content` holds the chosen value means Pool rendered and
   submitted the form.
