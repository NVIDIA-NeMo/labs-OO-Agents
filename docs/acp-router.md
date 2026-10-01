# The nooa-coder router and its workers

`nooa coder` serves the NOOA coding agent over the Agent Client Protocol (ACP)
on its standard input and output, or over WebSocket with `--http`. It runs in
one of four roles. A client such as Pool sees the same protocol in every role.

## Roles

**Router (the default).** `nooa coder [options]`. The router answers
`initialize` and `session/list` itself. Each `session/new` starts a worker
process, and the session runs there. `session/load` of a subagent's session
goes to the worker that runs its root, so a root and its subagents always
share one process. All other messages are forwarded, unchanged, to the
session's worker.

**Network (`--http`).** `nooa coder --http [--host H] [--port P] [options]`.
Clients connect over WebSocket at `ws://H:P/acp` (default
`ws://127.0.0.1:8765/acp`). Each connection gets its own router, with its
own workers, so a remote client gets what a client on standard input and
output gets. See "Serving over the network" below.

**Single process.** `nooa coder --single-process [options]`. All sessions run
in the one process that the client started. This is the P3 behaviour. Use it
to compare with the router or to debug.

**Worker.** `nooa coder [options] --worker-fd N --id-base B`. The router
starts workers; do not run this role by hand. A worker is a plain ACP server
on one end of a Unix socket pair (file descriptor `N`). Its requests to the
client (permission, elicitation, file and terminal requests) use ids from `B`
upwards, with `B = k << 32` for worker `k`. The router sends each reply to the
worker whose id range contains the reply's id.

The shared options are `--model` (or `NOOA_MODEL`, required), `--client-type`,
`--agent`, `--sessions-dir` (or `NOOA_SESSIONS_DIR`) and
`--tee`. The router starts
each worker by re-running its own command line (`sys.orig_argv`) with
`--worker-fd` and `--id-base` added. A worker therefore runs the same
installation with the same options. `--tee` on the router records the traffic
between the client and the router (see `docs/acp-tee.md`). Workers ignore it.
The MCP handoff trace (`NOOA_ACP_MCP_TRACE`) is also written by the router
only.

In the router and in single-process mode, the entry point keeps standard input
and output for ACP frames only. It duplicates the real descriptors for the
transport, points descriptor 0 at `/dev/null` and descriptor 1 at standard
error, so a stray `print` or a subprocess started from a cell cannot corrupt
or consume the client's stream.

## Where sessions live

Each workspace keeps its sessions in `<workspace>/.nooa/sessions`, the
directory the `nooa-acp` server and the TUI use, so sessions they wrote are
listed and can be loaded. A subagent's session file sits next to its root's.
`--sessions-dir DIR` (or `NOOA_SESSIONS_DIR`) puts the sessions of all
workspaces in one shared directory instead; each session records its
workspace, so `session/list` with a `cwd` still shows only that workspace.

The request's `cwd` picks the store: `session/new`, `session/load` and
`session/list` read and write the store of their `cwd`. There is no index of
every workspace yet, so `session/list` without a `cwd` lists the stores of
the workspaces the client has already named in this connection (in
`session/new`, `session/load`, `session/list` or a delete). The
`_nooa/session/delete` extension takes an optional `cwd`; without it, the
same named workspaces are searched, and the router passes the `cwd` from the
session's record on to the worker.

## Messages during a turn

There are three ways to send the agent text while a turn runs, following the
ACP RFD for message injection
([agent-client-protocol PR #1261](https://github.com/agentclientprotocol/agent-client-protocol/pull/1261)):

1. **`session/prompt`** queues the text. The running turn sees it as a pending
   user message in its queues context block and may take it; otherwise the
   next turn handles it. The request returns when the turn that took it
   ends, so two prompts return in order, each with its own stop reason.
   `session/cancel` answers a prompt whose text no turn took with
   `cancelled` and withdraws the text.
2. **`_nooa/session/inject`** with `{sessionId, mode: "queue", prompt}` (or
   `text`) queues the text and answers at once with
   `{messageId, delivered: "queued"}`. An injected message is not a prompt
   request, so `session/cancel` leaves it queued.
3. **`_nooa/session/inject`** with `mode: "steer"` hands the text to the
   running turn's next model call (`delivered: "steered"`), or queues it when
   no model call is coming.

`_nooa/session/revoke_inject` with `{sessionId, messageId}` takes back an
injected message that nothing has taken yet and answers `{revoked: true}`.
`initialize` advertises both in
`agentCapabilities._meta["dev.nooa/inject"] = {"queue": {}, "steer": {}, "revoke": {}}`,
from the router as well as in single-process mode.

Pool uses its own request. `initialize` also advertises
`agentCapabilities._meta["poolside/session_steer"] = true`, and Pool then sends
what the person types during a running prompt as `_poolside/session_steer`
with `{sessionId, inputId, prompt}` instead of keeping it in its own queue,
where Esc drops it. The text is queued for the next turn as in item 1, not
handed to the running turn's next model call. This is deliberate until the Pool team says
whether the request means queue or steer. The answer is `{inputId}`, sent as
soon as the message is admitted. A slash
command is refused with `invalid_params`, and Pool then sends it as a normal
prompt. The router routes the request by
`sessionId` like any other session request, so the prompt and the message are
handled by the same worker.

Pool keeps its turn open until the messages it handed over this way are
handled ("deferring turn close"), so the open `session/prompt` follows them:
it returns `end_turn` only after the turns that handle them, in order, with
their questions asked as for the prompt's own turn, except that a question is
not asked while a later message is still waiting: that message's turn comes
first, and the question stays as text in its context. A message the prompt's own
turn already took resolves with that turn. Stop withdraws the messages no turn
took (they would otherwise run with no prompt open) and lists them in one agent
message, "Stopped before these messages were handled"; the prompt answers
`cancelled`. A message sent this way with no prompt open is only queued.
`_nooa/session/inject` messages never hold a prompt open.

## Lifetime

- A worker runs in its own session and process group. Its standard input is
  `/dev/null`, and its standard output goes to standard error, so only the
  router writes to the client. Log lines from each process start with
  `nooa-coder router` or `nooa-coder worker k`. The router logs `worker k pid
  N started` for each worker. For each `session/new` and `session/load` it
  also logs `spawn_ms`, `handshake_ms` and `forward_ms`.
- Only the router stops workers. It stops a worker when the worker has no
  sessions left and no requests in flight: the root session was closed, a new
  or load request failed, or a session that was not open was deleted. To stop
  a worker, the router closes its end of the socket. The worker closes its
  sessions and exits with status 0. If the worker has not exited after 5
  seconds, the router kills its process group.
- If a worker exits unexpectedly, the client first receives every message the
  worker had already sent. Next, it receives a `session/update` of type
  `session_info_update` with `_meta.status = "worker_exited"` for each of the
  worker's sessions. Last, each of the worker's open requests fails with error
  `-32603`, except `session/close`, which succeeds.
- When the client closes the router's standard input, or the router receives
  SIGTERM, the router closes every worker's socket. The workers get 5 seconds
  in total to exit. Then the router kills every worker's process group and
  exits with status 0. A second SIGTERM kills the router at once.
- If the router itself is killed (SIGKILL), each worker detects that its
  parent process changed. It checks `os.getppid()` every 0.5 seconds, and
  then kills its own process group. This also works when a cell is keeping the
  worker's event loop busy.

## Serving over the network

`--http` follows the WebSocket profile of the ACP remote transport proposal
("Streamable HTTP & WebSocket Transport" in the ACP repository): one endpoint,
`/acp`, a WebSocket upgrade on it, and one JSON-RPC message per text frame,
`initialize` first. The proposal lets a server offer only WebSocket. The
Streamable HTTP profile (`POST` and server-sent events) is not served. The
`acp` Python library's WebSocket client works with it
(`acp.ws.create_websocket_stream`, with the `agent-client-protocol[http]`
extra).

Access:

- The token is read from `NOOA_CODER_TOKEN` and removed from the environment,
  so workers and the code they run do not see it. Clients send it as
  `Authorization: Bearer <token>`, or as `?token=<token>` where they cannot
  set headers (browsers). The server does not start without a token unless
  `--no-auth` is given, and `--no-auth` is refused on an address other than
  loopback.
- A request with an `Origin` header (browsers send one) is refused unless the
  origin is on loopback or given with `--allowed-origin`. So a web page cannot
  reach a server on this machine.
- The server binds to `127.0.0.1` by default and has no TLS. From another
  machine, use an SSH tunnel (`ssh -L 8765:127.0.0.1:8765 host`) or a proxy
  that terminates TLS.

Sessions are stored on disk as usual and outlive the connection: a client that
reconnects loads its sessions with `session/load`. The session lock keeps two
connections from running one session at once. Messages sent while a client
was disconnected are not replayed. SIGTERM or Ctrl-C closes every connection,
and each router stops its workers, which checkpoint their sessions.
`--single-process` and `--tee` do not work with `--http`.

The server accepts messages up to 50 MiB, as on standard input. The `acp`
library's WebSocket client keeps the `websockets` default of 1 MiB per
message, so a long transcript replayed by `session/load` can close its
connection (code 1009). A client that loads large sessions should raise its
limit (`max_size`).

```bash
export NOOA_CODER_TOKEN=$(openssl rand -hex 32)
uv run nooa coder --http --model MODEL_ALIAS
```

## Running it

From a checkout, with the environment created by `uv sync --all-extras`:

```bash
uv run nooa coder --model MODEL_ALIAS                    # router
uv run nooa coder --model MODEL_ALIAS --single-process   # one process
uv run nooa coder --model MODEL_ALIAS --http             # WebSocket, see above
uv run nooa coder --help                                 # lists the four roles
```

### Pool

Pool starts the router the same way it would start the single-process server.
This entry runs the router from the worktree `/localhome/local-pfurgale/dev/wt-p4`.
The worktree's environment was created with
`UV_PROJECT_ENVIRONMENT=.venv-host uv sync --all-extras`:

```yaml
agent_servers:
  nooa-coder:
    command: /usr/bin/env
    args:
      - UV_PROJECT_ENVIRONMENT=.venv-host
      - uv
      - run
      - --project
      - /localhome/local-pfurgale/dev/wt-p4
      - nooa
      - coder
      - --model
      - MODEL_ALIAS
      - --tee
      - /tmp/nooa-coder-acp.jsonl
```

Then run `pool --agent-server nooa-coder` in the repository you want to work
on. To compare with one process, add `- --single-process` to `args`.

To see the workers while Pool is connected:

```bash
ps -eo pid,ppid,pgid,stat,args | grep -e PGID -e 'bin/[n]ooa-coder'
```

Each root session has one line that contains `--worker-fd`. Its `PPID` is the
router's pid, and its `PGID` equals its own `PID`. The router's `PGID` is
different.

## Cost

Starting a worker costs about one Python start plus the `nooa` import. On the
development machine that was measured on, `session/new` took about 0.5 s in
one process and about 5 s through the router (`spawn_ms=9 handshake_ms=4430
forward_ms=504`). Almost all of that time is the new worker importing `nooa`.
If this cost matters, the next step is to keep one spare worker started in
advance.

## Not yet

- Workers are reached only through the socket pair their router created.
  Attaching a second client to a running worker needs a listening socket;
  that is stage 2 of the session tree design. The adapter already works
  from the Session's methods and its update stream only (the agent's events
  arrive as `AgentEventUpdate`), never from the agent, so a second client
  can be served from the same data.
- MCP over ACP (`mcpCapabilities.acp`) is not supported. Its messages are
  routed by connection id, not by session id.
