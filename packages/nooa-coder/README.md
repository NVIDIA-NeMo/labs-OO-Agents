# nooa-coder

`nooa-coder` holds the Session layer for NOOA interactive agents: a `Session`
that owns one agent and its turn loop, a `SessionRegistry` that keeps the tree
of sessions (a root and the children it delegates to), the agent-side port
(`self.session`) through which an agent creates and talks to children, and a
headless in-process host (`open_tree`).

Status: pre-release. The package is part of the workspace but is not published
yet, and its API may change until the hosts switch over to it. The coding
agent, the ACP adapter and the benchmark host move onto it in later changes.

The design is tracked in issue #388.
