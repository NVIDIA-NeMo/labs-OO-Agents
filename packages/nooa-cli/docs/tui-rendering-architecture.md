# TUI rendering and interactive-agent architecture

Status: as-built ownership map for the fullscreen renderer and direct Python agent interface.

## Why this architecture exists

Fullscreen mode gives prompt_toolkit sole ownership of the terminal. `TUIApplication`
uses `prompt_toolkit.Application(full_screen=True)`, and transcript output enters its
semantic model instead of a second terminal writer. The explicit `native` and
`native-replay` modes remain operational escape hatches. No mode uses terminal-position
queries: CPR, private renderer-position mutation, and additional transcript writers are
prohibited.

## Display modes

The application exposes three restart-only modes:

| Mode | prompt_toolkit mode | Transcript owner | Resize behavior |
| --- | --- | --- | --- |
| `native-replay` | inline (`full_screen=False`) | ordered `run_in_terminal` consumer | clear and replay retained blocks |
| `native` | inline (`full_screen=False`) | ordered `run_in_terminal` consumer | no historical replay |
| `fullscreen` | alternate screen (`full_screen=True`) | application transcript model/window | invalidate and reproject; never replay terminal bytes |

Resolution precedence is CLI `--display-mode`, explicit `tui.display_mode`, the
older `tui.full_screen` setting, then the application-owned `fullscreen` default.
The boolean setting maps `true` to `native-replay` and `false` to `native`.

Fullscreen owns every visible cell for its complete lifetime. Ordinary transcript
traffic, logging, and exception diagnostics enter the ordered presentation path; they
may not use `run_in_terminal`, stdout/stderr, destructive screen or scrollback clears,
semantic replay callbacks, or CPR. ANSI received from an agent, model, tool,
subprocess, or exception is untrusted. In fullscreen it is parsed into styled text
with an allow-list of SGR attributes and safe hyperlink metadata; OSC clipboard/title
commands, DCS/APC, C0/C1 controls, cursor/erase commands, and malformed or incomplete
escapes are rendered visibly or discarded, never emitted as terminal control bytes.

## Current ownership map

| Responsibility | Owner and important seams |
| --- | --- |
| PTK layout, focus, key bindings, subviews | `TUIApplication` renderer shell |
| Transcript ordering and presentation | one ordered consumer feeding either the native sink or `FullscreenTranscriptModel` |
| Agent state and direct controls | structural `InteractiveAgent` interface |
| Local dispatch, callbacks, and worker lifecycle | `LocalAgentRunner`, outside renderer classes |
| Observation replacement and stale-callback rejection | `AgentController` |
| Session/storage/command integration | `Session` composition root plus narrow `TUIHostServices` |

`Session` coordinates concrete-agent storage, event-manager, shell, and command
integration. Concrete queue and callback details are isolated in `LocalAgentRunner`.
The renderer receives an `InteractiveAgent`; it does not receive concrete queues,
storage backends, shell objects, event managers, or transport protocol objects.

## Python-native boundary

The host-neutral interface lives in `nooa_cli.interactive` and uses structural typing:

```python
class InteractiveAgent(Protocol):
    @property
    def state(self) -> AgentState: ...
    def observe(self, callback, scheduler, on_terminated=None) -> Observation: ...
    def submit(self, text: str) -> bool: ...
    def interrupt(self) -> bool: ...
    def withdraw_pending_input(self) -> str | None: ...
    def stop(self) -> bool: ...
```

The boundary is ordinary Python, not a transport model. There are no public action
unions, receipts, capability strings, event envelopes, sequence numbers, reconnect
states, or replay handshakes. A future subprocess or remote implementation may be a
proxy with the same Python surface, but transport concerns stay private to that proxy.
`nooa_acp` remains an edge adapter and is not the native TUI contract.

`AgentState` and every nested value are immutable. The UI reads complete snapshots
rather than reducing a public event stream. A successful direct command means that the
operation was admitted, not completed; its state effect is visible through `state`
before the method returns. `stop()` requests owned lifecycle shutdown, while closing an
`Observation` only stops that frontend from observing.

## Observation and switching invariants

* `observe()` atomically registers against the latest state and schedules an initial
  delivery after releasing the agent state lock.
* Each observation retains only its latest pending state. Publications coalesce into at
  most one scheduled or running drain, callbacks are serialized, and an older snapshot
  cannot overwrite a newer one.
* A scheduler may execute inline or raise. Scheduler/listener failure closes only that
  observation, records `Observation.failure`, and invokes `on_terminated` exactly once.
  The controller enters a visible disconnected state and gates commands rather than
  continuing against a frozen snapshot. Agent locks are never held while invoking
  scheduler or listener code.
* `Observation.close()` is idempotent. It prevents queued callbacks from starting;
  a callback already running may finish. It never stops the agent.
* `AgentController` installs replacement observation and captured state transactionally.
  Setup failure leaves the old agent active. On success it publishes the new generation,
  closes the old observation, and rejects every late old-generation callback. A reserved
  transition rejects callback-issued commands without holding a routing lock while it
  waits for callback completion; observation cleanup failure cannot poison controller
  bookkeeping or later close/replace operations.
* Agent switching does not reconstruct the renderer or stop the previous agent merely
  because it is no longer observed. The composition root owns resources it creates and
  closes them exactly once.
* UI mutation is marshalled onto the prompt_toolkit owner loop. Background producers do
  not mutate renderer state directly.

## Fullscreen transcript and viewport invariants

The transcript model stores safe immutable presentation records with stable IDs. It
supports append, streaming replacement/finalization, clear/tag removal, and bounded
retention. Live fullscreen retention is capped at 10,000 records or a conservative
16 MiB resident-text budget, whichever is reached first; durable event history remains
a separate authority. The byte budget charges source, rendered replay expansion, model
ANSI/plain copies, and the model's bounded projection/format caches. Source blocks and
model records are evicted together, including after resize replay changes their size.
Projection wraps records for the current cell width and caches only derived data.

At the tail, append and resize continue following the tail. While scrolled up, the
viewport preserves a logical record ID plus visual-line offset across append and
geometry changes. Prefix eviction also adjusts record-local selection offsets when it
removes a synthetic joining separator, so wholly retained selections keep their exact
payload. Eviction and clear have deterministic fallback anchors.

Unicode width uses terminal cells and grapheme clusters, including CJK, emoji, and
combining marks. Source and export text always preserve valid Unicode. If a grapheme is
physically wider than a one-cell viewport, only the screen projection uses a narrow
ellipsis; emitting the original wide glyph would desynchronize the terminal cursor from
prompt_toolkit's screen model. Tiny terminals keep the composer usable without
prompt_toolkit's `Window too small` replacement becoming the UI.

Fullscreen width observations do not replay static records: the transcript model already
projects those records for the requested width. Width-sensitive semantic callbacks are
coalesced until resize activity settles and cache their latest width result. This keeps
the ordinary 10,000-record resize callback constant-time while retaining exact semantic
rerendering for Rich output that genuinely depends on width.

## Validation and rollout

Unit tests cover immutable state, observation scheduling/coalescing/failure/close races,
controller replacement and stale-callback filtering, direct command routing, local
lifecycle races, transcript projection, and terminal safety. Component tests inspect
prompt_toolkit screen cells. A composition test exercises `Session.run()` through the
real local runner, direct agent observation, renderer, and first submit/output cycle.
Fault-injection tests prove that independent teardown failures cannot skip later cleanup,
that the primary application exception wins, and that terminal restoration still runs.
A POSIX PTY test verifies alternate-screen entry and restoration on normal exit.

EOF, interrupt, startup-failure PTY cases and tmux/SSH resize soak tests remain rollout
work. Explicit native modes retain characterization coverage while fullscreen is the
resolved default.

## Explicit non-goals

This project does not provide remote transport, authentication, durable event replay,
multi-client arbitration, generalized arbitrary-call RPC, an ACP-driven native API, a
Rust frontend, or a Textual rewrite. PyO3 is reserved for measured pure-computation
hotspots, not terminal ownership.
