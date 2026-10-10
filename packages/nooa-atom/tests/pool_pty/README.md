# Real Pool 1.0.16 form reproduction (opt-in)

This Linux PTY fixture uses the **actual current `pool_form_schema`**, an installed
Pool binary, a synthetic stdio ACP server and a fresh temporary HOME/XDG for every
case. No model calls, credentials, shared configuration, live sessions or MCP
servers are used. No proprietary binary is included. This is not part of the
mocked adapter test suite.

```sh
uv run --frozen python packages/nooa-atom/tests/pool_pty/run.py --pool /localhome/local-pfurgale/dev/pool
# Without --pool, looks on PATH; prints SKIP if absent.
```

The runner prints a new `/tmp/pool-final-mapper.*` artifact root. Each case saves
`case.json`, timestamped `wire.jsonl`, paired stage ANSI/screen projections,
`all.ansi`, `summary.json`, and isolated Pool logs under `state/`. `runtime.json`
records version/path/SHA-256. Waits are bounded (startup/form <=5s, keys 0.65s,
final response <=3s). Process groups are terminated/waited in `finally` with a
2-second SIGKILL fallback, including on assertions. Descendants receive SIGKILL
independently of leader exit, and `/proc` verifies no runnable group members
remain (zombies cannot run). Launch failures close both PTY descriptors;
absent groups are tolerated. Screens are a convenience
ANSI replay, not an independent screenshot oracle; consult raw ANSI/wire/logs.

## Cases and semantics

- Required strict picker: Ruby red/Ocean blue display, Down submits value `blue`.
- Required flexible picker: An apple/Mixed nuts display, Down submits `nuts`;
  a separate free-answer control types and submits `purple`.
- Intentional Escape decline: pane exists and awaits input, then `action:decline`.
- Optional strict/flexible and original seven-field mixed form: automatic decline
  **before any form-input bytes**, with `optional non-text property ... is unsupported`
  in the client log. Diagnostic probes explicitly bypass constructor validation
  via Pydantic `model_copy`/`model_construct` to recreate formerly valid descriptors;
  schemas still come from the actual mapper, not hand-crafted approximations.
- The normal constructors now reject both optional picker kinds with actionable
  guidance. This runner asserts that guard before launching cases.
- **Explicitly authored alternative**, not the original shape: replace optional
  pickers with optional `TextQuestion`s whose help lists titles/values. Seven
  questions plus review render; all three optional fields submit empty strings;
  required flexible custom text submits `pretzels`; count submits `not-a-number`.
- Optional text alternatives also accept `gold` and custom `custom-toast`.
  These are text inputs, not native optional pickers; domain validation remains
  the author's responsibility. No silent required-flag or widget-kind change.

Each supported case asserts there was no response before the first injected key,
exact content/action, expected screen markers, and no unknown ANSI controls.
Required picker controls also assert radio titles and the custom-text option
inside the pre-input question pane, not merely in echoed descriptors.
Unsupported cases inject no form keys and assert client diagnostics. The fixture
sends visible descriptor text before opening the form to demonstrate text
retention. Production event-bridge rendering/admission/replay is independently
covered by adapter tests; the synthetic server is not the full Atom runtime.
No automatic reask or validation loop is implemented.

## Results

Final reviewed run: `/tmp/pool-final-mapper.05bl9695`: **9/9 real UI cases passed** on Pool
1.0.16, SHA-256 `0c06c4f5a20f092193b9968172dc6c19fd90c1b2e22f53adebaeb97d6ac550d2`.
Before-change actual-mapper captures: `/tmp/pool-current-mapper.Jjnl4k` (five
cases). Required flexible picker without nested `type:string` already rendered;
optional picker rejection caused the original mixed failure. Its `after/` captures
were an interim automatic text-fallback experiment, **not the final design**;
that fallback was removed following review in favor of constructor rejection
and an explicitly authored text alternative.

The first final-run attempt `/tmp/pool-final-mapper.dqoy_aor` stopped after the
third case because searching raw ANSI for contiguous `purple` was unreliable
(incremental terminal updates). All groups were torn down. The final runner
checks reconstructed screens, includes reverse-index ANSI support, and reran
all nine cases from a fresh root successfully.

## Protocol limitation

Unsupported rendering and intentional Escape both return identical
`{"action":"decline"}` without a reason. Only the isolated experiment's input
schedule and client log distinguish them. Production must preserve the reported
action, not infer intent, rewrite it into a transport error or automatically
retry. Original descriptors remain visible/replayable text; a declined request
is not still pending for structured admission.
