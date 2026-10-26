# Round 1: prompt-only Atom state guidance

This fallback compares a short static instruction addition against an unchanged
runtime. **It does not enable persistent Python locals.**

## Revisions and scope

- Control: `68852a587388091f4354d5db0bb92fb6909d9a87`, based directly on
  `96397522436ed2ed03fe129f8cf9ea3c142118d7` plus the frozen O2 overlay commits,
  in order: `06d379d4`, `7b288a4ab`, `49640ac6b`, `6ab24e7f7`,
  `958e7a308`, `6a2be43f6`.
- Treatment: a child of that control commit. Its only production change is the
  additional static block in `AtomAgent`'s class docstring. The other additions
  are this document and `test_state_guidance.py`.
- The held persistent-locals candidate `1948fbb4` and its packaging branch are
  **not ancestors** of either arm. None of their namespace, factory, harness-mode,
  serialization or helper-refresh changes belong to this comparison.
- There is no new environment flag, constructor option, harness mode, or
  dependency change. Use different source revisions for the two arms and the
  same frozen runtime dependencies. O2 worker/budget behavior is identical.

## Exact treatment text

The addition is 194 whitespace-delimited words. Its canonical model-facing text
is the `State across turns:` paragraph through `Record durable facts, not handles.`,
with common docstring indentation removed (`inspect.cleandoc`), UTF-8 encoded,
LF line endings and **no trailing newline**.

SHA-256 of that text:
`60e8e48da66cbaa324877c9345210689011230b8cc6968ec61ee631c00b10989`

SHA-256 of the exact source insertion (four-space class-docstring indentation,
empty blank lines, including the final two LF characters):
`d37aba4912437caa4f9e06c894bb74c9ef40ae1e44a8d8898078af838561fb39`

The block explains that locals end with `handle`, small serializable facts and
stable IDs/names belong in `self.v`, and a named Shell must be re-looked up through
the same jobs manager after a turn. Its executable lookup/launch/send pattern
works both before and after a real Waiting boundary. It does not claim that a new
Shell retains the old environment, that a new command feeds stdin, or that jobs
survive session close/restart.

## Verification and prompt receipts

Always point subprocess imports at the selected checkout when reusing an existing
venv; the installed editable package paths may refer to another checkout:

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/nooa-atom/src:$PWD/packages/nooa-bench/src"
export UV_PROJECT_ENVIRONMENT=/path/to/existing/venv
uv run --no-sync pytest -q tests/strategies tests/runtime \
  packages/nooa-atom/tests/test_unified_agent.py \
  packages/nooa-atom/tests/test_atom_resume.py \
  packages/nooa-atom/tests/test_agent_factory.py \
  packages/nooa-bench/tests/test_gym.py
uv run --no-sync pytest -q packages/nooa-atom/tests/test_state_guidance.py
```

The focused tests verify the exact text/hash, executable code fence, actual
AtomHarness root/kept-child/one-shot-child model prompts, same bash PID/cwd/env
through Waiting and relookup, Todo IDs and small `self.v` facts, JSON snapshots
without live objects, and deliberate recreation of a cancelled named Shell.

The test module is also an executable cross-arm receipt probe. Run its treatment
file against either arm's PYTHONPATH, with `--expect-guidance present` for treatment
or `absent` for control, and `--output /path/to/receipt.json`. The receipt includes
actual imported source/revision, all three captured model prompts, exact guidance
text/hash when present, and named-shell boundary evidence. No live model is used.

## Limits

This tests executability and prompt delivery, not whether a live model follows the
advice or improves benchmark outcomes. No benchmark launch or outcome is claimed.
The controller's existing venv is not the packaged runtime; packaged imports and
smokes still need a separate gate. An exploratory all-Atom control run encountered
pre-existing CLI/ACP/environment and child-session failures; it is not an all-green
baseline. In particular `agent-client-protocol==0.11.0` in that venv lacks `acp.ws`,
so the websocket test file cannot collect. No dependency or runtime repair is part
of this prompt-only change.
