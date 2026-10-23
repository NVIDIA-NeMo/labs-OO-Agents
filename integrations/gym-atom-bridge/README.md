<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->
# Gym–Atom bridge

A separate package, **not a core or uv workspace member**, for the regular
NeMo Gym NOOA runner. It uses native Atom `handle_batch` sessions with the
Gym-injected policy model, captures root and delegated-child hooks, returns
Atom's `TaskResult`, rejects alternate model construction, and closes sessions.
The bridge module is copied unchanged from the tested external adapter.

Tested source pair:
- Atom: `2de12dfe180596bfab717b1e1cd87847a1ab3e46`.
- Gym PR4088: `eeaf3743ac9c070e57e741ff0fb935f340fa7d6e`.

Configure the existing runner's invocation fields:
```yaml
agent_class: gym_atom_bridge:GymAtomAgent
invocation_adapter: gym_atom_bridge:invoke_atom
init_kwargs: {}  # or {turn_method: handle_orchestrate}
```
`init_kwargs.turn_method` picks the root session's turn method:
`handle_batch` (the default) or `handle_orchestrate` (needs an Atom that has
it); other values are rejected. Delegated children run the turn method Atom
gives them (`handle_work` in Atom `7f9a09dc1` and later), not this setting.

`init_kwargs.agent_spec` picks the session's agent class as `module:Class`
(or `path/to/file.py:Class`); the default is `nooa_atom.agent:AtomAgent`, so
omitting it keeps the behavior above. Delegated children inherit the spec.
For example, the unified-work experiment overrides only the class:
```yaml
init_kwargs:
  turn_method: handle_orchestrate
  agent_spec: nooa_atom.agent.unified:UnifiedAtomAgent
```
The spec is loaded with Atom's `nooa_atom.session.loader.load_agent_class`
when Gym constructs `GymAtomAgent`, so a malformed spec, an unimportable
module, a missing class or a class that is not an `InteractiveAgent` fails
with `AgentSpecError` (a `ValueError`) before any model call. The class must
also define the chosen `turn_method`; that is not checked at construction.

Compatibility: `agent_spec` needs only `load_agent_class` and
`SessionOptions.agent_spec`, which Atom has had since before this bridge.
Offline-tested (section below) against Atom `7f9a09dc14a31c8b71ca620d53c07332eb50cb3c`
(default and file-spec classes, both turn methods) and the unified-work spike
`9ea274d7218bb6448d25dd56b63f6616a4904973` (additionally
`nooa_atom.agent.unified:UnifiedAtomAgent` with `handle_orchestrate`), each
with Gym `eeaf3743ac9c070e57e741ff0fb935f340fa7d6e`. `UnifiedAtomAgent` exists
only in the spike; naming it with an Atom that lacks it fails at construction.
Gym supplies `nemo_gym` and `responses_api_agents.nooa_agent` in its task
runtime archive. This package deliberately does **not** depend on full Gym.
It declares `nooa`, `nooa-atom`, and the runner's `jsonschema` requirement;
`nooa-cli` is a transitive Atom dependency. The version alone does not pin
the unpublished Atom APIs: install all three candidate distributions from
one exact source commit.

## Commit-pinned source installation

After the parent publishes the candidate and additive bridge branch to
`furgalep/labs-OO-Agents`, generate an external runtime requirements file from
the bridge checkout (no guessed bridge SHA):
```sh
ATOM_COMMIT=2de12dfe180596bfab717b1e1cd87847a1ab3e46
BRIDGE_COMMIT=$(git rev-parse HEAD) # the committed additive bridge checkout
ARCHIVE=https://github.com/furgalep/labs-OO-Agents/archive
cat > /tmp/gym-atom-requirements.txt <<EOF
jsonschema>=4.25.1,<5
nooa @ ${ARCHIVE}/${ATOM_COMMIT}.tar.gz
nooa-cli @ ${ARCHIVE}/${ATOM_COMMIT}.tar.gz#subdirectory=packages/nooa-cli
nooa-atom @ ${ARCHIVE}/${ATOM_COMMIT}.tar.gz#subdirectory=packages/nooa-atom
gym-atom-bridge @ ${ARCHIVE}/${BRIDGE_COMMIT}.tar.gz#subdirectory=integrations/gym-atom-bridge
EOF
```
These are standard pip-compatible archive/subdirectory requirements; for local
package management use `uv pip install -r /tmp/gym-atom-requirements.txt` in a
fresh Python **3.13.14** environment. Select this file via Gym's existing
`runtime_requirements_file` configuration. Selection replaces the default
requirements text. Keep required runtime dependencies as shown. Source-archive
builds of the candidate use its declared fallback version without Git metadata;
provenance is the commit URL, not that version.

Publication/reachability is a prerequisite: local wheels and controller-only
paths in requirements are **not** copied by Gym's runtime archive. Install the
bridge in the server environment too if that execution mode imports it there.
No bootstrap or Gym source modification is required or included here.

## Focused offline checks

Use a pristine Gym checkout at the exact commit above. In a new environment,
install that Gym checkout with its test dependencies **only for offline tests**,
the three exact candidate distributions, this package, and its `test` extra.
Then, from the Gym checkout (absolute paths below are caller-provided):
```sh
GYM_ROOT="$PWD" PYTHONPATH="$PWD" OTLP_ENDPOINT=http://127.0.0.1:1/v1/traces \
  uv run --no-project --python "$OFFLINE_ENV/bin/python" python -m pytest \
  -c "$BRIDGE_ROOT/pyproject.toml" --rootdir="$BRIDGE_ROOT" \
  --confcutdir="$BRIDGE_ROOT/tests" -q -s -rs "$BRIDGE_ROOT/tests/test_external_atom.py"
uv build --no-sources --wheel --sdist --out-dir "$BUILD_DIR" "$BRIDGE_ROOT"
```
The first two tests are adapted from the external `test_external_atom.py`: one checks
that Gym archives requirements text but not external wheels/Atom code; the other
runs the actual `InProcessNOOARunner` with a fake HTTP policy (two calls), native
root and delegated child, result/trajectory capture, policy URL/cookies and
131072 output-token propagation. The rest check `init_kwargs` with the same
runner and one fake policy call: the root turn method, the default
`agent_spec`, a file-spec `AtomAgent` subclass with each turn method, the
unified agent (skipped when the installed Atom lacks it; set
`GYM_ATOM_BRIDGE_REQUIRE_UNIFIED=1` to make that a failure), and invalid specs.
The `-c`/`--rootdir` options keep pytest from adopting this repository's root
configuration, under which the tests import the checkout's own `nooa_atom`
instead of the installed candidate; check the installed paths with `-s`. Temporary requirements and trajectories stay
in pytest's temporary directory. No model, benchmark, or job is called.

The unchanged bridge writes complete native traces and session state to a fresh
`gym-atom-native-*` temporary directory and enables process-global tracing.
These can contain task text, generated code and tool output; treat them as
sensitive, retain them only under the rollout's access/cleanup policy, and do
not publish them. No credentials, private host paths, or runtime logs are
bundled in this package. Set/disable external tracing explicitly for offline
checks; the closed loopback endpoint above avoids inherited remote exports.

The empty-tools fixture does not verify Gym resource-method propagation: the
bridge constructs a fresh native Atom agent with the supplied LLM, not the
construction shell's resource methods.

Offline success is not remote sandbox/bootstrap, model, benchmark or artifact
reachability validation. This package adds no runtime profile or infrastructure.
