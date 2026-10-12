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
```
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
  --confcutdir="$BRIDGE_ROOT/tests" -q -s "$BRIDGE_ROOT/tests/test_external_atom.py"
uv build --no-sources --wheel --sdist --out-dir "$BUILD_DIR" "$BRIDGE_ROOT"
```
The two tests are adapted from the external `test_external_atom.py`: one checks
that Gym archives requirements text but not external wheels/Atom code; the other
runs the actual `InProcessNOOARunner` with a fake HTTP policy (two calls), native
root and delegated child, result/trajectory capture, policy URL/cookies and
131072 output-token propagation. Temporary requirements and trajectories stay
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
