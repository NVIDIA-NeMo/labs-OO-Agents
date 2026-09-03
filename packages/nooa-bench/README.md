# nooa-bench

Benchmark agent (`BenchAgent`) and Harbor runner for
[NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents). Reproduces the SWE-bench
and Terminal-Bench results from the NOOA tech report.

```bash
uv add nooa-bench
nemo-harbor --help
```

See the [main repository](https://github.com/NVIDIA-NeMo/labs-OO-Agents) for
documentation.

## Scaled Evals bundle

Every published `vX.Y.Z` GitHub Release builds and smoke-tests a Bullseye
`linux/amd64` Scaled Evals sidecar, then publishes it as:

```text
ghcr.io/nvidia-nemo/nooa-bench-agent:X.Y.Z
```

Use the immutable digest from the release's
`nooa-bench-agent-X.Y.Z-manifest.json`, not the mutable tag, when promoting a
bundle. The release also carries the generated descriptor, source lock, and
checksums. The image installs `/installed-agent/bin/nemo-harbor` and reports
the same version as the Python packages in that release.

The bundle uses Debian 11 Bullseye so its Python runtime works in older
SWE-bench task images with GLIBC 2.31. A Bookworm runtime requires newer GLIBC
symbols and fails before `BenchAgent` starts.

The GHCR image is a release candidate, not a production Scaled Evals
signature. To use it in production, mirror the exact digest to the approved
internal registry, sign it with the Scaled Evals CI Toolkit, run the installed
runtime validation, and register the resulting image digest as a new agent
bundle. See [the bundle template](agent-bundle/README.md) for local preparation
and the boundary between GitHub publication and production promotion.

Apache-2.0 licensed.
