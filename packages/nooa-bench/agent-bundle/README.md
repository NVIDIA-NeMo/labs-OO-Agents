# Scaled Evals agent bundle

This directory is the version-independent template for the `nooa-bench`
Scaled Evals sidecar. Do not commit generated `descriptor.json`,
`source-lock.json`, or source trees here. A release job generates them from the
exact Git tag with:

```bash
uv run python scripts/prepare_nooa_bench_bundle.py \
  --version 0.0.10 \
  --source-ref v0.0.10 \
  --revision "$(git rev-parse v0.0.10^{commit})" \
  --output build/nooa-bench-agent
```

The output path must not already exist; this prevents an accidental recursive
delete when the command is run by hand.

The generated Docker context produces the sidecar image expected by Scaled
Evals. Its entrypoint validates the registered bundle identity and copies a
self-contained Python runtime to `/installed-agent` in the task sandbox.

The builder and final image both use Debian 11 Bullseye. This is intentional:
SWE-bench task images based on Ubuntu 20.04 provide GLIBC 2.31, while a
Bookworm-built Python runtime requires newer GLIBC symbols and fails before
`BenchAgent` starts.

GitHub Releases publish the built image to GHCR and attach its descriptor,
source lock, immutable image digest, and checksums. The GHCR image is not by
itself admitted to production Scaled Evals. Production promotion must mirror
the exact digest to the approved registry, apply the internal CI Toolkit
signature, validate the copied runtime, and register a new agent-bundle ID.

No model credential is needed to build the bundle. GitHub publication uses the
workflow's short-lived `GITHUB_TOKEN` with `packages: write`; the organization
may require a one-time package-creation or repository-linking change. Production
promotion separately needs the internal registry/signing identity and Scaled
Evals API access. A model credential is required only when a benchmark run is
submitted.
