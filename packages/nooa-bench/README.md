# nooa-bench

Benchmark agent (`BenchAgent`) and Harbor runner for
[NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents). Reproduces the SWE-bench
and Terminal-Bench results from the NOOA tech report.

```bash
uv add nooa-bench
nemo-harbor --help
```

Agent registry keys:

| Key | Use |
|-----|-----|
| `bench` | Existing context-enabled BenchAgent default |
| `bench-codeact` | CodeAct baseline with context-management APIs hidden |
| `bench-codeact-acm` | CodeAct with `self.context`, `self.events`, and event-collapse guidance exposed |

For a minimal SWE-bench-style A/B run without training, use the Harbor config
at `examples/benchmarks/harbor_codeact_context_ab.yaml` and keep model, tasks,
attempt count, and runtime settings fixed across both agent keys.
For a small Long-Horizon Terminal-Bench subset, use
`examples/benchmarks/harbor_lhtb_codeact_context_ab_small.yaml`.

See the [main repository](https://github.com/NVIDIA-NeMo/labs-OO-Agents) for
documentation.

Apache-2.0 licensed.
