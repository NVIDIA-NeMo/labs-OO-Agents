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

Two agent variants are available through `nemo-harbor --agent-type`:

- `bench` — compact CodeAct baseline with automatic summarization and optional delegation.
- `rlm` — the same controller plus explicit context-isolated coding workers.

Apache-2.0 licensed.
