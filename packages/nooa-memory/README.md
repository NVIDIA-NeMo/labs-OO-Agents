# nooa-memory

Long-term memory subsystem for [NOOA](https://github.com/NVIDIA-NeMo/labs-OO-Agents)
agents. Installs the `nemo.memory` skill, which gives an agent persistent recall
across sessions backed by vector search.

```bash
uv add nooa-memory
```

> **Windows:** use [WSL 2](https://learn.microsoft.com/windows/wsl/install). Native Windows is not supported; see the [main README](https://github.com/NVIDIA-NeMo/labs-OO-Agents#installation).

The default vector backend is numpy-only (no extra dependencies). `sqlite-vec`
and `chromadb` backends are imported lazily if you install them yourself.

See the [main repository](https://github.com/NVIDIA-NeMo/labs-OO-Agents) for
documentation, and `examples/arc_agi_3` for a worked usage example.

Apache-2.0 licensed.
