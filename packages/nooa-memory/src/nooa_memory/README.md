# `nooa_memory` — long-term memory for agents

An **opt-in, additive, brain-inspired long-term memory subsystem** for nooa
Agents. Toggle it on per agent; it changes no core framework code. Its defining
idea: **the agent authors and curates its own memories** through native tools —
rather than a harness extracting them behind the agent's back.

- Runnable demos + measured results: [`examples/memory_bench/`](../../../examples/memory_bench/)
  and the quickstart [`examples/quickstart/12_memory.py`](../../../../examples/quickstart/12_memory.py).

---

## Table of contents
1. [Why](#why) · 2. [Quick start](#quick-start) · 3. [Architecture](#architecture) ·
4. [Memory records & schema](#memory-records--schema) · 5. [The operations](#the-operations) ·
6. [Agent-facing tools](#agent-facing-tools) · 7. [Retrieval & scoring](#retrieval--scoring) ·
8. [Spontaneous retrieval](#spontaneous-retrieval-automatic-recall) · 9. [Reflection](#reflection-offline-consolidation) ·
10. [Forgetting](#forgetting) · 11. [Storage & embeddings](#storage--embedding-backends) ·
12. [Configuration](#configuration-reference) · 13. [Monitoring](#monitoring--debugging) ·
14. [What it's good for](#when-does-memory-help) · 15. [Module map](#module-map) · 16. [Limitations](#limitations)

---

## Why

A nooa agent is normally amnesic between method calls and sessions. This add-on
gives it durable memory across **long-horizon autonomous tasks** and **tasks
accreted over time**, modeled on how the brain uses memory.

| | other systems (Mem0, Zep, RAG) | **this system** |
|---|---|---|
| who creates memories | a harness pipeline extracts/stores them | **the agent** writes them via tools |
| instruction | implicit / external | the agent is **told it owns its memory** and given the schema (`MEMORY_SCHEMA_GUIDE`, injected at install) |
| refinement | system-side | the agent **`update_memory` / `forget` / `associate`** as it goes |

When `enabled=False` (or not installed), the agent is byte-for-byte unchanged — the
*additive guarantee* (regression-tested).

---

## Quick start

```python
from nooa import Agent
from nooa_memory import MemoryConfig, MemoryManager, MemoryToolsMixin

class MyAgent(MemoryToolsMixin, Agent, llm=my_llm):
    async def work(self, task: str) -> str:
        # Do {task}; recall with self.recall(...) and save with self.remember(...).
        ...

agent = MyAgent()
MemoryManager.install(agent, config=MemoryConfig(enabled=True))

# The agent now writes/reads its own memory; you can too:
agent.remember("Deploy with `make ship`.", type="skill", importance="HIGH")
hits = agent.recall("how do I deploy?")          # list[Memory]
report = agent.reflect()                          # offline consolidation
print(agent.memory_stats().summary())             # usage counters
```

Default config needs **no external services**: a local SQLite file + a
deterministic offline embedder. Point it at real models when you want (below).
`MemoryManager.install(...)` remains available as lower-level plumbing for hosts
that need manual lifecycle control, but normal users should enable `nemo.memory`
as a skill.

---

## Architecture

`MemoryManager.install(agent, config=...)` wires three surfaces onto the agent's
**existing** machinery (mirrors `agents/summarization.py`; **no core edits**):

```
                    ┌─────────────────────── your Agent ───────────────────────┐
   conscious tools  │  remember / recall / search / update_memory / forget /    │
   (MemoryToolsMixin)│  associate         ── visible via doc(self) ──            │
                    │                                                            │
   pre-turn inject  │  BeforeTurn ─▶ recall_for_context() ─▶ ContextManager     │  "spontaneous association"
                    │                  (dynamic block, configurable cadence)     │
                    │                                                            │
   write-on-event   │  EventManager.on("Error"/"Notification"/…) ─▶ remember()  │
                    │                                                            │
   post-task reflect│  intercept("agent_call", top-level) ─▶ reflect()          │  "consolidation"
                    └────────────────────────────┬───────────────────────────-─┘
                                                  │
                       MemoryManager  ──────────  owns:
                         • MemoryStore   (SQLite: records + graph + vectors)
                         • RetrievalEngine (hybrid recall + ACT-R scoring + k-hop)
                         • ReflectionEngine (merge · reconsolidate · abstract · prune)
                         • ForgettingEngine (decay + prune)
                         • Embedder      (hashing | litellm)
                         • VectorIndex   (numpy | sqlite-vec | chroma)
```

`install()` also injects `MEMORY_SCHEMA_GUIDE` (the "you own your memory + here's the
schema" instruction) as a static context block, and `register`s runtime
memory-events. `uninstall()` removes every hook and the instruction block.

---

## Memory records & schema

A memory (`schema.Memory`) is intentionally **loose** — only `id`, `type`,
`content`, `created_at` are required; everything else has sane defaults.

- **types** (`MemoryType`): `info` (semantic fact), `skill` (verified procedure),
  `episode` (a specific experience), `intent` (future reminder/TODO),
  `reflection` (insight distilled from episodes), `scratch` (transient working memory).
- **descriptors:** `importance` (verbal: CRITICAL/HIGH/MEDIUM/LOW/TRIVIAL), `salience`, `confidence`, `mood`, `strength`
  (spaced-repetition counter), `reinforcement_count`.
- **metadata:** `created_at`, `last_accessed_at`, `access_log`, `access_count`,
  `source_task_ref`, `related_files`, `chat_turn_ref`, `valid_from/valid_to`.
- **graph:** typed directed `edges` (`EdgeType`): causal `derived_from` / `created_by`
  / `causes`, plus `refines`, `supports`, `contradicts`, `related`, `precedes`,
  `part_of`, `triggers`.

The directed graph lets retrieval do **multi-hop associative spread** and lets
reflection record provenance.

---

## The operations

| operation | what | how it fires |
|---|---|---|
| **Encode (write)** | create a memory; dedup-on-write reinforces near-duplicates | conscious `remember()` + automatic event-driven |
| **Spontaneous association** | inject relevant memories into context each turn | `BeforeTurn` → dynamic block (cadence: `self_gated`/`per_task`/`every_turn`) |
| **Deliberate recall** | the agent looks something up | `recall()` / `search()` tools |
| **Reflection** | offline consolidation after a task | top-level `intercept("agent_call")` or `manager.reflect()` |
| **Forgetting** | decay + prune | online (lazy) + offline (in reflection) |

---

## Agent-facing tools

Mixed in by `MemoryToolsMixin`, enabled/disabled per `MemoryConfig.tools`:

| tool | signature | purpose |
|---|---|---|
| `remember` | `(content, *, type="info", importance="MEDIUM", tags=None, title=None) -> id` | write a new memory (dedup-on-write) |
| `recall` | `(query, k=5) -> list[Memory]` | associative recall (similarity + graph spread) |
| `search` | `(query, k=5) -> list[Memory]` | term/keyword recall (no graph hop) |
| `update_memory` | `(id, *, content=, importance=, type=, tags=) -> bool` | refine an existing memory (re-embeds if content changed) |
| `forget` | `(id) -> bool` | archive an obsolete/wrong memory |
| `associate` | `(a_id, b_id, relation="related") -> None` | add a directed graph edge |

The same operations are available programmatically on `agent._memory`
(`MemoryManager`), which also exposes `reflect()`, `recall_for_context()`,
`memory_stats()`, `log_summary()`.

---

## Retrieval & scoring

`recall(query)` runs (see `retrieval.py`):

1. **Hybrid candidates** — dense (embedding KNN) ∪ sparse (keyword) pools.
2. **ACT-R-style score** per candidate (each term min-max normalised, then weighted):
   ```
   rel = λ·cos(query, mem) + (1−λ)·context-overlap          # encoding specificity
   rec = σ( ln Σ_k (t_now − access_k)^−d )                  # ACT-R base-level activation
   imp = importance / 10
   score = α_rel·rel̂ + α_rec·reĉ + α_imp·imp̂
   ```
3. **Multi-hop spread** (`hops ≥ 1`) — activation propagates over graph edges with
   per-hop decay, surfacing memories linked to (but not directly similar to) the query.

On retrieval, memories are "touched" (recency + strength bumped) so hot paths
self-strengthen. Weights/decays/hops are all in `RetrievalConfig`/`ScoringWeights`.

---

## Spontaneous retrieval (automatic recall)

The agent does **not** have to call `recall()` to benefit from memory. Before each
turn, a `BeforeTurn` hook **synthesises a query from the conversation** ("the anchor"),
runs the **same full `recall()` pipeline** (dense ANN ∪ keyword → ACT-R score → graph
spread — *not* ANN-only), and injects the top hits into a context block the model reads
that turn (`manager.py` → `ContextManager.set_dynamic`). As the conversation moves, the
anchor changes and different memories surface — "spontaneous association".

**How to use it.** Nothing — it's automatic once memory is installed (`enabled=True`).
The recalled memories appear in the agent's context under the block:

```
## Recalled memories (associative)
- [info] prod DB is pg-east-1
- [skill] deploy with `make ship`
```

Use explicit `self.recall(query)` only when you want to *deliberately* fetch for a
specific query; spontaneous recall covers the passive "what's relevant right now" case.

**The anchor** is built by `SpontaneousConfig.query_strategies` (composable, de-duped):

| strategy | anchor it builds |
|---|---|
| `last_message` *(default)* | the most recent user-text message |
| `recent_events` | the last `recent_events_n` events, concatenated |
| `working_state` | recent `PythonOutput`/`LLMOutput` (the agent's scratch state) |
| `distilled` | LLM-distilled query — falls back to `recent_events` with no LLM |

If no anchor can be derived (e.g. empty turn) **nothing is injected** — it never
"retrieves from nothing". Spontaneous recall uses **`touch=False`**: passively surfaced
memories are *not* reinforced (only memories the agent deliberately `recall()`s get their
recency/strength bumped), so the ACT-R signal stays meaningful.

**How to configure** (`SpontaneousConfig`):

```python
from nooa_memory import MemoryConfig, MemoryManager
from nooa_memory.config import SpontaneousConfig

MemoryManager.install(agent, config=MemoryConfig(
    enabled=True,
    spontaneous=SpontaneousConfig(
        enabled=True,                       # set False to disable auto-injection entirely
        query_strategies=("last_message",), # anchor source(s), tried in order
        inject_cadence="self_gated",        # self_gated | per_task | every_turn
        top_k=5,                            # memories injected (0 -> retrieval.top_k)
        context_char_budget=2000,           # hard cap on the injected block
        context_block_key="recalled_memories",
        recent_events_n=5,                  # window for recent_events / working_state
    ),
))
```

- **`inject_cadence`** — `self_gated` (default): re-inject only when the derived anchor
  changes (cheapest); `per_task`: inject once per task; `every_turn`: refresh every turn.
- Turn it **off** with `spontaneous=SpontaneousConfig(enabled=False)` and rely on the
  agent's deliberate `recall`/`search` tools instead.

---

## Reflection (offline consolidation)

`manager.reflect()` (see `reflection.py`) runs an ordered, brain-inspired pipeline:

1. **merge** near-identical memories into one canonical record;
2. **reconsolidate** — cluster related memories and let an optional LLM `reconciler`
   resolve outdated/contradicted values (**keep-latest**: archive stale, store the
   current consolidated one);
3. **edge formation** — link memories whose embeddings are close;
4. **re-score** importance (salience + access-frequency aware);
5. **abstract** — an optional LLM `reasoner` turns clusters of episodes into compact
   `reflection` memories (the generative abstraction step);
6. **prune** — forget decayed memories.

Deterministic steps (1, 3, 4, 6) run with no LLM. The generative steps (2, 5) only
run when you pass `reasoner=` / `reconciler=` to `install()`.

**How to use it.** Two ways to trigger a pass:

- **Automatic (default):** with `ReflectionPolicy(trigger="post_task")`, a top-level
  `agent_call` interception runs `reflect()` after each task (only top-level calls, not
  nested subagent calls, when `only_top_level=True`).
- **Manual:** call `manager.reflect()` (or `self.memory.reflect()` via the skill) whenever
  you want — returns a `ReflectionReport` (`merged`, `reconciled`, `superseded`,
  `edges_added`, `rescored`, `created`, `pruned`).

**How to configure** (`ReflectionPolicy`):

```python
from nooa_memory import MemoryConfig, MemoryManager
from nooa_memory.config import ReflectionPolicy

MemoryManager.install(
    agent,
    config=MemoryConfig(enabled=True, reflection=ReflectionPolicy(
        enabled=True,
        trigger="post_task",        # post_task (auto) | manual
        only_top_level=True,        # nested subagent calls do NOT each reflect
        background=False,           # False = await inline (deterministic); True = create_task
        merge_threshold=0.95,       # cosine >= this -> merge duplicates
        edge_threshold=0.80,        # this <= cosine < merge_threshold -> add a 'related' edge
        recon_threshold=0.6,        # cluster radius for reconsolidation
        recon_max_cluster=6,
        max_episodes_per_reflection=50,
    )),
    reasoner=my_llm_reasoner,        # OPTIONAL — enables step 5 (abstraction)
    reconciler=my_llm_reconciler,    # OPTIONAL — enables step 2 (reconsolidation)
)
```

**LLM or not.** Without `reasoner`/`reconciler`, `reflect()` makes **zero LLM calls** —
the deterministic steps (merge, edges, re-score, prune) run over the already-stored
embeddings + ACT-R math (the skill installs this way by default). Wire the two callables
to enable the generative steps. They're plain callables you build from your LLM, e.g.:

```python
def my_llm_reasoner(episodes):                  # list[Memory] -> list[Memory] (new reflections)
    text = my_llm.call([{ "role": "user",
        "content": "Distil durable insights from these episodes:\n" + format(episodes)}]).content
    return [Memory(content=line, type=MemoryType.REFLECTION) for line in parse(text)]

def my_llm_reconciler(cluster):                 # list[Memory] (old->new) -> (current|None, [archive_ids])
    # ask the LLM which value is current; return the consolidated memory + ids to retire
    ...
```

See `examples/memory_bench/reflecting.py` (`make_llm_reasoner`) and `longmemeval.py`
(`make_llm_reconciler`) for working implementations.

**When does reflection help?** Empirically (real gpt-5.4):

| task shape | retrieval bottleneck? | reflect effect |
|---|---|---|
| synthesis of many scattered facts | yes (small `top_k` ≪ #facts) | **+50%** |
| aggregation / latest value, memories already fit budget | no | **+0%** (neutral) |
| one pinpoint fact | no | **−20%** (abstraction blurs it) |

Consolidation pays off **only when retrieval is the bottleneck** and the model
can't reconcile in-context. It's off by default for pure retrieval-QA.

---

## Forgetting

`forgetting.py`: every memory has a retention that **decays with time** (Ebbinghaus
`R = e^(−Δt/S)`), slowed by `strength` (retrieval boosts it). The offline prune
archives memories below threshold, with guards: never prune very young memories,
high-importance memories, or `protected_types` (default: `skill`). Archive (soft
tombstone) vs hard delete is configurable.

---

## Storage & embedding backends

**One SQLite file** holds records + the association graph + vector blobs (separate
from the agent's session DB; WAL). The vector index is pluggable behind one
protocol (`vector_backends.py`):

| `vector.backend` | what | extra dep |
|---|---|---|
| `numpy` (default) | exact brute-force cosine, in-process | none |
| `sqlite_vec` | ANN inside the same SQLite file | `sqlite-vec` |
| `chroma_embedded` | Chroma over a local dir | `chromadb` |
| `chroma_http` | a running Chroma server | `chromadb` |

All four return identical ranking; swap freely. Embeddings (`embeddings.py`):

| `embedding.backend` | what |
|---|---|
| `hashing` (default) | deterministic, offline, dependency-free (great for tests) |
| `litellm` | OpenAI-compatible `/embeddings` (e.g. `text-embedding-3-large` via the NVIDIA gateway) |

```python
MemoryConfig(
    enabled=True,
    vector=VectorConfig(backend="sqlite_vec"),
    embedding=EmbeddingConfig(backend="litellm",
                              model="openai/azure/openai/text-embedding-3-large",
                              endpoint="https://inference-api.nvidia.com/v1",
                              dimensions=1024),
)
```

---

## Configuration reference

`MemoryConfig` (frozen Pydantic; `.merge_with(**overrides)`) with nested sub-configs:

| group | key knobs |
|---|---|
| **top-level** | `enabled`, `path`, `tools`, `instruct`, `chunk_size/overlap` |
| **spontaneous** | `enabled`, `query_strategies` (`last_message`/`recent_events`/`working_state`), `inject_cadence`, `context_char_budget`, `top_k` |
| **retrieval** | `top_k`, `hops`, `n_dense`, `n_sparse`, `per_hop_decay`, `min_similarity`, `weights` (relevance/recency/importance/λ/decay/spread) |
| **embedding** | `backend`, `model`, `endpoint`, `api_key`, `dim`, `dimensions`, `batch_size` |
| **vector** | `backend`, `collection`, `host`, `port` |
| **write** | `on_events`, `salience_min`, `dedup_threshold`, `dedup_top_k`, `write_episodic` |
| **reflection** | `enabled`, `trigger`, `only_top_level`, `merge_threshold`, `edge_threshold`, `recon_threshold`, budgets |
| **forget** | `enabled`, `decay_half_life_hours`, `prune_activation_threshold`, `prune_min_age_hours`, `archive_vs_delete`, `protected_types` |

---

## Monitoring & debugging

Reuses the framework's existing observability (no new system):

- **logger** `nooa_memory` — `setLevel(logging.DEBUG)` for per-op traces.
- **events** on the agent's `EventManager`: `MemoryWritten`, `MemoryRecalled`,
  `MemoryInjected`, `ReflectionCompleted` (RUNTIME_EVENT role → visible to any telemetry
  subscriber, never injected into the LLM context).
- **counters** `manager.memory_stats()` → `MemoryStats` (writes, reinforced, recalls,
  recalled_items, injections, reflections, merged, edges_added, pruned, store_size);
  `manager.log_summary()` logs a one-liner.

```python
import logging; logging.getLogger("nooa_memory").setLevel(logging.DEBUG)
agent.event_manager.on("MemoryWritten", lambda e: print("wrote", e.memory_id, e.op))
```

---

## When does memory help?

Measured with real gpt-5.4 + text-embedding-3-large (see
[`examples/memory_bench/`](../../../examples/memory_bench/)):

| demo | what it isolates | result |
|---|---|---|
| `recall_qa.py` | unguessable cross-session facts | memory **+75%** (ON 75% / OFF 0%) |
| `locomo.py` | LoCoMo, agent-authored | memory **+44–60%** vs no-memory |
| `reflecting.py` | synthesis under a tight retrieval budget | reflection **+50%** |
| `longmemeval.py` | LongMemEval + reconsolidation | memory **+70%**, reflection neutral (+0%) |
| `locomo.py --reflect` | pinpoint lookup | reflection **−20%** |
| `memory_effect.py` | useful *and* detrimental | memory can backfire when stale |

**Principle:** memory helps when the answer isn't derivable in-context (past-session
or unguessable facts). Consolidation (reflection) is a deliberate tool — it helps
synthesis/aggregation under a retrieval bottleneck, is neutral when memories already
fit the budget, and *hurts* pinpoint lookups.

---

## Module map

| file | contents |
|---|---|
| `schema.py` | `Memory`, `MemoryType`, `Edge`, `EdgeType` |
| `config.py` | `MemoryConfig` + sub-configs (frozen Pydantic) |
| `embeddings.py` | `Embedder` protocol, `HashingEmbedder`, `LiteLLMEmbedder`, `get_embedder` |
| `store.py` | `MemoryStore` (SQLite: records + graph + vectors) |
| `vector_backends.py` | `VectorIndex` protocol + numpy / sqlite-vec / chroma impls + `make_vector_index` |
| `retrieval.py` | `RetrievalEngine` (hybrid recall, ACT-R scoring, k-hop), query strategies |
| `forgetting.py` | `ForgettingEngine`, `retention()` |
| `reflection.py` | `ReflectionEngine` (merge · reconsolidate · edges · re-score · abstract · prune), `ReflectionReport` |
| `monitoring.py` | `MemoryStats` + runtime memory events |
| `manager.py` | `MemoryManager` (install/hooks/ops) + `MemoryToolsMixin` + `MEMORY_SCHEMA_GUIDE` |

Tests: [`tests/memory/`](../../tests/memory/).

---

## Limitations

- **Agentic authoring costs LLM calls** (one per session/task) — the deterministic
  offline path (hashing embedder, no reasoner/reconciler) is free and good for tests.
- **Reconsolidation** only fires when contradictory memories cluster *and* the agent
  authored both values; a missing fact can't be reconciled.
- **Reflection's win** needs a real retrieval bottleneck (large/noisy store under a
  tight budget); on small stores a strong model reconciles in-context, so it's neutral.
- `numpy`/`sqlite-vec` backends are exact/brute-force — fine to ~10⁵ vectors; beyond
  that prefer a Chroma backend.

See [`examples/memory_bench/`](../../../examples/memory_bench/) for runnable
benchmarks and measured behavior.
